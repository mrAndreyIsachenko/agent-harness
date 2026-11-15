"""OpenRouter model routing policy.

Ported from `.tools/opencode_guard.py`: provider preference routing,
per-model reasoning / max_tokens caps, and the late-turn and error-recovery
reasoning budgets applied to chat completion payloads before dispatch.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path

DEFAULT_MODEL = 'openrouter/z-ai/glm-5.3-flash'
TEACHER_MODELS = {
    'qwen/qwen3-coder',
    'moonshotai/kimi-k2.5',
    'qwen/qwen-2.5-72b-instruct',
    'deepseek/deepseek-r1',
}
MODELS = {
    DEFAULT_MODEL,
    'openrouter/z-ai/glm-5.1',
    'openrouter/deepseek/deepseek-v4.1-flash-1017',
}
FP8_PROVIDER_ROUTING = {
    'quantizations': ['fp8', 'bf16', 'fp16'],
    'sort': 'throughput',
    'allow_fallbacks': True,
}
GLM_53_FLASH_PROVIDER_ROUTING = {
    'quantizations': ['fp8', 'bf16', 'fp16'],
    'order': ['z-ai', 'novita', 'siliconflow', 'gmicloud'],
    'ignore': ['baseten'],
    'allow_fallbacks': True,
}
GLM_51_PROVIDER_ROUTING = {
    'quantizations': ['fp8', 'bf16', 'fp16'],
    'order': ['chutes', 'z-ai', 'siliconflow'],
    'allow_fallbacks': True,
}
MODEL_PROVIDER_PREFS = {
    'openrouter/z-ai/glm-5.3-flash': GLM_53_FLASH_PROVIDER_ROUTING,
    'openrouter/z-ai/glm-5.1': GLM_51_PROVIDER_ROUTING,
    'openrouter/deepseek/deepseek-v4.1-flash-1017': FP8_PROVIDER_ROUTING,
}
DEFAULT_REASONING_MAX_TOKENS = 2048
LATE_TURN_REASONING_MAX_TOKENS = 768
LATE_TURN_START = 3
QUIET_LATE_TURN_REASONING_MAX_TOKENS = 512
ERROR_RECOVERY_REASONING_MAX_TOKENS = 1536
ERROR_SIGNAL_SUBSTRINGS = ('[AST PREFLIGHT ERROR', '[GUARD LOOP WARNING',
                           'Traceback (most recent call last)', 'SyntaxError',
                           'FAILED ', 'AssertionError')
GLM_51_REASONING_MAX_TOKENS = DEFAULT_REASONING_MAX_TOKENS
DEFAULT_MAX_TOKENS_CAP = 8192
GLM_51_MAX_TOKENS_CAP = DEFAULT_MAX_TOKENS_CAP
MODEL_REQUEST_CAPS = {
    'openrouter/z-ai/glm-5.3-flash': {
        'reasoning_max_tokens': DEFAULT_REASONING_MAX_TOKENS,
        'max_tokens_cap': DEFAULT_MAX_TOKENS_CAP,
    },
    'openrouter/deepseek/deepseek-v4.1-flash-1017': {
        'reasoning_max_tokens': DEFAULT_REASONING_MAX_TOKENS,
        'max_tokens_cap': DEFAULT_MAX_TOKENS_CAP,
    },
    'openrouter/z-ai/glm-5.1': {
        'reasoning_max_tokens': GLM_51_REASONING_MAX_TOKENS,
        'max_tokens_cap': GLM_51_MAX_TOKENS_CAP,
    },
    **{
        key: {
            'reasoning_max_tokens': DEFAULT_REASONING_MAX_TOKENS,
            'max_tokens_cap': DEFAULT_MAX_TOKENS_CAP,
        }
        for m in TEACHER_MODELS
        for key in (m, f'openrouter/{m}')
    },
}

DEFAULT_PROVIDER_HEALTH_TTL_SECONDS = 600.0


def _message_text(message):
    """Flatten one chat message to text for error-signal substring matching."""
    if not isinstance(message, dict):
        return ''
    chunks = []

    def _collect(value):
        if isinstance(value, str):
            chunks.append(value)
        elif isinstance(value, dict):
            for key in ('text', 'content'):
                child = value.get(key)
                if isinstance(child, str):
                    chunks.append(child)
                elif isinstance(child, list):
                    for part in child:
                        _collect(part)
        elif isinstance(value, list):
            for part in value:
                _collect(part)

    _collect(message.get('content'))
    return '\n'.join(chunks)


def select_escalated_model(
    messages,
    *,
    current_model: str | None,
    fallback_teacher_model: str | None = None,
    error_turn_threshold: int = 2,
) -> str | None:
    """Escalate to ``fallback_teacher_model`` when recent messages hit repeated error signals."""
    if not fallback_teacher_model or not isinstance(messages, list):
        return current_model
    window = max(8, error_turn_threshold * 3) if error_turn_threshold > 0 else len(messages)
    recent = messages[-window:] if window > 0 else []
    signals = (*ERROR_SIGNAL_SUBSTRINGS, '[SQL GUARD ERROR')
    error_count = sum(
        1 for msg in recent
        if isinstance(msg, dict) and any(sig in _message_text(msg) for sig in signals)
    )
    if error_count >= error_turn_threshold:
        return fallback_teacher_model
    return current_model


def apply_provider_routing(body, path, *, default_model=None, turn_index=1, reasoning_max_tokens=None,
                           degraded_providers=None, sticky_provider=None,
                           upstream_mode: str = "openrouter",
                           fallback_teacher_model: str | None = None) -> bytes:
    if upstream_mode not in ("openrouter", "vllm", "openai"):
        raise ValueError(f"unsupported upstream_mode: {upstream_mode!r}")
    if not body or not path.rstrip('/').endswith(('/chat/completions', '/responses')):
        return body
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, ValueError):
        return body
    if not isinstance(payload, dict):
        return body
    raw_model = payload.get('model') if isinstance(payload.get('model'), str) else default_model
    if not isinstance(raw_model, str) or not raw_model:
        return body
    changed = False
    selected_model = select_escalated_model(
        payload.get('messages'),
        current_model=raw_model,
        fallback_teacher_model=fallback_teacher_model,
    )
    if selected_model and selected_model != raw_model:
        payload['model'] = selected_model
        raw_model = selected_model
        changed = True
    canonical = raw_model if raw_model.startswith('openrouter/') else 'openrouter/' + raw_model
    if upstream_mode == "openrouter" and canonical in MODEL_PROVIDER_PREFS:
        desired_provider = json.loads(json.dumps(MODEL_PROVIDER_PREFS[canonical]))
        degraded = degraded_providers or set()
        if degraded and isinstance(desired_provider.get('order'), list):
            order = desired_provider['order']
            healthy = [p for p in order if p not in degraded]
            demoted = [p for p in order if p in degraded]
            desired_provider['order'] = healthy + demoted
        sticky = sticky_provider if isinstance(sticky_provider, str) and sticky_provider else None
        if sticky and sticky not in degraded:
            if isinstance(desired_provider.get('order'), list):
                order = desired_provider['order']
                healthy = [p for p in order if p not in degraded and p != sticky]
                demoted = [p for p in order if p in degraded]
                desired_provider['order'] = [sticky] + healthy + demoted
            else:
                # No explicit order (e.g. FP8 throughput sort): pin the provider by
                # replacing sort with a single-entry order so OpenRouter keeps the
                # KV-cached provider for subsequent turns.
                desired_provider['order'] = [sticky]
                desired_provider.pop('sort', None)
        if payload.get('provider') != desired_provider:
            payload['provider'] = desired_provider
            changed = True
    apply_caps = (
        canonical in MODEL_PROVIDER_PREFS
        or upstream_mode in ("vllm", "openai")
        or reasoning_max_tokens is not None
        or fallback_teacher_model is not None
    )
    if canonical in MODEL_REQUEST_CAPS and apply_caps:
        caps = MODEL_REQUEST_CAPS[canonical]
        if reasoning_max_tokens is not None:
            rmax = reasoning_max_tokens
        elif turn_index >= LATE_TURN_START and caps.get('reasoning_max_tokens'):
            base = caps['reasoning_max_tokens']
            messages = payload.get('messages')
            recent = ([m for m in messages[-2:] if isinstance(m, dict)]
                      if isinstance(messages, list) else [])
            error_hit = any(any(sig in _message_text(m) for sig in ERROR_SIGNAL_SUBSTRINGS)
                            for m in recent)
            last = recent[-1] if recent else None
            tool_result_last = False
            if isinstance(last, dict):
                if last.get('role') == 'tool':
                    tool_result_last = True
                elif isinstance(last.get('content'), list) and any(
                        isinstance(part, dict) and part.get('type') == 'tool_result'
                        for part in last['content']):
                    tool_result_last = True
            if error_hit:
                rmax = min(base, ERROR_RECOVERY_REASONING_MAX_TOKENS)
            elif tool_result_last and turn_index >= 4:
                rmax = min(base, QUIET_LATE_TURN_REASONING_MAX_TOKENS)
            else:
                rmax = min(base, LATE_TURN_REASONING_MAX_TOKENS)
        else:
            rmax = caps.get('reasoning_max_tokens')
        if rmax:
            if 'reasoning' not in payload:
                payload['reasoning'] = {'max_tokens': rmax}
                changed = True
            elif (isinstance(payload.get('reasoning'), dict)
                  and 'max_tokens' not in payload['reasoning']
                  and 'effort' not in payload['reasoning']
                  and payload['reasoning'].get('enabled') is not False):
                payload['reasoning'] = {**payload['reasoning'], 'max_tokens': rmax}
                changed = True
        mcap = caps.get('max_tokens_cap')
        if mcap:
            cur_max = payload.get('max_tokens')
            if cur_max is None or (type(cur_max) is int and cur_max > mcap):
                payload['max_tokens'] = mcap
                changed = True
    return json.dumps(payload).encode('utf-8') if changed else body


class ProviderHealthStore:
    """Persistent TTL store for degraded providers and sticky provider pins.

    JSON structure on disk::

        {'degraded': {provider: expiry_epoch},
         'sticky': {model_or_default: {'provider': name,
                                       'expiry': expiry_epoch}}}

    Thread-safe via an instance lock; process-safe via read-modify-write
    from disk plus atomic temp-file + ``os.replace`` persistence, so
    concurrent workers never observe a partially written store.
    """

    def __init__(self, path: str | Path, *, ttl_seconds: float = DEFAULT_PROVIDER_HEALTH_TTL_SECONDS):
        self.path = Path(path)
        self.ttl_seconds = float(ttl_seconds)
        self._lock = threading.Lock()

    @staticmethod
    def _now(now: float | None) -> float:
        return time.time() if now is None else float(now)

    def _load(self) -> dict:
        """Read the on-disk store, tolerating a missing or corrupt file."""
        try:
            with self.path.open('r', encoding='utf-8') as handle:
                raw = json.load(handle)
        except (OSError, ValueError):
            raw = {}
        if not isinstance(raw, dict):
            raw = {}
        degraded = raw.get('degraded')
        sticky = raw.get('sticky')
        return {
            'degraded': ({str(key): float(value) for key, value in degraded.items()
                          if isinstance(value, (int, float))}
                         if isinstance(degraded, dict) else {}),
            'sticky': ({str(key): value for key, value in sticky.items()
                        if isinstance(value, dict)}
                       if isinstance(sticky, dict) else {}),
        }

    def _persist(self, state: dict) -> None:
        """Atomically write the store: temp file in the same dir + replace."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(self.path.parent),
                                        prefix=self.path.name + '.', suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as handle:
                json.dump(state, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def active_degraded(self, *, now: float | None = None) -> set[str]:
        """Providers still inside their degradation TTL; prunes expired rows."""
        with self._lock:
            moment = self._now(now)
            state = self._load()
            expired = [p for p, expiry in state['degraded'].items() if expiry <= moment]
            if expired:
                for provider in expired:
                    state['degraded'].pop(provider, None)
                self._persist(state)
            return set(state['degraded'])

    def mark_degraded(self, provider: str, *, now: float | None = None) -> None:
        """Record ``provider`` as degraded for the TTL and drop its sticky pins."""
        with self._lock:
            moment = self._now(now)
            state = self._load()
            state['degraded'][provider] = moment + self.ttl_seconds
            stale = [key for key, entry in state['sticky'].items()
                     if isinstance(entry, dict) and entry.get('provider') == provider]
            for key in stale:
                state['sticky'].pop(key, None)
            self._persist(state)

    def get_sticky(self, model: str | None = None, *, now: float | None = None) -> str | None:
        """Sticky provider for ``model`` (or the default key), unless degraded."""
        with self._lock:
            moment = self._now(now)
            key = model if model else '_default'
            state = self._load()
            entry = state['sticky'].get(key)
            if not isinstance(entry, dict):
                return None
            if float(entry.get('expiry', 0)) <= moment:
                state['sticky'].pop(key, None)
                self._persist(state)
                return None
            provider = entry.get('provider')
            if not isinstance(provider, str) or not provider:
                return None
            if any(p == provider and expiry > moment
                   for p, expiry in state['degraded'].items()):
                return None
            return provider

    def set_sticky(self, provider: str, model: str | None = None, *,
                   now: float | None = None) -> None:
        """Pin ``provider`` for ``model`` (or the default key) if it is healthy."""
        with self._lock:
            moment = self._now(now)
            state = self._load()
            if any(p == provider and expiry > moment
                   for p, expiry in state['degraded'].items()):
                return
            key = model if model else '_default'
            state['sticky'][key] = {'provider': provider,
                                    'expiry': moment + self.ttl_seconds}
            self._persist(state)
