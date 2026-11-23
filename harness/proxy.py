"""OpenRouter guard proxy.

Ported from `.tools/opencode_guard.py`: a local HTTP proxy between the
OpenCode CLI and OpenRouter that gates dispatch, runs AST preflight and
duplicate-tool-call loop detection on the payload, applies provider routing,
and tracks per-turn cost / generation ids / provider health.
"""

import ast
import http.server
import json
import os
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path

from .guards import (
    check_allowed_files_syntax,
    check_cot_and_slop_integrity,
    check_egress_payload,
    check_readonly_sql_tool_calls,
    collect_python_top_symbols,
    compact_context_window,
    compact_stale_reads,
    detect_duplicate_tool_calls,
    detect_edit_oscillation,
    enforce_context_budget,
    inject_edit_failure_hints,
    inject_read_guard_hints,
    prune_noisy_tool_outputs,
)
try:
    from .routing import MODEL_PROVIDER_PREFS
except ImportError:  # pragma: no cover - routing module without provider prefs table
    MODEL_PROVIDER_PREFS = ()
from .routing import ProviderHealthStore, apply_provider_routing
from .telemetry import HarnessTracer

OPENROUTER_UPSTREAM = 'https://openrouter.ai/api/v1'
GEN_ID_RE = re.compile(rb'"id"\s*:\s*"(gen-[A-Za-z0-9_-]+)"')
STREAM_COST_RE = re.compile(rb'"usage"\s*:\s*\{[^{}]*"cost"\s*:\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)')
PROMPT_TOKENS_RE = re.compile(rb'"prompt_tokens"\s*:\s*(\d+)')
COMPLETION_TOKENS_RE = re.compile(rb'"completion_tokens"\s*:\s*(\d+)')
PROVIDER_LATENCY_THRESHOLD_SECONDS = 25.0


def _effective_latency_threshold():
    """Prefer the guard module's threshold so tests monkeypatching
    opencode_guard.PROVIDER_LATENCY_THRESHOLD_SECONDS take effect."""
    og = sys.modules.get('opencode_guard')
    if og is not None and hasattr(og, 'PROVIDER_LATENCY_THRESHOLD_SECONDS'):
        return og.PROVIDER_LATENCY_THRESHOLD_SECONDS
    return PROVIDER_LATENCY_THRESHOLD_SECONDS


class GuardError(Exception):
    pass


class OpenRouterGuardProxy(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, gate, *, upstream_url=OPENROUTER_UPSTREAM, request_timeout=300, default_model=None,
                 reasoning_max_tokens=None, allowed_files=None, health_store=None, tracer=None,
                 allowed_models: Iterable[str] | None = None, enforce_egress_guard: bool = False,
                 enforce_readonly_sql: bool = False, host: str = '127.0.0.1', port: int = 0,
                 upstream_mode: str = 'openrouter', fallback_teacher_model: str | None = None,
                 max_context_tokens: int | None = None,
                 sml_results_dir: str | Path | None = None):
        self.gate = gate
        self.upstream_url = upstream_url.rstrip('/')
        self.request_timeout = request_timeout
        self.default_model = default_model
        self.reasoning_max_tokens = reasoning_max_tokens
        self.allowed_files = list(allowed_files or [])
        self.allowed_models = list(allowed_models) if allowed_models is not None else None
        self.enforce_egress_guard = bool(enforce_egress_guard)
        self.enforce_readonly_sql = bool(enforce_readonly_sql)
        if upstream_mode not in ('openrouter', 'vllm', 'openai', 'llamacpp'):
            raise ValueError(f'unsupported upstream_mode: {upstream_mode!r}')
        self.upstream_mode = upstream_mode
        self.fallback_teacher_model = fallback_teacher_model
        self.max_context_tokens = max_context_tokens
        self.sml_results_dir = sml_results_dir
        self.baseline_symbols = collect_python_top_symbols(self.allowed_files)
        self.health_store = health_store
        self.tracer = tracer if tracer is not None else HarnessTracer()
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        self._turn_count = 0
        self._dispatched = []
        self._blocked = []
        self._blocked_reason = None
        self._auth_bearer = None
        self._compacted_reads = 0
        self._edit_hints_injected = 0
        self._cot_slop_warnings = 0
        self.degraded_providers = (set(health_store.active_degraded())
                                   if health_store is not None else set())
        self.sticky_provider = (health_store.get_sticky(default_model)
                                if health_store is not None else None)
        super().__init__((host, int(port)), _OpenRouterProxyHandler)

    @property
    def base_url(self):
        return f'http://{self.server_address[0]}:{self.server_address[1]}/api/v1'

    @property
    def auth_bearer(self):
        with self._lock:
            return self._auth_bearer

    @property
    def blocked_reason(self):
        with self._lock:
            return self._blocked_reason

    def reset_session(self) -> None:
        """Reset per-session turn and block state while preserving billing and sticky provider."""
        with self._lock:
            self._turn_count = 0
            self._blocked_reason = None

    def summary(self):
        with self._lock:
            return {'dispatched_count': len(self._dispatched),
                    'blocked_count': len(self._blocked),
                    'blocked_reason': self._blocked_reason,
                    'dispatched': [dict(d) for d in self._dispatched],
                    'blocked': [dict(b) for b in self._blocked],
                    'spans': list(self.tracer.spans),
                    'compacted_reads': self._compacted_reads,
                    'edit_hints_injected': self._edit_hints_injected,
                    'cot_slop_warnings': self._cot_slop_warnings}

    def task_summaries(self) -> dict[str, dict]:
        with self._lock:
            dispatched = [dict(d) for d in self._dispatched]
            blocked = [dict(b) for b in self._blocked]

        grouped: dict[str, dict] = {}

        def _bucket(task_id: str) -> dict:
            if task_id not in grouped:
                grouped[task_id] = {
                    'request_count': 0,
                    'blocked_count': 0,
                    'api_cost_usd': 0.0,
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'started_at': None,
                    'finished_at': None,
                    'wall_clock_seconds': 0.0,
                    'sml_request_ids': [],
                }
            return grouped[task_id]

        def _update_window(bucket: dict, start_ts: float | None, end_ts: float | None) -> None:
            for ts in (start_ts, end_ts):
                if ts is None:
                    continue
                if bucket['started_at'] is None or ts < bucket['started_at']:
                    bucket['started_at'] = ts
                if bucket['finished_at'] is None or ts > bucket['finished_at']:
                    bucket['finished_at'] = ts

        for item in dispatched:
            task_id = item.get('task_id')
            if not task_id:
                continue
            bucket = _bucket(task_id)
            bucket['request_count'] += 1
            if item.get('sml_request_id'):
                bucket['sml_request_ids'].append(item['sml_request_id'])
            bucket['api_cost_usd'] += float(item.get('stream_cost') or 0.0)
            bucket['prompt_tokens'] += int(item.get('prompt_tokens') or 0)
            bucket['completion_tokens'] += int(item.get('completion_tokens') or 0)
            _update_window(bucket, item.get('started_at'), item.get('finished_at') or item.get('started_at'))

        for item in blocked:
            task_id = item.get('task_id')
            if not task_id:
                continue
            bucket = _bucket(task_id)
            bucket['blocked_count'] += 1
            _update_window(bucket, item.get('time'), item.get('time'))

        for bucket in grouped.values():
            bucket['api_cost_usd'] = round(bucket['api_cost_usd'], 8)
            if bucket['started_at'] is not None and bucket['finished_at'] is not None:
                bucket['wall_clock_seconds'] = round(max(0.0, bucket['finished_at'] - bucket['started_at']), 6)

        return grouped

    def flush_sml_manifests(
        self,
        results_dir: str | Path,
        *,
        gpu_hours: float = 0.0,
        gpu_hourly_rate_usd: float = 0.0,
    ) -> dict[str, str]:
        base = Path(results_dir)
        written: dict[str, str] = {}
        compute_cost_usd = round(float(gpu_hours) * float(gpu_hourly_rate_usd), 8)
        for task_id, stats in self.task_summaries().items():
            api_cost_usd = round(float(stats['api_cost_usd']), 8)
            total_cost_usd = round(compute_cost_usd + api_cost_usd, 8)
            manifest = {
                'wall_clock_seconds': stats['wall_clock_seconds'],
                'gpu_hours': float(gpu_hours),
                'hourly_rate_usd': float(gpu_hourly_rate_usd),
                'HourlyRateUSD': float(gpu_hourly_rate_usd),
                'compute_cost_usd': compute_cost_usd,
                'api_cost_usd': api_cost_usd,
                'total_cost_usd': total_cost_usd,
                'prompt_tokens': int(stats['prompt_tokens']),
                'completion_tokens': int(stats['completion_tokens']),
                'request_count': int(stats['request_count']),
                'cost_accounting_status': 'estimated',
            }
            out_dir = base / task_id
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / 'cost.json'
            tmp_path = out_dir / f'.cost.{os.getpid()}.{threading.get_ident()}.tmp'
            tmp_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
            os.replace(tmp_path, out_path)
            written[task_id] = str(out_path)
        return written


class _OpenRouterProxyHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        return

    def _read_body(self):
        te = (self.headers.get('Transfer-Encoding') or '').lower()
        if 'chunked' in te:
            parts = []
            while True:
                line = self.rfile.readline()
                if not line:
                    break
                size = int(line.split(b';', 1)[0].strip(), 16)
                if size == 0:
                    self.rfile.readline()
                    break
                parts.append(self.rfile.read(size))
                self.rfile.read(2)
            return b''.join(parts)
        length = int(self.headers.get('Content-Length') or '0')
        return self.rfile.read(length) if length > 0 else b''

    def _target_url(self):
        rel = self.path
        if self.server.upstream_url.endswith('/api/v1') and rel.startswith('/api/v1'):
            rel = rel[len('/api/v1'):] or '/'
        return self.server.upstream_url + ('/' if not rel.startswith('/') else '') + rel

    def _capture_auth(self):
        auth = self.headers.get('Authorization') or ''
        if auth.lower().startswith('bearer '):
            token = auth[7:].strip()
            if token:
                with self.server._lock:
                    self.server._auth_bearer = token

    def _forward(self, method):
        self._capture_auth()
        raw_task_id = self.headers.get('X-Harness-Task-Id')
        task_id = raw_task_id.strip() if isinstance(raw_task_id, str) and raw_task_id.strip() else None
        raw_sml_req_id = self.headers.get('X-SML-Request-Id')
        sml_request_id = raw_sml_req_id.strip() if isinstance(raw_sml_req_id, str) and raw_sml_req_id.strip() else None
        if task_id is None:
            raw_sml_task_id = self.headers.get('X-SML-Task-Id')
            task_id = raw_sml_task_id.strip() if isinstance(raw_sml_task_id, str) and raw_sml_task_id.strip() else None
        if task_id is None and sml_request_id is not None and '-step-' in sml_request_id:
            task_id = sml_request_id.split('-step-', 1)[0].strip() or None
        body = self._read_body() if method == 'POST' else None
        entry = None
        if method == 'POST':
            try:
                reason = self.server.gate()
            except GuardError as exc:
                reason = str(exc)
            if reason:
                with self.server._lock:
                    blocked_item = {'time': time.time(), 'reason': reason, 'path': self.path}
                    if task_id is not None:
                        blocked_item['task_id'] = task_id
                    self.server._blocked.append(blocked_item)
                    if self.server._blocked_reason is None:
                        self.server._blocked_reason = reason
                self.server.stop_event.set()
                payload = json.dumps({'error': {'message': f'opencode_guard pre-dispatch stop: {reason}',
                                                'type': 'guard_stop', 'code': 429}}).encode('utf-8')
                self.send_response(429)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.send_header('Connection', 'close')
                self.end_headers()
                try:
                    self.wfile.write(payload)
                    self.wfile.flush()
                except OSError:
                    pass
                return
            try:
                payload = json.loads(body)
            except (UnicodeDecodeError, ValueError):
                payload = None
            if self.server.enforce_egress_guard and isinstance(payload, dict):
                # H-81(b): automatic pre-compaction pass for oversized payloads
                # so the egress guard can reject genuine policy violations
                # instead of raw context-window size.
                def _serialized_size() -> int:
                    try:
                        return len(json.dumps(payload, separators=(',', ':')).encode('utf-8'))
                    except (TypeError, ValueError):
                        return 0

                if _serialized_size() > 262_144:
                    if isinstance(payload.get('messages'), list):
                        payload['messages'], _ = compact_stale_reads(
                            payload['messages'],
                            turn_index=len(payload['messages']),
                            keep_recent_turns=1,
                            min_total_saved_chars=256,
                        )
                    enforce_context_budget(payload, max_tokens=16_000)
                    if _serialized_size() > 262_144:
                        enforce_context_budget(payload, max_tokens=8_000)
                    body = json.dumps(payload).encode('utf-8')
                egress_violations = check_egress_payload(payload, allowed_models=self.server.allowed_models)
                if egress_violations:
                    with self.server._lock:
                        blocked_item = {'time': time.time(), 'reason': 'egress_guard_blocked',
                                        'path': self.path, 'violations': egress_violations}
                        if task_id is not None:
                            blocked_item['task_id'] = task_id
                        self.server._blocked.append(blocked_item)
                        if self.server._blocked_reason is None:
                            self.server._blocked_reason = 'egress_guard_blocked'
                    self.server.tracer.record_span('guard.egress_guard', kind='GUARDRAIL', status='ERROR',
                                                   attributes={'guardrail.name': 'egress_guard',
                                                               'guardrail.triggered': True,
                                                               'guardrail.errors': '; '.join(egress_violations)})
                    err_payload = json.dumps({'error': {'message': f"egress_guard_blocked: {'; '.join(egress_violations)}",
                                                        'type': 'egress_guard_blocked', 'code': 400,
                                                        'violations': egress_violations}}).encode('utf-8')
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(err_payload)))
                    self.send_header('Connection', 'close')
                    self.end_headers()
                    try:
                        self.wfile.write(err_payload)
                        self.wfile.flush()
                    except OSError:
                        pass
                    return
            with self.server._lock:
                entry = {'seq': len(self.server._dispatched) + 1, 'started_at': time.time(),
                         'finished_at': None,
                         'http_status': None, 'generation_id': None, 'stream_cost': None,
                         'prompt_tokens': 0, 'completion_tokens': 0,
                         'completed': False, 'error': None}
                if task_id is not None:
                    entry['task_id'] = task_id
                if sml_request_id is not None:
                    entry['sml_request_id'] = sml_request_id
                self.server._dispatched.append(entry)
            if self.path.rstrip('/').endswith(('/chat/completions', '/responses')):
                with self.server._lock:
                    self.server._turn_count += 1
                    turn_index = self.server._turn_count
            else:
                turn_index = 1
            messages = payload.get('messages') if isinstance(payload, dict) else None
            notices = []
            if self.server.enforce_readonly_sql and isinstance(messages, list):
                if sql_errors := check_readonly_sql_tool_calls(messages):
                    notices.append(' '.join(f'[SQL GUARD ERROR: {err} — use read-only SELECT queries only]'
                                            for err in sql_errors))
                    self.server.tracer.record_span('guard.readonly_sql', kind='GUARDRAIL', status='ERROR',
                                                   attributes={'guardrail.name': 'readonly_sql',
                                                               'guardrail.triggered': True,
                                                               'guardrail.errors': '; '.join(sql_errors),
                                                               'harness.turn_index': turn_index})
            if turn_index >= 2:
                syntax_errors = (check_allowed_files_syntax(self.server.allowed_files,
                                                            baseline_symbols=self.server.baseline_symbols)
                                 if self.server.allowed_files else [])
                if syntax_errors:
                    notices.append(' '.join(f'[AST PREFLIGHT ERROR: {err} — fix before finishing]'
                                            for err in syntax_errors))
                    self.server.tracer.record_span('guard.ast_preflight', kind='GUARDRAIL', status='ERROR',
                                                   attributes={'guardrail.name': 'ast_preflight',
                                                               'guardrail.triggered': True,
                                                               'guardrail.errors': '; '.join(syntax_errors),
                                                               'harness.turn_index': turn_index})
                if isinstance(messages, list) and (loop_warnings := detect_duplicate_tool_calls(messages)):
                    notices.append(' '.join(f'[GUARD LOOP WARNING: {w} — change target lines/arguments or finish]'
                                            for w in loop_warnings))
                    self.server.tracer.record_span('guard.duplicate_tool_loop', kind='GUARDRAIL', status='ERROR',
                                                   attributes={'guardrail.name': 'duplicate_tool_loop',
                                                               'guardrail.triggered': True,
                                                               'guardrail.warnings': '; '.join(loop_warnings),
                                                               'harness.turn_index': turn_index})
                if isinstance(messages, list):
                    cot_warnings = check_cot_and_slop_integrity(messages)
                    if cot_warnings:
                        notices.append(' '.join(cot_warnings))
                        with self.server._lock:
                            self.server._cot_slop_warnings += len(cot_warnings)
                        self.server.tracer.record_span('guard.cot_slop_integrity', kind='GUARDRAIL', status='ERROR',
                                                       attributes={'guardrail.name': 'cot_slop_integrity',
                                                                   'guardrail.type': 'cot_slop_integrity',
                                                                   'guardrail.triggered': True,
                                                                   'guardrail.warnings': '; '.join(cot_warnings),
                                                                   'harness.turn_index': turn_index})
            # H-32: edit-oscillation / untested-edit plateau guard.
            if isinstance(messages, list):
                osc = detect_edit_oscillation(messages)
                if osc:
                    notices.append(
                        f"[HARNESS PLATEAU GUARD: you are oscillating on {osc['file']} "
                        f"({osc['reason']}) — run the tests or rethink the root cause instead of "
                        f"repeating edits]")
                    self.server.tracer.record_span('guard.edit_oscillation', kind='GUARDRAIL', status='ERROR',
                                                   attributes={'guardrail.name': 'edit_oscillation',
                                                               'guardrail.triggered': True,
                                                               'guardrail.file': osc['file'],
                                                               'guardrail.reason': osc['reason'],
                                                               'harness.turn_index': turn_index})
            if notices and messages:
                last = messages[-1]
                if isinstance(last, dict):
                    notice = '\n' + '\n'.join(notices)
                    content = last.get('content')
                    if isinstance(content, list):
                        content.append({'type': 'text', 'text': notice})
                    else:
                        last['content'] = (content if isinstance(content, str) else '') + notice
                    body = json.dumps(payload).encode('utf-8')
            # H-17: inject recovery hints on trailing edit-target mismatch results.
            if isinstance(payload, dict) and isinstance(payload.get('messages'), list):
                new_msgs, hints_injected = inject_edit_failure_hints(payload['messages'])
                if hints_injected > 0:
                    payload['messages'] = new_msgs
                    with self.server._lock:
                        self.server._edit_hints_injected += hints_injected
                    body = json.dumps(payload).encode('utf-8')
                    self.server.tracer.record_span(
                        'guard.edit_failure_hint',
                        kind='GUARDRAIL',
                        attributes={
                            'guardrail.name': 'edit_failure_hint',
                            'guardrail.triggered': True,
                            'harness.edit_hints_injected': hints_injected,
                            'harness.turn_index': turn_index,
                        },
                    )
            # H-21: inject read-guard hints (permission denied / read budget).
            if isinstance(payload, dict) and isinstance(payload.get('messages'), list):
                new_msgs, read_hints = inject_read_guard_hints(payload['messages'])
                if read_hints > 0:
                    payload['messages'] = new_msgs
                    body = json.dumps(payload).encode('utf-8')
                    self.server.tracer.record_span(
                        'guard.read_guard_hint',
                        kind='GUARDRAIL',
                        attributes={
                            'guardrail.name': 'read_guard_hint',
                            'guardrail.triggered': True,
                            'harness.read_hints_injected': read_hints,
                            'harness.turn_index': turn_index,
                        },
                    )
            # H-30: only on epoch turns (turn_index >= 6, every 3rd turn) and
            # only when the rewrite saves enough characters to justify busting
            # the upstream KV-cache prefix; minor re-reads stay untouched so
            # cached prefixes remain valid on intermediate turns.
            if (
                turn_index >= 6
                and turn_index % 3 == 0
                and isinstance(payload, dict)
                and isinstance(payload.get('messages'), list)
            ):
                new_msgs, compacted = compact_stale_reads(
                    payload['messages'], turn_index=turn_index, min_total_saved_chars=2400
                )
                if compacted > 0:
                    payload['messages'] = new_msgs
                    with self.server._lock:
                        self.server._compacted_reads += compacted
                    body = json.dumps(payload).encode('utf-8')
                    self.server.tracer.record_span('guard.stale_read_compaction', kind='GUARDRAIL',
                                                   attributes={'guardrail.name': 'stale_read_compaction',
                                                               'guardrail.triggered': True,
                                                               'harness.compacted_count': compacted,
                                                               'harness.turn_index': turn_index})
            # H-31: SWE-Pruner Lite — compact bloated test/traceback tool
            # outputs before the context-budget compaction runs, so oversized
            # pytest dumps never reach the upstream prompt.
            if isinstance(payload, dict) and isinstance(payload.get('messages'), list):
                pruned_msgs, pruned_count = prune_noisy_tool_outputs(payload['messages'])
                if pruned_count > 0:
                    payload['messages'] = pruned_msgs
                    body = json.dumps(payload).encode('utf-8')
                    self.server.tracer.record_span(
                        'guard.swe_pruner',
                        kind='GUARDRAIL',
                        attributes={
                            'guardrail.name': 'swe_pruner',
                            'guardrail.triggered': True,
                            'harness.pruned_tool_outputs': pruned_count,
                            'harness.turn_index': turn_index,
                        },
                    )
            if (
                self.server.max_context_tokens is not None
                and isinstance(payload, dict)
                and isinstance(payload.get('messages'), list)
            ):
                new_msgs, stats = compact_context_window(
                    payload['messages'], max_context_tokens=self.server.max_context_tokens
                )
                if stats['truncated_messages'] > 0:
                    payload['messages'] = new_msgs
                    with self.server._lock:
                        self.server._compacted_reads += stats['truncated_messages']
                    body = json.dumps(payload).encode('utf-8')
                    self.server.tracer.record_span(
                        'guard.context_budget',
                        kind='GUARDRAIL',
                        attributes={
                            'guardrail.name': 'context_budget',
                            'guardrail.triggered': True,
                            'harness.truncated_messages': stats['truncated_messages'],
                            'harness.tokens_before': stats['tokens_before'],
                            'harness.tokens_after': stats['tokens_after'],
                            'harness.max_context_tokens': self.server.max_context_tokens,
                        },
                    )
            # Provider routing runs last so notices injected above land inside the
            # payload that apply_provider_routing scans for error signals, letting
            # the same turn immediately use the error-recovery reasoning budget.
            with self.server._lock:
                degraded_snapshot = set(self.server.degraded_providers)
                sticky_snapshot = self.server.sticky_provider
            body = apply_provider_routing(body, self.path, default_model=self.server.default_model,
                                          turn_index=turn_index,
                                          reasoning_max_tokens=self.server.reasoning_max_tokens,
                                          degraded_providers=degraded_snapshot,
                                          sticky_provider=sticky_snapshot,
                                          upstream_mode=self.server.upstream_mode,
                                          fallback_teacher_model=self.server.fallback_teacher_model)
            front_provider = None
            routed_model = self.server.default_model or ''
            applied_reasoning_max_tokens = None
            try:
                routed_payload = json.loads(body)
                if isinstance(routed_payload, dict):
                    provider = routed_payload.get('provider')
                    if isinstance(provider, dict) and isinstance(provider.get('order'), list) and provider['order']:
                        front_provider = provider['order'][0]
                    if isinstance(routed_payload.get('model'), str) and routed_payload['model']:
                        routed_model = routed_payload['model']
                    reasoning = routed_payload.get('reasoning')
                    if isinstance(reasoning, dict) and isinstance(reasoning.get('max_tokens'), (int, float)):
                        applied_reasoning_max_tokens = reasoning['max_tokens']
            except (UnicodeDecodeError, ValueError):
                pass
        headers = {'Accept-Encoding': 'identity', 'Connection': 'close'}
        for h in ('Authorization', 'Content-Type', 'Accept', 'HTTP-Referer', 'X-Title', 'User-Agent'):
            val = self.headers.get(h)
            if val is not None:
                headers[h] = val
        req = urllib.request.Request(self._target_url(), data=body, headers=headers, method=method)
        req_start = time.monotonic()
        degraded_provider = [front_provider]

        def _mark_degraded(name):
            if not name:
                return
            with self.server._lock:
                self.server.degraded_providers.add(name)
                if self.server.sticky_provider == name:
                    self.server.sticky_provider = None
            if self.server.health_store is not None:
                self.server.health_store.mark_degraded(name)
            self.server.tracer.record_span('guard.provider_circuit_breaker', kind='GUARDRAIL', status='ERROR',
                                           attributes={'guardrail.name': 'provider_circuit_breaker',
                                                       'guardrail.triggered': True,
                                                       'llm.provider': name})

        def _provider_from_body(raw):
            if not raw:
                return None
            try:
                err_payload = json.loads(raw)
            except (UnicodeDecodeError, ValueError):
                return None
            if isinstance(err_payload, dict):
                for key in ('provider', 'provider_name'):
                    val = err_payload.get(key)
                    if isinstance(val, str) and val:
                        return val
                    if isinstance(val, dict) and isinstance(val.get('name'), str):
                        return val['name']
            return None

        req_start_ns = time.time_ns()
        resp = None
        try:
            for attempt in range(2):
                try:
                    resp = urllib.request.urlopen(req, timeout=self.server.request_timeout)
                    break
                except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
                    if (isinstance(exc, urllib.error.URLError)
                            and not isinstance(exc.reason, (TimeoutError, socket.timeout))):
                        raise
                    if attempt == 0:
                        failed_provider = degraded_provider[0] or self.server.sticky_provider
                        if (failed_provider
                                or (body and (self.server.sticky_provider
                                              or routed_model in MODEL_PROVIDER_PREFS))):
                            _mark_degraded(failed_provider)
                            if body:
                                with self.server._lock:
                                    degraded_snapshot = set(self.server.degraded_providers)
                                    sticky_snapshot = self.server.sticky_provider
                                retry_body = apply_provider_routing(
                                    body, self.path,
                                    default_model=self.server.default_model,
                                    turn_index=turn_index,
                                    reasoning_max_tokens=self.server.reasoning_max_tokens,
                                    degraded_providers=degraded_snapshot,
                                    sticky_provider=sticky_snapshot,
                                    upstream_mode=self.server.upstream_mode,
                                    fallback_teacher_model=self.server.fallback_teacher_model)
                                next_provider = None
                                try:
                                    retry_payload = json.loads(retry_body)
                                    if isinstance(retry_payload, dict):
                                        provider = retry_payload.get('provider')
                                        if (isinstance(provider, dict)
                                                and isinstance(provider.get('order'), list)
                                                and provider['order']):
                                            next_provider = provider['order'][0]
                                except (UnicodeDecodeError, ValueError):
                                    pass
                                if (next_provider
                                        and next_provider not in degraded_snapshot
                                        and next_provider != failed_provider):
                                    body = retry_body
                                    degraded_provider[0] = next_provider
                                    req = urllib.request.Request(self._target_url(), data=body,
                                                                 headers=headers, method=method)
                                    continue
                    raise
            with resp:
                status = getattr(resp, 'status', 200)
                if entry is not None:
                    with self.server._lock:
                        entry['http_status'] = status
                hdr_provider = resp.headers.get('X-OpenRouter-Provider')
                if hdr_provider:
                    degraded_provider[0] = hdr_provider
                if status == 200 and degraded_provider[0]:
                    with self.server._lock:
                        if (self.server.sticky_provider is None
                                and degraded_provider[0] not in self.server.degraded_providers):
                            self.server.sticky_provider = degraded_provider[0]
                    if (self.server.health_store is not None
                            and self.server.sticky_provider == degraded_provider[0]):
                        self.server.health_store.set_sticky(degraded_provider[0], self.server.default_model)
                self.send_response(status)
                ctype = resp.headers.get('Content-Type')
                if ctype:
                    self.send_header('Content-Type', ctype)
                is_sse = bool(ctype and 'event-stream' in ctype.lower())
                if is_sse:
                    self.send_header('Cache-Control', 'no-cache')
                self.send_header('Connection', 'close')
                self.end_headers()
                tail = b''
                non_sse_chunks = [] if (entry is not None and not is_sse) else None
                non_sse_bytes = 0

                def _apply_usage_obj(obj):
                    if not isinstance(obj, dict):
                        return
                    gen_id = obj.get('id')
                    prov = obj.get('provider') or obj.get('provider_name')
                    if isinstance(prov, str) and prov and not hdr_provider:
                        degraded_provider[0] = prov
                    usage = obj.get('usage')
                    with self.server._lock:
                        if isinstance(gen_id, str) and gen_id and entry['generation_id'] is None:
                            entry['generation_id'] = gen_id
                        if isinstance(usage, dict):
                            pt = usage.get('prompt_tokens')
                            ct = usage.get('completion_tokens')
                            cost = usage.get('cost')
                            if isinstance(pt, (int, float)):
                                entry['prompt_tokens'] = int(pt)
                            if isinstance(ct, (int, float)):
                                entry['completion_tokens'] = int(ct)
                            if isinstance(cost, (int, float)):
                                entry['stream_cost'] = float(cost)

                while True:
                    # Never touch resp.fp.read1 directly: HTTPResponse.readline / read
                    # decode upstream HTTP/1.1 Transfer-Encoding: chunked frames safely.
                    chunk = resp.readline() if is_sse else resp.read(4096)
                    if not chunk:
                        if non_sse_chunks:
                            try:
                                _apply_usage_obj(json.loads(b''.join(non_sse_chunks)))
                            except (UnicodeDecodeError, ValueError):
                                pass
                        if entry is not None:
                            with self.server._lock:
                                entry['completed'] = True
                        break
                    if entry is not None:
                        if non_sse_chunks is not None and non_sse_bytes < 2_000_000:
                            non_sse_chunks.append(chunk)
                            non_sse_bytes += len(chunk)
                        if is_sse:
                            stripped = chunk.strip()
                            if stripped.startswith(b'data:'):
                                raw_json = stripped[5:].strip()
                                if raw_json and raw_json != b'[DONE]':
                                    try:
                                        _apply_usage_obj(json.loads(raw_json))
                                    except (UnicodeDecodeError, ValueError):
                                        pass
                        window = tail + chunk
                        m_id = GEN_ID_RE.search(window)
                        m_cost = STREAM_COST_RE.search(window)
                        m_pt = PROMPT_TOKENS_RE.search(window)
                        m_ct = COMPLETION_TOKENS_RE.search(window)
                        m_prov = re.search(rb'"provider(?:_name)?"\s*:\s*"([^"]+)"', window)
                        if m_prov and not hdr_provider:
                            degraded_provider[0] = m_prov.group(1).decode('utf-8', 'replace')
                        if m_id or m_cost or m_pt or m_ct:
                            with self.server._lock:
                                if m_id and entry['generation_id'] is None:
                                    entry['generation_id'] = m_id.group(1).decode('ascii')
                                if m_cost:
                                    try:
                                        entry['stream_cost'] = float(m_cost.group(1))
                                    except ValueError:
                                        pass
                                if m_pt and not entry['prompt_tokens']:
                                    try:
                                        entry['prompt_tokens'] = int(m_pt.group(1))
                                    except ValueError:
                                        pass
                                if m_ct and not entry['completion_tokens']:
                                    try:
                                        entry['completion_tokens'] = int(m_ct.group(1))
                                    except ValueError:
                                        pass
                        tail = window[-160:]
                    try:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    except OSError:
                        if entry is not None:
                            with self.server._lock:
                                entry['error'] = 'client_disconnected'
                        break
                if time.monotonic() - req_start > _effective_latency_threshold():
                    _mark_degraded(degraded_provider[0])
                elif status == 200:
                    observed = degraded_provider[0]
                    if observed:
                        with self.server._lock:
                            if (self.server.sticky_provider is None
                                    and observed not in self.server.degraded_providers):
                                self.server.sticky_provider = observed
                        if (self.server.health_store is not None
                                and self.server.sticky_provider == observed):
                            self.server.health_store.set_sticky(observed, self.server.default_model)
        except urllib.error.HTTPError as exc:
            err_body = exc.read() if exc.fp else b''
            if exc.code == 429 or exc.code >= 500:
                _mark_degraded(exc.headers.get('X-OpenRouter-Provider') or _provider_from_body(err_body) or degraded_provider[0])
            if entry is not None:
                m_id = GEN_ID_RE.search(err_body)
                with self.server._lock:
                    entry['http_status'] = exc.code
                    entry['error'] = f'HTTP_{exc.code}'
                    if m_id and entry['generation_id'] is None:
                        entry['generation_id'] = m_id.group(1).decode('ascii')
            try:
                self.send_response(exc.code)
                self.send_header('Content-Type', exc.headers.get('Content-Type', 'application/json'))
                self.send_header('Content-Length', str(len(err_body)))
                self.send_header('Connection', 'close')
                self.end_headers()
                self.wfile.write(err_body)
                self.wfile.flush()
            except OSError:
                pass
        except OSError as exc:
            _mark_degraded(degraded_provider[0])
            if entry is not None:
                with self.server._lock:
                    entry['error'] = type(exc).__name__
            try:
                payload = json.dumps({'error': {'message': type(exc).__name__}}).encode('utf-8')
                self.send_response(502)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.send_header('Connection', 'close')
                self.end_headers()
                self.wfile.write(payload)
                self.wfile.flush()
            except OSError:
                pass
        finally:
            if entry is not None:
                with self.server._lock:
                    entry['finished_at'] = time.time()
                span_attrs = {
                    'gen_ai.system': 'openrouter',
                    'gen_ai.request.model': routed_model,
                    'llm.model_name': routed_model,
                    'llm.provider': degraded_provider[0] or front_provider or '',
                    'gen_ai.response.id': entry.get('generation_id') or '',
                    'gen_ai.usage.cost_usd': entry.get('stream_cost') or 0.0,
                    'harness.turn_index': turn_index,
                    'harness.reasoning_max_tokens': applied_reasoning_max_tokens,
                    'http.status_code': entry.get('http_status') or 0,
                }
                if task_id is not None:
                    span_attrs['harness.task_id'] = task_id
                self.server.tracer.record_span('gen_ai.chat', kind='LLM',
                                               start_time_ns=req_start_ns,
                                               end_time_ns=time.time_ns(),
                                               status='ERROR' if entry.get('error') else 'OK',
                                               attributes=span_attrs)

    def do_POST(self):
        self._forward('POST')

    def do_GET(self):
        self._forward('GET')


@contextmanager
def openrouter_proxy(gate, *, upstream_url=OPENROUTER_UPSTREAM, request_timeout=300, default_model=None,
                     reasoning_max_tokens=None, allowed_files=None,
                     health_store: ProviderHealthStore | None = None, tracer: HarnessTracer | None = None,
                     allowed_models: Iterable[str] | None = None, enforce_egress_guard: bool = False,
                     enforce_readonly_sql: bool = False, host: str = '127.0.0.1', port: int = 0,
                     upstream_mode: str = 'openrouter', fallback_teacher_model: str | None = None,
                     max_context_tokens: int | None = None,
                     sml_results_dir: str | Path | None = None):
    srv = OpenRouterGuardProxy(gate, upstream_url=upstream_url, request_timeout=request_timeout,
                               default_model=default_model, reasoning_max_tokens=reasoning_max_tokens,
                               allowed_files=allowed_files, health_store=health_store, tracer=tracer,
                               allowed_models=allowed_models, enforce_egress_guard=enforce_egress_guard,
                               enforce_readonly_sql=enforce_readonly_sql, host=host, port=int(port),
                               upstream_mode=upstream_mode, fallback_teacher_model=fallback_teacher_model,
                               max_context_tokens=max_context_tokens,
                               sml_results_dir=sml_results_dir)
    thread = threading.Thread(target=srv.serve_forever, kwargs={'poll_interval': 0.05}, daemon=True)
    thread.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=2)
        if sml_results_dir is not None:
            srv.flush_sml_manifests(sml_results_dir)
