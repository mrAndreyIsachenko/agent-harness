"""OpenRouter guard proxy.

Ported from `.tools/opencode_guard.py`: a local HTTP proxy between the
OpenCode CLI and OpenRouter that gates dispatch, runs AST preflight and
duplicate-tool-call loop detection on the payload, applies provider routing,
and tracks per-turn cost / generation ids / provider health.
"""

import ast
import http.server
import json
import re
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
    check_egress_payload,
    check_readonly_sql_tool_calls,
    collect_python_top_symbols,
    compact_stale_reads,
    detect_duplicate_tool_calls,
)
from .routing import ProviderHealthStore, apply_provider_routing
from .telemetry import HarnessTracer

OPENROUTER_UPSTREAM = 'https://openrouter.ai/api/v1'
GEN_ID_RE = re.compile(rb'"id"\s*:\s*"(gen-[A-Za-z0-9_-]+)"')
STREAM_COST_RE = re.compile(rb'"usage"\s*:\s*\{[^{}]*"cost"\s*:\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)')
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
                 upstream_mode: str = 'openrouter', fallback_teacher_model: str | None = None):
        self.gate = gate
        self.upstream_url = upstream_url.rstrip('/')
        self.request_timeout = request_timeout
        self.default_model = default_model
        self.reasoning_max_tokens = reasoning_max_tokens
        self.allowed_files = list(allowed_files or [])
        self.allowed_models = list(allowed_models) if allowed_models is not None else None
        self.enforce_egress_guard = bool(enforce_egress_guard)
        self.enforce_readonly_sql = bool(enforce_readonly_sql)
        self.upstream_mode = upstream_mode
        self.fallback_teacher_model = fallback_teacher_model
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

    def summary(self):
        with self._lock:
            return {'dispatched_count': len(self._dispatched),
                    'blocked_count': len(self._blocked),
                    'blocked_reason': self._blocked_reason,
                    'dispatched': [dict(d) for d in self._dispatched],
                    'blocked': [dict(b) for b in self._blocked],
                    'spans': list(self.tracer.spans),
                    'compacted_reads': self._compacted_reads}


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
        body = self._read_body() if method == 'POST' else None
        entry = None
        if method == 'POST':
            try:
                reason = self.server.gate()
            except GuardError as exc:
                reason = str(exc)
            if reason:
                with self.server._lock:
                    self.server._blocked.append({'time': time.time(), 'reason': reason, 'path': self.path})
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
                egress_violations = check_egress_payload(payload, allowed_models=self.server.allowed_models)
                if egress_violations:
                    with self.server._lock:
                        self.server._blocked.append({'time': time.time(), 'reason': 'egress_guard_blocked',
                                                     'path': self.path, 'violations': egress_violations})
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
                         'http_status': None, 'generation_id': None, 'stream_cost': None,
                         'completed': False, 'error': None}
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
            # H-8: on later turns, compact stale read outputs the guard cannot
            # rewrite in place (notice injection already re-encoded the body).
            if turn_index >= 5 and isinstance(payload, dict) and isinstance(payload.get('messages'), list):
                new_msgs, compacted = compact_stale_reads(payload['messages'], turn_index=turn_index)
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

        try:
            with urllib.request.urlopen(req, timeout=self.server.request_timeout) as resp:
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
                while True:
                    # Never touch resp.fp.read1 directly: HTTPResponse.readline / read
                    # decode upstream HTTP/1.1 Transfer-Encoding: chunked frames safely.
                    chunk = resp.readline() if is_sse else resp.read(4096)
                    if not chunk:
                        if entry is not None:
                            with self.server._lock:
                                entry['completed'] = True
                        break
                    if entry is not None:
                        window = tail + chunk
                        m_id = GEN_ID_RE.search(window)
                        m_cost = STREAM_COST_RE.search(window)
                        m_prov = re.search(rb'"provider(?:_name)?"\s*:\s*"([^"]+)"', window)
                        if m_prov and not hdr_provider:
                            degraded_provider[0] = m_prov.group(1).decode('utf-8', 'replace')
                        if m_id or m_cost:
                            with self.server._lock:
                                if m_id and entry['generation_id'] is None:
                                    entry['generation_id'] = m_id.group(1).decode('ascii')
                                if m_cost:
                                    try:
                                        entry['stream_cost'] = float(m_cost.group(1))
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
                self.server.tracer.record_span('gen_ai.chat', kind='LLM',
                                               status='ERROR' if entry.get('error') else 'OK',
                                               attributes={'gen_ai.system': 'openrouter',
                                                           'gen_ai.request.model': routed_model,
                                                           'llm.model_name': routed_model,
                                                           'llm.provider': degraded_provider[0] or front_provider or '',
                                                           'gen_ai.response.id': entry.get('generation_id') or '',
                                                           'gen_ai.usage.cost_usd': entry.get('stream_cost') or 0.0,
                                                           'harness.turn_index': turn_index,
                                                           'harness.reasoning_max_tokens': applied_reasoning_max_tokens,
                                                           'http.status_code': entry.get('http_status') or 0})

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
                     upstream_mode: str = 'openrouter', fallback_teacher_model: str | None = None):
    srv = OpenRouterGuardProxy(gate, upstream_url=upstream_url, request_timeout=request_timeout,
                               default_model=default_model, reasoning_max_tokens=reasoning_max_tokens,
                               allowed_files=allowed_files, health_store=health_store, tracer=tracer,
                               allowed_models=allowed_models, enforce_egress_guard=enforce_egress_guard,
                               enforce_readonly_sql=enforce_readonly_sql, host=host, port=int(port),
                               upstream_mode=upstream_mode, fallback_teacher_model=fallback_teacher_model)
    thread = threading.Thread(target=srv.serve_forever, kwargs={'poll_interval': 0.05}, daemon=True)
    thread.start()
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=2)
