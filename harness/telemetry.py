"""OpenTelemetry GenAI + Arize AX / Phoenix OpenInference telemetry.

Zero-dependency (stdlib only) tracer for the agent harness: records spans
that carry dual-standard attributes (``openinference.span.kind`` /
``openinference.project.name`` alongside ``gen_ai.*`` / ``llm.*`` /
``guardrail.*`` / ``harness.*`` keys), optionally appends them to a JSONL
file, converts them to OTLP/HTTP JSON ``ExportTraceServiceRequest`` format,
and flushes them to an Arize AX or Phoenix collector.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Local Phoenix collector used when no explicit endpoint (argument or env) is set.
DEFAULT_PHOENIX_ENDPOINT = 'http://127.0.0.1:6006/v1/traces'

OPENINFERENCE_SPAN_KINDS = ('AGENT', 'LLM', 'GUARDRAIL', 'CHAIN', 'TOOL')
_SCOPE_NAME = 'harness.telemetry'
_SCOPE_VERSION = '0.1.0'
# OTLP SpanKind enum: 1=INTERNAL, 3=CLIENT. LLM and TOOL calls are outgoing
# client calls; agent/chain/guardrail orchestration stays INTERNAL.
_OTLP_SPAN_KINDS = {'AGENT': 1, 'CHAIN': 1, 'GUARDRAIL': 1, 'LLM': 3, 'TOOL': 3}
_OTLP_STATUS_CODES = {'OK': 1, 'ERROR': 2}
# Attribute keys surfaced by format_trace_tree, in display order.
_PREVIEW_ATTR_KEYS = (
    'gen_ai.system', 'gen_ai.request.model', 'gen_ai.response.id',
    'gen_ai.usage.cost_usd', 'llm.model_name', 'llm.provider',
    'guardrail.name', 'guardrail.triggered',
    'harness.turn_index', 'harness.reasoning_max_tokens',
)


class HarnessTracer:
    """Minimal OTel-compatible tracer with OpenInference/GenAI attributes."""

    def __init__(self, service_name: str = 'agent-harness', *, trace_id: str | None = None,
                 project_name: str | None = None, jsonl_path: str | Path | None = None,
                 otlp_endpoint: str | None = None, arize_api_key: str | None = None,
                 arize_space_id: str | None = None):
        self.service_name = service_name
        self.trace_id = trace_id or secrets.token_hex(16)
        self.root_span_id = secrets.token_hex(8)
        self.project_name = (project_name
                             or os.environ.get('ARIZE_PROJECT_NAME')
                             or os.environ.get('PHOENIX_PROJECT_NAME')
                             or 'agent-harness')
        self.jsonl_path = Path(jsonl_path) if jsonl_path is not None else None
        self.otlp_endpoint = (otlp_endpoint
                              or os.environ.get('ARIZE_COLLECTOR_ENDPOINT')
                              or os.environ.get('PHOENIX_COLLECTOR_ENDPOINT')
                              or os.environ.get('OTEL_EXPORTER_OTLP_ENDPOINT')
                              or DEFAULT_PHOENIX_ENDPOINT)
        self.arize_api_key = arize_api_key or os.environ.get('ARIZE_API_KEY')
        self.arize_space_id = arize_space_id or os.environ.get('ARIZE_SPACE_ID')
        self.spans: list[dict] = []
        self._lock = threading.Lock()

    def record_span(self, name: str, *, kind: str = 'CHAIN', parent_span_id: str | None = None,
                    start_time_ns: int | None = None, end_time_ns: int | None = None,
                    status: str = 'OK', attributes: dict | None = None) -> dict:
        """Record one span; returns the stored span dict.

        Always stamps ``openinference.span.kind`` (validated against
        OPENINFERENCE_SPAN_KINDS) and ``openinference.project.name`` on top of
        the caller-provided attributes (``gen_ai.*``, ``llm.*``, ...).
        """
        if kind not in OPENINFERENCE_SPAN_KINDS:
            raise ValueError(f"kind must be one of {OPENINFERENCE_SPAN_KINDS}, got {kind!r}")
        span = {
            'name': name,
            'span_id': secrets.token_hex(8),
            'trace_id': self.trace_id,
            'parent_span_id': parent_span_id,
            'start_time_ns': start_time_ns if start_time_ns is not None else time.time_ns(),
            'end_time_ns': end_time_ns if end_time_ns is not None else time.time_ns(),
            'status': status,
            'attributes': {
                'openinference.span.kind': kind,
                'openinference.project.name': self.project_name,
                **(attributes or {}),
            },
        }
        with self._lock:
            self.spans.append(span)
            if self.jsonl_path is not None:
                self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                with self.jsonl_path.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(span, sort_keys=True) + '\n')
        return span

    def to_otlp_payload(self) -> dict:
        """Convert recorded spans into OTLP/HTTP JSON ExportTraceServiceRequest."""
        with self._lock:
            spans = list(self.spans)
        resource_attributes = [
            {'key': 'service.name', 'value': {'stringValue': self.service_name}},
            {'key': 'openinference.project.name', 'value': {'stringValue': self.project_name}},
        ]
        return {
            'resourceSpans': [{
                'resource': {'attributes': resource_attributes},
                'scopeSpans': [{
                    'scope': {'name': _SCOPE_NAME, 'version': _SCOPE_VERSION},
                    'spans': [self._span_to_otlp(span) for span in spans],
                }],
            }],
        }

    @staticmethod
    def _otlp_value(value) -> dict:
        # proto3 JSON mapping: int64 values are encoded as decimal strings.
        if isinstance(value, bool):
            return {'boolValue': value}
        if isinstance(value, int):
            return {'intValue': str(value)}
        if isinstance(value, float):
            return {'doubleValue': value}
        if isinstance(value, str):
            return {'stringValue': value}
        return {'stringValue': json.dumps(value, sort_keys=True, default=str)}

    @classmethod
    def _span_to_otlp(cls, span: dict) -> dict:
        attributes = span.get('attributes') or {}
        otlp_span = {
            'traceId': span.get('trace_id', ''),
            'spanId': span.get('span_id', ''),
            'name': span.get('name', ''),
            'kind': _OTLP_SPAN_KINDS.get(attributes.get('openinference.span.kind'), 1),
            'startTimeUnixNano': str(span.get('start_time_ns', 0)),
            'endTimeUnixNano': str(span.get('end_time_ns', 0)),
            'attributes': [{'key': key, 'value': cls._otlp_value(value)}
                           for key, value in attributes.items()],
            'status': {'code': _OTLP_STATUS_CODES.get(span.get('status', 'OK'), 0)},
        }
        if span.get('parent_span_id'):
            otlp_span['parentSpanId'] = span['parent_span_id']
        return otlp_span

    def flush_otlp(self, *, endpoint: str | None = None, timeout: float = 1.5) -> bool:
        """POST the current spans as OTLP/HTTP JSON; True on 2xx, never raises."""
        target = endpoint or self.otlp_endpoint
        if not target:
            return False
        headers = {'Content-Type': 'application/json'}
        if self.arize_api_key:
            headers['api_key'] = self.arize_api_key
            headers['x-arize-api-key'] = self.arize_api_key
        if self.arize_space_id:
            headers['space_id'] = self.arize_space_id
            headers['x-arize-space-id'] = self.arize_space_id
        request = urllib.request.Request(
            target, data=json.dumps(self.to_otlp_payload()).encode('utf-8'),
            headers=headers, method='POST')
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return 200 <= getattr(response, 'status', 200) < 300
        except (urllib.error.URLError, OSError, ValueError):
            return False


def format_trace_tree(spans: list[dict]) -> str:
    """Render recorded spans as a human-readable ASCII tree."""
    if not spans:
        return '(no spans)'
    by_id = {span.get('span_id'): span for span in spans if span.get('span_id')}
    children: dict[str, list[dict]] = {}
    roots: list[dict] = []
    for span in spans:
        parent = span.get('parent_span_id')
        if parent and parent in by_id:
            children.setdefault(parent, []).append(span)
        else:
            roots.append(span)
    ordered = sorted(roots, key=lambda s: s.get('start_time_ns') or 0)
    for group in children.values():
        group.sort(key=lambda s: s.get('start_time_ns') or 0)
    lines = []
    if spans[0].get('trace_id'):
        lines.append(f"trace {spans[0]['trace_id']}")

    def _render(span, prefix: str, connector: str) -> None:
        attributes = span.get('attributes') or {}
        kind = attributes.get('openinference.span.kind', 'CHAIN')
        start, end = span.get('start_time_ns'), span.get('end_time_ns')
        latency_ms = ((end - start) / 1e6) if (start is not None and end is not None) else 0.0
        extras = ', '.join(f"{key}={attributes[key]}"
                           for key in _PREVIEW_ATTR_KEYS if key in attributes)
        suffix = f'  {extras}' if extras else ''
        lines.append(f"{prefix}{connector}[{kind}] {span.get('name', '?')}"
                     f"  latency_ms={latency_ms:.1f}"
                     f"  status={span.get('status', 'OK')}{suffix}")
        kids = children.get(span.get('span_id'), [])
        for index, child in enumerate(kids):
            last = index == len(kids) - 1
            child_prefix = prefix if connector == '' else prefix + ('    ' if last else '│   ')
            _render(child, child_prefix, '└── ' if last else '├── ')

    for index, root in enumerate(ordered):
        _render(root, '', '')
    return '\n'.join(lines)


# --- Historical run trace synthesis and Phoenix sync ------------------------

# Key spellings accepted in ``result.json`` / ``requests.json`` records.
_TASK_ID_KEYS = ('task_id', 'taskId', 'task')
_MODEL_KEYS = ('model', 'model_name', 'model_id', 'gen_ai.request.model', 'llm.model_name')
_STATUS_KEYS = ('status', 'final_status', 'result_status', 'validation_status')
_STEPS_KEYS = ('steps', 'step_count', 'model_steps')
_COST_KEYS = ('cost_usd', 'total_cost_usd', 'cost', 'costUSD', 'spend_usd', 'usd_cost')
_STARTED_AT_KEYS = ('started_at', 'started_at_utc', 'start_time', 'created_at', 'start')
_FINISHED_AT_KEYS = ('finished_at', 'finished_at_utc', 'end_time', 'completed_at', 'updated_at', 'finish')


def _first(mapping: dict, keys) -> object:
    """Return the first present, non-None value among ``keys``."""
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _to_int(value) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _timestamp_to_ns(value) -> int | None:
    """Best-effort conversion of epoch numbers or ISO strings to nanoseconds."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds <= 0:
            return None
        if seconds < 1e11:  # epoch seconds
            return int(seconds * 1e9)
        if seconds < 1e14:  # epoch milliseconds
            return int(seconds * 1e6)
        if seconds < 1e17:  # epoch microseconds
            return int(seconds * 1e3)
        return int(seconds)  # already nanoseconds
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace('Z', '+00:00'))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1e9)


def _parse_jsonl_spans(text: str) -> list[dict]:
    """Parse spans from JSONL text, tolerating blank/unparseable lines."""
    stripped = text.strip()
    if not stripped:
        return []
    try:
        payload = json.loads(stripped)
    except ValueError:
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get('spans'), list):
        return [span for span in payload['spans'] if isinstance(span, dict)]
    if isinstance(payload, list):
        return [span for span in payload if isinstance(span, dict)]
    spans = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            span = json.loads(line)
        except ValueError:
            continue
        if isinstance(span, dict):
            spans.append(span)
    return spans


def _deterministic_hex(seed: str, width: int) -> str:
    """Stable hex identifier derived from ``seed`` (keeps re-synthesis idempotent)."""
    return hashlib.sha256(seed.encode('utf-8')).hexdigest()[:width]


def _resolve_otlp_endpoint(endpoint: str | None) -> str:
    """Pick the OTLP endpoint: argument, then env, then the Phoenix default."""
    return (endpoint
            or os.environ.get('ARIZE_COLLECTOR_ENDPOINT')
            or os.environ.get('PHOENIX_COLLECTOR_ENDPOINT')
            or os.environ.get('OTEL_EXPORTER_OTLP_ENDPOINT')
            or DEFAULT_PHOENIX_ENDPOINT)


def _synthesis_project_name() -> str:
    return (os.environ.get('ARIZE_PROJECT_NAME')
            or os.environ.get('PHOENIX_PROJECT_NAME')
            or 'agent-harness')


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + '.tmp')
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(tmp_path, path)


def _load_run_requests(run_path: Path) -> list:
    """Load the sibling ``requests.json`` request records, when present."""
    try:
        payload = json.loads((run_path / 'requests.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ('requests', 'records', 'items'):
            if isinstance(payload.get(key), list):
                return payload[key]
    return []


def synthesize_spans_from_run_dir(run_dir: str | Path) -> list[dict]:
    """Return spans for a guard run directory, synthesizing them if needed.

    If ``traces.jsonl`` already exists and holds parseable spans, those are
    returned unchanged. Otherwise ``result.json`` (plus the sibling
    ``requests.json`` when present) is converted into a root AGENT span
    (``harness.task``) and per-request LLM spans (``gen_ai.chat``), and the
    spans are written to ``traces.jsonl`` so historical runs gain a trace
    file automatically. Span/trace ids are derived deterministically from the
    run id, keeping re-synthesis idempotent.
    """
    run_path = Path(run_dir)
    traces_path = run_path / 'traces.jsonl'
    if traces_path.is_file():
        try:
            spans = _parse_jsonl_spans(traces_path.read_text(encoding='utf-8'))
        except OSError:
            spans = []
        if spans:
            return spans

    try:
        payload = json.loads((run_path / 'result.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return []
    if not isinstance(payload, dict):
        return []

    run_id = run_path.name
    trace_id = _deterministic_hex(f'{run_id}:trace', 32)
    root_span_id = _deterministic_hex(f'{run_id}:root', 16)
    started_at = _first(payload, _STARTED_AT_KEYS)
    finished_at = _first(payload, _FINISHED_AT_KEYS)
    start_ns = _timestamp_to_ns(started_at)
    end_ns = _timestamp_to_ns(finished_at)
    if start_ns is None:
        start_ns = end_ns if end_ns is not None else time.time_ns()
    if end_ns is None:
        end_ns = start_ns
    if end_ns < start_ns:
        end_ns = start_ns

    task_model = str(_first(payload, _MODEL_KEYS) or 'unknown')
    root_span = {
        'name': 'harness.task',
        'span_id': root_span_id,
        'trace_id': trace_id,
        'parent_span_id': None,
        'start_time_ns': start_ns,
        'end_time_ns': end_ns,
        'status': 'OK',
        'attributes': {
            'openinference.span.kind': 'AGENT',
            'openinference.project.name': _synthesis_project_name(),
            'harness.task_id': str(_first(payload, _TASK_ID_KEYS) or run_id),
            'gen_ai.request.model': task_model,
            'harness.status': str(_first(payload, _STATUS_KEYS) or 'unknown'),
            'harness.steps': _to_int(_first(payload, _STEPS_KEYS)),
            'gen_ai.usage.cost_usd': _to_float(_first(payload, _COST_KEYS)),
            'harness.started_at': str(started_at) if started_at is not None else '',
            'harness.finished_at': str(finished_at) if finished_at is not None else '',
        },
    }
    spans = [root_span]

    requests = _load_run_requests(run_path)
    if requests:
        window = max((end_ns - start_ns) // (len(requests) + 1), 1)
        for index, record in enumerate(requests):
            child_start = start_ns + index * window
            child_end = child_start + window
            if isinstance(record, dict):
                record_start = _timestamp_to_ns(_first(record, _STARTED_AT_KEYS))
                record_end = _timestamp_to_ns(_first(record, _FINISHED_AT_KEYS))
                if record_start is not None:
                    child_start = record_start
                if record_end is not None:
                    child_end = max(record_end, child_start)
            record_model = _first(record, _MODEL_KEYS) if isinstance(record, dict) else None
            spans.append({
                'name': 'gen_ai.chat',
                'span_id': _deterministic_hex(f'{run_id}:request:{index}', 16),
                'trace_id': trace_id,
                'parent_span_id': root_span_id,
                'start_time_ns': child_start,
                'end_time_ns': child_end,
                'status': 'OK',
                'attributes': {
                    'openinference.span.kind': 'LLM',
                    'openinference.project.name': _synthesis_project_name(),
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.request.model': str(record_model or task_model),
                    'gen_ai.usage.cost_usd': (_to_float(_first(record, _COST_KEYS))
                                              if isinstance(record, dict) else 0.0),
                },
            })

    try:
        traces_path.parent.mkdir(parents=True, exist_ok=True)
        with traces_path.open('w', encoding='utf-8') as handle:
            for span in spans:
                handle.write(json.dumps(span, sort_keys=True) + '\n')
    except OSError:
        pass
    return spans


def _sha256_file(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(65536), b''):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _candidate_run_dirs(state_dir: Path) -> list[Path]:
    """Run directories to sync: ``state_dir`` itself if it is one, else newest runs/ entries."""
    if state_dir.is_file():
        return [state_dir.parent]
    if state_dir.is_dir() and ((state_dir / 'traces.jsonl').is_file()
                               or (state_dir / 'result.json').is_file()):
        return [state_dir]
    runs_dir = state_dir / 'runs'
    if not runs_dir.is_dir():
        return []
    candidates = [entry for entry in runs_dir.iterdir() if entry.is_dir()]
    try:
        candidates.sort(key=lambda entry: entry.stat().st_mtime, reverse=True)
    except OSError:
        pass
    return candidates


def sync_state_traces_to_phoenix(state_dir: str | Path, *, endpoint: str | None = None,
                                 timeout: float = 1.5, max_runs: int = 200) -> dict:
    """Sync historical run traces to the Phoenix collector, skipping synced runs.

    Scans run directories under ``state_dir`` (or treats ``state_dir`` itself
    as a single run dir / traces.jsonl file), ensures each has a
    ``traces.jsonl`` (synthesizing from ``result.json`` when missing), and
    flushes new or changed traces to the OTLP endpoint. A
    ``phoenix-sync-state.json`` ledger (``{'synced': {run_id: sha256}}``)
    keeps already-synced runs from being re-posted; the first unreachable
    collector stops further network attempts in the pass.
    """
    state = Path(state_dir)
    target = _resolve_otlp_endpoint(endpoint)
    ledger_base = state.parent if state.is_file() else state
    ledger_path = ledger_base / 'phoenix-sync-state.json'
    synced: dict[str, str] = {}
    try:
        payload = json.loads(ledger_path.read_text(encoding='utf-8'))
        if isinstance(payload, dict) and isinstance(payload.get('synced'), dict):
            synced = {str(key): str(value) for key, value in payload['synced'].items()}
    except (OSError, ValueError):
        pass

    candidates = _candidate_run_dirs(state)[:max_runs]
    total = newly_synced = already_synced = pending = failed_sync = 0
    for run_path in candidates:
        spans = synthesize_spans_from_run_dir(run_path)
        if not spans:
            continue
        traces_path = run_path / 'traces.jsonl'
        digest = _sha256_file(traces_path)
        if digest is None:
            continue
        run_id = run_path.name
        total += 1
        if synced.get(run_id) == digest:
            already_synced += 1
            continue
        if failed_sync:
            pending += 1
            continue
        tracer = HarnessTracer(otlp_endpoint=target)
        tracer.spans = list(spans)  # spans keep their own trace_id/span_id
        if tracer.flush_otlp(endpoint=target, timeout=timeout):
            synced[run_id] = digest
            newly_synced += 1
        else:
            failed_sync += 1
            pending += 1

    _atomic_write_json(ledger_path, {'synced': synced})
    return {
        'endpoint': target,
        'total_checked': total,
        'newly_synced': newly_synced,
        'already_synced': already_synced,
        'pending_unsynced': pending,
        'collector_reachable': failed_sync == 0,
    }
