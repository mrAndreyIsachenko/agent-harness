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
import http.client
import json
import os
import secrets
import struct
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
        status_obj = {'code': _OTLP_STATUS_CODES.get(span.get('status', 'OK'), 0)}
        if span.get('status_message'):
            status_obj['message'] = str(span['status_message'])
        otlp_span = {
            'traceId': span.get('trace_id', ''),
            'spanId': span.get('span_id', ''),
            'name': span.get('name', ''),
            'kind': _OTLP_SPAN_KINDS.get(attributes.get('openinference.span.kind'), 1),
            'startTimeUnixNano': str(span.get('start_time_ns', 0)),
            'endTimeUnixNano': str(span.get('end_time_ns', 0)),
            'attributes': [{'key': key, 'value': cls._otlp_value(value)}
                           for key, value in attributes.items()],
            'status': status_obj,
        }
        if span.get('parent_span_id'):
            otlp_span['parentSpanId'] = span['parent_span_id']
        return otlp_span

    @staticmethod
    def _pb_varint(value: int) -> bytes:
        value &= 0xFFFFFFFFFFFFFFFF
        out = bytearray()
        while value > 0x7F:
            out.append((value & 0x7F) | 0x80)
            value >>= 7
        out.append(value)
        return bytes(out)

    @classmethod
    def _pb_tag(cls, field_number: int, wire_type: int) -> bytes:
        return cls._pb_varint((field_number << 3) | wire_type)

    @classmethod
    def _pb_bytes(cls, field_number: int, data: bytes) -> bytes:
        return cls._pb_tag(field_number, 2) + cls._pb_varint(len(data)) + data

    @classmethod
    def _pb_str(cls, field_number: int, text: str) -> bytes:
        return cls._pb_bytes(field_number, text.encode('utf-8'))

    @classmethod
    def _pb_any_value(cls, value) -> bytes:
        if isinstance(value, bool):
            return cls._pb_tag(2, 0) + cls._pb_varint(1 if value else 0)
        if isinstance(value, int):
            return cls._pb_tag(3, 0) + cls._pb_varint(value)
        if isinstance(value, float):
            return cls._pb_tag(4, 1) + struct.pack('<d', value)
        if isinstance(value, str):
            return cls._pb_str(1, value)
        return cls._pb_str(1, json.dumps(value, sort_keys=True, default=str))

    @classmethod
    def _pb_kv(cls, key: str, value) -> bytes:
        return cls._pb_str(1, key) + cls._pb_bytes(2, cls._pb_any_value(value))

    @staticmethod
    def _hex_to_bytes(hex_str: str, length: int) -> bytes:
        try:
            raw = bytes.fromhex(str(hex_str or '').strip())
        except ValueError:
            raw = b''
        if len(raw) == length:
            return raw
        if len(raw) > length:
            return raw[:length]
        return raw.rjust(length, b'\x00')

    @classmethod
    def _span_to_pb(cls, span: dict) -> bytes:
        attributes = span.get('attributes') or {}
        parts = [
            cls._pb_bytes(1, cls._hex_to_bytes(span.get('trace_id', ''), 16)),
            cls._pb_bytes(2, cls._hex_to_bytes(span.get('span_id', ''), 8)),
        ]
        if span.get('parent_span_id'):
            parts.append(cls._pb_bytes(4, cls._hex_to_bytes(span['parent_span_id'], 8)))
        parts.append(cls._pb_str(5, str(span.get('name', ''))))
        kind = _OTLP_SPAN_KINDS.get(attributes.get('openinference.span.kind'), 1)
        parts.append(cls._pb_tag(6, 0) + cls._pb_varint(kind))
        start_ns = max(0, int(span.get('start_time_ns') or 0))
        end_ns = max(0, int(span.get('end_time_ns') or 0))
        parts.append(cls._pb_tag(7, 1) + struct.pack('<Q', start_ns))
        parts.append(cls._pb_tag(8, 1) + struct.pack('<Q', end_ns))
        for key, value in attributes.items():
            parts.append(cls._pb_bytes(9, cls._pb_kv(str(key), value)))
        status_code = _OTLP_STATUS_CODES.get(span.get('status', 'OK'), 0)
        status_parts = []
        if span.get('status_message'):
            status_parts.append(cls._pb_str(2, str(span['status_message'])))
        status_parts.append(cls._pb_tag(3, 0) + cls._pb_varint(status_code))
        parts.append(cls._pb_bytes(15, b''.join(status_parts)))
        return b''.join(parts)

    def to_otlp_protobuf(self) -> bytes:
        """Encode recorded spans into OTLP Protobuf ExportTraceServiceRequest bytes."""
        with self._lock:
            spans = list(self.spans)
        resource_bytes = (
            self._pb_bytes(1, self._pb_kv('service.name', self.service_name))
            + self._pb_bytes(1, self._pb_kv('openinference.project.name', self.project_name))
        )
        scope_bytes = self._pb_str(1, _SCOPE_NAME) + self._pb_str(2, _SCOPE_VERSION)
        scope_spans_parts = [self._pb_bytes(1, scope_bytes)]
        for span in spans:
            scope_spans_parts.append(self._pb_bytes(2, self._span_to_pb(span)))
        resource_spans_bytes = (
            self._pb_bytes(1, resource_bytes)
            + self._pb_bytes(2, b''.join(scope_spans_parts))
        )
        return self._pb_bytes(1, resource_spans_bytes)

    def flush_otlp(self, *, endpoint: str | None = None, timeout: float = 1.5) -> bool:
        """POST spans as OTLP/HTTP Protobuf (falling back to JSON if rejected); True on 2xx."""
        target = endpoint or self.otlp_endpoint
        if not target:
            return False
        base_headers = {}
        if self.arize_api_key:
            base_headers['api_key'] = self.arize_api_key
            base_headers['x-arize-api-key'] = self.arize_api_key
        if self.arize_space_id:
            base_headers['space_id'] = self.arize_space_id
            base_headers['x-arize-space-id'] = self.arize_space_id

        def _post(body: bytes, content_type: str) -> bool:
            headers = {**base_headers, 'Content-Type': content_type}
            request = urllib.request.Request(target, data=body, headers=headers, method='POST')
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return 200 <= getattr(response, 'status', 200) < 300

        try:
            return _post(self.to_otlp_protobuf(), 'application/x-protobuf')
        except urllib.error.HTTPError as exc:
            code = exc.code
            exc.close()
            if code not in (400, 415, 422, 500, 501):
                return False
        except http.client.HTTPException:
            pass
        except (urllib.error.URLError, OSError, ValueError):
            return False

        try:
            return _post(json.dumps(self.to_otlp_payload()).encode('utf-8'), 'application/json')
        except urllib.error.HTTPError as exc:
            exc.close()
            return False
        except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError):
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


_OK_STATUSES = {
    'validation_passed_review_required',
    'patch_ready_review_required',
    'configuration_verified_no_model_call',
    'succeeded',
    'ok',
}


def _load_task_json(run_path: Path) -> dict:
    try:
        payload = json.loads((run_path / 'task.json').read_text(encoding='utf-8'))
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError):
        return {}


def _load_run_events_times(run_path: Path) -> tuple[int | None, int | None]:
    events_path = run_path / 'events.jsonl'
    if not events_path.is_file():
        return None, None
    first_ns = last_ns = started_ns = finished_ns = None
    try:
        with events_path.open('r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(ev, dict):
                    continue
                ts_ns = _timestamp_to_ns(ev.get('time'))
                if ts_ns is None:
                    continue
                if first_ns is None:
                    first_ns = ts_ns
                last_ns = ts_ns
                ev_name = ev.get('event')
                if ev_name == 'run_started' and started_ns is None:
                    started_ns = ts_ns
                elif ev_name == 'run_finished':
                    finished_ns = ts_ns
    except OSError:
        return None, None
    return (started_ns if started_ns is not None else first_ns,
            finished_ns if finished_ns is not None else last_ns)


def _needs_resynthesis(spans: list[dict], run_path: Path) -> bool:
    if not (run_path / 'result.json').is_file():
        return False
    if any((s.get('end_time_ns') or 0) < (s.get('start_time_ns') or 0) for s in spans):
        return True
    root = next((s for s in spans if s.get('name') == 'harness.task'), None)
    if root is not None:
        attrs = root.get('attributes') or {}
        if 'input.value' not in attrs:
            return True
        if root.get('start_time_ns') == root.get('end_time_ns'):
            # Re-synthesize if duration_seconds or events indicate non-zero duration.
            return (run_path / 'events.jsonl').is_file() or (run_path / 'requests.json').is_file()
        return False
    return (run_path / 'requests.json').is_file() or (run_path / 'events.jsonl').is_file()


def synthesize_spans_from_run_dir(run_dir: str | Path) -> list[dict]:
    """Return spans for a guard run directory, synthesizing them if needed.

    If ``traces.jsonl`` already exists and holds complete spans, those are
    returned unchanged. Otherwise ``result.json`` (plus ``task.json``,
    ``events.jsonl``, and ``requests.json`` when present) is converted into a
    root AGENT span (``harness.task``) and per-request LLM spans
    (``gen_ai.chat``), preserving any existing guardrail spans, and written to
    ``traces.jsonl``.
    """
    run_path = Path(run_dir)
    traces_path = run_path / 'traces.jsonl'
    existing_spans: list[dict] = []
    if traces_path.is_file():
        try:
            existing_spans = _parse_jsonl_spans(traces_path.read_text(encoding='utf-8'))
        except OSError:
            existing_spans = []
        if existing_spans and not _needs_resynthesis(existing_spans, run_path):
            return existing_spans

    result_path = run_path / 'result.json'
    try:
        payload = json.loads(result_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return existing_spans
    if not isinstance(payload, dict):
        return existing_spans

    run_id = run_path.name
    trace_id = _deterministic_hex(f'{run_id}:trace', 32)
    root_span_id = _deterministic_hex(f'{run_id}:root', 16)
    requests = _load_run_requests(run_path)
    task_json = _load_task_json(run_path)

    started_at = _first(payload, _STARTED_AT_KEYS)
    finished_at = _first(payload, _FINISHED_AT_KEYS)
    start_ns = _timestamp_to_ns(started_at)
    end_ns = _timestamp_to_ns(finished_at)

    if start_ns is None or end_ns is None:
        ev_start_ns, ev_end_ns = _load_run_events_times(run_path)
        if start_ns is None:
            start_ns = ev_start_ns
        if end_ns is None:
            end_ns = ev_end_ns

    if (start_ns is None or end_ns is None) and requests:
        req_starts, req_ends = [], []
        for record in requests:
            if not isinstance(record, dict):
                continue
            time_obj = record.get('time') if isinstance(record.get('time'), dict) else {}
            rs = (_timestamp_to_ns(time_obj.get('created'))
                  or _timestamp_to_ns(_first(record, _STARTED_AT_KEYS)))
            re = (_timestamp_to_ns(time_obj.get('completed'))
                  or _timestamp_to_ns(time_obj.get('streamed'))
                  or _timestamp_to_ns(_first(record, _FINISHED_AT_KEYS)))
            if rs is not None:
                req_starts.append(rs)
            if re is not None:
                req_ends.append(re)
        if start_ns is None and req_starts:
            start_ns = min(req_starts)
        if end_ns is None and req_ends:
            end_ns = max(req_ends)

    if end_ns is None:
        try:
            end_ns = int(result_path.stat().st_mtime * 1e9)
        except OSError:
            end_ns = time.time_ns()
    if start_ns is None:
        dur = _to_float(payload.get('duration_seconds'))
        start_ns = (end_ns - int(dur * 1e9)) if dur > 0 else end_ns
    if end_ns < start_ns:
        end_ns = start_ns

    project_name = _synthesis_project_name()
    task_id = str(_first(payload, _TASK_ID_KEYS) or run_id)
    task_model = str(_first(payload, _MODEL_KEYS) or task_json.get('model') or 'unknown')
    raw_status = str(_first(payload, _STATUS_KEYS) or 'unknown')
    otel_status = 'OK' if raw_status in _OK_STATUSES else 'ERROR'
    reason = str(payload.get('reason') or '')
    steps = _to_int(_first(payload, _STEPS_KEYS))
    total_cost = _to_float(_first(payload, _COST_KEYS))
    if total_cost == 0.0 and payload.get('observed_cost') is not None:
        total_cost = _to_float(payload.get('observed_cost'))

    child_spans: list[dict] = []
    sum_prompt_tokens = 0
    sum_completion_tokens = 0
    sum_total_tokens = 0
    sum_req_cost = 0.0

    if requests:
        window = max((end_ns - start_ns) // (len(requests) + 1), 1)
        for index, record in enumerate(requests):
            child_start = start_ns + index * window
            child_end = child_start + window
            record_dict = record if isinstance(record, dict) else {}
            time_obj = record_dict.get('time') if isinstance(record_dict.get('time'), dict) else {}
            record_start = (_timestamp_to_ns(time_obj.get('created'))
                            or _timestamp_to_ns(_first(record_dict, _STARTED_AT_KEYS)))
            record_end = (_timestamp_to_ns(time_obj.get('completed'))
                          or _timestamp_to_ns(time_obj.get('streamed'))
                          or _timestamp_to_ns(_first(record_dict, _FINISHED_AT_KEYS)))
            if record_start is not None:
                child_start = record_start
            if record_end is not None:
                child_end = max(record_end, child_start)
            if child_end < child_start:
                child_end = child_start

            record_model = str(_first(record_dict, _MODEL_KEYS) or task_model)
            rec_cost = _to_float(_first(record_dict, _COST_KEYS))
            sum_req_cost += rec_cost

            tokens = record_dict.get('tokens') if isinstance(record_dict.get('tokens'), dict) else {}
            cache = tokens.get('cache') if isinstance(tokens.get('cache'), dict) else {}
            rec_prompt = _to_int(tokens.get('input')) + _to_int(cache.get('read'))
            rec_completion = _to_int(tokens.get('output')) + _to_int(tokens.get('reasoning'))
            rec_total = rec_prompt + rec_completion
            sum_prompt_tokens += rec_prompt
            sum_completion_tokens += rec_completion
            sum_total_tokens += rec_total

            rec_error = bool(record_dict.get('error') or record_dict.get('state') == 'error')
            rec_status = 'ERROR' if rec_error else 'OK'
            rec_input = json.dumps({
                'ordinal': record_dict.get('ordinal', index + 1),
                'model': record_model,
            }, ensure_ascii=False)
            rec_output = json.dumps({
                'finish': record_dict.get('finish'),
                'state': record_dict.get('state'),
                'provider_request_ids': record_dict.get('provider_request_ids'),
                'tokens': tokens,
                'cost': rec_cost,
            }, ensure_ascii=False)

            child_spans.append({
                'name': 'gen_ai.chat',
                'span_id': _deterministic_hex(f'{run_id}:request:{index}', 16),
                'trace_id': trace_id,
                'parent_span_id': root_span_id,
                'start_time_ns': child_start,
                'end_time_ns': child_end,
                'status': rec_status,
                'attributes': {
                    'openinference.span.kind': 'LLM',
                    'openinference.project.name': project_name,
                    'gen_ai.operation.name': 'chat',
                    'gen_ai.request.model': record_model,
                    'llm.model_name': record_model,
                    'gen_ai.usage.cost_usd': rec_cost,
                    'llm.cost.total': rec_cost,
                    'llm.token_count.prompt': rec_prompt,
                    'llm.token_count.completion': rec_completion,
                    'llm.token_count.total': rec_total,
                    'input.value': rec_input,
                    'input.mime_type': 'application/json',
                    'output.value': rec_output,
                    'output.mime_type': 'application/json',
                },
            })

    if total_cost == 0.0 and sum_req_cost > 0.0:
        total_cost = sum_req_cost

    prompt_text = str(task_json.get('prompt') or '')
    if len(prompt_text) > 4000:
        prompt_text = prompt_text[:4000] + '...[truncated]'
    input_value = json.dumps({
        'task_id': task_id,
        'model': task_model,
        'files': task_json.get('files', []),
        'prompt': prompt_text,
    }, ensure_ascii=False)
    output_value = json.dumps({
        'status': raw_status,
        'reason': reason,
        'steps': steps,
        'cost': total_cost,
        'test_return_code': payload.get('test_return_code'),
    }, ensure_ascii=False)

    root_attrs = {
        'openinference.span.kind': 'AGENT',
        'openinference.project.name': project_name,
        'harness.task_id': task_id,
        'gen_ai.request.model': task_model,
        'llm.model_name': task_model,
        'harness.status': raw_status,
        'harness.steps': steps,
        'gen_ai.usage.cost_usd': total_cost,
        'llm.cost.total': total_cost,
        'llm.token_count.prompt': sum_prompt_tokens,
        'llm.token_count.completion': sum_completion_tokens,
        'llm.token_count.total': sum_total_tokens,
        'otel.status_code': otel_status,
        'otel.status_description': reason,
        'input.value': input_value,
        'input.mime_type': 'application/json',
        'output.value': output_value,
        'output.mime_type': 'application/json',
        'harness.started_at': str(started_at) if started_at is not None else '',
        'harness.finished_at': str(finished_at) if finished_at is not None else '',
    }
    root_span = {
        'name': 'harness.task',
        'span_id': root_span_id,
        'trace_id': trace_id,
        'parent_span_id': None,
        'start_time_ns': start_ns,
        'end_time_ns': end_ns,
        'status': otel_status,
        'status_message': reason if otel_status == 'ERROR' else '',
        'attributes': root_attrs,
    }
    spans = [root_span, *child_spans]

    # Preserve any guardrail spans recorded live by the proxy.
    for span in existing_spans:
        attrs = span.get('attributes') or {}
        if attrs.get('openinference.span.kind') == 'GUARDRAIL' or str(span.get('name', '')).startswith('guard.'):
            preserved = dict(span)
            preserved['trace_id'] = trace_id
            if not preserved.get('parent_span_id'):
                preserved['parent_span_id'] = root_span_id
            spans.append(preserved)

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
                                 timeout: float = 1.5, max_runs: int | None = None) -> dict:
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

    candidates = _candidate_run_dirs(state)
    if max_runs is not None:
        candidates = candidates[:max_runs]
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
