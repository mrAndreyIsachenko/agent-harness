"""Finops core for guarded OpenCode runs.

Request-ledger capture, billing reconciliation, and budget enforcement:
request-ledger capture, OpenRouter billing reconciliation, and telemetry
violation checks over the local opencode.db. Standard library only.
"""

import fcntl
import hashlib
import json
import math
import os
import sqlite3
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from .proxy import GuardError
from .routing import is_local_or_sml_model

DB = Path(os.environ.get("HARNESS_DB") or os.environ.get("OPENCODE_DB") or (Path.home() / ".local/share/opencode/opencode.db"))
STATE = Path(os.environ.get("HARNESS_STATE_DIR") or (Path.home() / ".local/share/agent-harness/state"))
LIMITS = {'max_steps': (16, 25), 'max_cost_usd': (.15, .30),
          'max_task_cost_usd': (.40, .60), 'timeout_seconds': (240, 360),
          'test_timeout_seconds': (120, 300)}

_GEN_ID_CHARS = frozenset(
    'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-')


def _is_generation_id(value):
    """Equivalent of re.fullmatch(r'gen-[A-Za-z0-9_-]+', value)."""
    if not isinstance(value, str) or not value.startswith('gen-') or len(value) == 4:
        return False
    return all(c in _GEN_ID_CHARS for c in value[4:])


def _is_local_zero_cost(model=None, upstream_mode='openrouter'):
    """Local/SML models and direct vLLM/llama.cpp upstreams are billed at $0 API cost."""
    return upstream_mode in ('vllm', 'llamacpp') or bool(is_local_or_sml_model(model))


class _Counter(dict):
    """Minimal collections.Counter replacement: missing keys count as 0."""

    def __missing__(self, key):
        return 0


@contextmanager
def locks(state, files, task_id):
    root = Path(state) / 'locks'
    root.mkdir(parents=True, exist_ok=True)
    # Files and task IDs are independent locks; different tasks can share a file.
    keys = ['file:' + str(Path(p).resolve()) for p in files] + ['task:' + str(task_id)]
    fds = []
    try:
        for key in sorted(set(keys)):
            name = hashlib.sha256(key.encode()).hexdigest() + '.lock'
            fd = os.open(root / name, os.O_CREAT | os.O_RDWR, 0o600)
            fds.append(fd)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise GuardError('files_or_task_already_running') from None
        yield
    finally:
        for fd in reversed(fds):
            os.close(fd)  # Never unlink: another process may hold the old inode.


def number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise GuardError(f'invalid_{name}')
    return value

def private_write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def private_log(path):
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'w+')

def event(run_dir, kind, **fields):
    with os.fdopen(os.open(Path(run_dir) / 'events.jsonl',
                           os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600), 'a') as f:
        f.write(json.dumps({'time': time.time(), 'event': kind, **fields}, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())

def request_records(snap, proxy_summary=None, *, model=None, upstream_mode='openrouter'):
    """Capture billing metadata only, never prompts, tool outputs or reasoning."""
    summary = proxy_summary if isinstance(proxy_summary, dict) else {}
    if upstream_mode == 'openrouter' and 'upstream_mode' in summary:
        upstream_mode = summary.get('upstream_mode') or upstream_mode
    if model is None:
        model = summary.get('default_model')
    local = _is_local_zero_cost(model, upstream_mode)
    records = []
    def ids(value):
        if isinstance(value, dict):
            found = []
            for key, item in value.items():
                if key in {'content', 'parsed', 'input', 'text'}:
                    continue
                if isinstance(item, str) and key in {'id', 'request_id', 'response_id', 'generation_id'}:
                    if _is_generation_id(item):
                        found.append(item)
                elif isinstance(item, (dict, list)):
                    found.extend(ids(item))
            return found
        if isinstance(value, list):
            return [found for item in value for found in ids(item)]
        return []
    raw_dispatched = (proxy_summary or {}).get('dispatched') or []
    dispatched = [
        d for d in raw_dispatched
        if not (d.get('http_status') is None and not d.get('generation_id') and d.get('error') == 'URLError')
    ]
    blocked_count = int((proxy_summary or {}).get('blocked_count') or 0)
    idx = 0
    for kind, data in snap['messages']:
        if kind != 'assistant':
            continue
        local_ids = sorted(set(ids(data)))
        proxy_gid = dispatched[idx].get('generation_id') if idx < len(dispatched) else None
        provider_ids = sorted(set(local_ids + ([proxy_gid] if proxy_gid else [])))
        state = 'failed' if data.get('error') else 'finished' if data.get('finish') else 'in_flight'
        cost = data.get('cost')
        if local and cost is None and state == 'finished':
            cost = 0.0
        if local_ids:
            identity_status = 'recorded'
        elif proxy_gid:
            identity_status = 'recorded_by_proxy'
        elif (proxy_summary is not None and idx >= len(dispatched)
              and blocked_count >= (idx - len(dispatched) + 1)
              and cost is None and state in {'failed', 'in_flight'}):
            state = 'blocked_locally'
            cost = 0.0
            identity_status = 'blocked_before_dispatch'
        else:
            identity_status = 'not_exposed_by_opencode'
        records.append({'ordinal': len(records) + 1, 'time': data.get('time'),
                        'state': state, 'finish': data.get('finish'), 'cost': cost,
                        'tokens': data.get('tokens'), 'provider_request_ids': provider_ids,
                        'identity_status': identity_status})
        idx += 1
    while idx < len(dispatched):
        d = dispatched[idx]
        proxy_gid = d.get('generation_id')
        records.append({'ordinal': len(records) + 1, 'time': {'started_at': d.get('started_at')},
                        'state': 'in_flight', 'finish': None, 'cost': None, 'tokens': None,
                        'provider_request_ids': [proxy_gid] if proxy_gid else [],
                        'identity_status': 'recorded_by_proxy' if proxy_gid else 'not_exposed_by_opencode'})
        idx += 1
    return records

def capture_requests(run_dir, sid, snap, proxy_summary=None):
    value = {'session_id': sid, 'observed_cost': snap['cost'],
             'requests': request_records(snap, proxy_summary)}
    if proxy_summary is not None:
        value['proxy'] = proxy_summary
    path = Path(run_dir) / 'requests.json'
    previous = json.loads(path.read_text()) if path.exists() else None
    if previous != value:
        private_write(path, value)
        event(run_dir, 'request_state', **value)

def resolved_cost(path, result):
    if result.get('cost') is not None:
        return number(result['cost'], 'prior_cost')
    reconciliation = Path(path).parent / 'billing-reconciliation.json'
    if not reconciliation.exists():
        raise GuardError('prior_cost_unknown_reconcile_before_retry')
    data = json.loads(reconciliation.read_text())
    if data.get('result_sha256') != hashlib.sha256(Path(path).read_bytes()).hexdigest():
        raise GuardError('reconciliation_source_changed')
    if (data.get('schema') != 'verified_billing.v1' or data.get('session_id') != result.get('session_id')
            or data.get('task_id') != result.get('task_id')):
        raise GuardError('invalid_billing_reconciliation')
    cost = number(data.get('cost'), 'reconciled_cost')
    if cost + 1e-10 < number(result.get('observed_cost', 0), 'observed_cost'):
        raise GuardError('reconciliation_below_observed_cost')
    return cost

def fetch_generation(generation_id, *, key=None):
    key = key or os.environ.get('OPENROUTER_API_KEY')
    if not key:
        raise GuardError('OPENROUTER_API_KEY_required_for_billing_only')
    request = urllib.request.Request('https://openrouter.ai/api/v1/generation?id=' + generation_id,
                                     headers={'Authorization': 'Bearer ' + key})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.load(response)['data']
    except urllib.error.HTTPError as exc:
        raise GuardError(f'billing_unavailable_HTTP_{exc.code}') from None
    except (OSError, ValueError, KeyError):
        raise GuardError('billing_response_unavailable') from None

def fetch_generation_settled(generation_id, *, key=None, attempts=4, delays=(1.0, 2.0, 3.0)):
    last_exc = None
    for attempt in range(attempts):
        try:
            bill = fetch_generation(generation_id, key=key)
            if bill.get('id') == generation_id and (bill.get('native_finish_reason')
                    or bill.get('finish_reason') or bill.get('cancelled') is True):
                return bill
            last_exc = GuardError('billing_not_final_or_wrong_identity')
        except GuardError as exc:
            last_exc = exc
        if attempt + 1 < attempts:
            wait_s = delays[min(attempt, len(delays) - 1)] if delays else 0.6
            time.sleep(wait_s)
    raise last_exc or GuardError('billing_response_unavailable')

def _reconcile_locked(run_dir, path, result, evidence, *, fetch=fetch_generation, db=DB):
    task_id = result['task_id']
    target = run_dir / 'billing-reconciliation.json'
    if target.exists() or result.get('cost') is not None:
        raise GuardError('already_reconciled')
    if result.get('status') == 'pending':
        raise GuardError('worker_must_be_stopped_before_reconciliation')
    ledger_path = run_dir / 'requests.json'
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
        if ledger.get('proxy') and any(
            isinstance(d, dict) and d.get('http_status') is None and not d.get('generation_id') and d.get('error') == 'URLError'
            for d in (ledger['proxy'].get('dispatched') or [])
        ):
            snap = snapshot(db, result['session_id'])
            if snap is not None:
                ledger = {**ledger, 'requests': request_records(snap, ledger['proxy'])}
    else:
        snap = snapshot(db, result['session_id'])
        if snap is None:
            raise GuardError('missing_request_ledger')
        ledger = {'session_id': result['session_id'], 'requests': request_records(snap)}
    rows = ledger['requests']
    proxy_by_gid = {
        d['generation_id']: d
        for d in ((ledger.get('proxy') or {}).get('dispatched') or [])
        if isinstance(d, dict) and isinstance(d.get('generation_id'), str)
    }
    supplied = evidence.get('requests', [])
    if (not rows and evidence.get('empty_ledger_resolution') == 'conservative_step_cap'
            and evidence.get('session_id') == result.get('session_id')
            and ledger.get('session_id') == result.get('session_id')
            and supplied == []):
        if not all(isinstance(evidence.get(k), str) and evidence[k].strip()
                   for k in ('mapping_source', 'verified_by')):
            raise GuardError('mapping_evidence_required')
        cap = number(evidence.get('capped_cost'), 'capped_cost')
        if cap + 1e-10 < 0.01:
            raise GuardError('conservative_cap_too_low')
        bills = [{'id': None, 'resolution': 'conservative_step_cap', 'ordinal': 0, 'total_cost': cap}]
    else:
        if (not rows or evidence.get('session_id') != result.get('session_id')
                or ledger.get('session_id') != result.get('session_id')
                or not isinstance(supplied, list) or len(supplied) != len(rows)
                or [r.get('ordinal') for r in supplied] != [r['ordinal'] for r in rows]):
            raise GuardError('complete_request_coverage_required')
        min_step_cap = max([number(r['cost'], 'local_request_cost')
                            for r in rows if r.get('cost') is not None] + [0.01])
        bills, seen = [], set()
        for row, mapping in zip(rows, supplied):
            resolution = mapping.get('resolution')
            if resolution == 'blocked_before_dispatch':
                if (row.get('state') != 'blocked_locally'
                        or row.get('identity_status') != 'blocked_before_dispatch'):
                    raise GuardError('invalid_blocked_before_dispatch_claim')
                bills.append({'id': None, 'ordinal': row['ordinal'],
                              'resolution': 'blocked_before_dispatch', 'total_cost': 0.0})
                continue
            if resolution == 'local_telemetry_finished':
                if (row.get('provider_request_ids') or row.get('state') != 'finished'
                        or row.get('cost') is None):
                    raise GuardError('local_telemetry_resolution_requires_unexposed_finished_request')
                if not all(isinstance(mapping.get(k), str) and mapping[k].strip()
                           for k in ('mapping_source', 'verified_by')):
                    raise GuardError('mapping_evidence_required')
                local_cost = number(row['cost'], 'local_request_cost')
                settled = (number(mapping['settled_cost'], 'settled_cost')
                           if mapping.get('settled_cost') is not None else local_cost)
                if settled + 1e-10 < local_cost:
                    raise GuardError('reconciliation_below_observed_cost')
                bills.append({'id': None, 'ordinal': row['ordinal'],
                              'resolution': 'local_telemetry_finished', 'total_cost': settled})
                continue
            if resolution == 'conservative_step_cap':
                if row.get('provider_request_ids'):
                    gid0 = row['provider_request_ids'][0]
                    p0 = proxy_by_gid.get(gid0)
                    if not (p0 and p0.get('completed') is False and p0.get('stream_cost') is None
                            and not os.environ.get('OPENROUTER_API_KEY')):
                        raise GuardError('provider_id_present_fetch_required')
                if row.get('cost') is not None:
                    raise GuardError('finished_request_must_use_actual_cost')
                if not all(isinstance(mapping.get(k), str) and mapping[k].strip()
                           for k in ('mapping_source', 'verified_by')):
                    raise GuardError('mapping_evidence_required')
                cap = number(mapping.get('capped_cost'), 'capped_cost')
                if cap + 1e-10 < min_step_cap:
                    raise GuardError('conservative_cap_too_low')
                bills.append({'id': None, 'ordinal': row['ordinal'],
                              'resolution': 'conservative_step_cap', 'total_cost': cap})
                continue
            if resolution == 'proxy_stream_usage':
                gid = mapping.get('generation_id')
                if not _is_generation_id(gid) or gid in seen:
                    raise GuardError('invalid_or_duplicate_generation_id')
                if gid not in (row.get('provider_request_ids') or []):
                    raise GuardError('provider_request_identity_mismatch')
                pentry = proxy_by_gid.get(gid)
                if not pentry or pentry.get('completed') is not True or pentry.get('stream_cost') is None:
                    raise GuardError('proxy_stream_usage_incomplete')
                seen.add(gid)
                cost = number(pentry['stream_cost'], 'proxy_stream_cost')
                if row.get('cost') is not None and cost + 1e-10 < number(row['cost'], 'local_request_cost'):
                    raise GuardError('reconciliation_below_observed_cost')
                bills.append({'id': gid, 'ordinal': row['ordinal'],
                              'resolution': 'proxy_stream_usage', 'total_cost': cost})
                continue
            if resolution is not None:
                raise GuardError('unsupported_reconciliation_resolution')
            gid = mapping.get('generation_id')
            if not _is_generation_id(gid) or gid in seen:
                raise GuardError('invalid_or_duplicate_generation_id')
            seen.add(gid)
            if row['provider_request_ids']:
                if gid not in row['provider_request_ids']:
                    raise GuardError('provider_request_identity_mismatch')
            elif not all(isinstance(mapping.get(k), str) and mapping[k].strip()
                         for k in ('mapping_source', 'verified_by')):
                raise GuardError('mapping_evidence_required')
            try:
                bill = fetch(gid)
            except GuardError as exc:
                pentry = proxy_by_gid.get(gid) if evidence.get('allow_proxy_stream_fallback') else None
                if (str(exc).startswith('billing_unavailable_HTTP_404') and pentry
                        and pentry.get('completed') is True and pentry.get('stream_cost') is not None):
                    cost = number(pentry['stream_cost'], 'proxy_stream_cost')
                    if row.get('cost') is not None and cost + 1e-10 < number(row['cost'], 'local_request_cost'):
                        raise GuardError('reconciliation_below_observed_cost') from None
                    bills.append({'id': gid, 'ordinal': row['ordinal'],
                                  'resolution': 'proxy_stream_usage', 'total_cost': cost})
                    continue
                if ((str(exc).startswith('billing_unavailable_HTTP_404')
                        or str(exc) == 'billing_not_final_or_wrong_identity')
                        and pentry and pentry.get('completed') is False
                        and pentry.get('stream_cost') is None and row.get('cost') is None):
                    bills.append({'id': gid, 'ordinal': row['ordinal'],
                                  'resolution': 'aborted_stream_conservative_cap', 'total_cost': min_step_cap})
                    continue
                raise
            if bill.get('id') != gid or not (bill.get('native_finish_reason') or bill.get('finish_reason')
                                            or bill.get('cancelled') is True):
                raise GuardError('billing_not_final_or_wrong_identity')
            cost = number(bill.get('total_cost'), 'provider_cost')
            if row.get('cost') is not None and cost + 1e-10 < number(row['cost'], 'local_request_cost'):
                raise GuardError('reconciliation_below_observed_cost')
            bills.append({k: bill.get(k) for k in ('id', 'total_cost', 'created_at', 'provider_name',
                     'native_finish_reason', 'finish_reason', 'cancelled', 'tokens_prompt', 'tokens_completion')})
    total = math.fsum(b['total_cost'] for b in bills)
    if total + 1e-10 < number(result.get('observed_cost', 0), 'observed_cost'):
        raise GuardError('reconciliation_below_observed_cost')
    record = {'schema': 'verified_billing.v1', 'task_id': task_id, 'session_id': result['session_id'],
              'result_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'cost': total,
              'verified_at': time.time(), 'evidence': evidence, 'billing': bills}
    # Publish only a complete file, without ever replacing earlier evidence.
    temporary = run_dir / ('billing.' + hashlib.sha256(os.urandom(16)).hexdigest() + '.pending')
    try:
        private_write(temporary, record)
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    event(run_dir, 'billing_reconciled', cost=total)
    return record

def reconcile_run(run_dir, evidence, *, fetch=fetch_generation, state=STATE, db=DB):
    """Append verified billing without rewriting the original result or launching work."""
    run_dir = Path(run_dir).resolve()
    if run_dir.parent != (Path(state).resolve() / 'runs'):
        raise GuardError('run_outside_journal')
    path = run_dir / 'result.json'
    result = json.loads(path.read_text())
    with locks(state, [], result.get('task_id') or run_dir.name):
        return _reconcile_locked(run_dir, path, result, evidence, fetch=fetch, db=db)

def build_unexposed_settlement_evidence(run_dir, *, verified_by, mapping_source, db=DB):
    if not isinstance(verified_by, str) or not verified_by.strip():
        raise GuardError('verified_by_required')
    if not isinstance(mapping_source, str) or not mapping_source.strip():
        raise GuardError('mapping_source_required')
    run_dir = Path(run_dir).resolve()
    result = json.loads((run_dir / 'result.json').read_text())
    ledger_path = run_dir / 'requests.json'
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
        if ledger.get('proxy') and any(
            isinstance(d, dict) and d.get('http_status') is None and not d.get('generation_id') and d.get('error') == 'URLError'
            for d in (ledger['proxy'].get('dispatched') or [])
        ):
            snap = snapshot(db, result['session_id'])
            if snap is not None:
                ledger = {**ledger, 'requests': request_records(snap, ledger['proxy'])}
    else:
        snap = snapshot(db, result['session_id'])
        if snap is None:
            raise GuardError('missing_request_ledger')
        ledger = {'session_id': result['session_id'], 'requests': request_records(snap)}
    rows = ledger['requests']
    proxy_by_gid = {
        d['generation_id']: d
        for d in ((ledger.get('proxy') or {}).get('dispatched') or [])
        if isinstance(d, dict) and isinstance(d.get('generation_id'), str)
    }
    has_api_key = bool(os.environ.get('OPENROUTER_API_KEY'))
    meta = {'mapping_source': mapping_source.strip(), 'verified_by': verified_by.strip()}
    if not rows:
        return {'session_id': result['session_id'], 'requests': [],
                'empty_ledger_resolution': 'conservative_step_cap',
                'capped_cost': max(number(result.get('observed_cost', 0), 'observed_cost'), 0.01), **meta}
    step_cap = max([number(r['cost'], 'local_request_cost')
                    for r in rows if r.get('cost') is not None] + [0.01])
    supplied = []
    for row in rows:
        if row.get('state') == 'blocked_locally' and row.get('identity_status') == 'blocked_before_dispatch':
            supplied.append({'ordinal': row['ordinal'], 'resolution': 'blocked_before_dispatch'})
        elif row.get('provider_request_ids'):
            gid = row['provider_request_ids'][0]
            pentry = proxy_by_gid.get(gid)
            if not has_api_key and pentry and pentry.get('completed') is True and pentry.get('stream_cost') is not None:
                supplied.append({'ordinal': row['ordinal'], 'generation_id': gid,
                                 'resolution': 'proxy_stream_usage'})
            elif not has_api_key and pentry and pentry.get('completed') is False and pentry.get('stream_cost') is None and row.get('cost') is None:
                supplied.append({'ordinal': row['ordinal'], 'resolution': 'conservative_step_cap',
                                 'capped_cost': step_cap, **meta})
            else:
                supplied.append({'ordinal': row['ordinal'], 'generation_id': gid})
        elif row.get('state') == 'finished' and row.get('cost') is not None:
            supplied.append({'ordinal': row['ordinal'], 'resolution': 'local_telemetry_finished', **meta})
        else:
            supplied.append({'ordinal': row['ordinal'], 'resolution': 'conservative_step_cap',
                             'capped_cost': step_cap, **meta})
    return {'session_id': result['session_id'], 'allow_proxy_stream_fallback': True, 'requests': supplied}

def auto_reconcile_from_proxy(run_dir, path, result, *, auth_bearer=None, fetch=None, db=DB):
    ledger_path = Path(run_dir) / 'requests.json'
    if not ledger_path.exists() or result.get('cost') is not None or not result.get('session_id'):
        return None
    ledger = json.loads(ledger_path.read_text())
    rows = ledger.get('requests') or []
    if not rows:
        return None
    raw_dispatched = ((ledger.get('proxy') or {}).get('dispatched') or [])
    dispatched = [
        d for d in raw_dispatched
        if isinstance(d, dict)
        and not (d.get('http_status') is None and not d.get('generation_id') and d.get('error') == 'URLError')
    ]
    proxy_by_gid = {
        d['generation_id']: d
        for d in raw_dispatched
        if isinstance(d, dict) and isinstance(d.get('generation_id'), str)
    }
    step_cap = max([number(r['cost'], 'local_request_cost')
                    for r in rows if r.get('cost') is not None] + [0.01])
    key = os.environ.get('OPENROUTER_API_KEY') or auth_bearer
    supplied = []
    for idx, row in enumerate(rows):
        if row.get('state') == 'blocked_locally' and row.get('identity_status') == 'blocked_before_dispatch':
            supplied.append({'ordinal': row['ordinal'], 'resolution': 'blocked_before_dispatch'})
        elif row.get('provider_request_ids'):
            gid = row['provider_request_ids'][0]
            pentry = proxy_by_gid.get(gid)
            if fetch is None and not key:
                if pentry and pentry.get('completed') is True and pentry.get('stream_cost') is not None:
                    supplied.append({'ordinal': row['ordinal'], 'generation_id': gid,
                                     'resolution': 'proxy_stream_usage'})
                elif pentry and pentry.get('completed') is False and pentry.get('stream_cost') is None and row.get('cost') is None:
                    supplied.append({'ordinal': row['ordinal'], 'resolution': 'conservative_step_cap',
                                     'capped_cost': step_cap, 'verified_by': 'openrouter_guard_proxy',
                                     'mapping_source': 'proxy_aborted_stream'})
                else:
                    return None
            else:
                supplied.append({'ordinal': row['ordinal'], 'generation_id': gid})
        elif row.get('state') == 'finished' and row.get('cost') is not None:
            supplied.append({'ordinal': row['ordinal'], 'resolution': 'local_telemetry_finished',
                             'verified_by': 'openrouter_guard_proxy',
                             'mapping_source': 'proxy_local_telemetry_finished'})
        elif (ledger.get('proxy') and row.get('cost') is None
              and (idx >= len(dispatched)
                   or (dispatched[idx].get('completed') is False
                       and not dispatched[idx].get('generation_id')
                       and dispatched[idx].get('stream_cost') is None))):
            supplied.append({'ordinal': row['ordinal'], 'resolution': 'conservative_step_cap',
                             'capped_cost': step_cap, 'verified_by': 'openrouter_guard_proxy',
                             'mapping_source': 'proxy_aborted_before_generation_id'})
        else:
            return None
    if fetch is None and key:
        fetch = lambda gid: fetch_generation_settled(gid, key=key)
    evidence = {'session_id': result['session_id'], 'allow_proxy_stream_fallback': True,
                'auto_reconciled_by': 'openrouter_guard_proxy', 'requests': supplied}
    return _reconcile_locked(Path(run_dir), Path(path), result, evidence,
                             fetch=fetch or fetch_generation, db=db)

def prior_cost(state, task_id):
    total = 0.0
    for path in (Path(state) / 'runs').glob('*/result.json'):
        try:
            result = json.loads(path.read_text())
        except (OSError, ValueError):
            raise GuardError('unreadable_cost_journal') from None
        if not isinstance(result, dict):
            raise GuardError('invalid_cost_journal')
        if result.get('task_id') != task_id:
            continue
        total += resolved_cost(path, result)
    return total

def connect(db):
    return sqlite3.connect(Path(db).resolve().as_uri() + '?mode=ro', uri=True, timeout=.5)

def verify_db(db):
    c = connect(db)
    try:
        c.execute('SELECT id,cost FROM session_v2 LIMIT 0')
        c.execute('SELECT session_id,type,data FROM session_message LIMIT 0')
    finally:
        c.close()

def snapshot(db, sid):
    c = connect(db)
    try:
        row = c.execute('SELECT cost FROM session_v2 WHERE id=?', (sid,)).fetchone()
        if row is None:
            return None
        cost = number(row[0], 'telemetry_cost')
        rows = c.execute('SELECT type,data FROM session_message WHERE session_id=? ORDER BY rowid', (sid,)).fetchall()
        messages = [(kind, json.loads(raw)) for kind, raw in rows]
        if any(not isinstance(d, dict) for _, d in messages):
            raise GuardError('invalid_telemetry_message')
        return {'cost': cost, 'messages': messages}
    finally:
        c.close()

def clear_session_busy(db, sid, outcome='failed'):
    if not db or not sid:
        return False
    db_path = Path(db)
    if not db_path.exists():
        return False
    try:
        conn = sqlite3.connect(str(db_path), timeout=1.0)
        try:
            cols = {r[1] for r in conn.execute('PRAGMA table_info(session_v2)').fetchall()}
            if 'time_busy' not in cols:
                return False
            if 'idle_outcome' in cols:
                cur = conn.execute(
                    'UPDATE session_v2 SET time_busy = NULL, idle_outcome = COALESCE(idle_outcome, ?) WHERE id = ? AND time_busy IS NOT NULL',
                    (outcome, sid),
                )
            else:
                cur = conn.execute(
                    'UPDATE session_v2 SET time_busy = NULL WHERE id = ? AND time_busy IS NOT NULL',
                    (sid,),
                )
            conn.commit()
            return bool(cur.rowcount)
        finally:
            conn.close()
    except sqlite3.Error:
        return False

_EDIT_TOOL_NAMES = {'edit', 'write', 'multiedit', 'patch', 'apply_patch',
                    'harness_multiedit', 'harness_apply_patch'}

_RECOVERABLE_EDIT_ERROR_MARKERS = ('could not find oldstring', 'no changes to apply', 'identical')

def _is_recoverable_edit_error(tool_name, error_text):
    """Recoverable edit/write failures (mismatch, no-op, ambiguous match)."""
    if tool_name not in _EDIT_TOOL_NAMES:
        return False
    if not isinstance(error_text, str):
        return False
    low = error_text.lower()
    if any(marker in low for marker in _RECOVERABLE_EDIT_ERROR_MARKERS):
        return True
    # "Found N matches" ambiguous-match pattern ('Found ' together with ' matches').
    return 'found ' in low and ' matches' in low

def metrics(snap):
    errors, reads = _Counter(), _Counter()
    file_reads = _Counter()
    window_reads = _Counter()
    recoverable_errors = _Counter()
    steps = 0
    write_edit_calls = 0
    steps_since_last_edit = 0
    reasoning_only_streak = 0
    seen_edit_or_read = False
    compact = model_error = unfinished = False
    for kind, data in snap['messages']:
        if kind == 'compaction':
            compact = True
        if kind != 'assistant':
            continue
        steps += 1
        model_error |= bool(data.get('error'))
        unfinished |= not bool(data.get('finish'))
        turn_input_errors = _Counter()
        turn_recoverable_errors = _Counter()
        turn_file_read_inputs = _Counter()
        turn_has_tool = False
        turn_has_read = turn_has_edit = False
        for part in data.get('content', []):
            if part.get('type') != 'tool':
                continue
            turn_has_tool = True
            st = part.get('state', {})
            if st.get('status') == 'error':
                error_text = st.get('error') if isinstance(st.get('error'), str) else json.dumps(st.get('error'))
                if _is_recoverable_edit_error(part.get('name'), error_text):
                    # Keyed by tool name only: distinct recoverable messages
                    # (mismatch, ambiguous match, no-op) accumulate together.
                    turn_recoverable_errors[part.get('name')] += 1
                else:
                    key = (part.get('name'), json.dumps(st.get('error'), sort_keys=True))
                    inp = json.dumps(st.get('input'), sort_keys=True)
                    turn_input_errors[(key, inp)] += 1
            if part.get('name') == 'read' and st.get('status') == 'completed':
                turn_has_read = True
                reads[(json.dumps(st.get('input'), sort_keys=True),
                       json.dumps(st.get('content'), sort_keys=True))] += 1
                inp = st.get('input') or {}
                path = inp.get('path') or inp.get('filePath') or inp.get('file_path')
                if path:
                    inp_json = json.dumps(st.get('input'), sort_keys=True)
                    turn_file_read_inputs[(path, inp_json)] += 1
                    offset = inp.get('offset') if inp.get('offset') is not None else inp.get('start_line')
                    limit = inp.get('limit') if inp.get('limit') is not None else inp.get('end_line')
                    window_reads[(path, offset, limit)] += 1
            elif part.get('name') in _EDIT_TOOL_NAMES and st.get('status') == 'completed':
                turn_has_edit = True
                write_edit_calls += 1
                edit_inp = st.get('input') or {}
                edit_path = edit_inp.get('path') or edit_inp.get('filePath') or edit_inp.get('file_path')
                if edit_path:
                    file_reads.pop(edit_path, None)
                    for wkey in [k for k in window_reads if k[0] == edit_path]:
                        window_reads.pop(wkey, None)
                    for rkey in [k for k in turn_file_read_inputs if k[0] == edit_path]:
                        turn_file_read_inputs.pop(rkey, None)
                else:
                    file_reads.clear()
                    window_reads.clear()
                reads.clear()
                errors.clear()
                recoverable_errors.clear()
                turn_input_errors.clear()
                turn_recoverable_errors.clear()
                turn_file_read_inputs.clear()
        per_key_max = {}
        for (key, _inp), cnt in turn_input_errors.items():
            if cnt > per_key_max.get(key, 0):
                per_key_max[key] = cnt
        for key, inc in per_key_max.items():
            errors[key] += inc
        for key, inc in turn_recoverable_errors.items():
            recoverable_errors[key] += inc
        per_path_max = {}
        for (path, _inp), cnt in turn_file_read_inputs.items():
            if cnt > per_path_max.get(path, 0):
                per_path_max[path] = cnt
        for path, inc in per_path_max.items():
            file_reads[path] += inc
        if turn_has_edit:
            steps_since_last_edit = 0
            seen_edit_or_read = True
        elif seen_edit_or_read:
            steps_since_last_edit += 1
        if turn_has_read:
            seen_edit_or_read = True
        reasoning_only_streak = 0 if turn_has_tool else reasoning_only_streak + 1
    return dict(cost=snap['cost'], steps=steps, compact=compact, model_error=model_error,
                unfinished=unfinished, errors=max(errors.values(), default=0), reads=max(reads.values(), default=0),
                file_reads=max(file_reads.values(), default=0),
                window_reads=max(window_reads.values(), default=0),
                recoverable_errors=max(recoverable_errors.values(), default=0),
                write_edit_calls=write_edit_calls,
                steps_since_last_edit=steps_since_last_edit,
                reasoning_only_streak=reasoning_only_streak)

def step_cost(model=None, *, upstream_mode='openrouter', cost=None):
    """Effective API step cost: local/SML models and vllm/llamacpp upstreams are $0."""
    return 0.0 if _is_local_zero_cost(model, upstream_mode) else cost


def budget_gate(m_or_snap, budget=0.15, max_steps=16, *, model=None, upstream_mode='openrouter'):
    """Check a metrics dict (or a telemetry snapshot) against the budget gate.

    Accepts either a metrics dict (contains ``'steps'``), a snapshot dict
    (contains ``'messages'``, run through :func:`metrics`), or a request-level
    dict with ``'model'``/``'upstream_mode'``/``'cost'``/``'steps'`` keys.
    Local/SML models and vllm/llamacpp upstreams have their cost forced to
    ``0.0`` (never ``unpriced_model``) while ``max_steps`` stays strict.
    """
    if not isinstance(m_or_snap, dict):
        raise GuardError('budget_gate_requires_metrics_or_snapshot')
    if model is None:
        model = m_or_snap.get('model')
    if upstream_mode == 'openrouter' and 'upstream_mode' in m_or_snap:
        upstream_mode = m_or_snap.get('upstream_mode') or 'openrouter'
    if 'messages' in m_or_snap and 'steps' not in m_or_snap:
        m = dict(metrics(m_or_snap))
    else:
        m = dict(m_or_snap)
    if _is_local_zero_cost(model, upstream_mode):
        m['cost'] = 0.0
        m['unpriced_model'] = False
    return violation(m, budget, max_steps, model=model, upstream_mode=upstream_mode)


def violation(m, budget, max_steps, *, model=None, upstream_mode='openrouter'):
    local = _is_local_zero_cost(model, upstream_mode)
    # Local/SML models and vllm/llamacpp upstreams have no API pricing: treat
    # the step API cost as $0 and never block as 'unpriced_model'.
    api_cost = 0.0 if local else (m.get('cost') or 0.0)
    unpriced = False if local else (bool(m.get('unpriced_model')) or m.get('cost') is None)
    for condition, reason in [(api_cost >= budget, 'cost_limit'), (unpriced, 'unpriced_model'),
                              (m.get('steps', 0) > max_steps, 'step_limit'),
                              (m.get('compact'), 'compaction'), (m.get('model_error'), 'model_error'),
                              (m.get('errors', 0) >= 2 or m.get('recoverable_errors', 0) >= 3, 'repeated_tool_error'),
                              (m.get('reads', 0) >= 3, 'repeated_identical_read'),
                              (m.get('window_reads', 0) >= 4 or m.get('file_reads', 0) >= 12,
                               'repeated_file_read_without_write'),
                              (m.get('reasoning_only_streak', 0) >= 4, 'reasoning_stall_no_tool_call')]:
        if condition:
            return reason
    return None

def can_settle_from_proxy_local(snap, proxy_summary, *, model=None, upstream_mode='openrouter'):
    if snap is None or proxy_summary is None:
        return False
    raw_dispatched = proxy_summary.get('dispatched') or []
    dispatched = []
    for d in raw_dispatched:
        if d.get('http_status') is None and not d.get('generation_id') and d.get('error') == 'URLError':
            continue
        dispatched.append(d)
    blocked_count = int(proxy_summary.get('blocked_count') or 0)
    if blocked_count <= 0:
        return False
    records = request_records(snap, proxy_summary, model=model, upstream_mode=upstream_mode)
    k = len(dispatched)
    if k == 0:
        return all(r.get('state') == 'blocked_locally' for r in records)
    if len(records) < k:
        return False
    if not all(d.get('completed') and d.get('http_status') == 200 and not d.get('error') for d in dispatched):
        return False
    if not all(r.get('state') == 'finished' and r.get('cost') is not None for r in records[:k]):
        return False
    if not all(r.get('state') == 'blocked_locally' for r in records[k:]):
        return False
    local_sum = math.fsum(number(r['cost'], 'local_request_cost') for r in records[:k])
    telemetry_cost = snap.get('cost')
    telemetry_cost = 0.0 if telemetry_cost is None else number(telemetry_cost, 'telemetry_cost')
    return telemetry_cost + 1e-10 >= local_sum


def _extract_tokens(tokens: object) -> tuple[int, int]:
    """Extract (prompt_tokens, completion_tokens) from a request or billing token dict."""
    if not isinstance(tokens, dict):
        return 0, 0
    prompt = 0
    for key in ('prompt_tokens', 'input_tokens', 'input', 'prompt', 'tokens_prompt'):
        val = tokens.get(key)
        if isinstance(val, (int, float)) and math.isfinite(val) and val >= 0:
            prompt = int(val)
            break
    completion = 0
    for key in ('completion_tokens', 'output_tokens', 'output', 'completion', 'tokens_completion'):
        val = tokens.get(key)
        if isinstance(val, (int, float)) and math.isfinite(val) and val >= 0:
            completion = int(val)
            break
    return prompt, completion


def export_sml_cost_manifest(
    run_dir: str | Path,
    output_path: str | Path | None = None,
    *,
    gpu_hours: float = 0.0,
    gpu_hourly_rate_usd: float = 0.0,
    hardware_name: str = 'local_or_openrouter',
    num_gpus: int = 1,
    price_as_of: str = '2026-11-20',
    billed_instance_seconds: float | None = None,
    estimated_pre_run_cost_usd: float = 0.0,
) -> dict:
    """Export a Small Model Lab CostRecord manifest from a harness run directory."""
    run_path = Path(run_dir)
    gpu_h = float(number(gpu_hours, 'gpu_hours'))
    hourly_rate = float(number(gpu_hourly_rate_usd, 'gpu_hourly_rate_usd'))

    result_path = run_path / 'result.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    if not isinstance(result, dict):
        result = {}

    ledger = None
    for candidate in ('requests.json', 'proxy-requests.json'):
        req_path = run_path / candidate
        if req_path.exists():
            ledger = json.loads(req_path.read_text())
            break

    wall_clock_seconds = 0.0
    for key in ('wall_clock_seconds', 'duration_seconds', 'elapsed_seconds', 'duration'):
        val = result.get(key)
        if isinstance(val, (int, float)) and math.isfinite(val) and val >= 0:
            wall_clock_seconds = float(val)
            break
    else:
        started = result.get('started_at')
        finished = result.get('finished_at') or result.get('ended_at')
        if (isinstance(started, (int, float)) and isinstance(finished, (int, float))
                and math.isfinite(started) and math.isfinite(finished) and finished >= started):
            wall_clock_seconds = float(finished - started)

    prompt_tokens = 0
    completion_tokens = 0
    request_count = 0
    rows = ledger.get('requests') if isinstance(ledger, dict) and isinstance(ledger.get('requests'), list) else []
    dispatched = None
    if isinstance(ledger, dict):
        proxy_obj = ledger.get('proxy')
        if isinstance(proxy_obj, dict) and isinstance(proxy_obj.get('dispatched'), list):
            dispatched = proxy_obj['dispatched']
        elif isinstance(ledger.get('dispatched'), list):
            dispatched = ledger['dispatched']

    if rows:
        request_count = len(rows)
        for row in rows:
            if isinstance(row, dict):
                p_tok, c_tok = _extract_tokens(row.get('tokens'))
                prompt_tokens += p_tok
                completion_tokens += c_tok
    elif isinstance(dispatched, list):
        request_count = len(dispatched)

    if prompt_tokens == 0 and completion_tokens == 0 and isinstance(dispatched, list):
        for entry in dispatched:
            if not isinstance(entry, dict):
                continue
            p_tok, c_tok = _extract_tokens(entry.get('usage') or entry.get('stream_usage') or entry.get('tokens') or entry)
            prompt_tokens += p_tok
            completion_tokens += c_tok

    reconciliation_path = run_path / 'billing-reconciliation.json'
    reconciliation = json.loads(reconciliation_path.read_text()) if reconciliation_path.exists() else None
    if prompt_tokens == 0 and completion_tokens == 0 and isinstance(reconciliation, dict):
        for bill in reconciliation.get('billing') or []:
            if isinstance(bill, dict):
                p_tok, c_tok = _extract_tokens(bill)
                prompt_tokens += p_tok
                completion_tokens += c_tok

    if prompt_tokens == 0 and completion_tokens == 0:
        p_tok, c_tok = _extract_tokens(result)
        prompt_tokens += p_tok
        completion_tokens += c_tok
    if request_count == 0 and isinstance(result.get('request_count'), int) and result['request_count'] >= 0:
        request_count = result['request_count']

    api_cost_raw = None
    cost_accounting_status = 'unreconciled'
    if result_path.exists():
        try:
            api_cost_raw = float(resolved_cost(result_path, result))
            cost_accounting_status = 'settled'
        except GuardError as exc:
            if str(exc) != 'prior_cost_unknown_reconcile_before_retry':
                raise
    elif isinstance(reconciliation, dict) and reconciliation.get('cost') is not None:
        api_cost_raw = float(number(reconciliation['cost'], 'reconciled_cost'))
        cost_accounting_status = 'settled'

    if api_cost_raw is None:
        row_costs = [
            float(number(r['cost'], 'request_cost'))
            for r in rows
            if isinstance(r, dict) and r.get('cost') is not None
        ]
        if row_costs:
            api_cost_raw = math.fsum(row_costs)
            cost_accounting_status = 'estimated'
        elif isinstance(dispatched, list):
            stream_costs = [
                float(number(d.get('stream_cost', d.get('cost')), 'stream_cost'))
                for d in dispatched
                if isinstance(d, dict) and (d.get('stream_cost') is not None or d.get('cost') is not None)
            ]
            if stream_costs:
                api_cost_raw = math.fsum(stream_costs)
                cost_accounting_status = 'estimated'
        if api_cost_raw is None:
            observed = result.get('observed_cost')
            if observed is None and isinstance(ledger, dict):
                observed = ledger.get('observed_cost')
            if observed is not None:
                api_cost_raw = float(number(observed, 'observed_cost'))
                cost_accounting_status = 'estimated'
            else:
                api_cost_raw = 0.0

    api_cost_usd = round(api_cost_raw, 6)
    compute_cost_usd = round(gpu_h * hourly_rate, 6)
    total_cost_usd = round(compute_cost_usd + api_cost_usd, 6)

    num_gpus_int = int(num_gpus)
    active_sec = wall_clock_seconds
    billed_sec = float(billed_instance_seconds) if billed_instance_seconds is not None else wall_clock_seconds
    active_gpu_h = gpu_h if gpu_h > 0 else round((active_sec / 3600.0) * num_gpus_int, 6)
    billed_gpu_h = gpu_h if (gpu_h > 0 and billed_instance_seconds is None) else round((billed_sec / 3600.0) * num_gpus_int, 6)
    infra_overhead = round(max(0.0, billed_sec - active_sec) / billed_sec, 6) if billed_sec > 0 else 0.0
    active_cost = round(compute_cost_usd + api_cost_usd, 6)
    billed_cost = round(billed_gpu_h * hourly_rate + api_cost_usd, 6)

    manifest = {
        'wall_clock_seconds': wall_clock_seconds,
        'gpu_hours': gpu_h,
        'hourly_rate_usd': round(hourly_rate, 6),
        'HourlyRateUSD': round(hourly_rate, 6),
        'compute_cost_usd': compute_cost_usd,
        'api_cost_usd': api_cost_usd,
        'total_cost_usd': total_cost_usd,
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'request_count': request_count,
        'cost_accounting_status': cost_accounting_status,
        'hardware_name': str(hardware_name),
        'price_as_of': str(price_as_of),
        'num_gpus': num_gpus_int,
        'active_training_seconds': active_sec,
        'billed_instance_seconds': billed_sec,
        'active_gpu_hours': active_gpu_h,
        'billed_gpu_hours': billed_gpu_h,
        'infrastructure_overhead_fraction': infra_overhead,
        'active_cost_usd': active_cost,
        'billed_cost_usd': billed_cost,
        'estimated_pre_run_cost_usd': float(estimated_pre_run_cost_usd),
    }
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        private_write(out, manifest)
    return manifest
