"""Guarded OpenCode worker supervisor: sandboxed task execution, cost accounting and billing reconciliation.

Ported from the former single-module ``.tools/opencode_guard.py``. Signatures,
guard-error messages and behavior are preserved verbatim so the launcher,
journal and test suites keep working unchanged.
"""

import argparse
import ast
import base64
import difflib
import fcntl
import fnmatch
import hashlib
import json
import math
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from contextlib import ExitStack, contextmanager
from pathlib import Path

from .telemetry import sync_state_traces_to_phoenix, synthesize_spans_from_run_dir

from .finops import (
    DB,
    LIMITS,
    STATE as _FINOPS_STATE,
    _reconcile_locked,
    auto_reconcile_from_proxy,
    build_unexposed_settlement_evidence,
    can_settle_from_proxy_local,
    capture_requests,
    clear_session_busy,
    connect,
    event,
    fetch_generation,
    fetch_generation_settled,
    locks,
    metrics,
    number,
    prior_cost,
    private_log,
    private_write,
    reconcile_run,
    request_records,
    resolved_cost,
    snapshot,
    verify_db,
    violation,
)
from .guards import (build_file_skeletons, build_preloaded_file_excerpts,
                     extract_compact_test_failure, extract_failing_test_excerpts)
from .proxy import OPENROUTER_UPSTREAM, GuardError, openrouter_proxy
from .report import build_repo_memory_hint
from .routing import DEFAULT_MODEL, MODELS, TEACHER_MODELS, ProviderHealthStore

MAX_STEPS = LIMITS['max_steps'][1]

ESCALATABLE_BLOCK_REASONS = frozenset({
    'validation_failed',
    'worker_timeout',
    'repeated_file_read_without_write',
    'repeated_tool_error',
    'step_limit',
    'model_error',
    'no_change',
})

SALVAGEABLE_BLOCK_REASONS = frozenset({
    'step_limit',
    'worker_timeout',
    'repeated_tool_error',
    'repeated_file_read_without_write',
    'model_error',
})

CLI = os.environ.get("HARNESS_CLI") or os.environ.get("OPENCODE_CLI") or "/Applications/OpenCode.app/Contents/Resources/opencode-cli"
# The durable journal stays next to the launcher facade when loaded via opencode_guard;
# otherwise use finops.STATE (which honors HARNESS_STATE_DIR).
_OG_FILE = getattr(sys.modules.get('opencode_guard'), '__file__', None)
STATE = (
    Path(os.environ["HARNESS_STATE_DIR"])
    if os.environ.get("HARNESS_STATE_DIR")
    else (Path(_OG_FILE).resolve().parent / 'opencode-guard-state' if _OG_FILE else _FINOPS_STATE)
)


def executable_argv(value):
    if not isinstance(value, list) or not value or any(
            not isinstance(x, str) or '\x00' in x for x in value):
        raise GuardError('invalid_argv')
    if not Path(value[0]).is_absolute() or not Path(value[0]).is_file() or not os.access(value[0], os.X_OK):
        raise GuardError('executable_must_be_an_existing_absolute_path')
    return value

_PYTEST_PROBE_CACHE: dict[str, bool] = {}

def _python_has_pytest(python_bin) -> bool:
    resolved = os.path.realpath(str(python_bin))
    cached = _PYTEST_PROBE_CACHE.get(resolved)
    if cached is not None:
        return cached
    try:
        proc = subprocess.run([str(python_bin), '-c', 'import pytest'],
                              capture_output=True, timeout=5)
        has = proc.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        has = False
    _PYTEST_PROBE_CACHE[resolved] = has
    return has

def resolve_pytest_test_argv(test_argv, directory):
    argv = list(test_argv)
    is_pytest = len(argv) >= 3 and (
        argv[1:3] == ['-m', 'pytest'] or
        (Path(argv[0]).name.startswith('python') and
         any(argv[i:i + 2] == ['-m', 'pytest'] for i in range(1, min(len(argv) - 1, 4)))))
    if not is_pytest:
        return argv
    if _python_has_pytest(argv[0]):
        return argv
    base = Path(directory).resolve()
    for parent in [base, *list(base.parents)[:4]]:
        for name in ('python', 'python3'):
            candidate = parent / '.venv' / 'bin' / name
            if candidate.is_file() and os.access(candidate, os.X_OK) and _python_has_pytest(candidate):
                return [str(candidate), *argv[1:]]
    raise GuardError('test_runner_missing_pytest')

_PYTEST_VALUE_FLAGS = frozenset({
    '-k', '-m', '-c', '-o', '-p', '-W', '-n',
    '--tb', '--rootdir', '--confcutdir', '--basetemp', '--junitxml', '--junit-prefix',
    '--maxfail', '--durations', '--durations-min', '--log-level', '--log-format',
    '--log-date-format', '--ignore', '--ignore-glob', '--deselect', '--import-mode',
    '--override-ini', '--pythonwarnings', '--dist', '--tx',
    '--cov', '--cov-report', '--cov-config', '--asyncio-mode',
    '--timeout', '--timeout-method',
})

def build_targeted_pytest_argv(test_argv, test_log, *, max_nodeids=5):
    """H-56: build a targeted pytest invocation from failed nodeids in a test log.

    Returns None when test_argv is not a pytest invocation, when the log has no
    (or too many) failed nodeids, when collection itself failed, or when the
    targeted argv would be identical to the original.
    """
    argv = list(test_argv)
    if not argv:
        return None
    if Path(argv[0]).name.startswith('pytest'):
        prefix = [argv[0]]
        rest = argv[1:]
    else:
        idx = next((i for i in range(len(argv) - 1)
                    if argv[i:i + 2] == ['-m', 'pytest']), None)
        if idx is None:
            return None
        prefix = list(argv[:idx + 2])
        rest = argv[idx + 2:]
    nodeids = []
    for m in re.finditer(r'^FAILED\s+(\S+::\S+)', test_log, re.MULTILINE):
        if m.group(1) not in nodeids:
            nodeids.append(m.group(1))
    if not 1 <= len(nodeids) <= max_nodeids:
        return None
    if 'Interrupted:' in test_log or 'ERROR collecting' in test_log:
        return None
    flag_args = []
    skip_next = False
    for i, arg in enumerate(rest):
        if skip_next:
            skip_next = False
            continue
        if arg == '--':
            break
        if not arg.startswith('-'):
            continue  # positional test path: drop it
        flag_args.append(arg)
        name = arg.split('=', 1)[0]
        if name in _PYTEST_VALUE_FLAGS and '=' not in arg:
            nxt = rest[i + 1] if i + 1 < len(rest) else None
            if nxt is not None and not nxt.startswith('-'):
                flag_args.append(nxt)
                skip_next = True
    targeted = prefix + flag_args + nodeids
    return targeted if targeted != list(test_argv) else None

def infer_read_only_companion_files(files, *, directory='', existing_read_only=(), worktree=False, max_files=5):
    """H-62: infer existing implementation companion files for test_<mod>.py editable targets."""
    editable_set = {str(Path(f).resolve()) for f in files}
    result = [str(Path(f).resolve()) for f in existing_read_only]
    seen = set(editable_set) | set(result)
    dir_path = Path(directory).resolve() if directory else None
    for f in files:
        if len(result) >= max_files:
            break
        p = Path(f).resolve()
        if not (p.suffix == '.py' and p.name.startswith('test_') and len(p.name) > len('test_.py')):
            continue
        mod_name = p.name[len('test_'):]
        candidates = []
        if p.parent.name in ('tests', 'test'):
            candidates.append(p.parent.parent / mod_name)
            if dir_path is not None:
                candidates.append(dir_path / mod_name)
                candidates.append(dir_path / 'src' / mod_name)
        candidates.append(p.parent / mod_name)
        for cand in candidates:
            if len(result) >= max_files:
                break
            try:
                resolved = cand.resolve()
            except OSError:
                continue
            if str(resolved) in seen or not resolved.is_file():
                continue
            if dir_path is not None and not resolved.is_relative_to(dir_path):
                continue
            seen.add(str(resolved))
            result.append(str(resolved))
    # H-65: second pass — infer companions from local imports of editable .py files.
    if len(result) < max_files:
        for f in files:
            if len(result) >= max_files:
                break
            p = Path(f).resolve()
            if p.suffix != '.py' or not p.is_file():
                continue
            try:
                tree = ast.parse(p.read_text(encoding='utf-8', errors='ignore'))
            except (SyntaxError, OSError, ValueError):
                continue
            roots = [p.parent]
            if p.parent.parent != p.parent:
                roots.append(p.parent.parent)
            if dir_path is not None:
                roots.append(dir_path)
            mod_candidates = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.level and node.level > 0:
                        base = p.parent
                        for _ in range(node.level - 1):
                            base = base.parent
                        if node.module:
                            base = base / node.module.replace('.', '/')
                        mod_candidates.append(base)
                        for alias in node.names:
                            mod_candidates.append(base / alias.name)
                    elif node.module:
                        rel = node.module.replace('.', '/')
                        mod_candidates.append(rel)
                        for alias in node.names:
                            mod_candidates.append(rel + '/' + alias.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        mod_candidates.append(alias.name.replace('.', '/'))
            for cand in mod_candidates:
                if len(result) >= max_files:
                    break
                cand_paths = ([root / cand for root in roots]
                              if isinstance(cand, str) else [cand])
                for cand_path in cand_paths:
                    m_path = cand_path if cand_path.suffix == '.py' else cand_path.with_suffix('.py')
                    try:
                        resolved = m_path.resolve()
                    except OSError:
                        continue
                    if str(resolved) in seen or not resolved.is_file():
                        continue
                    if dir_path is not None and not resolved.is_relative_to(dir_path):
                        continue
                    seen.add(str(resolved))
                    result.append(str(resolved))
                    break
    return sorted(result)


def validate_task(raw):
    if not isinstance(raw, dict):
        raise GuardError('task_must_be_an_object')
    t = dict(raw)
    if not isinstance(t.get('task_id'), str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', t['task_id']):
        raise GuardError('invalid_task_id')
    prompt_replacements = []
    file_aliases = []
    read_only_file_aliases = []
    directory = t.get('directory')
    if not isinstance(directory, str) or not Path(directory).is_absolute() or not Path(directory).is_dir():
        raise GuardError('invalid_working_directory')
    t['directory'] = str(Path(directory).resolve())
    if os.path.abspath(directory) != t['directory']:
        prompt_replacements.append((os.path.abspath(directory), t['directory']))
    files = t.get('files')
    if not isinstance(files, list) or not 1 <= len(files) <= 3:
        raise GuardError('exactly_one_to_three_files_required')
    normalized = []
    for f in files:
        if not isinstance(f, str) or '\x00' in f or not Path(f).is_absolute():
            raise GuardError('file_paths_must_be_absolute')
        p = Path(f).resolve()
        if any(x in str(p) for x in '*?') or not p.parent.is_dir() or (p.exists() and not p.is_file()):
            raise GuardError('invalid_file_target')
        normalized.append(str(p))
        if os.path.abspath(f) != str(p):
            file_aliases.append(os.path.abspath(f))
            prompt_replacements.append((os.path.abspath(f), str(p)))
    if len(set(normalized)) != len(normalized):
        raise GuardError('duplicate_file_or_symlink_alias')
    t['files'] = sorted(normalized)
    ro_raw = t.get('read_only_files', [])
    if not isinstance(ro_raw, list) or not 0 <= len(ro_raw) <= 5:
        raise GuardError('read_only_files_must_be_a_list_of_up_to_five_files')
    normalized_ro = []
    for f in ro_raw:
        if not isinstance(f, str) or '\x00' in f or not Path(f).is_absolute():
            raise GuardError('file_paths_must_be_absolute')
        p = Path(f).resolve()
        if any(x in str(p) for x in '*?') or not p.is_file():
            raise GuardError('invalid_file_target')
        if str(p) in t['files']:
            raise GuardError('read_only_file_overlaps_editable_files')
        normalized_ro.append(str(p))
        if os.path.abspath(f) != str(p):
            read_only_file_aliases.append(os.path.abspath(f))
            prompt_replacements.append((os.path.abspath(f), str(p)))
    t['read_only_files'] = sorted(set(normalized_ro))
    t['test_argv'] = executable_argv(t.get('test_argv'))
    t.setdefault('mode', 'model')
    if t['mode'] not in ('model', 'patch', 'command'):
        raise GuardError('invalid_mode')
    if t['mode'] == 'command':
        t['command_argv'] = executable_argv(t.get('command_argv'))
    elif not isinstance(t.get('prompt'), str) or not t['prompt'].strip():
        raise GuardError('prompt_required')
    if t['mode'] == 'patch':
        p = t.get('patch_file')
        if (not isinstance(p, str) or not Path(p).is_absolute() or
                not Path(p).parent.is_dir() or Path(p).exists() or Path(p).is_symlink()):
            raise GuardError('patch_file_must_be_new_absolute_path')
        t['patch_file'] = str(Path(p).resolve())
        if t['patch_file'] in t['files']:
            raise GuardError('patch_cannot_replace_source')
        if os.path.abspath(p) != t['patch_file']:
            prompt_replacements.append((os.path.abspath(p), t['patch_file']))
    if isinstance(t.get('prompt'), str) and prompt_replacements:
        for raw_abs, resolved in sorted(set(prompt_replacements), key=lambda item: len(item[0]), reverse=True):
            t['prompt'] = t['prompt'].replace(raw_abs, resolved)
    t['file_aliases'] = sorted(set(file_aliases))
    t['read_only_file_aliases'] = sorted(set(read_only_file_aliases))
    t.setdefault('model', DEFAULT_MODEL)
    if t['model'] not in MODELS | TEACHER_MODELS:
        raise GuardError('unsupported_model')
    t.setdefault('strategy', 'single')
    if t['strategy'] not in ('single', 'best_of_2'):
        raise GuardError('invalid_strategy')
    t.setdefault('auto_escalate', False)
    if not isinstance(t['auto_escalate'], bool):
        raise GuardError('invalid_auto_escalate_flag')
    t.setdefault('candidate_models', None)
    if t['candidate_models'] is not None:
        cm = t['candidate_models']
        if (not isinstance(cm, list) or not 1 <= len(cm) <= 2
                or any(not isinstance(m, str) or m not in MODELS | TEACHER_MODELS for m in cm)):
            raise GuardError('invalid_candidate_models')
    if (t['strategy'] == 'best_of_2' or t['auto_escalate']) and t['candidate_models'] is None:
        t['candidate_models'] = [t['model'],
                                 t.get('fallback_teacher_model') or 'openrouter/deepseek/deepseek-v4.1-flash-1017']
    t.setdefault('max_context_tokens', 32000)
    if t['max_context_tokens'] is not None:
        if type(t['max_context_tokens']) is not int or t['max_context_tokens'] < 0:
            raise GuardError('invalid_max_context_tokens')
        if t['max_context_tokens'] == 0:
            t['max_context_tokens'] = None
    t.setdefault('fallback_teacher_model', None)
    if t['fallback_teacher_model'] is not None and \
            t['fallback_teacher_model'] not in TEACHER_MODELS | MODELS:
        raise GuardError('invalid_fallback_teacher_model')
    t.setdefault('enforce_egress_guard', True)
    if not isinstance(t['enforce_egress_guard'], bool):
        raise GuardError('invalid_enforce_egress_guard')
    t.setdefault('enforce_readonly_sql', False)
    if not isinstance(t['enforce_readonly_sql'], bool):
        raise GuardError('invalid_enforce_readonly_sql')
    t.setdefault('upstream_mode', 'openrouter')
    if t['upstream_mode'] not in ('openrouter', 'vllm', 'openai'):
        raise GuardError('invalid_upstream_mode')
    if t.get('reasoning_max_tokens') is not None:
        val = t['reasoning_max_tokens']
        if type(val) is not int or val <= 0:
            raise GuardError('invalid_reasoning_max_tokens')
    t.setdefault('worktree', False)
    if not isinstance(t['worktree'], bool):
        raise GuardError('invalid_worktree_flag')
    if t['worktree']:
        if t['mode'] == 'patch':
            raise GuardError('worktree_not_supported_for_patch_mode')
        if any(not Path(f).is_relative_to(Path(t['directory'])) for f in t['files']):
            raise GuardError('worktree_files_must_be_under_directory')
        if any(not Path(f).is_relative_to(Path(t['directory'])) for f in t['read_only_files']):
            raise GuardError('worktree_files_must_be_under_directory')
    t.setdefault('auto_infer_read_only', True)
    if not isinstance(t['auto_infer_read_only'], bool):
        raise GuardError('invalid_auto_infer_read_only_flag')
    if t['mode'] == 'model' and t['auto_infer_read_only'] and len(t['read_only_files']) < 5:
        t['read_only_files'] = infer_read_only_companion_files(
            t['files'], directory=t['directory'], existing_read_only=t['read_only_files'],
            worktree=t['worktree'], max_files=5)
    t.setdefault('include_repo_memory', True)
    if not isinstance(t['include_repo_memory'], bool):
        raise GuardError('invalid_include_repo_memory_flag')
    t.setdefault('auto_micro_repair', True)
    if not isinstance(t['auto_micro_repair'], bool):
        raise GuardError('invalid_auto_micro_repair_flag')
    for key, (default, maximum) in LIMITS.items():
        value = number(t.setdefault(key, default), key)
        if not 0 < value <= maximum or (key == 'max_steps' and type(value) is not int):
            raise GuardError(f'invalid_{key}')
        if value > default and (not isinstance(t.get('limit_reason'), str) or not t['limit_reason'].strip()):
            raise GuardError('limit_increase_requires_written_reason')
    if t['mode'] == 'model' and t.get('adaptive_steps', True):
        t['max_steps'] = min(MAX_STEPS, max(int(t['max_steps']), 4 + 3 * len(t['files'])))
    return t

def capture_files(run_dir, files):
    """Keep exact bytes, including new/deleted files, independently of exit status."""
    run_dir = Path(run_dir)
    current = {}
    for name in files:
        p = Path(name)
        data = p.read_bytes() if p.exists() else None
        current[name] = {'sha256': hashlib.sha256(data).hexdigest() if data is not None else None,
                         'base64': base64.b64encode(data).decode() if data is not None else None}
    initial_path, latest_path = run_dir / 'files.before.json', run_dir / 'files.latest.json'
    if not initial_path.exists():
        private_write(initial_path, current)
    previous = json.loads(latest_path.read_text()) if latest_path.exists() else None
    if previous == current:
        return
    private_write(run_dir / ('files.' + uuid.uuid4().hex + '.json'), current)
    private_write(latest_path, current)
    before = json.loads(initial_path.read_text())
    diff = []
    for name, item in current.items():
        try:
            old = base64.b64decode(before[name]['base64'] or '').decode('utf-8')
            new = base64.b64decode(item['base64'] or '').decode('utf-8')
        except UnicodeDecodeError:
            continue  # Exact binary bytes are still in the snapshots.
        diff.extend(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                    fromfile=name, tofile=name))
    with private_log(run_dir / 'changes.patch') as f:
        f.write(''.join(line if line.endswith('\n') else line + '\n\\ No newline at end of file\n'
                       for line in diff))
        f.flush()
        os.fsync(f.fileno())
    event(run_dir, 'files_captured', hashes={p: d['sha256'] for p, d in current.items()})

def check_patch(task):
    """Validate a proposal against exact paths without applying it."""
    patch_file = task['patch_file']
    root = subprocess.run(['/usr/bin/git', 'rev-parse', '--show-toplevel'], cwd=task['directory'],
                          capture_output=True, timeout=15)
    if root.returncode or Path(root.stdout.decode().strip()).resolve() != Path(task['directory']).resolve():
        raise GuardError('patch_directory_must_be_repository_root')
    def git(*args):
        return subprocess.run(['/usr/bin/git', 'apply', *args, patch_file], cwd=task['directory'],
                              capture_output=True, timeout=15)
    stats = git('--numstat', '-z')
    if stats.returncode or not stats.stdout:
        raise GuardError('invalid_patch')
    for entry in stats.stdout.split(b'\0'):
        if not entry:
            continue
        parts = entry.decode('utf-8').split('\t', 2)
        if len(parts) != 3 or not parts[2] or parts[0] == '-' or parts[1] == '-':
            raise GuardError('binary_or_rename_patch_not_supported')
        path = (Path(task['directory']) / parts[2]).resolve()
        if str(path) not in task['files']:
            raise GuardError('patch_outside_allowed_files')
    if git('--check').returncode:
        raise GuardError('patch_does_not_apply_cleanly')

def config_for(task, *, base_url=None):
    permissions = [{'action': '*', 'resource': '*', 'effect': 'deny'},
                   {'action': 'external_directory', 'resource': '*', 'effect': 'allow'}]
    actions = ('read',) if task['mode'] == 'patch' else ('read', 'edit')
    permissions += [{'action': action, 'resource': path, 'effect': 'allow'}
                    for path in task['files'] for action in actions]
    permissions += [{'action': 'read', 'resource': path, 'effect': 'allow'}
                    for path in task.get('read_only_files', ())]
    permissions += [{'action': action, 'resource': path, 'effect': 'allow'}
                    for path in task.get('file_aliases', ()) for action in actions]
    permissions += [{'action': 'read', 'resource': path, 'effect': 'allow'}
                    for path in task.get('read_only_file_aliases', ())]
    if task['mode'] == 'patch':
        permissions += [{'action': a, 'resource': task['patch_file'], 'effect': 'allow'} for a in ('read', 'edit')]
    cfg = {'model': task['model'], 'small_model': task['model'],
           'agents': {'guarded_worker': {'description': 'Scoped bounded coding worker',
           'mode': 'primary', 'steps': task['max_steps'], 'permissions': permissions}}}
    if base_url:
        cfg['providers'] = {'openrouter': {'settings': {'baseURL': base_url}}}
    return cfg

def verify_permissions(payload, task):
    for a in payload.get('data', []):
        m = a.get('model')
        if isinstance(m, dict) and f"{m.get('providerID')}/{m.get('id')}" not in MODELS:
            raise GuardError('unsupported_agent_model')
    agent = next((a for a in payload.get('data', []) if a.get('id') == 'guarded_worker'), None)
    if not agent or agent.get('steps') != task['max_steps']:
        raise GuardError('effective_agent_or_steps_mismatch')
    rules = agent.get('permissions', [])
    deny = [i for i, r in enumerate(rules) if r == {'action': '*', 'resource': '*', 'effect': 'deny'}]
    if not deny:
        raise GuardError('effective_catch_all_deny_missing')
    allowed = {(r['action'], r['resource']) for r in config_for(task)['agents']['guarded_worker']['permissions']
               if r['effect'] == 'allow'}
    tail = rules[deny[-1]:]
    for r in tail:
        if r.get('effect') == 'deny':
            continue
        if r.get('effect') != 'allow' or (r.get('action'), r.get('resource')) not in allowed:
            raise GuardError('effective_permissions_too_broad')
    for action, resource in allowed:
        matching = [r for r in tail if fnmatch.fnmatchcase(action, r.get('action', ''))
                    and fnmatch.fnmatchcase(resource, r.get('resource', ''))]
        if not matching or matching[-1].get('effect') != 'allow':
            raise GuardError('effective_file_permissions_missing')

def stop_group(proc):
    # Reap an exited leader before signaling its group (macOS zombie-only EPERM).
    proc.poll()
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass
        if sig == signal.SIGTERM:
            try:
                proc.wait(timeout=.3)
            except subprocess.TimeoutExpired:
                pass
    proc.wait(timeout=5)

@contextmanager
def owned_process(argv, directory, log):
    # OpenCode run uses PWD for the session location, even when cwd is correct.
    proc = subprocess.Popen(argv, cwd=directory, env=dict(os.environ, PWD=str(directory)),
                            stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    try:
        yield proc
    finally:
        stop_group(proc)  # Includes descendants after the group leader exits.

@contextmanager
def interruptions():
    def interrupted(signum, frame):
        raise GuardError('interrupted')
    saved = {s: signal.signal(s, interrupted) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for s, handler in saved.items():
            signal.signal(s, handler)

def execute(argv, directory, timeout, path):
    with private_log(path) as log, owned_process(argv, directory, log) as proc:
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            raise GuardError('command_timeout') from None

def preflight(cli, task, run_dir, *, base_url=None):
    executable_argv([str(cli)])
    if task.get('mode', 'model') == 'model':
        task['test_argv'] = resolve_pytest_test_argv(task['test_argv'], task['directory'])
    private_write(run_dir / 'opencode.jsonc', config_for(task, base_url=base_url))
    path = run_dir / 'config-check.log'
    rc = execute([str(cli), 'api', '--standalone', 'GET', '/api/agent'], run_dir, 30, path)
    if rc:
        raise GuardError('configuration_check_failed')
    try:
        payload = json.loads(path.read_text())
        if not any(a.get('id') == 'guarded_worker' for a in payload.get('data', [])):
            rc2 = execute([str(cli), 'debug', 'agents'], run_dir, 30, path)
            if rc2 == 0:
                alt = json.loads(path.read_text())
                if isinstance(alt, list):
                    payload = {'data': alt}
        verify_permissions(payload, task)
    except (ValueError, TypeError, AttributeError, KeyError):
        raise GuardError('configuration_response_invalid') from None

def model_run(cli, db, task, run_dir, result, budget, *, proxy=None, started_ref=None):
    sid = result['session_id']
    skeletons = build_file_skeletons(task['files'])
    if '[CURRENT_PATCH]' not in task['prompt']:
        excerpts = build_preloaded_file_excerpts(
            list(task['files']) + list(task.get('read_only_files', ())), task['prompt'],
            read_only_files=task.get('read_only_files', []))
    else:
        excerpts = ''
    ro_line = ''
    if task.get('read_only_files'):
        ro_line = ('\nRead-only reference files (do NOT edit): '
                   + json.dumps(task['read_only_files']))
    if 'COMPLETE FILE PRELOADED' in excerpts:
        read_instruction = (
            'Files marked COMPLETE FILE PRELOADED above are already provided in full — '
            'do NOT call read on them; edit directly. Batch any other needed reads into '
            'your first turn and batch edits together.\n'
        )
    else:
        read_instruction = (
            'Batch all needed file reads into your first turn and batch edits together; '
            'do not loop one small read/edit per turn.\n'
        )
    prompt = ('Read/edit ONLY the listed files; do NOT run find/glob or search for Python interpreters.\n'
              + read_instruction
              + json.dumps(task['files']) + ro_line + '\n' + task['prompt']
              + '\nNo shell is available. The supervisor runs this exact validation AFTER you exit: '
              + json.dumps(task['test_argv'])
              + '\nDo not claim tests passed. Stop immediately after the required edits.')
    if task['mode'] == 'patch':
        prompt = ('Read ONLY these source files: ' + json.dumps(task['files']) + ro_line + '\n' + task['prompt']
                  + '\nWrite ONLY a small unified diff to ' + task['patch_file']
                  + '. Source files are read-only. Do not reconstruct whole files. '
                  'No shell or network. Exit immediately after writing the patch; no extra summary call. '
                  'The supervisor checks the patch without applying it; testing follows reviewed application.')
    prompt = skeletons + excerpts + prompt
    argv = [str(cli), 'run', '--standalone', '--auto', '--format', 'json', '--agent', 'guarded_worker',
            '--session', sid, '--title', task['task_id'], '--model', task['model'], prompt]
    result['cost'] = None
    private_write(run_dir / 'result.json', result)  # An abrupt crash leaves unknown cost.
    event(run_dir, 'model_session_started', session_id=sid, model=task['model'])
    started = time.monotonic()
    if started_ref is not None:
        started_ref[0] = started
    naturally_exited = False
    try:
        with private_log(run_dir / 'worker.log') as log, owned_process(argv, run_dir, log) as proc:
            while True:
                elapsed = time.monotonic() - started
                if elapsed >= task['timeout_seconds']:
                    raise GuardError('worker_timeout')
                rc = proc.poll()
                naturally_exited = rc is not None
                capture_files(run_dir, task['files'] + ([task['patch_file']] if task['mode'] == 'patch' else []))
                snap = snapshot(db, sid)
                proxy_summary = proxy.summary() if proxy is not None else None
                if snap is not None:
                    capture_requests(run_dir, sid, snap, proxy_summary)
                    m = metrics(snap)
                    result.update(steps=m['steps'], observed_cost=m['cost'])
                    if proxy is not None and proxy.blocked_reason:
                        raise GuardError(proxy.blocked_reason)
                    reason = violation(m, budget, task['max_steps'])
                    if reason:
                        raise GuardError(reason)
                elif proxy is not None and proxy.blocked_reason:
                    raise GuardError(proxy.blocked_reason)
                elif elapsed >= 30:
                    raise GuardError('missing_telemetry')
                if rc is not None:
                    # Process completion between polls must not bypass final guards.
                    final = snapshot(db, sid)
                    proxy_summary = proxy.summary() if proxy is not None else None
                    if final is None:
                        raise GuardError('missing_final_telemetry')
                    capture_requests(run_dir, sid, final, proxy_summary)
                    m = metrics(final)
                    result.update(steps=m['steps'], observed_cost=m['cost'])
                    if proxy is not None and proxy.blocked_reason:
                        raise GuardError(proxy.blocked_reason)
                    reason = violation(m, budget, task['max_steps'])
                    if reason:
                        raise GuardError(reason)
                    if rc or not m['steps'] or m['unfinished']:
                        raise GuardError('worker_incomplete_or_failed')
                    result['cost'] = m['cost']
                    break
                if proxy is not None:
                    proxy.stop_event.wait(.25)
                else:
                    time.sleep(.25)
    finally:
        try:
            proxy_summary = proxy.summary() if proxy is not None else None
            final = snapshot(db, sid)
            if final is not None:
                capture_requests(run_dir, sid, final, proxy_summary)
                m = metrics(final)
                result.update(steps=m['steps'], observed_cost=m['cost'])
                if (naturally_exited and proc.returncode == 0 and m['steps']
                        and not m['unfinished'] and not m['model_error']):
                    result['cost'] = m['cost']
                elif proxy_summary is not None and can_settle_from_proxy_local(final, proxy_summary):
                    result['cost'] = m['cost']
            elif proxy_summary is not None and (proxy_summary['dispatched'] or proxy_summary['blocked']):
                capture_requests(run_dir, sid, {'cost': 0.0, 'messages': []}, proxy_summary)
                if proxy_summary['dispatched_count'] == 0 and proxy_summary['blocked_count'] > 0:
                    result['cost'] = 0.0
        except (OSError, ValueError, sqlite3.Error, GuardError):
            result['cost'] = None
        finally:
            clear_session_busy(db, sid, outcome='succeeded' if (naturally_exited and proc.returncode == 0) else 'failed')

def file_hashes(files):
    return {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).exists() else None for p in files}

def _python_top_symbols(source):
    """Top-level function/class names mapped to body hashes (H-33)."""
    import ast
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {}
    symbols = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            try:
                symbols[node.name] = hashlib.sha256(ast.dump(node).encode()).hexdigest()
            except (AttributeError, RecursionError, ValueError):
                continue
    return symbols


def _python_symbol_spans(source):
    """Top-level function/class names mapped to (start_line, end_line) (H-53)."""
    import ast
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {}
    spans = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            dec_lines = [d.lineno for d in getattr(node, 'decorator_list', ()) if hasattr(d, 'lineno')]
            start = min([node.lineno] + dec_lines)
            end = getattr(node, 'end_lineno', None) or node.lineno
            spans[node.name] = (start, end)
    return spans


def build_micro_repair_prompt(task, run_dir, originals=None):
    """Assemble a zero-read micro-repair prompt (H-41).

    Includes the current patch, a Python symbol diff against the pre-attempt
    snapshot, preloaded on-disk excerpts and the compact test failure so the
    repair attempt applies a targeted edit without re-reading files.
    """
    run_dir = Path(run_dir)
    try:
        test_log_text = (run_dir / 'test.log').read_text(errors='replace')
    except OSError:
        test_log_text = ''
    compact_failure = extract_compact_test_failure(test_log_text, '')
    try:
        current_patch = (run_dir / 'changes.patch').read_text(errors='replace')[:3000]
    except OSError:
        current_patch = ''
    trajspec_lines: list[str] = []
    trajspec_seed_tokens: list[str] = []
    if originals:
        file_names = sorted(set(originals) | set(task.get('files', ())))
        modified: list[str] = []
        added: list[str] = []
        dropped: list[str] = []
        for name in file_names:
            if not str(name).endswith('.py'):
                continue
            before_src = originals.get(name)
            before_symbols = (
                _python_top_symbols(before_src.decode('utf-8', errors='replace'))
                if isinstance(before_src, (bytes, bytearray)) else {}
            )
            p = Path(name)
            after_src = p.read_bytes() if p.is_file() else None
            after_text = after_src.decode('utf-8', errors='replace') if isinstance(after_src, (bytes, bytearray)) else ''
            after_symbols = _python_top_symbols(after_text) if after_text else {}
            after_spans = _python_symbol_spans(after_text) if after_text else {}
            for symbol in sorted(set(before_symbols) | set(after_symbols)):
                is_dropped = symbol in before_symbols and symbol not in after_symbols
                is_added = symbol not in before_symbols and symbol in after_symbols
                is_modified = (symbol in before_symbols and symbol in after_symbols
                               and before_symbols[symbol] != after_symbols[symbol])
                if is_dropped:
                    dropped.append(symbol)
                elif is_added or is_modified:
                    span = after_spans.get(symbol)
                    label = f'{symbol} (L{span[0]}-L{span[1]})' if span else symbol
                    (added if is_added else modified).append(label)
                    trajspec_seed_tokens.append(f'`{symbol}`')
                    if span:
                        trajspec_seed_tokens.append(f'L{span[0]}-L{span[1]}')
        if modified:
            trajspec_lines.append('Modified symbols: ' + ', '.join(modified))
        if added:
            trajspec_lines.append('Added symbols: ' + ', '.join(added))
        if dropped:
            trajspec_lines.append('Dropped symbols: ' + ', '.join(dropped))
    # Long files (>150 lines) are only preloaded when the query carries an
    # explicit line range; seed the query with ranges parsed from the raw
    # test log (tracebacks / "file.py:155:" patterns) so the failing lines
    # always land in [CURRENT_ON_DISK_EXCERPTS].
    try:
        from harness.guards import _parse_prompt_line_ranges
        range_markers = ['L%d-L%d' % (a, b)
                         for a, b in _parse_prompt_line_ranges(test_log_text)][:8]
    except Exception:
        range_markers = []
    excerpts_query = task['prompt'] + '\n' + compact_failure
    if trajspec_seed_tokens:
        excerpts_query += '\n' + ' '.join(trajspec_seed_tokens)
    if range_markers:
        excerpts_query += '\n' + ' '.join(range_markers)
    excerpts = build_preloaded_file_excerpts(
        list(task['files']) + list(task.get('read_only_files', ())),
        excerpts_query,
        read_only_files=task.get('read_only_files', []),
    )
    prompt = (
        task['prompt']
        + '\n\n[CURRENT_PATCH]\n'
        + current_patch
    )
    if trajspec_lines:
        prompt += '\n\n[TRAJSPEC_MODIFIED_SYMBOLS]\n' + '\n'.join(trajspec_lines)
    if excerpts:
        prompt += '\n\n[CURRENT_ON_DISK_EXCERPTS]\n' + excerpts
    failing_test_excerpts = extract_failing_test_excerpts(
        test_log_text,
        task.get('directory'),
        exclude_files=list(task.get('files', ())) + list(task.get('read_only_files', ())),
    )
    if failing_test_excerpts:
        prompt += '\n\n' + failing_test_excerpts
    prompt += (
        '\n\n[TEST_ARGV FAILURE — fix only the failing assertion/import]:\n'
        'Apply the minimal targeted edit directly to fix the failure above; '
        'do NOT re-read entire files from scratch.\n'
        + compact_failure
    )
    return prompt


def _snapshot_files_for_feedback(run_dir, files, tag):
    """Save plain-bytes snapshots of Python files under run_dir/<tag>/ (H-33)."""
    out = Path(run_dir) / tag
    try:
        out.mkdir(parents=True, exist_ok=True)
        for name in files:
            if not str(name).endswith('.py'):
                continue
            p = Path(name)
            safe = str(name).replace('/', '__').replace('\\', '__')
            target = out / safe
            if p.exists():
                target.write_bytes(p.read_bytes())
            elif target.exists():
                target.unlink()
    except OSError:
        pass

def _extract_read_windows_from_worker_log(run_dir, *, max_windows=6):
    """H-63: extract deduplicated 'L<start>-L<end>' read windows from worker.log tool events."""
    log_path = Path(run_dir) / 'worker.log'
    if not log_path.is_file():
        return []
    windows = []
    seen = set()
    try:
        lines = log_path.read_text(encoding='utf-8', errors='replace').splitlines()
    except OSError:
        return []
    for line in lines:
        line = line.strip()
        if not line.startswith('{'):
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        part = rec.get('part') if isinstance(rec.get('part'), dict) else rec
        tool_name = part.get('tool') or part.get('name')
        if tool_name != 'read':
            continue
        state = part.get('state') if isinstance(part.get('state'), dict) else {}
        inp = state.get('input') if isinstance(state.get('input'), dict) else part.get('input')
        if isinstance(inp, str):
            try:
                inp = json.loads(inp)
            except ValueError:
                inp = None
        if not isinstance(inp, dict):
            continue
        path = inp.get('path') or ''
        offset = inp.get('offset')
        limit = inp.get('limit')
        if isinstance(offset, int) and offset >= 1:
            span = limit if (isinstance(limit, int) and limit >= 1) else 80
            end = offset + min(span, 200) - 1
            label = f'{Path(path).name} L{offset}-L{end}' if path else f'L{offset}-L{end}'
            if label not in seen:
                seen.add(label)
                windows.append(label)
                if len(windows) >= max_windows:
                    break
    return windows


def build_prior_attempt_feedback(state, task_id, *, max_patch_chars=1500, max_chars=1200):
    """Compact feedback from earlier completed runs of the same task_id."""
    runs_dir = Path(state) / 'runs'
    matches = []
    for result_path in runs_dir.glob('*/result.json'):
        try:
            res = json.loads(result_path.read_text())
        except (OSError, ValueError):
            continue
        if res.get('task_id') == task_id and res.get('status') not in (
                'configuration_verified_no_model_call', 'pending', None):
            try:
                matches.append((res, result_path.parent))
            except OSError:
                continue
    if not matches:
        return ''
    matches.sort(key=lambda item: (str(item[0].get('started_at') or ''),
                                   item[1].stat().st_mtime))
    last_res, last_dir = matches[-1]
    if last_res.get('status') in ('validation_passed_review_required',
                                  'patch_ready_review_required', 'succeeded'):
        return ''
    if last_res.get('status') != 'blocked':
        return ''
    reason = str(last_res.get('reason') or 'blocked')
    model = str(last_res.get('model') or 'unknown')
    sections = ''
    try:
        test_log = last_dir / 'test.log'
        if test_log.is_file() and test_log.stat().st_size:
            sections += '\n[Test failure]:\n' + extract_compact_test_failure(
                test_log.read_text(errors='replace'), '')
    except OSError:
        pass
    try:
        patch_file = last_dir / 'changes.patch'
        if patch_file.is_file() and patch_file.stat().st_size:
            sections += '\n[Previous patch (truncated)]:\n' + patch_file.read_text(
                errors='replace')[:max_patch_chars]
    except OSError:
        pass
    # H-33: symbol-level diff between before/after Python snapshots of the
    # failed attempt, plus a one-line guardrail/salvage event summary.
    extra = ''
    try:
        before_dir, after_dir = last_dir / 'before', last_dir / 'after'
        if before_dir.is_dir() and after_dir.is_dir():
            names = sorted({p.name for p in before_dir.iterdir()} | {p.name for p in after_dir.iterdir()})
            added, modified, dropped = [], [], []
            for name in names:
                b_path, a_path = before_dir / name, after_dir / name
                b = _python_top_symbols(b_path.read_text(errors='replace')) if b_path.is_file() else {}
                a = _python_top_symbols(a_path.read_text(errors='replace')) if a_path.is_file() else {}
                added += [f'{sym}()' for sym in sorted(set(a) - set(b))]
                modified += [f'{sym}()' for sym in sorted(set(a) & set(b)) if a[sym] != b[sym]]
                dropped += sorted(set(b) - set(a))
            diff_lines = []
            if modified:
                diff_lines.append('Modified symbols: ' + ', '.join(modified))
            if added:
                diff_lines.append('Added symbols: ' + ', '.join(added))
            if dropped:
                diff_lines.append('Dropped symbols: ' + ', '.join(dropped))
            if diff_lines:
                extra += '\n[SYMBOL DIFF vs last attempt]:\n' + '\n'.join(diff_lines)
    except OSError:
        pass
    try:
        events_path = last_dir / 'events.jsonl'
        if events_path.is_file():
            notable = []
            for line in events_path.read_text(errors='replace').splitlines():
                try:
                    name = str(json.loads(line).get('event') or '')
                except ValueError:
                    continue
                if name and any(k in name for k in ('guard', 'salvage', 'repair', 'hint', 'reconcile')) \
                        and name not in notable:
                    notable.append(name)
            if notable:
                extra += '\n[Previous attempt guardrail events]: ' + ', '.join(notable[:8])
    except OSError:
        pass
    read_windows = _extract_read_windows_from_worker_log(last_dir)
    if read_windows:
        extra += '\n[Visited read windows]: ' + ', '.join(read_windows)
    feedback = (f'\n\n[PRIOR_ATTEMPT_FEEDBACK — previous run on this task_id failed with '
                f'reason={reason} on model={model}; avoid repeating the same mistake]:'
                + extra + sections)
    return feedback[:max_chars] if len(feedback) > max_chars else feedback

@contextmanager
def isolated_worktree(task, run_dir):
    if not task.get('worktree'):
        yield task
        return
    top = subprocess.run(['/usr/bin/git', 'rev-parse', '--show-toplevel'], cwd=task['directory'],
                         capture_output=True, timeout=15)
    if top.returncode:
        raise GuardError('worktree_requires_git_repository')
    repo_root = Path(top.stdout.decode().strip()).resolve()
    dir_rel = Path(task['directory']).resolve().relative_to(repo_root)
    wt_root = (Path(run_dir).parent.parent / 'worktrees' / Path(run_dir).name).resolve()
    wt_root.parent.mkdir(parents=True, exist_ok=True)
    add = subprocess.run(['/usr/bin/git', 'worktree', 'add', '--detach', str(wt_root), 'HEAD'],
                         cwd=repo_root, capture_output=True, timeout=30)
    if add.returncode:
        raise GuardError('worktree_creation_failed')
    try:
        wt_dir = (wt_root / dir_rel).resolve()
        wt_files = []
        for src in task['files']:
            rel = Path(src).resolve().relative_to(repo_root)
            dst = (wt_root / rel).resolve()
            dst.parent.mkdir(parents=True, exist_ok=True)
            if Path(src).exists():
                dst.write_bytes(Path(src).read_bytes())
            elif dst.exists():
                dst.unlink()
            wt_files.append(str(dst))
        wt_ro_files = []
        for src in task.get('read_only_files', ()):
            rel = Path(src).resolve().relative_to(repo_root)
            dst = (wt_root / rel).resolve()
            dst.parent.mkdir(parents=True, exist_ok=True)
            if Path(src).exists():
                dst.write_bytes(Path(src).read_bytes())
            elif dst.exists():
                dst.unlink()
            wt_ro_files.append(str(dst))
        prefix = str(repo_root)
        _fh = getattr(sys.modules.get('opencode_guard'), 'file_hashes', file_hashes)
        def remap(argv):
            return [x.replace(prefix, str(wt_root)) for x in argv]
        wt_task = dict(task, directory=str(wt_dir), files=wt_files,
                       prompt=task.get('prompt', '').replace(prefix, str(wt_root)),
                       origin_files=list(task['files']), origin_hashes=_fh(task['files']),
                       read_only_files=wt_ro_files,
                       test_argv=remap(task['test_argv']))
        # H-65: keep original repo paths reachable as read-only aliases so a
        # worker that references task['files'] / task['read_only_files'] paths
        # is not permission-denied inside the worktree. Original paths must
        # NEVER gain edit access via file_aliases (config_for grants 'edit' on
        # file_aliases in model mode), so both sets go to read_only_file_aliases.
        wt_task['file_aliases'] = list(task.get('file_aliases') or ())
        wt_task['read_only_file_aliases'] = sorted(
            {*(task.get('read_only_file_aliases') or ()),
             *task.get('read_only_files', ()), *task.get('files', ())})
        if 'command_argv' in task:
            wt_task['command_argv'] = remap(task['command_argv'])
        yield wt_task
    finally:
        subprocess.run(['/usr/bin/git', 'worktree', 'remove', '--force', str(wt_root)],
                       cwd=repo_root, capture_output=True, timeout=30)

def compute_repair_max_steps(task: dict) -> int:
    """Adaptive micro-repair budget: tighter than the primary run but enough
    for the worker to re-read only the failing region and apply one edit."""
    return min(MAX_STEPS, max(6, 4 + 2 * len(task.get('files', ()))))

def compute_repair_timeout_seconds(task: dict) -> int:
    """Bounded proportional wall-clock timeout for micro_repair: 15s per
    repair step (floor 45s), capped at the task's own timeout."""
    repair_steps = compute_repair_max_steps(task)
    return min(int(task.get('timeout_seconds', 180)), max(45, 15 * repair_steps))


def should_auto_escalate(raw_task: dict) -> bool:
    """Decide whether a raw task dict should run through the best_of_2 path."""
    if not isinstance(raw_task, dict) or raw_task.get('mode', 'model') != 'model':
        return False
    if raw_task.get('strategy') == 'best_of_2':
        return True
    if raw_task.get('strategy') == 'single' and raw_task.get('auto_escalate') is not True:
        return False
    if isinstance(raw_task.get('auto_escalate'), bool):
        return raw_task['auto_escalate']
    return os.environ.get('HARNESS_AUTO_ESCALATE', '').strip().lower() in ('1', 'true', 'yes', 'on')


def run_task(task_path, *, cli=None, db=None, state=None, check=False, upstream_url=None, fetch=None,
             _quiet=False, _return_result=False):
    if cli is None:
        cli = CLI
    if db is None:
        db = DB
    if state is None:
        state = STATE
    if not check:
        try:
            raw_strategy = json.loads(Path(task_path).read_text())
        except (OSError, ValueError):
            raw_strategy = {}
        if should_auto_escalate(raw_strategy):
            return run_best_of_2(task_path, cli=cli, db=db, state=state,
                                 upstream_url=upstream_url, fetch=fetch)
    result = {'status': 'rejected', 'reason': '', 'artifact': None}
    reconciled_cost = None
    _fh = getattr(sys.modules.get('opencode_guard'), 'file_hashes', file_hashes)
    try:
        task = validate_task(json.loads(Path(task_path).read_text()))
        state = Path(state)
        lock_files = [] if task.get('worktree') else (task['files'] + ([task['patch_file']] if task['mode'] == 'patch' else []))
        with interruptions(), locks(state, lock_files, task['task_id']), ExitStack() as stack:
            spent = prior_cost(state, task['task_id']) if task['mode'] != 'command' else 0
            if spent >= task['max_task_cost_usd']:
                raise GuardError('task_budget_exhausted')
            budget = min(task['max_cost_usd'], task['max_task_cost_usd'] - spent)
            run_dir = state / 'runs' / uuid.uuid4().hex
            run_dir.mkdir(parents=True, mode=0o700)
            started = time.monotonic()
            started_at = time.time()
            if not check:
                task = stack.enter_context(isolated_worktree(task, run_dir))
            tracked = task['files'] + ([task['patch_file']] if task['mode'] == 'patch' else [])
            ro_files = task.get('read_only_files', ())
            ro_hashes_before = _fh(ro_files)
            result = dict(task_id=task['task_id'], session_id=None, status='pending', reason='',
                          model=task['model'] if task['mode'] != 'command' else None,
                          cost=0, steps=0, observed_cost=0, prior_task_cost=spent,
                          limit_reason=task.get('limit_reason'),
                          limits={k: task[k] for k in LIMITS}, test_return_code=None,
                          artifact=str(run_dir), hashes_before=_fh(task['files']),
                          worktree=task.get('worktree', False),
                          started_at=started_at)
            result['limits']['available_cost_usd'] = budget
            private_write(run_dir / 'result.json', result)
            private_write(run_dir / 'task.json', task)
            capture_files(run_dir, tracked)
            _snapshot_files_for_feedback(run_dir, task['files'], 'before')
            event(run_dir, 'run_started', task_id=task['task_id'], mode=task['mode'])
            validation_attempted = False
            proxy = None
            is_real_cli = False
            try:
                is_real_cli = Path(cli).exists() and Path(CLI).exists() and Path(cli).resolve() == Path(CLI).resolve()
            except OSError:
                is_real_cli = False
            use_proxy = (not check and task['mode'] != 'command' and (upstream_url is not None or is_real_cli))
            started_ref = [None]
            active_max_steps = [task['max_steps']]
            active_timeout_seconds = [task['timeout_seconds']]
            try:
                if task['mode'] != 'command':
                    verify_db(db)
                    if use_proxy:
                        def _gate():
                            if started_ref[0] is not None and (time.monotonic() - started_ref[0]) >= active_timeout_seconds[0]:
                                return 'worker_timeout'
                            sid_cur = result.get('session_id')
                            if not sid_cur:
                                return None
                            snap_cur = snapshot(db, sid_cur)
                            if snap_cur is None:
                                return None
                            m_cur = metrics(snap_cur)
                            reason = violation(m_cur, budget, active_max_steps[0])
                            if reason:
                                return reason
                            if m_cur['steps'] >= active_max_steps[0] and not m_cur['unfinished']:
                                return 'step_limit'
                            return None
                        health_store = ProviderHealthStore(state / 'provider_health.json')
                        proxy = stack.enter_context(openrouter_proxy(
                            _gate, upstream_url=upstream_url or OPENROUTER_UPSTREAM,
                            request_timeout=max(15, min(int(task['timeout_seconds']), 60)),
                            default_model=task['model'],
                            reasoning_max_tokens=task.get('reasoning_max_tokens'),
                            allowed_files=task.get('files'),
                            health_store=health_store,
                            max_context_tokens=task.get('max_context_tokens'),
                            fallback_teacher_model=task.get('fallback_teacher_model'),
                            enforce_egress_guard=task.get('enforce_egress_guard', True),
                            enforce_readonly_sql=task.get('enforce_readonly_sql', False),
                            upstream_mode=task.get('upstream_mode', 'openrouter')))
                    base_url = (proxy.base_url if proxy is not None
                                else ('http://127.0.0.1:9/api/v1' if (check and is_real_cli) else None))
                    preflight(cli, task, run_dir, base_url=base_url)
                if check:
                    result['status'] = 'configuration_verified_no_model_call'
                else:
                    salvage_candidate = None
                    if task['mode'] == 'command':
                        rc = execute(task['command_argv'], task['directory'], task['timeout_seconds'], run_dir / 'command.log')
                        if rc:
                            raise GuardError('prepared_command_failed')
                    else:
                        sid = 'ses_' + uuid.uuid4().hex[:26]
                        if snapshot(db, sid) is not None:
                            raise GuardError('session_id_collision')
                        result['session_id'] = sid
                        originals = {name: (Path(name).read_bytes() if Path(name).is_file() else None)
                                     for name in task['files']}
                        if task.get('include_prior_feedback', True):
                            prior_feedback = build_prior_attempt_feedback(state, task['task_id'])
                            if prior_feedback:
                                task['prompt'] += prior_feedback
                                event(run_dir, 'prior_attempt_feedback_injected')
                        if task.get('include_repo_memory', True):
                            repo_hint = build_repo_memory_hint(state, task.get('directory'))
                            if repo_hint:
                                task['prompt'] += repo_hint
                                event(run_dir, 'repo_memory_hint_injected')
                        try:
                            model_run(cli, db, task, run_dir, result, budget, proxy=proxy, started_ref=started_ref)
                        except GuardError as exc:
                            if (task['mode'] == 'model'
                                    and str(exc) in SALVAGEABLE_BLOCK_REASONS
                                    and _fh(task['files']) != result['hashes_before']):
                                capture_files(run_dir, tracked)
                                _snapshot_files_for_feedback(run_dir, task['files'], 'after')
                                salvage_candidate = str(exc)
                                event(run_dir, 'salvage_validation_started', reason=salvage_candidate)
                            else:
                                raise
                        if task['mode'] == 'patch':
                            if _fh(task['files']) != result['hashes_before']:
                                raise GuardError('patch_worker_modified_source')
                            check_patch(task)
                        elif salvage_candidate is None and _fh(task['files']) == result['hashes_before']:
                            raise GuardError('no_change')
                    if task['mode'] == 'patch':
                        result['status'] = 'patch_ready_review_required'
                    else:
                        validation_attempted = True
                        result['test_return_code'] = execute(task['test_argv'], task['directory'],
                                                            task['test_timeout_seconds'], run_dir / 'test.log')
                        if result['test_return_code']:
                            capture_files(run_dir, tracked)
                            if salvage_candidate is not None:
                                raise GuardError(salvage_candidate)
                            repair_salvage_reason = None
                            if (task.get('auto_micro_repair', True)
                                    and task['mode'] == 'model'
                                    and (run_dir / 'changes.patch').exists()
                                    and (run_dir / 'changes.patch').stat().st_size > 0
                                    and result.get('cost') is not None
                                    and result.get('steps', 0) <= task['max_steps']
                                    and result['cost'] <= min(budget * 0.8, 0.08)):
                                repair_max_steps = compute_repair_max_steps(task)
                                active_max_steps[0] = repair_max_steps
                                repair_timeout = compute_repair_timeout_seconds(task)
                                active_timeout_seconds[0] = repair_timeout
                                repair_task = dict(
                                    task,
                                    max_steps=repair_max_steps,
                                    timeout_seconds=repair_timeout,
                                    prompt=build_micro_repair_prompt(task, run_dir, originals),
                                )
                                primary_steps = result.get('steps', 0)
                                primary_cost = result.get('cost')
                                primary_observed = result.get('observed_cost')
                                primary_sid = result['session_id']
                                sid2 = 'ses_' + uuid.uuid4().hex[:26]
                                result['session_id'] = sid2
                                result['micro_repair_session_id'] = sid2
                                result['micro_repair_attempted'] = True
                                hashes_before_repair = _fh(task['files'])
                                if proxy is not None and hasattr(proxy, 'reset_session'):
                                    proxy.reset_session()
                                try:
                                    event(run_dir, 'micro_repair_started', session_id=sid2)
                                    try:
                                        model_run(cli, db, repair_task, run_dir, result,
                                                  max(0.0, budget - primary_cost),
                                                  proxy=proxy, started_ref=started_ref)
                                    except GuardError as exc:
                                        if (str(exc) in SALVAGEABLE_BLOCK_REASONS
                                                and _fh(task['files']) != hashes_before_repair):
                                            capture_files(run_dir, tracked)
                                            _snapshot_files_for_feedback(run_dir, task['files'], 'after')
                                            repair_salvage_reason = str(exc)
                                            event(run_dir, 'micro_repair_salvage_started',
                                                  reason=repair_salvage_reason)
                                        else:
                                            raise
                                finally:
                                    result['steps'] = primary_steps + (result.get('steps') or 0)
                                    repair_cost = result.get('cost')
                                    result['cost'] = (None if primary_cost is None or repair_cost is None
                                                      else primary_cost + repair_cost)
                                    repair_observed = result.get('observed_cost')
                                    result['observed_cost'] = (None if primary_observed is None
                                                               or repair_observed is None
                                                               else primary_observed + repair_observed)
                                    result['session_id'] = primary_sid
                                if _fh(task['files']) == hashes_before_repair:
                                    event(run_dir, 'micro_repair_no_change')
                                else:
                                    # H-56: two-stage fast-fail verification. First
                                    # re-run only the previously failed nodeids; if
                                    # any still fail, skip the full suite.
                                    first_test_log_text = ''
                                    try:
                                        first_test_log_text = (run_dir / 'test.log').read_text(errors='replace')
                                    except OSError:
                                        first_test_log_text = ''
                                    targeted_argv = build_targeted_pytest_argv(
                                        task['test_argv'], first_test_log_text)
                                    if targeted_argv is not None:
                                        rc_fast = execute(targeted_argv, task['directory'],
                                                          min(30, task['test_timeout_seconds']),
                                                          run_dir / 'test.log')
                                        if rc_fast != 0:
                                            result['test_return_code'] = rc_fast
                                            event(run_dir, 'micro_repair_fast_fail',
                                                  return_code=rc_fast)
                                        else:
                                            result['test_return_code'] = execute(
                                                task['test_argv'], task['directory'],
                                                task['test_timeout_seconds'], run_dir / 'test.log')
                                    else:
                                        result['test_return_code'] = execute(
                                            task['test_argv'], task['directory'],
                                            task['test_timeout_seconds'], run_dir / 'test.log')
                            if result['test_return_code']:
                                raise GuardError(repair_salvage_reason or 'validation_failed')
                            if repair_salvage_reason is not None:
                                result['salvaged_from'] = f'micro_repair_{repair_salvage_reason}'
                        if ro_files and _fh(ro_files) != ro_hashes_before:
                            raise GuardError('read_only_file_modified')
                        if task.get('origin_files'):
                            if _fh(task['origin_files']) != task['origin_hashes']:
                                raise GuardError('worktree_origin_modified_concurrently')
                            for src, dst in zip(task['files'], task['origin_files']):
                                if Path(src).exists():
                                    Path(dst).parent.mkdir(parents=True, exist_ok=True)
                                    Path(dst).write_bytes(Path(src).read_bytes())
                                elif Path(dst).exists():
                                    Path(dst).unlink()
                        if salvage_candidate is not None:
                            result['salvaged_from'] = salvage_candidate
                        result['status'] = 'validation_passed_review_required'
            except (GuardError, OSError, ValueError, TypeError, sqlite3.Error) as exc:
                result.update(status='blocked', reason=str(exc) if isinstance(exc, GuardError) else type(exc).__name__)
            finally:
                result['duration_seconds'] = round(time.monotonic() - started, 3)
                try:
                    result['hashes_after'] = _fh(task['files'])
                    capture_files(run_dir, tracked)
                    _snapshot_files_for_feedback(run_dir, task['files'], 'after')
                    # A failed worker can leave useful changes. Validate them once,
                    # but never turn that into execution success or settled billing.
                    if (not check and task['mode'] == 'model' and result['session_id']
                            and not validation_attempted and result['hashes_after'] != result['hashes_before']):
                        validation_attempted = True
                        try:
                            result['test_return_code'] = execute(task['test_argv'], task['directory'],
                                task['test_timeout_seconds'], run_dir / 'test.log')
                        except (GuardError, OSError) as exc:
                            result['validation_error'] = str(exc) if isinstance(exc, GuardError) else type(exc).__name__
                except OSError:
                    result['artifact_error'] = 'cannot_capture_final_files'
                    result['status'] = 'blocked'
                    result['reason'] = result['reason'] or 'cannot_capture_final_files'
                result['duration_seconds'] = round(time.monotonic() - started, 3)
                result['finished_at'] = time.time()
                result['validation_attempted'] = validation_attempted
                result_path = run_dir / 'result.json'
                event(run_dir, 'run_finished', status=result['status'], reason=result['reason'],
                      cost=result['cost'], test_return_code=result['test_return_code'])
                private_write(result_path, result)
                if proxy is not None and result.get('cost') is None:
                    try:
                        rec = auto_reconcile_from_proxy(
                            run_dir, result_path, result,
                            auth_bearer=proxy.auth_bearer, fetch=fetch, db=db)
                        if rec is not None:
                            reconciled_cost = rec['cost']
                    except (GuardError, OSError, ValueError, sqlite3.Error) as exc:
                        event(run_dir, 'auto_reconcile_deferred',
                              reason=str(exc) if isinstance(exc, GuardError) else type(exc).__name__)
                try:
                    synthesize_spans_from_run_dir(run_dir)
                    sync_state_traces_to_phoenix(state, timeout=1.5, max_runs=25)
                except Exception:
                    pass
    except (GuardError, OSError, ValueError, TypeError, sqlite3.Error) as exc:
        error = str(exc) if isinstance(exc, GuardError) else type(exc).__name__
        if result.get('reason'):
            result['supervisor_error'] = error
        else:
            result['reason'] = error
        result['status'] = 'blocked'
    out = {k: result.get(k) for k in ('status', 'reason', 'artifact', 'cost', 'observed_cost', 'steps')}
    if reconciled_cost is not None:
        out['reconciled_cost'] = reconciled_cost
    if _return_result:
        return result
    if not _quiet:
        print(json.dumps(out))
    return 0 if result['status'] in ('validation_passed_review_required', 'patch_ready_review_required', 'configuration_verified_no_model_call') else 1

def _build_candidate1_trajspec_feedback(res, originals, after_snap):
    """H-39: compact TraSpec handoff so Candidate 2 avoids repeating
    Candidate 1's validation mistake."""
    status = str(res.get('status') or 'blocked')
    reason = str(res.get('reason') or '')
    added, modified, dropped = [], [], []
    for name in sorted(set(originals) | set(after_snap)):
        if not str(name).endswith('.py'):
            continue
        before = originals.get(name)
        after = after_snap.get(name)
        before_src = before.decode('utf-8', errors='replace') if before is not None else ''
        after_src = after.decode('utf-8', errors='replace') if after is not None else ''
        before_syms = _python_top_symbols(before_src) if before_src else {}
        after_syms = _python_top_symbols(after_src) if after_src else {}
        after_spans = _python_symbol_spans(after_src) if after_src else {}
        for sym in sorted(set(after_syms) - set(before_syms)):
            span = after_spans.get(sym)
            added.append(f'{sym} (L{span[0]}-L{span[1]})' if span else sym)
        for sym in sorted(set(before_syms) & set(after_syms)):
            if before_syms[sym] != after_syms[sym]:
                span = after_spans.get(sym)
                modified.append(f'{sym} (L{span[0]}-L{span[1]})' if span else sym)
        for sym in sorted(set(before_syms) - set(after_syms)):
            dropped.append(sym)
    parts = ['\n\n[CANDIDATE_1_TRAJSPEC_FEEDBACK]',
             'Candidate 1 status: %s' % status]
    if reason:
        parts.append('Candidate 1 reason: %s' % reason)
    if modified:
        parts.append('Modified symbols: %s' % ', '.join(modified))
    if added:
        parts.append('Added symbols: %s' % ', '.join(added))
    if dropped:
        parts.append('Dropped symbols: %s' % ', '.join(dropped))
    artifact = res.get('artifact')
    if artifact:
        test_log = Path(artifact) / 'test.log'
        try:
            if test_log.exists() and test_log.stat().st_size > 0:
                parts.append(extract_compact_test_failure(
                    test_log.read_text(errors='replace'), '', max_chars=1500))
        except OSError:
            pass
        read_windows = _extract_read_windows_from_worker_log(artifact)
        if read_windows:
            parts.append('Visited read windows: %s' % ', '.join(read_windows))
    return '\n'.join(parts)

def run_best_of_2(task_path, *, cli=None, db=None, state=None, upstream_url=None, fetch=None):
    """Verifier-guided best_of_2 strategy: run candidate 1; only if it does not
    pass validation cleanly, restore the original file bytes and run candidate 2.
    Keep the best candidate by validation outcome, then absence of salvage,
    then lower steps, then lower cost."""
    if cli is None:
        cli = CLI
    if db is None:
        db = DB
    if state is None:
        state = STATE
    first_bytes = None
    try:
        raw_task = json.loads(Path(task_path).read_text())
        task = validate_task(raw_task)
        candidates = task['candidate_models'] or [
            task['model'],
            task.get('fallback_teacher_model') or 'openrouter/deepseek/deepseek-v4.1-flash-1017',
        ]

        def _snapshot():
            snap = {}
            for name in task['files']:
                p = Path(name)
                snap[name] = p.read_bytes() if p.exists() else None
            return snap

        def _restore(snap):
            for name, data in snap.items():
                p = Path(name)
                if data is None:
                    if p.exists():
                        p.unlink()
                else:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(data)

        originals = _snapshot()

        def _passes(res):
            return res.get('status') == 'validation_passed_review_required'

        def _better(a, b):
            pa, pb = _passes(a[0]), _passes(b[0])
            if pa != pb:
                return a if pa else b
            if pa:
                sa = not a[0].get('salvaged_from')
                sb = not b[0].get('salvaged_from')
                if sa != sb:
                    return a if sa else b
                ka = (a[0].get('steps') or 0,
                      float('inf') if a[0].get('cost') is None else a[0]['cost'])
                kb = (b[0].get('steps') or 0,
                      float('inf') if b[0].get('cost') is None else b[0]['cost'])
                return a if ka <= kb else b
            return a

        results = []
        handoff_feedback = ''
        for index, model in enumerate(candidates):
            candidate_path = Path(task_path).with_name(
                Path(task_path).stem + ('.candidate%d.json' % (index + 1)))
            candidate_task = dict(raw_task, strategy='single', auto_escalate=False, model=model)
            candidate_task.pop('candidate_models', None)
            if index > 0 and handoff_feedback:
                candidate_task['prompt'] = (candidate_task.get('prompt') or '') + handoff_feedback
            candidate_path.write_text(json.dumps(candidate_task))
            try:
                res = run_task(candidate_path, cli=cli, db=db, state=state,
                               upstream_url=upstream_url, fetch=fetch,
                               _quiet=True, _return_result=True)
            finally:
                try:
                    candidate_path.unlink()
                except OSError:
                    pass
            results.append((res, model))
            if _passes(res) and not res.get('salvaged_from'):
                break
            if index == 0 and res.get('status') == 'blocked' and res.get('reason') not in ESCALATABLE_BLOCK_REASONS:
                break
            if index == 0 and len(candidates) > 1:
                first_bytes = _snapshot()
                handoff_feedback = _build_candidate1_trajspec_feedback(res, originals, first_bytes)
                _restore(originals)
        winner = results[0]
        for other in results[1:]:
            winner = _better(winner, other)
        if len(results) > 1 and winner is results[0] and first_bytes is not None:
            _restore(first_bytes)

        # H-34 fix: aggregate across ALL executed candidates, not just passed ones.
        def _sum(key):
            pool = [r[0] for r in results]
            vals = [r.get(key) for r in pool]
            if key == 'steps':
                return sum(v or 0 for v in vals)
            # For cost keys, fall back per-candidate to observed_cost when
            # the settled cost was never recorded (e.g. a blocked attempt).
            if key == 'cost' and any(v is None for v in vals):
                vals = [v if v is not None else r.get('observed_cost')
                        for r, v in zip(pool, vals)]
            if all(v is None for v in vals):
                return None
            total = sum(v for v in vals if v is not None)
            return round(total, 8) if key == 'cost' else total
        summary = {
            'status': winner[0].get('status'),
            'reason': winner[0].get('reason', ''),
            'artifact': winner[0].get('artifact'),
            'winning_model': winner[1],
            'candidates_run': len(results),
            'cost': _sum('cost'),
            'observed_cost': _sum('observed_cost'),
            'steps': _sum('steps'),
        }
        print(json.dumps(summary))
        return 0 if _passes(winner[0]) else 1
    except (GuardError, OSError, ValueError, TypeError, sqlite3.Error, KeyError) as exc:
        error = str(exc) if isinstance(exc, GuardError) else type(exc).__name__
        print(json.dumps({'status': 'blocked', 'reason': error, 'candidates_run': 0,
                          'winning_model': None, 'artifact': None,
                          'cost': None, 'observed_cost': None, 'steps': None}))
        return 1

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'check', 'inspect', 'reconcile'])
    parser.add_argument('task_json', type=Path)
    parser.add_argument('--evidence', type=Path, help='Request-to-provider mapping; billing is fetched read-only')
    parser.add_argument('--settle-unexposed', action='store_true',
                        help='Settle unexposed/aborted requests using local finished telemetry + conservative step cap')
    parser.add_argument('--verified-by', type=str, help='Operator identifier for --settle-unexposed')
    parser.add_argument('--reason', type=str, help='Audit justification / mapping source for --settle-unexposed')
    args = parser.parse_args()
    if args.action in {'inspect', 'reconcile'}:
        try:
            if args.action == 'reconcile':
                if args.evidence is not None and args.settle_unexposed:
                    raise GuardError('choose_either_evidence_or_settle_unexposed')
                if args.settle_unexposed:
                    evidence = build_unexposed_settlement_evidence(
                        args.task_json, verified_by=args.verified_by, mapping_source=args.reason)
                elif args.evidence is not None:
                    evidence = json.loads(args.evidence.read_text())
                else:
                    raise GuardError('evidence_file_required')
                r = reconcile_run(args.task_json, evidence)
                print(json.dumps({'status': 'billing_reconciled_no_model_call', 'cost': r['cost']}))
            else:
                path = args.task_json / 'result.json'
                r = json.loads(path.read_text())
                try:
                    cost, billing = resolved_cost(path, r), 'known'
                except GuardError as exc:
                    cost, billing = None, str(exc)
                print(json.dumps({'task_id': r.get('task_id'), 'status': r.get('status'),
                    'reason': r.get('reason'), 'cost': cost, 'observed_cost': r.get('observed_cost'),
                    'billing': billing, 'test_return_code': r.get('test_return_code'),
                    'changed_files': [p for p, h in r.get('hashes_after', {}).items()
                                      if h != r.get('hashes_before', {}).get(p)],
                    'next_action': 'Review saved files and validation; obtain complete billing evidence before paid retry'
                                   if cost is None else 'Review changes and validation before any next attempt'}))
            return 0
        except (GuardError, OSError, ValueError, KeyError, TypeError) as exc:
            print(json.dumps({'status': 'blocked', 'reason': str(exc) if isinstance(exc, GuardError) else type(exc).__name__}))
            return 1
    return run_task(args.task_json, check=args.action == 'check')

if __name__ == '__main__':
    sys.exit(main())
