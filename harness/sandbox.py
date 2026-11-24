"""Guarded OpenCode worker supervisor: sandboxed task execution, cost accounting and billing reconciliation.

Ported from the former single-module ``.tools/opencode_guard.py``. Signatures,
guard-error messages and behavior are preserved verbatim so the launcher,
journal and test suites keep working unchanged.
"""

import argparse
import ast
import base64
import concurrent.futures
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
from datetime import datetime, timezone
from pathlib import Path

from .telemetry import sync_state_traces_to_phoenix, synthesize_spans_from_run_dir

from .herdr import report_herdr_task

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
from .guards import (audit_patch_for_reward_hacking,
                     build_cross_file_symbol_context, build_file_skeletons,
                     build_preloaded_file_excerpts, extract_compact_test_failure,
                     extract_failing_test_excerpts,
                     extract_runtime_assertion_locals)
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

REPAIRABLE_SALVAGE_REASONS = frozenset({
    'worker_timeout',
    'step_limit',
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


def _extract_leading_env_pythonpath(argv) -> tuple[list[str], list[str]]:
    """Return (pythonpath_entries, remaining_cmd_argv) for leading `env PYTHONPATH=...`."""
    tokens = list(argv or ())
    env_pp: list[str] = []
    if tokens and Path(tokens[0]).name == 'env':
        idx = 1
        while idx < len(tokens) and '=' in tokens[idx] and not tokens[idx].startswith('-'):
            key, val = tokens[idx].split('=', 1)
            if key == 'PYTHONPATH' and val:
                env_pp.extend(p for p in val.split(os.pathsep) if p)
            idx += 1
        return env_pp, tokens[idx:]
    return env_pp, tokens


def _top_level_imported_modules(py_path: Path) -> set[str]:
    """Return top-level module names imported by a Python file."""
    try:
        tree = ast.parse(py_path.read_text(encoding='utf-8', errors='replace'))
    except (OSError, SyntaxError, ValueError):
        return set()
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    mods.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                mods.add(node.module.split('.')[0])
    return mods


def _has_module_in_dir(directory: Path, module_name: str) -> bool:
    return (directory / f'{module_name}.py').is_file() or (directory / module_name / '__init__.py').is_file()


def build_test_env_overrides(task: dict) -> dict[str, str]:
    """Build PYTHONPATH env overrides for test/validation runs."""
    directory = str(task.get('directory') or '')
    base = Path(directory) if directory else None

    caller_entries, _ = _extract_leading_env_pythonpath(task.get('test_argv'))
    env_pp = os.environ.get('PYTHONPATH', '')
    if env_pp:
        caller_entries.extend(p for p in env_pp.split(os.pathsep) if p)

    candidates: list[str] = []
    if base is not None:
        candidates.append(str(base))
        src_dir = base / 'src'
        if src_dir.is_dir():
            candidates.append(str(src_dir))

    all_task_paths = (
        list(task.get('files') or ())
        + list(task.get('read_only_files') or ())
        + list(task.get('test_argv') or ())
    )
    under_ai = any(
        '/ai/' in str(p).replace('\\', '/')
        or str(p).replace('\\', '/').startswith('ai/')
        or str(p).replace('\\', '/').endswith('/ai')
        for p in all_task_paths
    )
    ai_subdirs = ('ai/mcp', 'ai/serving', 'ai/tools', 'ai/coding_worker')
    if base is not None and under_ai:
        for sub in ai_subdirs:
            sub_dir = base / sub
            if sub_dir.is_dir():
                candidates.append(str(sub_dir))

    # Inspect .py files for bare top-level sibling imports and ai/* module imports.
    py_candidates = (
        list(task.get('files') or ())
        + list(task.get('read_only_files') or ())
        + list(task.get('test_argv') or ())
    )
    for raw_path in py_candidates:
        s = str(raw_path)
        if not s.endswith('.py'):
            continue
        p = Path(s)
        if not p.is_absolute() and base is not None:
            p = base / p
        if not p.is_file():
            continue
        mods = _top_level_imported_modules(p)
        if not mods:
            continue
        parent = p.parent
        if any(_has_module_in_dir(parent, mod) for mod in mods):
            candidates.append(str(parent))
        if base is not None:
            for sub in ('src', *ai_subdirs):
                sub_dir = base / sub
                if sub_dir.is_dir() and any(_has_module_in_dir(sub_dir, mod) for mod in mods):
                    candidates.append(str(sub_dir))

    merged: list[str] = []
    seen: set[str] = set()
    for entry in [*caller_entries, *candidates]:
        if not entry:
            continue
        norm = os.path.normpath(os.path.abspath(entry))
        if norm not in seen:
            seen.add(norm)
            merged.append(entry)
    return {'PYTHONPATH': os.pathsep.join(merged)} if merged else {}


def _usage_has_required_positional(help_output: str) -> bool:
    """Return True if an argparse-style usage synopsis requires a positional arg."""
    lines = help_output.splitlines()
    usage_lines: list[str] = []
    in_usage = False
    for line in lines:
        if not in_usage:
            m = re.match(r'^\s*usage:\s*(.*)$', line, re.IGNORECASE)
            if m:
                in_usage = True
                usage_lines.append(m.group(1))
        else:
            if not line.strip() or not line[0].isspace():
                break
            usage_lines.append(line.strip())
    if not usage_lines:
        return False
    usage_text = ' '.join(usage_lines)
    while True:
        stripped = re.sub(r'\[[^\[\]]*\]', ' ', usage_text)
        if stripped == usage_text:
            break
        usage_text = stripped
    tokens = usage_text.split()
    if len(tokens) <= 1:
        return False
    skip_next = False
    for tok in tokens[1:]:
        if skip_next:
            skip_next = False
            continue
        if tok.startswith('-'):
            if '=' not in tok:
                skip_next = True
            continue
        if tok in ('options', '...'):
            continue
        return True
    return False


def preflight_test_argv(task: dict, run_dir: Path) -> None:
    """Validate pytest targets/collection and non-editable local .py test scripts before calling the LLM."""
    del run_dir
    argv = list(task.get('test_argv') or ())
    if not argv:
        return
    _, cmd = _extract_leading_env_pythonpath(argv)
    if not cmd:
        return

    directory = str(task.get('directory') or '')
    editable_real = {
        os.path.realpath(str(Path(directory) / f if directory and not Path(f).is_absolute() else f))
        for f in (task.get('files') or ())
    }

    pytest_rest: list[str] | None = None
    if Path(cmd[0]).name.startswith('pytest'):
        pytest_rest = cmd[1:]
    else:
        idx = next((i for i in range(len(cmd) - 1) if cmd[i:i + 2] == ['-m', 'pytest']), None)
        if idx is not None:
            pytest_rest = cmd[idx + 2:]

    if pytest_rest is not None:
        positional_targets: list[str] = []
        has_k_filter = False
        skip_next = False
        after_double_dash = False
        for i, arg in enumerate(pytest_rest):
            if skip_next:
                skip_next = False
                continue
            if not after_double_dash and arg == '--':
                after_double_dash = True
                continue
            if not after_double_dash and arg.startswith('-'):
                name = arg.split('=', 1)[0]
                if name == '-k' or arg.startswith('-k'):
                    has_k_filter = True
                if name in _PYTEST_VALUE_FLAGS and '=' not in arg:
                    nxt = pytest_rest[i + 1] if i + 1 < len(pytest_rest) else None
                    if nxt is not None and not nxt.startswith('-'):
                        skip_next = True
                continue
            target_str = arg.split('::', 1)[0]
            if target_str:
                positional_targets.append(target_str)

        target_reals: list[str] = []
        for target_str in positional_targets:
            target_path = Path(target_str)
            if not target_path.is_absolute() and directory:
                target_path = Path(directory) / target_path
            target_real = os.path.realpath(str(target_path))
            target_reals.append(target_real)
            if not target_path.exists() and target_real not in editable_real:
                raise GuardError('missing_test_target')

        has_missing_editable_test = False
        for f in (task.get('files') or ()):
            fp = Path(directory) / f if directory and not Path(f).is_absolute() else Path(f)
            if ((fp.name.startswith('test_') and fp.name.endswith('.py'))
                    or fp.name.endswith('_test.py')) and not fp.exists():
                has_missing_editable_test = True
                break

        if (directory
                and (positional_targets or has_k_filter)
                and not any(r in editable_real for r in target_reals)
                and not has_missing_editable_test):
            env = worker_env(directory, env_overrides=build_test_env_overrides(task))
            try:
                proc = subprocess.run(
                    [*cmd, '--collect-only', '-q'],
                    cwd=directory,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
            except (OSError, subprocess.TimeoutExpired):
                return
            output = f'{proc.stdout or ""}\n{proc.stderr or ""}'
            if proc.returncode == 5 or (
                proc.returncode == 4
                and 'ImportError' not in output
                and 'SyntaxError' not in output
            ):
                raise GuardError('empty_pytest_collection')
        return

    python_bin: str | None = None
    script_token: str | None = None
    flag_tokens: list[str] = []
    if Path(cmd[0]).name.startswith('python'):
        for i in range(1, len(cmd)):
            tok = cmd[i]
            if tok.startswith('-'):
                if tok in ('-m', '-c'):
                    return
                continue
            if tok.endswith('.py'):
                python_bin = cmd[0]
                script_token = tok
                flag_tokens = cmd[i + 1:]
            break
    elif cmd[0].endswith('.py'):
        python_bin = sys.executable
        script_token = cmd[0]
        flag_tokens = cmd[1:]

    if not python_bin or not script_token:
        return

    script_path = Path(script_token)
    if not script_path.is_absolute() and directory:
        script_path = Path(directory) / script_path
    if not script_path.is_file():
        return

    if os.path.realpath(str(script_path)) in editable_real:
        return

    flags: list[str] = []
    positional_args: list[str] = []
    after_double_dash = False
    for tok in flag_tokens:
        if not after_double_dash and tok == '--':
            after_double_dash = True
            continue
        if not after_double_dash and tok.startswith('-'):
            if tok == '--help':
                continue
            m = re.match(r'^(--[a-zA-Z0-9_-]+)(?:=.*)?$', tok)
            if m:
                flags.append(m.group(1))
        else:
            positional_args.append(tok)
    if not flags and positional_args:
        return

    env = worker_env(directory, env_overrides=build_test_env_overrides(task)) if directory else None
    try:
        proc = subprocess.run(
            [python_bin, str(script_path), '--help'],
            cwd=directory or None,
            env=env,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return

    if proc.returncode != 0:
        return
    help_output = f'{proc.stdout or ""}\n{proc.stderr or ""}'
    if 'usage:' not in help_output.lower():
        return

    for flag in flags:
        if not re.search(r'(?<![a-zA-Z0-9_-])' + re.escape(flag) + r'(?![a-zA-Z0-9_-])', help_output):
            raise GuardError('invalid_test_argv_flags')
    if not positional_args and _usage_has_required_positional(help_output):
        raise GuardError('missing_test_script_args')

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


_FRAME_RE = re.compile(r'^([^\s:][^:]*\.py):(\d+):', re.MULTILINE)
_TB_FRAME_RE = re.compile(r'File "([^"]+\.py)", line (\d+)')


def synthesize_heal_task(directory: str, test_argv: list, test_log: str, *,
                         task_id: str | None = None, edit_tests: bool = False,
                         model: str = DEFAULT_MODEL, worktree: bool = False) -> dict | None:
    """H-68: synthesize a guarded heal task from a failing pytest log.

    Returns a validated task dict, or None when the log shows no failures or
    no in-repo Python source files can be identified as heal targets.
    """
    failure_markers = ('FAILED ', 'ERROR collecting', 'ERROR ', 'Traceback (most recent call last)')
    has_failures = (
        'FAILED ' in test_log
        or 'ERROR collecting' in test_log
        or 'ERROR ' in test_log
        or ('Traceback (most recent call last)' in test_log
            and re.search(r'^(FAILED|ERROR)', test_log, re.MULTILINE))
        or re.search(r'^=+ FAILURES =+', test_log, re.MULTILINE)
        or re.search(r'^=+ ERRORS =+', test_log, re.MULTILINE)
        or re.search(r'=+ \d+ failed', test_log)
    )
    if not has_failures:
        return None
    base_dir = Path(directory).resolve()

    nodeids: list[str] = []
    for m in re.finditer(r'^FAILED\s+(\S+)', test_log, re.MULTILINE):
        if m.group(1) not in nodeids:
            nodeids.append(m.group(1))
    for m in re.finditer(r'^(?:ERROR collecting\s+(\S+)|ERROR\s+(\S+))', test_log, re.MULTILINE):
        nid = m.group(1) or m.group(2)
        if nid and nid not in nodeids:
            nodeids.append(nid)

    def _existing_repo_file(ref: str):
        if not ref or not ref.endswith('.py'):
            return None
        p = Path(ref)
        try:
            if not p.is_absolute():
                p = base_dir / ref
            p = p.resolve()
        except OSError:
            return None
        parts = set(p.parts)
        if '.venv' in parts or 'site-packages' in parts:
            return None
        if not (p.is_file() and p.is_relative_to(base_dir)):
            return None
        return p

    frame_paths: list[Path] = []
    seen_paths: set[str] = set()

    def _add_frame(p):
        if p is None:
            return
        key = str(p)
        if key not in seen_paths:
            seen_paths.add(key)
            frame_paths.append(p)

    for m in _FRAME_RE.finditer(test_log):
        _add_frame(_existing_repo_file(m.group(1)))
    for m in _TB_FRAME_RE.finditer(test_log):
        _add_frame(_existing_repo_file(m.group(1)))
    for nid in nodeids:
        path_part = nid.split('::', 1)[0]
        _add_frame(_existing_repo_file(path_part))

    if not frame_paths:
        return None
    # Deepest frames first: reverse extraction order, dedupe preserving first-seen order.
    ordered = list(reversed(frame_paths))

    def _is_test_file(p: Path) -> bool:
        name = p.name
        return name.startswith('test_') or name.endswith('_test.py') or name == 'conftest.py'

    test_files = [str(p) for p in ordered if _is_test_file(p)]
    impl_files = [str(p) for p in ordered if not _is_test_file(p)]

    if not edit_tests:
        if impl_files:
            files = impl_files[:3]
            read_only_files = (test_files + impl_files[3:])[:5]
        else:
            files = test_files[:3]
            read_only_files = test_files[3:8]
    else:
        candidates = ordered
        files = [str(p) for p in candidates[:3]]
        read_only_files = [str(p) for p in candidates[3:8]]
    if not files:
        return None

    targeted = build_targeted_pytest_argv(test_argv, test_log) or list(test_argv)

    # Compact error excerpt: E-lines plus the tail of the FAILURES/ERRORS section.
    error_lines = [line for line in test_log.splitlines() if line.startswith('E   ')]
    section_match = re.search(r'^=+ (?:FAILURES|ERRORS) =+\s*$', test_log, re.MULTILINE)
    if section_match:
        section = test_log[section_match.end():]
        section = re.split(r'^=+ .* =+\s*$', section, maxsplit=1, flags=re.MULTILINE)[0]
        tail = [line for line in section.splitlines() if line.strip()][-25:]
    else:
        tail = [line for line in test_log.splitlines() if line.strip()][-25:]
    excerpt_lines = error_lines + [''] + tail if error_lines else tail

    nodeid_lines = [f'- {nid}' for nid in nodeids] or ['- (no explicit nodeids captured)']
    prompt_lines = [
        'Heal the failing tests in this repository.',
        '',
        'Failed nodeids:',
        *nodeid_lines,
        '',
        'Error excerpt:',
        *(excerpt_lines),
        '',
        f'Editable files (fix the root cause here with a minimal patch): {files}',
        f'Read-only reference files (do NOT edit): {read_only_files}',
        '',
        'Fix the root cause of the failures with a minimal patch.',
    ]
    prompt = '\n'.join(prompt_lines)

    tid = task_id or f'heal-{uuid.uuid4().hex[:8]}'
    raw = {'task_id': tid, 'directory': str(base_dir), 'files': files,
           'read_only_files': read_only_files, 'test_argv': targeted,
           'prompt': prompt, 'model': model, 'worktree': worktree}
    return validate_task(raw)


def heal_directory(directory, test_argv, *, dry_run=False, edit_tests=False,
                   model=DEFAULT_MODEL, worktree=False, runner=None) -> dict:
    """H-68: run tests in a directory and, on failure, synthesize and run a heal task."""
    if runner is None:
        runner = execute
    resolved_argv = resolve_pytest_test_argv(test_argv, directory)
    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / 'initial_test.log'
        rc = execute(resolved_argv, directory, 120, log_path)
        test_log = log_path.read_text(encoding='utf-8', errors='replace')
    if rc == 0:
        return {'status': 'already_passing', 'returncode': 0}
    task = synthesize_heal_task(directory, test_argv, test_log,
                               edit_tests=edit_tests, model=model, worktree=worktree)
    if task is None:
        return {'status': 'no_heal_target', 'returncode': rc, 'initial_test_output': test_log}
    if dry_run:
        return {'status': 'task_synthesized', 'task': task, 'initial_test_output': test_log}
    if runner is execute:
        with tempfile.TemporaryDirectory() as td:
            task_path = Path(td) / 'task.json'
            task_path.write_text(json.dumps(task), encoding='utf-8')
            res = run_task(task_path, _quiet=True, _return_result=True)
    else:
        res = runner(task)
    return dict(res, synthesized_task=task)


_JS_TS_SUFFIXES = ('.mjs', '.js', '.cjs', '.ts', '.tsx')
_DBT_SKIP_DIRS = frozenset({'.git', 'target', 'node_modules', '.venv', '__pycache__', 'dbt_packages'})
_SAFE_ENV_NAMES = frozenset({'.env.example', '.env.sample', '.env.template'})
_JS_IMPORT_FROM_RE = re.compile(r'''(?:import|export)\b[^;'"]*?\bfrom\s*['"](\.\.?/[^'"]+)['"]''')
_JS_DYNAMIC_IMPORT_RE = re.compile(r'''\b(?:import|require)\s*\(\s*['"](\.\.?/[^'"]+)['"]\s*\)''')
_JS_EXPORT_DECL_RE = re.compile(
    r'''\bexport\s+(?:default\s+)?(?:async\s+)?(?:function|class|const|let|var|type|interface)\s+([A-Za-z_$][A-Za-z0-9_$]*)'''
)
_JS_EXPORT_BRACE_RE = re.compile(r'''\bexport\s*\{([^}]+)\}''')
_DBT_REF_RE = re.compile(r'''\{\{\s*ref\(\s*['"]([a-zA-Z0-9_]+)['"]\s*\)\s*\}\}''')


def _is_sensitive_companion_path(path: Path) -> bool:
    name = path.name.lower()
    if name in _SAFE_ENV_NAMES:
        return False
    if name == '.env' or name.startswith('.env.'):
        return True
    return (
        fnmatch.fnmatch(name, '*secret*')
        or fnmatch.fnmatch(name, '*credentials*')
        or fnmatch.fnmatch(name, '*key*')
    )


def _build_dbt_models_index(
    dir_path: Path | None, seed_files: list[Path]
) -> tuple[dict[str, list[Path]], dict[Path, list[Path]]]:
    models_root: Path | None = None
    if dir_path is not None and (dir_path / 'models').is_dir():
        models_root = (dir_path / 'models').resolve()
    else:
        for sf in seed_files:
            for anc in [sf.parent, *list(sf.parents)]:
                if dir_path is not None and not anc.is_relative_to(dir_path):
                    break
                if anc.name == 'models' and anc.is_dir():
                    models_root = anc.resolve()
                    break
            if models_root is not None:
                break
    sql_by_stem: dict[str, list[Path]] = {}
    yml_by_dir: dict[Path, list[Path]] = {}
    if models_root is None:
        return sql_by_stem, yml_by_dir
    scanned = 0
    for root, dirs, filenames in os.walk(models_root):
        dirs[:] = sorted(d for d in dirs if d not in _DBT_SKIP_DIRS)
        root_path = Path(root)
        for fname in sorted(filenames):
            scanned += 1
            if scanned > 400:
                return sql_by_stem, yml_by_dir
            fpath = (root_path / fname).resolve()
            if _is_sensitive_companion_path(fpath):
                continue
            suf = fpath.suffix.lower()
            if suf == '.sql':
                sql_by_stem.setdefault(fpath.stem, []).append(fpath)
            elif suf in ('.yml', '.yaml'):
                yml_by_dir.setdefault(fpath.parent, []).append(fpath)
    return sql_by_stem, yml_by_dir


def _extract_exported_symbols_for_ranking(path: Path) -> set[str]:
    try:
        text = path.read_text(encoding='utf-8', errors='ignore')
    except OSError:
        return set()
    suf = path.suffix.lower()
    if suf == '.py':
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            return set()
        syms: set[str] = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not node.name.startswith('_'):
                    syms.add(node.name)
            elif isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and not tgt.id.startswith('_'):
                        syms.add(tgt.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if not node.target.id.startswith('_'):
                    syms.add(node.target.id)
        return syms
    if suf in _JS_TS_SUFFIXES:
        syms = set(_JS_EXPORT_DECL_RE.findall(text))
        for grp in _JS_EXPORT_BRACE_RE.findall(text):
            for part in grp.split(','):
                name = part.strip().split(' as ')[-1].strip()
                if re.fullmatch(r'[A-Za-z_$][A-Za-z0-9_$]*', name):
                    syms.add(name)
        return syms
    if suf == '.sql':
        return {path.stem}
    return set()


def _local_dependency_edges(
    src: Path, dir_path: Path | None, dbt_index_getter
) -> list[Path]:
    if not src.is_file():
        return []
    suf = src.suffix.lower()
    edges: list[Path] = []
    seen_edges: set[Path] = set()

    def _add_if_valid(cand: Path) -> None:
        try:
            resolved = cand.resolve()
        except OSError:
            return
        if resolved == src or resolved in seen_edges or not resolved.is_file():
            return
        if dir_path is not None and not resolved.is_relative_to(dir_path):
            return
        if _is_sensitive_companion_path(resolved):
            return
        seen_edges.add(resolved)
        edges.append(resolved)

    if suf == '.py':
        try:
            tree = ast.parse(src.read_text(encoding='utf-8', errors='ignore'))
        except (SyntaxError, OSError, ValueError):
            return []
        roots = [src.parent]
        if src.parent.parent != src.parent:
            roots.append(src.parent.parent)
        if dir_path is not None:
            roots.extend([dir_path, dir_path / 'src'])
        mod_candidates: list[Path | str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    base = src.parent
                    for _ in range(node.level - 1):
                        base = base.parent
                    if node.module:
                        base = base / node.module.replace('.', '/')
                    mod_candidates.append(base)
                    for alias in node.names:
                        if alias.name != '*':
                            mod_candidates.append(base / alias.name)
                elif node.module:
                    rel = node.module.replace('.', '/')
                    mod_candidates.append(rel)
                    for alias in node.names:
                        if alias.name != '*':
                            mod_candidates.append(rel + '/' + alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    mod_candidates.append(alias.name.replace('.', '/'))
        for cand in mod_candidates:
            cand_paths = [root / cand for root in roots] if isinstance(cand, str) else [cand]
            for cand_path in cand_paths:
                py_cand = cand_path if cand_path.suffix == '.py' else cand_path.with_suffix('.py')
                _add_if_valid(py_cand)
                _add_if_valid(cand_path / '__init__.py')
    elif suf in _JS_TS_SUFFIXES:
        try:
            text = src.read_text(encoding='utf-8', errors='ignore')
        except OSError:
            return []
        specs = _JS_IMPORT_FROM_RE.findall(text) + _JS_DYNAMIC_IMPORT_RE.findall(text)
        for spec in specs:
            raw_target = src.parent / spec
            candidates = [raw_target]
            if raw_target.suffix == '':
                candidates.extend(Path(str(raw_target) + ext) for ext in _JS_TS_SUFFIXES)
            candidates.extend(raw_target / ('index' + ext) for ext in _JS_TS_SUFFIXES)
            for cand in candidates:
                if cand.is_file():
                    _add_if_valid(cand)
                    break
    elif suf in ('.sql', '.yml', '.yaml'):
        try:
            text = src.read_text(encoding='utf-8', errors='ignore')
        except OSError:
            return []
        refs = _DBT_REF_RE.findall(text)
        if refs:
            sql_by_stem, yml_by_dir = dbt_index_getter()
            for model_name in refs:
                for sql_path in sql_by_stem.get(model_name, ()):
                    _add_if_valid(sql_path)
                    for yml_path in yml_by_dir.get(sql_path.parent, ()):
                        _add_if_valid(yml_path)
    return edges


def infer_read_only_companion_files(files, *, directory='', existing_read_only=(), worktree=False, max_files=5, prompt=''):
    """H-62/H-79/H-95: infer companion files via test/conftest heuristics and a 2-hop polyglot dependency graph."""
    editable_paths = [Path(f).resolve() for f in files]
    editable_set = {str(p) for p in editable_paths}
    result = [str(Path(f).resolve()) for f in existing_read_only]
    seen = set(editable_set) | set(result)
    dir_path = Path(directory).resolve() if directory else None
    for p in editable_paths:
        if len(result) >= max_files:
            break
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
            if _is_sensitive_companion_path(resolved):
                continue
            seen.add(str(resolved))
            result.append(str(resolved))
    # H-79: add existing conftest.py ancestors for test-file targets.
    if len(result) < max_files:
        for p in editable_paths:
            if len(result) >= max_files:
                break
            if not (p.name.startswith('test_') or p.name.endswith('_test.py')
                    or 'tests' in p.parts):
                continue
            visited = 0
            for anc in [p.parent, *list(p.parent.parents)]:
                if dir_path is not None:
                    if not anc.is_relative_to(dir_path):
                        break
                elif visited >= 2:
                    break
                visited += 1
                cand = anc / 'conftest.py'
                try:
                    resolved = cand.resolve()
                except OSError:
                    continue
                if str(resolved) in seen or not resolved.is_file():
                    continue
                if dir_path is not None and not resolved.is_relative_to(dir_path):
                    continue
                if _is_sensitive_companion_path(resolved):
                    continue
                seen.add(str(resolved))
                result.append(str(resolved))
                if len(result) >= max_files:
                    break
    # H-95: 2-hop polyglot dependency graph ranked by hop weight, in-degree, and prompt overlap.
    if len(result) < max_files:
        dbt_cache: list[tuple[dict[str, list[Path]], dict[Path, list[Path]]]] = []

        def _get_dbt_index():
            if not dbt_cache:
                dbt_cache.append(_build_dbt_models_index(dir_path, editable_paths))
            return dbt_cache[0]

        hop1: set[Path] = set()
        incoming: dict[Path, set[Path]] = {}
        for src in editable_paths:
            for tgt in _local_dependency_edges(src, dir_path, _get_dbt_index):
                if str(tgt) not in editable_set:
                    hop1.add(tgt)
                    incoming.setdefault(tgt, set()).add(src)
        hop2: set[Path] = set()
        for h1 in sorted(hop1):
            for tgt in _local_dependency_edges(h1, dir_path, _get_dbt_index):
                if str(tgt) not in editable_set:
                    if tgt not in hop1:
                        hop2.add(tgt)
                    incoming.setdefault(tgt, set()).add(h1)

        prompt_tokens = set(re.findall(r'[A-Za-z_][A-Za-z0-9_]*', prompt or ''))
        prompt_lower = {tok.lower() for tok in prompt_tokens}
        scored: list[tuple[float, int, str, bool]] = []
        for cand in (hop1 | hop2):
            cand_str = str(cand)
            if cand_str in seen:
                continue
            hop_num = 1 if cand in hop1 else 2
            hop_weight = 1.0 if hop_num == 1 else 0.45
            in_deg = len(incoming.get(cand, ()))
            prompt_match = False
            if prompt_tokens:
                if (
                    cand.stem in prompt_tokens
                    or (len(cand.stem) >= 3 and cand.stem.lower() in prompt_lower)
                    or bool(_extract_exported_symbols_for_ranking(cand) & prompt_tokens)
                ):
                    prompt_match = True
            score = hop_weight + 0.35 * in_deg + (0.5 if prompt_match else 0.0)
            scored.append((score, hop_num, cand_str, prompt_match))

        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        has_2hop_prompt = any(hop_num == 2 and pmatch for _, hop_num, _, pmatch in scored)
        effective_max = 6 if (max_files == 5 and has_2hop_prompt) else max_files
        for _, _, cand_str, _ in scored:
            if len(result) >= effective_max:
                break
            seen.add(cand_str)
            result.append(cand_str)
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
    t['explicit_read_only_files'] = list(t['read_only_files'])
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
            worktree=t['worktree'], max_files=5,
            prompt=t['prompt'] if isinstance(t.get('prompt'), str) else '')
    t.setdefault('include_repo_memory', True)
    if not isinstance(t['include_repo_memory'], bool):
        raise GuardError('invalid_include_repo_memory_flag')
    t.setdefault('include_symbol_context', True)
    if not isinstance(t['include_symbol_context'], bool):
        raise GuardError('invalid_include_symbol_context_flag')
    t.setdefault('auto_micro_repair', True)
    if not isinstance(t['auto_micro_repair'], bool):
        raise GuardError('invalid_auto_micro_repair_flag')
    t.setdefault('os_sandbox', os.environ.get('HARNESS_OS_SANDBOX', 'off'))
    if t['os_sandbox'] not in ('off', 'test_only', 'strict'):
        raise GuardError('invalid_os_sandbox')
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

def render_guard_plugin(allowed_files=(), read_only_files=()) -> str:
    """Render the in-loop TypeScript guard plugin injected into OpenCode runs."""
    allowed_json = json.dumps([str(a) for a in allowed_files])
    read_only_json = json.dumps([str(r) for r in read_only_files])
    harness_root_json = json.dumps(str(Path(__file__).resolve().parent.parent))
    body = r'''// Harness in-loop guard plugin (auto-generated; do not edit by hand).
// Provides immediate syntax checks after write-like tools, targeted
// recovery hints after failed edit calls, and transactional H-94 tools.
import { realpathSync } from "node:fs";
import * as nodePath from "node:path";

function resolveCanonical(p: string): string {
  try {
    return realpathSync(p);
  } catch {
    return nodePath.resolve(p);
  }
}

function isAllowedTarget(filePath: string): { ok: boolean; resolved: string; error?: string } {
  const resolved = resolveCanonical(filePath);
  const allowed = (__HARNESS_ALLOWED_JSON__ as string[]).map(resolveCanonical);
  const readOnly = (__HARNESS_READ_ONLY_JSON__ as string[]).map(resolveCanonical);
  if (readOnly.includes(resolved) && !allowed.includes(resolved)) {
    return { ok: false, resolved, error: `Permission denied: ${filePath} is read-only.` };
  }
  if (allowed.length > 0 && !allowed.includes(resolved)) {
    return { ok: false, resolved, error: `Permission denied: ${filePath} is not in allowed_files.` };
  }
  return { ok: true, resolved };
}

async function runHarnessEdit(
  mode: "multiedit" | "patch",
  filePath: string,
  payload: unknown,
): Promise<string> {
  const gate = isAllowedTarget(filePath);
  if (!gate.ok) return gate.error || "Permission denied.";
  const resolved = gate.resolved;
  let orig = "";
  try {
    orig = await Bun.file(resolved).text();
  } catch (err: any) {
    return `Failed to read ${filePath}: ${String(err?.message || err)}`;
  }

  const pyCode = [
    "import json, sys",
    "from harness.guards import apply_transactional_multiedit, apply_context_patch",
    "data = json.load(sys.stdin)",
    "fn = apply_transactional_multiedit if data['mode'] == 'multiedit' else apply_context_patch",
    "ok, updated, msg = fn(data['content'], data['payload'])",
    "print(json.dumps({'ok': ok, 'updated': updated, 'msg': msg}))",
  ].join("\n");

  const proc = Bun.spawnSync(["python3", "-c", pyCode], {
    stdin: new TextEncoder().encode(JSON.stringify({ mode, content: orig, payload })),
    env: { ...process.env, PYTHONPATH: __HARNESS_REPO_ROOT__ },
  });
  if (proc.exitCode !== 0) {
    const stderr = new TextDecoder().decode(proc.stderr || new Uint8Array());
    return `[HARNESS ${mode.toUpperCase()} ERROR] ${stderr || "Python helper failed"}`;
  }
  let res: { ok: boolean; updated: string; msg: string };
  try {
    res = JSON.parse(new TextDecoder().decode(proc.stdout || new Uint8Array()));
  } catch {
    return `[HARNESS ${mode.toUpperCase()} ERROR] Invalid response from Python helper`;
  }
  if (!res.ok) {
    return `[HARNESS ${mode.toUpperCase()} REJECTED] ${res.msg}`;
  }

  await Bun.write(resolved, res.updated);
  let syntaxOk = true;
  let syntaxErr = "";
  if (resolved.endsWith(".py")) {
    const chk = Bun.spawnSync(["python3", "-m", "py_compile", resolved]);
    if (chk.exitCode !== 0) {
      syntaxOk = false;
      syntaxErr = new TextDecoder().decode(chk.stderr || new Uint8Array());
    }
  } else if (resolved.endsWith(".json")) {
    try {
      JSON.parse(res.updated);
    } catch (err: any) {
      syntaxOk = false;
      syntaxErr = String(err?.message || err);
    }
  } else if (resolved.endsWith(".js") || resolved.endsWith(".mjs") || resolved.endsWith(".cjs")) {
    const chk = Bun.spawnSync(["node", "--check", resolved]);
    if (chk.exitCode !== 0) {
      syntaxOk = false;
      syntaxErr = new TextDecoder().decode(chk.stderr || new Uint8Array());
    }
  }

  if (!syntaxOk) {
    await Bun.write(resolved, orig);
    return `[HARNESS ${mode.toUpperCase()} ROLLED BACK] Post-write syntax check failed; original file restored.\n${syntaxErr}`;
  }
  return res.msg;
}

export default async function harnessGuardPlugin() {
  let toolDefs: Record<string, any> | undefined;
  try {
    const mod: any = await import("@opencode-ai/plugin");
    const toolFn = mod?.tool;
    if (typeof toolFn === "function" && toolFn.schema) {
      const z = toolFn.schema;
      toolDefs = {
        harness_multiedit: toolFn({
          description: "Atomically apply multiple sequential {oldString, newString} edits to a single allowed file with rollback on mismatch or syntax error.",
          args: {
            filePath: z.string().describe("Path to the target file"),
            edits: z.array(z.object({
              oldString: z.string().describe("Exact or indentation-aligned text to replace"),
              newString: z.string().describe("Replacement text"),
            })).describe("Sequential list of edits to apply atomically"),
          },
          async execute(args: any) {
            return await runHarnessEdit("multiedit", String(args.filePath || ""), args.edits || []);
          },
        }),
        harness_apply_patch: toolFn({
          description: "Atomically apply a context-only V4A @@ patch (with optional @@ <anchor> disambiguation) to a single allowed file with rollback on mismatch or syntax error.",
          args: {
            filePath: z.string().describe("Path to the target file"),
            patch: z.string().describe("Context-only V4A @@ patch text"),
          },
          async execute(args: any) {
            return await runHarnessEdit("patch", String(args.filePath || ""), String(args.patch || ""));
          },
        }),
      };
    }
  } catch {
    // @opencode-ai/plugin unavailable in standalone mock/test environments.
  }

  return {
    ...(toolDefs ? { tool: toolDefs } : {}),
    "tool.execute.after": async (input: any, output: any) => {
      try {
        const tool = String(input?.tool || "");
        const filePath = String(input?.args?.filePath || input?.args?.file_path || input?.args?.path || "");

        // (a) Immediate polyglot syntax & semantic checks for write-like tools.
        if (["edit", "write", "multiedit", "patch", "apply_patch", "harness_multiedit", "harness_apply_patch"].includes(tool)) {
          if (filePath.endsWith(".py")) {
            const proc = Bun.spawnSync(["python3", "-m", "py_compile", filePath]);
            if (proc.exitCode !== 0) {
              const stderr = new TextDecoder().decode(proc.stderr || new Uint8Array());
              output.output += `\n[HARNESS IN-LOOP SYNTAX CHECK FAILED]\n${stderr}\nFix this syntax error immediately before finishing.`;
            } else {
              // H-93: fast-path ruff check when available.
              try {
                if (Bun.which("ruff")) {
                  const ruffProc = Bun.spawnSync([
                    "ruff",
                    "check",
                    "--select",
                    "F,E9",
                    "--isolated",
                    "--output-format",
                    "concise",
                    filePath,
                  ]);
                  if (ruffProc.exitCode !== 0) {
                    const ruffOut = new TextDecoder().decode(ruffProc.stdout || new Uint8Array()).trim();
                    if (ruffOut) {
                      output.output += `\n[HARNESS IN-LOOP SEMANTIC LINT]\n${ruffOut}`;
                    }
                  }
                }
              } catch {
                // Lint must never break the tool result.
              }
              // H-86 + H-93: pure-stdlib undefined-name, missing self.<method>, and call-arity lint.
              try {
                const lintScript = [
                  'import ast,sys,builtins',
                  'tree=ast.parse(open(sys.argv[1]).read())',
                  'skip_undef=any(isinstance(n,ast.ImportFrom) and any(a.name=="*" for a in n.names) for n in ast.walk(tree))',
                  'if not skip_undef:',
                  '    names=set(dir(builtins))|{"__file__","__name__","__doc__","__package__","__all__","__spec__","__loader__","__cached__","__annotations__"}',
                  '    for n in ast.walk(tree):',
                  '        if isinstance(n,ast.Name) and isinstance(n.ctx,(ast.Store,ast.Del)): names.add(n.id)',
                  '        elif isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)): names.add(n.name)',
                  '        elif isinstance(n,(ast.Import,ast.ImportFrom)):',
                  '            for a in n.names: names.add(a.asname or a.name.split(".")[0])',
                  '        elif isinstance(n,ast.ExceptHandler) and n.name: names.add(n.name)',
                  '        elif isinstance(n,ast.arg): names.add(n.arg)',
                  '        elif isinstance(n,(ast.Global,ast.Nonlocal)): names.update(n.names)',
                  '        elif isinstance(n,ast.MatchAs) and n.name: names.add(n.name)',
                  '        elif isinstance(n,ast.MatchStar) and n.name: names.add(n.name)',
                  '        elif isinstance(n,ast.MatchMapping) and n.rest: names.add(n.rest)',
                  '    undef=[f"L{n.lineno}:{n.id}" for n in ast.walk(tree) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load) and n.id not in names]',
                  '    if undef: print("UNDEF:" + ",".join(sorted(set(undef))))',
                  'sem=[]',
                  'top_fn={}',
                  'for s in tree.body:',
                  '    if isinstance(s,(ast.FunctionDef,ast.AsyncFunctionDef)): top_fn[s.name]=None if s.name in top_fn else s',
                  'def chk_call(c,cname,fd,is_m):',
                  '    if fd.decorator_list or fd.args.vararg or fd.args.kwarg: return',
                  '    if any(isinstance(a,ast.Starred) for a in c.args) or any(k.arg is None for k in c.keywords): return',
                  '    po=list(fd.args.posonlyargs); rg=list(fd.args.args)',
                  '    if is_m:',
                  '        if po: po=po[1:]',
                  '        elif rg: rg=rg[1:]',
                  '        else: return',
                  '    ap=po+rg; mx=len(ap); mr=max(0,mx-len(fd.args.defaults))',
                  '    vkw={a.arg for a in rg}|{a.arg for a in fd.args.kwonlyargs}',
                  '    ckw={k.arg for k in c.keywords if k.arg is not None}',
                  '    for k in c.keywords:',
                  '        if k.arg is not None and k.arg not in vkw: sem.append(f"L{c.lineno}:unexpected kwarg {k.arg} in {cname}")',
                  '    if len(c.args)>mx: sem.append(f"L{c.lineno}:too many args in {cname}")',
                  '    else:',
                  '        pn={a.arg for a in po}',
                  '        miss=[p.arg for i,p in enumerate(ap[:mr]) if i>=len(c.args) and (p.arg in pn or p.arg not in ckw)]',
                  '        if miss: sem.append(f"L{c.lineno}:missing arg(s) {\',\'.join(miss)} in {cname}")',
                  'for cnode in ast.walk(tree):',
                  '    if not isinstance(cnode,ast.ClassDef): continue',
                  '    ms={}; ca=set(); sa=set()',
                  '    for s in cnode.body:',
                  '        if isinstance(s,(ast.FunctionDef,ast.AsyncFunctionDef)): ms[s.name]=None if s.name in ms else s',
                  '        elif isinstance(s,ast.Assign): ca.update(t.id for t in s.targets if isinstance(t,ast.Name))',
                  '        elif isinstance(s,ast.AnnAssign) and isinstance(s.target,ast.Name): ca.add(s.target.id)',
                  '    for s in cnode.body:',
                  '        if isinstance(s,(ast.FunctionDef,ast.AsyncFunctionDef)):',
                  '            for sub in ast.walk(s):',
                  '                if isinstance(sub,ast.Assign):',
                  '                    for t in sub.targets:',
                  '                        if isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id=="self": sa.add(t.attr)',
                  '                elif isinstance(sub,ast.AnnAssign) and isinstance(sub.target,ast.Attribute) and isinstance(sub.target.value,ast.Name) and sub.target.value.id=="self": sa.add(sub.target.attr)',
                  '    chk_self=(len(cnode.bases)==0 and "__getattr__" not in ms and "__getattribute__" not in ms)',
                  '    known=set(ms)|ca|sa',
                  '    for s in cnode.body:',
                  '        if not isinstance(s,(ast.FunctionDef,ast.AsyncFunctionDef)): continue',
                  '        ps=s.args.posonlyargs+s.args.args',
                  '        if not ps or ps[0].arg!="self": continue',
                  '        for sub in ast.walk(s):',
                  '            if isinstance(sub,ast.Call) and isinstance(sub.func,ast.Attribute) and isinstance(sub.func.value,ast.Name) and sub.func.value.id=="self":',
                  '                mn=sub.func.attr',
                  '                if mn in ms and ms[mn] is not None: chk_call(sub,f"self.{mn}",ms[mn],True)',
                  '                elif chk_self and mn not in known: sem.append(f"L{sub.lineno}:{cnode.name} has no method {mn}")',
                  'for s in tree.body:',
                  '    for sub in ast.walk(s):',
                  '        if isinstance(sub,ast.Call) and isinstance(sub.func,ast.Name) and sub.func.id in top_fn and top_fn[sub.func.id] is not None:',
                  '            chk_call(sub,sub.func.id,top_fn[sub.func.id],False)',
                  'if sem: print("SEM:" + "; ".join(sorted(set(sem))))',
                ].join("\n");
                const lintProc = Bun.spawnSync(["python3", "-c", lintScript, filePath]);
                const lintOut = new TextDecoder().decode(lintProc.stdout || new Uint8Array()).trim();
                for (const line of lintOut.split("\n")) {
                  if (line.startsWith("UNDEF:")) {
                    output.output += `\n[HARNESS IN-LOOP SEMANTIC LINT] Undefined name(s) detected: ${line.slice(6)} — add the missing import or fix the identifier before finishing.`;
                  } else if (line.startsWith("SEM:")) {
                    output.output += `\n[HARNESS IN-LOOP SEMANTIC LINT] ${line.slice(4)}`;
                  }
                }
              } catch {
                // Lint must never break the tool result.
              }
            }
          } else if (filePath.endsWith(".json")) {
            try {
              JSON.parse(await Bun.file(filePath).text());
            } catch (err: any) {
              const errMsg = String(err?.message || err);
              output.output += `\n[HARNESS IN-LOOP JSON CHECK FAILED]\n${errMsg}\nFix this invalid JSON immediately.`;
            }
          } else if (filePath.endsWith(".mjs") || filePath.endsWith(".js") || filePath.endsWith(".cjs")) {
            try {
              if (Bun.which("node")) {
                const jsProc = Bun.spawnSync(["node", "--check", filePath]);
                if (jsProc.exitCode !== 0 && jsProc.exitCode !== null) {
                  const stderr = new TextDecoder().decode(jsProc.stderr || new Uint8Array());
                  output.output += `\n[HARNESS IN-LOOP SYNTAX CHECK FAILED]\n${stderr}\nFix this syntax error immediately before finishing.`;
                }
              }
            } catch {
              // Gracefully skip if node binary is missing / ENOENT.
            }
          } else if (filePath.endsWith(".ts") || filePath.endsWith(".tsx")) {
            try {
              const tsSrc = await Bun.file(filePath).text();
              new Bun.Transpiler({ loader: filePath.endsWith(".tsx") ? "tsx" : "ts" }).scan(tsSrc);
            } catch (err: any) {
              const errMsg = String(err?.message || err);
              output.output += `\n[HARNESS IN-LOOP SYNTAX CHECK FAILED]\n${errMsg}\nFix this syntax error immediately before finishing.`;
            }
          } else if (filePath.endsWith(".yaml") || filePath.endsWith(".yml") || filePath.endsWith(".sql")) {
            try {
              const txt = await Bun.file(filePath).text();
              const isSql = filePath.endsWith(".sql");
              const issues: string[] = [];
              // Jinja delimiter balance check ({{ }}, {% %}, {# #})
              let state: string | null = null;
              let openLine = 0;
              let quote: string | null = null;
              let lineNo = 1;
              let braceDepth = 0;
              for (let i = 0; i < txt.length; i++) {
                const ch = txt[i];
                if (ch === "\n") { lineNo++; continue; }
                if (state === "{#") {
                  if (txt.startsWith("#}", i)) { state = null; i++; }
                  continue;
                }
                if (state === "{{" || state === "{%") {
                  if (quote) {
                    if (ch === "\\" && i + 1 < txt.length) { i++; continue; }
                    if (ch === quote) quote = null;
                    continue;
                  }
                  if (ch === "'" || ch === '"') { quote = ch; continue; }
                  const closeTok = state === "{{" ? "}}" : "%}";
                  if (txt.startsWith(closeTok, i)) { state = null; i++; continue; }
                  if (txt.startsWith("{{", i) || txt.startsWith("{%", i) || txt.startsWith("{#", i)) {
                    issues.push(`L${openLine}: unclosed Jinja '${state}' delimiter`);
                    state = txt.slice(i, i + 2);
                    openLine = lineNo;
                    quote = null;
                    i++;
                  }
                  continue;
                }
                if (txt.startsWith("{{", i) || txt.startsWith("{%", i) || txt.startsWith("{#", i)) {
                  state = txt.slice(i, i + 2);
                  openLine = lineNo;
                  quote = null;
                  i++;
                  continue;
                }
                if (txt.startsWith("%}", i) || txt.startsWith("#}", i)) {
                  issues.push(`L${lineNo}: unmatched Jinja '${txt.slice(i, i + 2)}' delimiter`);
                  i++;
                  continue;
                }
                if (txt.startsWith("}}", i)) {
                  if (braceDepth >= 2) braceDepth -= 2;
                  else issues.push(`L${lineNo}: unmatched Jinja '}}' delimiter`);
                  i++;
                  continue;
                }
                if (ch === "{") braceDepth++;
                else if (ch === "}" && braceDepth > 0) braceDepth--;
              }
              if (state) issues.push(`L${openLine}: unclosed Jinja '${state}' delimiter`);
              if (!isSql) {
                const bunYaml = (Bun as any).YAML;
                if (bunYaml && typeof bunYaml.parse === "function") {
                  try { bunYaml.parse(txt); } catch (yErr: any) { issues.push(String(yErr?.message || yErr)); }
                }
                const frames: Array<[number, Map<string, number>]> = [];
                let blockIndent: number | null = null;
                const keyRe = /^([A-Za-z0-9_.\-/]+|"(?:[^"\\]|\\.)*"|'(?:[^']|'')*')\s*:(?:\s|$)(.*)$/;
                const blockRe = /^(?:[^:#'"]*:\s*|-\s+)[|>][+-]?\d*\s*(?:#.*)?$/;
                const lines = txt.split("\n");
                for (let li = 0; li < lines.length; li++) {
                  const raw = lines[li];
                  const trimmed = raw.trim();
                  const ln = li + 1;
                  if (blockIndent !== null) {
                    if (!trimmed) continue;
                    const sp = raw.length - raw.replace(/^ +/, "").length;
                    if (raw.startsWith("\t")) { issues.push(`L${ln}: tab indentation is not allowed in YAML`); blockIndent = null; }
                    else if (sp > blockIndent) continue;
                    else blockIndent = null;
                  }
                  if (!trimmed || trimmed.startsWith("#")) continue;
                  if (trimmed === "---" || trimmed === "...") { frames.length = 0; continue; }
                  const leadLen = raw.length - raw.replace(/^[ \t]+/, "").length;
                  if (raw.slice(0, leadLen).includes("\t")) { issues.push(`L${ln}: tab indentation is not allowed in YAML`); continue; }
                  if (trimmed.startsWith("{%") || trimmed.startsWith("{#")) continue;
                  if (trimmed === "-" || trimmed.startsWith("- ")) {
                    while (frames.length > 0 && frames[frames.length - 1][0] >= leadLen) frames.pop();
                    if (trimmed === "-") continue;
                    const rest = trimmed.slice(2).replace(/^ +/, "");
                    const itemIndent = leadLen + (trimmed.length - rest.length);
                    const km = rest.match(keyRe);
                    if (km) {
                      const k = km[1].replace(/^['"]|['"]$/g, "");
                      frames.push([itemIndent, new Map([[k, ln]])]);
                      if (blockRe.test(rest)) blockIndent = leadLen;
                    } else if (blockRe.test(trimmed)) {
                      blockIndent = leadLen;
                    }
                    continue;
                  }
                  const km = trimmed.match(keyRe);
                  if (km) {
                    const k = km[1].replace(/^['"]|['"]$/g, "");
                    while (frames.length > 0 && frames[frames.length - 1][0] > leadLen) frames.pop();
                    if (frames.length > 0 && frames[frames.length - 1][0] === leadLen) {
                      const seen = frames[frames.length - 1][1];
                      if (seen.has(k)) issues.push(`L${ln}: duplicate YAML mapping key '${k}' (first at L${seen.get(k)})`);
                      else seen.set(k, ln);
                    } else {
                      frames.push([leadLen, new Map([[k, ln]])]);
                    }
                    if (blockRe.test(trimmed)) blockIndent = leadLen;
                  }
                }
              }
              if (issues.length > 0) {
                output.output += `\n[HARNESS IN-LOOP YAML/JINJA CHECK FAILED]\n${issues.join("\n")}\nFix these YAML/Jinja issues immediately before finishing.`;
              }
            } catch {
              // Lint must never break the tool result.
            }
          }
        }

        // (b) Targeted context hint after a failed edit call.
        if (tool === "edit") {
          const out = String(output?.output || "");
          const failed = out.includes("Could not find oldString")
            || (out.includes("Found ") && out.includes(" matches"));
          const multiMatch = out.includes("Found ") && out.includes(" matches");
          if (failed) {
            const oldString = String(input?.args?.oldString || input?.args?.old_string || "");
            const needle = oldString.split("\n").map((l: string) => l.trim())
              .find((l: string) => l.length > 0) || "";
            if (needle) {
              try {
                const text = await Bun.file(filePath).text();
                const lines = text.split("\n");
                // H-80: exact multi-match disambiguation via indexOf.
                let exactShown = false;
                let autoApplied = false;
                if (multiMatch && oldString.length > 0) {
                  const spans: Array<[number, number]> = [];
                  let pos = 0;
                  while (spans.length < 4) {
                    const idx = text.indexOf(oldString, pos);
                    if (idx < 0) break;
                    const startLine = text.slice(0, idx).split("\n").length;
                    const endLine = startLine + oldString.split("\n").length - 1;
                    spans.push([startLine, endLine]);
                    pos = idx + oldString.length;
                  }
                  if (spans.length > 0) {
                    const parts: string[] = ["", "[HARNESS EDIT HINT] Exact multi-match locations for oldString (include surrounding context lines to disambiguate):"];
                    let k = 0;
                    for (const [startLine, endLine] of spans) {
                      k++;
                      const ctxStart = Math.max(1, startLine - 3);
                      const ctxEnd = Math.min(lines.length, endLine + 3);
                      parts.push(`--- match #${k} at lines ${startLine}-${endLine} (context ${ctxStart}-${ctxEnd}) ---`);
                      for (let i = ctxStart; i <= ctxEnd; i++) {
                        parts.push(`${i}: ${lines[i - 1]}`);
                      }
                    }
                    output.output += "\n" + parts.join("\n");
                    exactShown = true;
                  }
                }
                // H-85: normalized-indentation auto-apply for failed exact edits.
                if (!exactShown && !multiMatch && oldString.length > 0) {
                  const allowed: string[] = __HARNESS_ALLOWED_JSON__;
                  if (allowed.length === 0 || allowed.includes(filePath)) {
                    try {
                      const rawOldLines = oldString.split("\n");
                      while (rawOldLines.length > 0 && rawOldLines[0].trim().length === 0) rawOldLines.shift();
                      while (rawOldLines.length > 0 && rawOldLines[rawOldLines.length - 1].trim().length === 0) rawOldLines.pop();
                      let baseOldIndent = Infinity;
                      for (const l of rawOldLines) {
                        if (l.trim().length === 0) continue;
                        const ind = l.match(/^[ ]*/)![0].length;
                        if (ind < baseOldIndent) baseOldIndent = ind;
                      }
                      const relOldLines: string[] = rawOldLines.map((l: string) =>
                        l.trim().length === 0 ? "" : l.slice(baseOldIndent).replace(/\s+$/, ""));
                      if (relOldLines.length >= 1) {
                        const matches: Array<{ start: number; deltaIndent: number }> = [];
                        for (let start = 0; start + relOldLines.length <= lines.length; start++) {
                          for (let deltaIndent = 0; deltaIndent <= 64; deltaIndent++) {
                            let ok = true;
                            for (let k = 0; k < relOldLines.length; k++) {
                              const rel = relOldLines[k];
                              const cur = lines[start + k];
                              if (rel === "") {
                                if (cur.trim().length !== 0) { ok = false; break; }
                              } else if (cur.replace(/\s+$/, "") !== " ".repeat(deltaIndent) + rel) {
                                ok = false; break;
                              }
                            }
                            if (ok) { matches.push({ start, deltaIndent }); break; }
                          }
                        }
                        if (matches.length === 1) {
                          const { start, deltaIndent } = matches[0];
                          const shift = deltaIndent - baseOldIndent;
                          const newString = String(input?.args?.newString || input?.args?.new_string || "");
                          const newLines = newString.split("\n").map((l: string) => {
                            if (l.trim().length === 0) return "";
                            if (shift >= 0) return " ".repeat(shift) + l;
                            return l.slice(Math.min(-shift, l.match(/^[ ]*/)![0].length));
                          });
                          const updatedText = lines.slice(0, start)
                            .concat(newLines, lines.slice(start + relOldLines.length)).join("\n");
                          await Bun.write(filePath, updatedText);
                          let valid = false;
                          if (filePath.endsWith(".json")) {
                            try { JSON.parse(updatedText); valid = true; } catch {}
                          } else if (filePath.endsWith(".py")) {
                            const chk = Bun.spawnSync(["python3", "-m", "py_compile", filePath]);
                            valid = chk.exitCode === 0;
                          } else {
                            valid = true;
                          }
                          if (valid) {
                            output.output = `\n[HARNESS AUTO-APPLIED NORMALIZED EDIT] Automatically aligned indentation and applied edit at lines ${start + 1}-${start + relOldLines.length} in ${filePath}. Do NOT re-issue this edit.`;
                            autoApplied = true;
                          } else {
                            await Bun.write(filePath, text);
                          }
                        }
                      }
                    } catch {
                      // Auto-apply must never break the edit-failure hint path.
                    }
                  }
                }
                if (!exactShown && !autoApplied) {
                  const candidates: number[] = [];
                  for (let i = 0; i < lines.length; i++) {
                    const line = lines[i];
                    if (line.trim() === needle || line.includes(needle)) {
                      candidates.push(i + 1);
                      if (candidates.length >= 3) break;
                    }
                  }
                  if (candidates.length > 0) {
                    const parts: string[] = ["", `[HARNESS EDIT HINT] ${needle ? "near-matching lines" : "matches"} for: ${needle}`];
                    for (const n of candidates) {
                      const start = Math.max(1, n - 5);
                      const end = Math.min(lines.length, n + 5);
                      parts.push(`--- lines ${start}-${end} (match at line ${n}) ---`);
                      for (let i = start; i <= end; i++) {
                        parts.push(`${i}: ${lines[i - 1]}`);
                      }
                    }
                    output.output += "\n" + parts.join("\n");
                  } else {
                    output.output += `\n[HARNESS EDIT HINT] No matching or near-matching line found for: ${needle}\nCheck the target file path and exact whitespace/punctuation.`;
                  }
                }
              } catch {
                // File unreadable; leave the failure message untouched.
              }
            }
          }
        }

        // (b2) H-92: chunked-edit recovery hint on JSON parse / invalid-args failures.
        {
          const rawOut = String(output?.output || "");
          if ((rawOut.includes('Invalid arguments for tool "edit"')
              || rawOut.includes('Invalid arguments for tool "write"')
              || rawOut.includes("could not be parsed as JSON")
              || rawOut.includes("stream ended without finish_reason"))
              && !rawOut.includes("[HARNESS CHUNKED-EDIT RECOVERY]")) {
            output.output += "\n[HARNESS CHUNKED-EDIT RECOVERY] Your previous `write`/`edit` tool call failed JSON serialization because the payload was too large or had unescaped quotes/newlines. Do NOT retry a full-file `write` or >80-line `edit`. Split the change into 2–3 small surgical `edit` calls (<= 40 lines each) with minimal unique context.";
          }
        }

        // (b2) H-92: chunked-edit recovery hint on JSON parse / invalid-args failures.
        {
          const rawOut = String(output?.output || "");
          if ((rawOut.includes('Invalid arguments for tool "edit"')
              || rawOut.includes('Invalid arguments for tool "write"')
              || rawOut.includes("could not be parsed as JSON")
              || rawOut.includes("stream ended without finish_reason"))
              && !rawOut.includes("[HARNESS CHUNKED-EDIT RECOVERY]")) {
            output.output += "\n[HARNESS CHUNKED-EDIT RECOVERY] Your previous `write`/`edit` tool call failed JSON serialization because the payload was too large or had unescaped quotes/newlines. Do NOT retry a full-file `write` or >80-line `edit`. Split the change into 2–3 small surgical `edit` calls (<= 40 lines each) with minimal unique context.";
          }
        }

        // (c) H-79: read-guard hint after a denied read.
        if (tool === "read" && String(output?.output || "").includes("Permission denied")) {
          const allowed: string[] = __HARNESS_ALLOWED_JSON__;
          const readOnly: string[] = __HARNESS_READ_ONLY_JSON__;
          if (allowed.length > 0 || readOnly.length > 0) {
            output.output += `\n[HARNESS READ GUARD] Do NOT retry reading denied paths. Editable files: ${allowed.join(", ")}. Read-only files: ${readOnly.join(", ")}.`;
          } else {
            output.output += `\n[HARNESS READ GUARD] Do NOT retry reading denied paths. Use only the allowed files specified in the prompt.`;
          }
        }
      } catch {
        // Never let the guard plugin itself break a tool result.
      }
      return output;
    },
  };
}
'''
    return (
        body.replace('__HARNESS_ALLOWED_JSON__', allowed_json)
        .replace('__HARNESS_READ_ONLY_JSON__', read_only_json)
        .replace('__HARNESS_REPO_ROOT__', harness_root_json)
    )


def sandbox_root_for(run_dir) -> Path:
    """Short per-run isolation root under /tmp (keeps AF_UNIX socket paths < 104 bytes)."""
    return Path('/tmp') / f"hb-{hashlib.sha1(str(run_dir).encode('utf-8')).hexdigest()[:12]}"


def stage_isolation_sandbox(run_dir) -> dict:
    """Create tmp/python-user/npm-global dirs and return env overrides pointing at them."""
    sb = sandbox_root_for(run_dir)
    for sub in ('tmp', 'python-user', 'npm-global'):
        (sb / sub).mkdir(parents=True, exist_ok=True)
    return {'TMPDIR': str(sb / 'tmp'), 'PYTHONUSERBASE': str(sb / 'python-user'),
            'NPM_CONFIG_PREFIX': str(sb / 'npm-global')}


def cleanup_isolation_sandbox(run_dir) -> None:
    shutil.rmtree(sandbox_root_for(run_dir), ignore_errors=True)


DEFAULT_SEATBELT_DENIED_READ_PATHS = (
    '~/.ssh',
    '~/.aws',
    '~/.gnupg',
    '~/.kube',
    '~/.config/gcloud',
)

_SEATBELT_SYSTEM_WRITABLE_PATHS = (
    '/dev/null',
    '/private/dev/null',
    '/dev/tty',
    '/dev/dtracehelper',
    '/private/var/folders',
)


def is_seatbelt_available() -> bool:
    """Return True when macOS /usr/bin/sandbox-exec is available and executable."""
    return sys.platform == 'darwin' and os.access('/usr/bin/sandbox-exec', os.X_OK)


def _sbpl_quote(value: str) -> str:
    """Escape a string for inclusion inside an SBPL double-quoted literal."""
    return value.replace('\\', '\\\\').replace('"', '\\"')


def build_seatbelt_profile(
    *,
    writable_paths=(),
    denied_read_paths=(),
    allow_localhost_only: bool = True,
    allowed_tcp_ports=(),
) -> str:
    """Build a macOS Seatbelt (SBPL) profile for filesystem and network confinement."""
    lines = [
        '(version 1)',
        '(allow default)',
        '(deny file-write*)',
    ]

    canonical_writable: list[str] = []
    seen_writable: set[str] = set()
    for raw_path in writable_paths:
        if not raw_path:
            continue
        canon = os.path.realpath(os.path.expanduser(str(raw_path)))
        if canon not in seen_writable:
            seen_writable.add(canon)
            canonical_writable.append(canon)
            quoted = _sbpl_quote(canon)
            if os.path.isdir(canon):
                lines.append(f'(allow file-write* (subpath "{quoted}"))')
            elif os.path.exists(canon):
                lines.append(f'(allow file-write* (literal "{quoted}"))')
            elif os.path.basename(canon) in ('.pytest_cache', '__pycache__'):
                lines.append(f'(allow file-write* (subpath "{quoted}"))')
            else:
                lines.append(f'(allow file-write* (literal "{quoted}"))')

    denied_candidates = list(DEFAULT_SEATBELT_DENIED_READ_PATHS)
    for raw_path in denied_read_paths:
        if raw_path:
            denied_candidates.append(str(raw_path))

    seen_denied: set[str] = set()
    for raw_path in denied_candidates:
        canon = os.path.realpath(os.path.expanduser(str(raw_path)))
        if canon in seen_denied:
            continue
        if any(canon == w or canon.startswith(w + os.sep) for w in canonical_writable):
            continue
        seen_denied.add(canon)
        lines.append(f'(deny file-read* (subpath "{_sbpl_quote(canon)}"))')

    if allow_localhost_only:
        lines.append('(allow network* (local unix))')
        lines.append('(deny network-outbound (remote ip))')
        ports = sorted({int(p) for p in allowed_tcp_ports if p is not None})
        if ports:
            for port in ports:
                lines.append(f'(allow network-outbound (remote ip "localhost:{port}"))')
        else:
            lines.append('(allow network-outbound (remote ip "localhost:*"))')

    return '\n'.join(lines) + '\n'


def wrap_argv_with_seatbelt(argv: list[str], profile_path: Path) -> list[str]:
    """Wrap argv with /usr/bin/sandbox-exec using the given SBPL profile."""
    return ['/usr/bin/sandbox-exec', '-f', str(profile_path), *argv]


def _stage_test_seatbelt_profile(task: dict, run_dir: Path) -> Path:
    """Stage seatbelt-test.sb in run_dir and return its path."""
    writable = [
        sandbox_root_for(run_dir),
        run_dir,
        *_SEATBELT_SYSTEM_WRITABLE_PATHS,
    ]
    if task.get('worktree'):
        writable.append(task['directory'])
    else:
        writable.extend(task.get('files', ()))
        writable.append(Path(task['directory']) / '.pytest_cache')
        writable.append(Path(task['directory']) / '__pycache__')
        for f in task.get('files', ()):
            writable.append(Path(f).parent / '__pycache__')
    profile = build_seatbelt_profile(writable_paths=writable, allow_localhost_only=True)
    profile_path = Path(run_dir) / 'seatbelt-test.sb'
    private_write(profile_path, profile)
    return profile_path


def _stage_worker_seatbelt_profile(task: dict, run_dir: Path, db, *, proxy=None) -> Path:
    """Stage seatbelt-worker.sb in run_dir and return its path."""
    writable = [
        *task.get('files', ()),
        run_dir,
        sandbox_root_for(run_dir),
        *_SEATBELT_SYSTEM_WRITABLE_PATHS,
    ]
    if task.get('mode') == 'patch' and task.get('patch_file'):
        writable.append(task['patch_file'])
    if db is not None:
        writable.append(Path(db).parent)
    ports: tuple[int, ...] = ()
    if proxy is not None and getattr(proxy, 'base_url', None):
        from urllib.parse import urlparse
        parsed_port = urlparse(proxy.base_url).port
        if parsed_port is not None:
            ports = (parsed_port,)
    profile = build_seatbelt_profile(
        writable_paths=writable,
        allow_localhost_only=True,
        allowed_tcp_ports=ports,
    )
    profile_path = Path(run_dir) / 'seatbelt-worker.sb'
    private_write(profile_path, profile)
    return profile_path


def _seatbelt_test_argv(task: dict, run_dir: Path, argv: list[str]) -> list[str]:
    """Return argv wrapped with seatbelt-test.sb when OS sandboxing is active."""
    if task.get('os_sandbox', 'off') in ('test_only', 'strict') and is_seatbelt_available():
        profile_path = Path(run_dir) / 'seatbelt-test.sb'
        if not profile_path.is_file():
            profile_path = _stage_test_seatbelt_profile(task, run_dir)
        return wrap_argv_with_seatbelt(argv, profile_path)
    return argv


def stage_guard_plugin(run_dir, *, allowed_files=(), read_only_files=()) -> Path:
    """Stage the rendered guard plugin under run_dir and return its path."""
    path = Path(run_dir) / '.opencode' / 'plugin' / 'harness_guard.ts'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_guard_plugin(allowed_files=allowed_files,
                                        read_only_files=read_only_files), encoding='utf-8')
    return path


def config_for(task, *, base_url=None, run_dir=None):
    permissions = [{'action': '*', 'resource': '*', 'effect': 'deny'},
                   {'action': 'external_directory', 'resource': '*', 'effect': 'allow'},
                   {'action': 'read', 'resource': '/tmp/hb-*/**', 'effect': 'allow'}]
    task_dir = task['directory']
    # Allow reads across the task directory (bounded by sensitive-path denies below).
    permissions += [{'action': 'read', 'resource': task_dir, 'effect': 'allow'},
                    {'action': 'read', 'resource': f'{task_dir}/**', 'effect': 'allow'}]
    # Deny secrets and credentials under the task directory (last matching rule wins).
    sensitive_patterns = ('.env', '.env.*', '*.pem', '*.key', '*credentials*', '*secret*')
    permissions += [{'action': action,
                     'resource': pattern,
                     'effect': 'deny'}
                    for p in sensitive_patterns
                    for pattern in (f'{task_dir}/{p}', f'{task_dir}/**/{p}')
                    for action in ('read', 'edit')]
    # Re-allow safe env templates that would otherwise match the deny patterns.
    safe_env_patterns = ('.env.example', '.env.sample', '.env.template')
    permissions += [{'action': 'read',
                     'resource': pattern,
                     'effect': 'allow'}
                    for p in safe_env_patterns
                    for pattern in (f'{task_dir}/{p}', f'{task_dir}/**/{p}')]
    actions = (
        ('read',)
        if task['mode'] == 'patch'
        else (
            'read',
            'edit',
            'multiedit',
            'patch',
            'apply_patch',
            'harness_multiedit',
            'harness_apply_patch',
        )
    )
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
    cfg = {'model': task['model'], 'small_model': task['model'], 'snapshot': False,
           'agents': {'guarded_worker': {'description': 'Scoped bounded coding worker',
           'mode': 'primary', 'steps': task['max_steps'], 'permissions': permissions}}}
    if base_url:
        cfg['providers'] = {'openrouter': {'settings': {'baseURL': base_url}}}
    if run_dir is not None:
        plugin_path = stage_guard_plugin(
            Path(run_dir), allowed_files=task.get('files', ()),
            read_only_files=task.get('read_only_files', ()))
        cfg['plugin'] = list(cfg.get('plugin', [])) + [plugin_path.resolve().as_uri()]
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

def _count_guard_hits(proxy_summary):
    """Count guard interventions from a proxy summary for herdr reporting."""
    summary = proxy_summary or {}
    return (int(summary.get('compacted_reads', 0))
            + int(summary.get('edit_hints_injected', 0))
            + int(summary.get('cot_slop_warnings', 0))
            + int(summary.get('blocked_count', 0)))

def worker_env(directory, *, env_overrides=None):
    # OpenCode run uses PWD for the session location, even when cwd is correct.
    env = dict(os.environ, PWD=str(directory))
    env.setdefault('OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX', '64000')
    if env_overrides:
        env.update(env_overrides)
    return env

@contextmanager
def owned_process(argv, directory, log, *, env_overrides=None):
    proc = subprocess.Popen(argv, cwd=directory, env=worker_env(directory, env_overrides=env_overrides),
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

def execute(argv, directory, timeout, path, *, env_overrides=None):
    with private_log(path) as log, owned_process(argv, directory, log, env_overrides=env_overrides) as proc:
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            raise GuardError('command_timeout') from None

def preflight(cli, task, run_dir, *, base_url=None, check=False):
    executable_argv([str(cli)])
    if task.get('mode', 'model') == 'model':
        task['test_argv'] = resolve_pytest_test_argv(task['test_argv'], task['directory'])
    if not check and task.get('preflight_test_argv', True) and task.get('test_argv'):
        preflight_test_argv(task, run_dir)
    private_write(run_dir / 'opencode.jsonc', config_for(task, base_url=base_url, run_dir=run_dir))
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
    symbol_context = ''
    if '[CURRENT_PATCH]' not in task['prompt'] and task.get('include_symbol_context', True):
        symbol_context = build_cross_file_symbol_context(
            task['files'], task['prompt'], directory=task.get('directory', ''),
            read_only_files=task.get('read_only_files', ()))
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
    prompt = skeletons + excerpts + (symbol_context + '\n\n' if symbol_context else '') + prompt
    argv = [str(cli), 'run', '--standalone', '--auto', '--format', 'json', '--agent', 'guarded_worker',
            '--session', sid, '--title', task['task_id'], '--model', task['model'], prompt]
    result['cost'] = None
    private_write(run_dir / 'result.json', result)  # An abrupt crash leaves unknown cost.
    event(run_dir, 'model_session_started', session_id=sid, model=task['model'])
    started = time.monotonic()
    if started_ref is not None:
        started_ref[0] = started
    naturally_exited = False
    last_reported_step = -1
    isolation_env = stage_isolation_sandbox(run_dir)
    if task.get('os_sandbox', 'off') == 'strict' and is_seatbelt_available():
        worker_sb = _stage_worker_seatbelt_profile(task, run_dir, db, proxy=proxy)
        argv = wrap_argv_with_seatbelt(argv, worker_sb)
    try:
        with private_log(run_dir / 'worker.log') as log, owned_process(argv, run_dir, log, env_overrides=isolation_env) as proc:
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
                    if m['steps'] != last_reported_step:
                        last_reported_step = m['steps']
                        try:
                            private_write(run_dir / 'result.json', result)
                        except Exception:
                            pass
                        try:
                            report_herdr_task(task['task_id'], 'working', model=task.get('model'),
                                              step=m['steps'], max_steps=task['max_steps'],
                                              cost=m['cost'], budget=budget,
                                              guard_hits=_count_guard_hits(proxy_summary),
                                              session_id=sid)
                        except Exception:
                            pass
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
        cleanup_isolation_sandbox(run_dir)

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


def _capture_assertion_locals(task, run_dir, test_env=None):
    """H-96: capture a bounded -vv --showlocals snapshot of failing assertion tests."""
    if not task.get('capture_assertion_locals', True):
        return
    run_dir = Path(run_dir)
    try:
        test_log_text = (run_dir / 'test.log').read_text(errors='replace')
    except OSError:
        return
    if 'AssertionError' not in test_log_text and 'assert ' not in test_log_text:
        return
    targeted = build_targeted_pytest_argv(task['test_argv'], test_log_text, max_nodeids=3)
    if not targeted:
        return
    if '--showlocals' not in targeted:
        targeted = [arg for arg in targeted if arg not in ('-q', '--quiet')]
        targeted.extend(['-vv', '--tb=short', '--showlocals', '--no-header'])
    diag_timeout = min(15, task['test_timeout_seconds'])
    try:
        execute(targeted, task['directory'], diag_timeout,
                run_dir / 'test_assertion_locals.log', env_overrides=test_env)
    except GuardError as exc:
        if str(exc) != 'command_timeout':
            raise
    except OSError:
        pass


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
    diag_log_path = run_dir / 'test_assertion_locals.log'
    if diag_log_path.is_file():
        try:
            diag_text = diag_log_path.read_text(errors='replace')
        except OSError:
            diag_text = ''
        assertion_locals = extract_runtime_assertion_locals(diag_text)
        if assertion_locals:
            prompt += '\n\n[RUNTIME_ASSERTION_LOCALS]\n' + assertion_locals
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
                       origin_directory=task.get('directory'),
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


_LESSON_ERROR_MARKERS = (
    'AssertionError', 'ImportError', 'ModuleNotFoundError', 'SyntaxError',
    'IndentationError', 'TypeError', 'ValueError', 'KeyError', 'AttributeError',
    'NameError', 'RuntimeError', 'ValidationError', 'FileNotFoundError',
    'PermissionError', 'ZeroDivisionError', 'IndexError', 'LookupError',
    'OSError', 'FAILED ', 'ERROR:',
)

_LESSON_PROGRESS_RE = re.compile(r'^[.sFxXEs]+(?:\s+\[\s*\d+%\])?$')
_LESSON_SUMMARY_RE = re.compile(r'\b\d+\s+(?:passed|deselected|skipped|warnings?)\b')
_LESSON_EXCEPTION_RE = re.compile(
    r'^[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Failure):\s+.+')
_LESSON_WORKTREE_PREFIX_RE = re.compile(
    r'/[^\s"\x27]*?(?:\.tools/opencode-guard-state/worktrees|\.worktrees|worktrees)/[^/\s"\x27]+/')


def _lesson_error_signature(failed_test_log: str) -> str:
    """First concise error line from a failing test log, truncated to 160 chars.

    Never returns noise from passing/progress output: pytest dot/progress
    lines, ``N passed`` summary lines (without failures), banner lines,
    tracebacks and warning lines are ignored. Returns ``''`` when no
    actionable error/failure line is present.
    """
    def _normalize(line: str) -> str:
        return _LESSON_WORKTREE_PREFIX_RE.sub('', line)

    raw_lines = [line.strip() for line in (failed_test_log or '').splitlines()]
    noise = {'', 'Traceback (most recent call last):'}
    error_lines: list[str] = []
    e_lines: list[str] = []
    for line in raw_lines:
        if not line or line in noise or line.startswith(('====', '----')):
            continue
        if 'ResourceWarning:' in line or 'DeprecationWarning:' in line or 'PytestWarning' in line:
            continue
        if _LESSON_PROGRESS_RE.match(line):
            continue
        if _LESSON_SUMMARY_RE.search(line):
            continue
        if line.startswith('E   '):
            e_lines.append(line)
        elif (any(marker in line for marker in _LESSON_ERROR_MARKERS)
                or _LESSON_EXCEPTION_RE.match(line)):
            error_lines.append(line)
    if e_lines:
        # Prefer a descriptive E   line over a bare 'E   AssertionError'.
        descriptive = [ln for ln in e_lines if ln != 'E   AssertionError']
        return _normalize((descriptive or e_lines)[0])[:160]
    if error_lines:
        return _normalize(error_lines[0])[:160]
    return ''


def _norm_lesson_path(value) -> str:
    text = str(value or '').strip()
    if not text:
        return ''
    # Normalize ephemeral worktree paths back to the canonical repo root:
    # /repo/.worktrees/<branch>[/sub] -> /repo[/sub]
    for marker in ('/.tools/opencode-guard-state/worktrees/', '/.worktrees/', '/worktrees/'):
        if marker in text:
            head, _, tail = text.partition(marker)
            _, sep, rest = tail.partition('/')
            if sep:
                text = head.rstrip('/') + '/' + rest
            else:
                text = head
            if not text:
                text = '/'
            break
    try:
        return str(Path(text).resolve())
    except OSError:
        return text


_SUBPROJECT_TOP2_ROOTS = frozenset({'ai', 'airflow', 'tools', 'scripts', 'dbt'})
_SUBPROJECT_TOP1_ROOTS = frozenset({'sml', 'harness', 'tests', 'tenet'})
_SUBPROJECT_STRIP_ROOTS = ('sml', 'harness', 'tenet', 'ai', 'airflow', 'dbt', 'tools', 'scripts', 'tests')
_MONOREPO_PROJECTS = frozenset({'medvidi/analytics-gitops', 'medvidi/analytics-airflow-app'})


def resolve_project_identity(task_dir, files=()) -> tuple[str, set[str]]:
    """H-84: map a task directory (plus files) to ``(project_id, subprojects)``.

    Worktrees of the same repo share one canonical ``project_id`` while
    different projects never collide; ``subprojects`` carries the top path
    segments (e.g. ``'ai/serving'``, ``'harness'``) so lessons can be scoped
    inside multi-package monorepos.
    """
    raw = str(task_dir or '').strip()
    norm_dir = _norm_lesson_path(raw)
    norm_lower = norm_dir.lower()
    raw_lower = raw.lower()
    file_strs = [str(f) for f in (files or ()) if str(f).strip()]
    file_lowers = [f.lower().replace('\\', '/') for f in file_strs]

    if ('small model lab' in raw_lower or 'small model lab' in norm_lower
            or any(f.startswith('sml/') or f.startswith('tenet')
                   or '/sml/' in f or '/tenet/' in f for f in file_lowers)):
        project_id = 'personal/small-model-lab'
    elif norm_lower.endswith('/harness') or '/harness/' in norm_lower:
        project_id = 'personal/harness'
    elif norm_lower.endswith('/herdr') or '/herdr/' in norm_lower:
        project_id = 'personal/herdr'
    elif ('analytics-airflow-app' in norm_lower
          or '/wt-' in raw_lower
          or any(f.startswith(('airflow/dags', 'tools/metric_spec'))
                 or '/airflow/dags/' in f or '/tools/metric_spec/' in f for f in file_lowers)):
        project_id = 'medvidi/analytics-airflow-app'
    elif ('analytics-gitops' in norm_lower
          or '/worktrees/gitops-' in raw_lower
          or '/.worktrees/' in raw_lower
          or '/.tools/opencode-guard-state/worktrees/' in raw_lower
          or any(f.startswith(('ai/serving', 'ai/mcp', 'ai/local_copilot', 'ai/graphiti',
                               'ai/security', 'ai/web', 'ai/dbt_authoring', 'ai/coding_worker',
                               'ai/tools', 'ai/sql', 'ai/openspec'))
                 or '/ai/' in f or '/context/' in f for f in file_lowers)):
        project_id = 'medvidi/analytics-gitops'
    else:
        project_id = norm_dir

    subprojects: set[str] = set()
    # A task_dir nested inside a subproject (e.g. .../ai/web/capri/src) itself
    # identifies that subproject.
    dir_parts = [p for p in Path(norm_dir).parts if p not in ('/',)]
    for i, seg in enumerate(dir_parts):
        if seg in _SUBPROJECT_TOP2_ROOTS and i + 1 < len(dir_parts):
            subprojects.add(f'{seg}/{dir_parts[i + 1]}')
            break
        if seg in _SUBPROJECT_TOP1_ROOTS:
            subprojects.add(seg)
            break

    for f in file_strs:
        text = str(f).replace('\\', '/')
        if not Path(text).is_absolute():
            rel = text
        else:
            norm_f = _norm_lesson_path(text).replace('\\', '/')
            try:
                rel = str(Path(norm_f).resolve().relative_to(norm_dir)) if norm_dir else norm_f
            except (OSError, ValueError):
                rel = norm_f
                for root in _SUBPROJECT_STRIP_ROOTS:
                    idx = norm_f.find('/' + root + '/')
                    if idx != -1:
                        rel = norm_f[idx + 1:]
                        break
        parts = [p for p in Path(rel).parts if p not in ('/', '.')]
        if not parts:
            continue
        top = parts[0]
        if top in _SUBPROJECT_TOP1_ROOTS:
            subprojects.add(top)
        elif top in _SUBPROJECT_TOP2_ROOTS and len(parts) >= 2:
            subprojects.add(f'{top}/{parts[1]}')
        elif len(parts) >= 2:
            subprojects.add(top)
    return project_id, subprojects


def _extract_patch_fix_summary(patch_text: str, max_chars: int = 120) -> str:
    """Compact 1-line summary of a unified diff: enclosing symbol + added lines."""
    if not patch_text:
        return ''
    symbol = ''
    added: list[str] = []
    for line in patch_text.splitlines():
        if line.startswith('@@') and '@@' in line[2:]:
            header = line.rsplit('@@', 1)[-1].strip()
            match = re.search(r'\b(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)', header)
            symbol = f'{header.split(" ")[0]} {match.group(1)}' if match else ''
            continue
        if not line.startswith('+') or line.startswith('+++'):
            continue
        candidate = line[1:].strip()
        if (not candidate or candidate.startswith('#')
                or candidate.startswith('import ') or candidate.startswith('from ')):
            continue
        added.append(candidate)
        if len(added) >= 2:
            break
    parts: list[str] = []
    if symbol:
        parts.append(f'in {symbol}:')
    parts.extend('+' + a for a in added)
    if not parts:
        return ''
    return ' '.join(parts)[:max_chars]


def record_repair_lesson(state_dir, task, failed_test_log, repaired_files,
                         patch_text='') -> None:
    """H-76: append a compact repair lesson to ``lessons.jsonl`` (deduplicated).

    Skips writing when an identical ``(task_dir, error_signature)`` pair is
    already recorded, or when the log carries no actionable error signature
    (e.g. passing pytest output).
    """
    signature = _lesson_error_signature(failed_test_log)
    if not signature:
        return
    state = Path(state_dir)
    task_dir = str(task.get('task_dir') or task.get('directory') or '')
    identity_files = [str(f) for f in (repaired_files or [])] or [
        str(f) for f in (task.get('files') or [])]
    project_id, subprojects = resolve_project_identity(task_dir, identity_files)
    base = Path(task_dir) if task_dir else None
    files: list[str] = []
    for item in (repaired_files or []):
        p = Path(str(item))
        rel = str(item)
        if base is not None:
            try:
                rel = str(p.resolve().relative_to(Path(base).resolve()))
            except (OSError, ValueError):
                rel = str(item)
        files.append(rel)
    lessons_path = state / 'lessons.jsonl'
    if lessons_path.is_file():
        for line in lessons_path.read_text(encoding='utf-8', errors='replace').splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if not isinstance(rec, dict):
                continue
            if str(rec.get('error_signature') or '') != signature:
                continue
            # H-84: dedupe by (project_id, signature); fall back to
            # (task_dir, signature) for records written before project_id.
            rec_project = str(rec.get('project_id') or '')
            if rec_project:
                if rec_project == project_id:
                    return
            elif str(rec.get('task_dir') or '') == task_dir:
                return
    record = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'task_id': task.get('task_id', ''),
        'task_dir': task_dir,
        'project_id': project_id,
        'subprojects': sorted(subprojects),
        'files': files,
        'error_signature': signature,
        'model': task.get('model'),
    }
    if patch_text:
        fix_summary = _extract_patch_fix_summary(patch_text)
        if fix_summary:
            record['fix_summary'] = fix_summary
    with lessons_path.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(record) + '\n')


def load_workspace_lessons(state_dir, task_dir, files=None, max_items=3, max_chars=600) -> str:
    """H-76: compact guardrail block from prior repair lessons for this workspace.

    Filters lessons by ``task_dir`` (normalizing paths), prioritizes recent
    lessons that share any file in ``files``, deduplicates by
    ``error_signature``, and caps the block at ``max_items`` entries /
    ``max_chars`` characters. Returns ``''`` when no lessons match.
    """
    lessons_path = Path(state_dir) / 'lessons.jsonl'
    want_dir = _norm_lesson_path(task_dir)
    if not lessons_path.is_file() or not want_dir:
        return ''
    file_set = {_norm_lesson_path(f) for f in (files or []) if str(f).strip()}
    want_project, want_subprojects = resolve_project_identity(task_dir, files or ())
    matched: list[tuple[int, dict]] = []
    lines = [line for line in lessons_path.read_text(encoding='utf-8', errors='replace').splitlines()
             if line.strip()]
    for line in reversed(lines):  # most recent first; tier sorting below
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if not isinstance(rec, dict):
            continue
        rec_files_raw = [str(f) for f in (rec.get('files') or []) if str(f).strip()]
        rec_project_id, rec_subprojects_raw = resolve_project_identity(
            rec.get('task_dir'), rec_files_raw)
        rec_project = str(rec.get('project_id') or '') or rec_project_id
        rec_subprojects = set(rec.get('subprojects') or []) or rec_subprojects_raw
        # H-84: strict project isolation — lessons from another project NEVER
        # match, even when file paths happen to overlap.
        if rec_project != want_project:
            continue
        rec_files = set()
        for f in rec_files_raw:
            fp = Path(f)
            resolved_f = fp if fp.is_absolute() else (Path(want_dir) / fp)
            rec_files.add(_norm_lesson_path(resolved_f))
        # Match tiers: 0 = exact file, 1 = same subproject, 2 = same project.
        tier = 2
        if file_set and rec_files:
            exact = bool(rec_files & file_set) or any(
                len(Path(f).parts) >= 2
                and any(wf.endswith('/' + '/'.join(Path(f).parts[-2:]))
                        for wf in file_set)
                for f in rec_files)
            if exact:
                tier = 0
        if tier != 0 and want_subprojects and (rec_subprojects & want_subprojects):
            tier = 1
        if tier == 2 and file_set:
            # For multi-subproject monorepos with an explicit file scope,
            # require a file or subproject match — do not inject lessons from
            # unrelated subprojects (e.g. ai/local_copilot into ai/serving).
            if want_project in _MONOREPO_PROJECTS and want_subprojects:
                continue
        matched.append((tier, rec))
    if not matched:
        return ''
    matched.sort(key=lambda item: item[0])  # stable: lower tier first, recent within tier
    header = '### Workspace Guardrails (from prior repairs)'
    out_lines: list[str] = []
    seen_signatures: set[str] = set()
    total_chars = len(header)
    for _, rec in matched:
        if len(out_lines) >= max_items:
            break
        signature = str(rec.get('error_signature') or '').strip()
        if not signature or signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        file_hint = ', '.join(str(f) for f in (rec.get('files') or []) if str(f).strip())
        fix_summary = str(rec.get('fix_summary') or '').strip()
        entry = '- ' + signature
        if fix_summary:
            entry += f' -> Fix: {fix_summary}'
        entry += (f' (files: {file_hint})' if file_hint else '')
        if total_chars + len(entry) + 1 > max_chars:
            break
        out_lines.append(entry)
        total_chars += len(entry) + 1
    if not out_lines:
        return ''
    return header + '\n' + '\n'.join(out_lines)


def _active_locked_or_sibling_editable_files(state: Path, current_task_id: str) -> set[str]:
    """H-90: return editable files owned by active pending sibling runs in state/runs."""
    runs_dir = Path(state) / 'runs'
    if not runs_dir.is_dir():
        return set()
    now = time.time()
    owned: set[str] = set()
    try:
        entries = list(runs_dir.iterdir())
    except OSError:
        return set()
    for d in entries:
        try:
            if not d.is_dir() or (now - d.stat().st_mtime) > 600:
                continue
            res = json.loads((d / 'result.json').read_text(encoding='utf-8'))
            task_data = json.loads((d / 'task.json').read_text(encoding='utf-8'))
        except (OSError, ValueError, TypeError):
            continue
        if not isinstance(res, dict) or not isinstance(task_data, dict):
            continue
        if res.get('status') != 'pending' or res.get('task_id') == current_task_id:
            continue
        for f in task_data.get('files') or ():
            if isinstance(f, str):
                try:
                    owned.add(str(Path(f).resolve()))
                except OSError:
                    owned.add(f)
    return owned


def _concurrent_sibling_editable_files(
    state: Path, current_task_id: str, window_start: float, window_end: float
) -> set[str]:
    """H-90: return editable files of sibling runs overlapping [window_start, window_end]."""
    runs_dir = Path(state) / 'runs'
    if not runs_dir.is_dir():
        return set()
    owned: set[str] = set()
    try:
        entries = list(runs_dir.iterdir())
    except OSError:
        return set()
    for d in entries:
        try:
            if not d.is_dir():
                continue
            res_path = d / 'result.json'
            task_path = d / 'task.json'
            res = json.loads(res_path.read_text(encoding='utf-8'))
            task_data = json.loads(task_path.read_text(encoding='utf-8'))
        except (OSError, ValueError, TypeError):
            continue
        if not isinstance(res, dict) or not isinstance(task_data, dict):
            continue
        if res.get('task_id') == current_task_id:
            continue
        sib_start = res.get('started_at')
        if not isinstance(sib_start, (int, float)):
            try:
                sib_start = d.stat().st_mtime
            except OSError:
                continue
        if sib_start > window_end:
            continue
        if res.get('status') == 'pending':
            sib_end = window_end
        else:
            dur = res.get('duration_seconds')
            if isinstance(dur, (int, float)):
                sib_end = sib_start + max(0.0, float(dur))
            else:
                try:
                    sib_end = res_path.stat().st_mtime
                except OSError:
                    sib_end = sib_start
        if sib_end < window_start:
            continue
        for key in ('files', 'origin_files'):
            for f in task_data.get(key) or ():
                if isinstance(f, str):
                    try:
                        owned.add(str(Path(f).resolve()))
                    except OSError:
                        owned.add(f)
    return owned


def run_semantic_diff_critic(task: dict, patch_text: str, *, base_url: str | None) -> dict:
    """H-97 Tier-2 single-shot LLM diff critic (best-effort)."""
    if not base_url or not isinstance(patch_text, str) or not patch_text.strip():
        return {'verdict': 'pass', 'issues': []}
    url = base_url if base_url.rstrip('/').endswith('/chat/completions') else base_url.rstrip('/') + '/chat/completions'
    model = task.get('diff_critic_model') or task.get('model') or 'openrouter/z-ai/glm-5.3-flash'
    prompt_text = str(task.get('prompt', '') or '')[:4000]
    diff_excerpt = patch_text[:12000]
    system_msg = (
        'You are a strict semantic diff critic. Respond ONLY with a single JSON object: '
        '{"verdict": "pass"|"fail", "issues": [...]}. '
        'Check ONLY: (a) hardcoded test literals instead of real logic, and '
        '(b) explicitly numbered requirements in the task prompt completely missing from the diff. '
        'If neither is present, return {"verdict": "pass", "issues": []}.'
    )
    user_msg = f'Task prompt:\n{prompt_text}\n\nDiff:\n{diff_excerpt}'
    payload = json.dumps({
        'model': model,
        'messages': [
            {'role': 'system', 'content': system_msg},
            {'role': 'user', 'content': user_msg},
        ],
        'max_tokens': 200,
        'temperature': 0,
        'stream': False,
    }).encode('utf-8')
    headers = {'Content-Type': 'application/json'}
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            raw = json.loads(resp.read().decode('utf-8', errors='replace'))
        content = str(raw['choices'][0]['message']['content'] or '').strip()
        if content.startswith('```'):
            content = re.sub(r'^```(?:json)?\s*|\s*```$', '', content, flags=re.DOTALL).strip()
        parsed = json.loads(content)
        if isinstance(parsed, dict) and parsed.get('verdict') == 'fail':
            raw_issues = parsed.get('issues')
            issues = [str(x) for x in raw_issues if x] if isinstance(raw_issues, list) else []
            return {'verdict': 'fail', 'issues': issues or ['semantic_diff_critic_failed']}
    except Exception:
        return {'verdict': 'pass', 'issues': []}
    return {'verdict': 'pass', 'issues': []}


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
            if not task.get('worktree') and task.get('read_only_files'):
                explicit_ro = set(task.get('explicit_read_only_files', ()))
                sibling_editable = _active_locked_or_sibling_editable_files(state, task['task_id'])
                if sibling_editable:
                    task['read_only_files'] = [
                        f for f in task['read_only_files']
                        if f in explicit_ro or f not in sibling_editable
                    ]
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
            os_sandbox = task.get('os_sandbox', 'off')
            if not check and os_sandbox in ('test_only', 'strict') and is_seatbelt_available():
                _stage_test_seatbelt_profile(task, run_dir)
                event(run_dir, 'seatbelt_sandbox_enabled', mode=os_sandbox)
            try:
                report_herdr_task(task['task_id'], 'working',
                                  model=task.get('model') if task['mode'] != 'command' else None,
                                  step=0, max_steps=task['max_steps'], cost=0.0,
                                  budget=budget, phase='starting')
            except Exception:
                pass
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
                    preflight(cli, task, run_dir, base_url=base_url, check=check)
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
                            repo_hint = build_repo_memory_hint(
                                state, task.get('origin_directory') or task.get('directory'))
                            if repo_hint:
                                task['prompt'] += repo_hint
                                event(run_dir, 'repo_memory_hint_injected')
                        if task.get('include_workspace_lessons', True):
                            try:
                                lesson_hint = load_workspace_lessons(
                                    state,
                                    task.get('origin_directory') or task.get('directory'),
                                    files=task.get('origin_files') or task.get('files'))
                            except Exception:
                                lesson_hint = ''
                            if lesson_hint:
                                task['prompt'] += lesson_hint
                                event(run_dir, 'workspace_lessons_injected')
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
                        test_env = build_test_env_overrides(task)
                        result['test_return_code'] = execute(_seatbelt_test_argv(task, run_dir, task['test_argv']),
                                                            task['directory'],
                                                            task['test_timeout_seconds'], run_dir / 'test.log',
                                                            env_overrides=test_env)

                        def _evaluate_diff_critic() -> list[str]:
                            capture_files(run_dir, tracked)
                            patch_path = run_dir / 'changes.patch'
                            patch_text = (
                                patch_path.read_text(encoding='utf-8', errors='replace')
                                if patch_path.is_file() else ''
                            )
                            blocking: list[str] = []
                            warnings: list[str] = []
                            if task.get('anti_tamper_audit', True):
                                audit = audit_patch_for_reward_hacking(patch_text, task.get('prompt', ''))
                                blocking.extend(audit.get('blocking_issues', []))
                                warnings.extend(audit.get('warnings', []))
                            if task.get('enable_diff_critic') or os.environ.get('HARNESS_DIFF_CRITIC') == '1':
                                critic = run_semantic_diff_critic(
                                    task, patch_text, base_url=upstream_url or OPENROUTER_UPSTREAM
                                )
                                if critic.get('verdict') == 'fail':
                                    blocking.extend(critic.get('issues') or ['semantic_diff_critic_failed'])
                            if warnings:
                                existing = result.get('critic_warnings') or []
                                result['critic_warnings'] = list(dict.fromkeys([*existing, *warnings]))
                            return blocking

                        critic_issues: list[str] = []
                        if result['test_return_code'] == 0:
                            critic_issues = _evaluate_diff_critic()
                            if critic_issues:
                                with private_log(run_dir / 'test.log') as f:
                                    f.write('[DIFF_CRITIC_FAILURE]\n' + '\n'.join(critic_issues) + '\n')
                                event(run_dir, 'diff_critic_rejected', issues=critic_issues)

                        if result['test_return_code'] or critic_issues:
                            capture_files(run_dir, tracked)
                            repair_eligible = (
                                task.get('auto_micro_repair', True)
                                and task['mode'] == 'model'
                                and not result.get('micro_repair_attempted')
                                and (run_dir / 'changes.patch').exists()
                                and (run_dir / 'changes.patch').stat().st_size > 0
                                and result.get('cost') is not None
                                and (salvage_candidate == 'step_limit'
                                     or result.get('steps', 0) <= task['max_steps'])
                                and result['cost'] <= min(budget * 0.8, 0.08)
                            )
                            if critic_issues and not repair_eligible:
                                raise GuardError('reward_hacking_detected')
                            if not critic_issues and salvage_candidate is not None and (
                                salvage_candidate not in REPAIRABLE_SALVAGE_REASONS
                                or result['test_return_code'] == 124
                                or not repair_eligible
                            ):
                                raise GuardError(salvage_candidate)
                            repair_salvage_reason = None
                            if repair_eligible:
                                if salvage_candidate is not None:
                                    event(run_dir, 'salvage_micro_repair_started', reason=salvage_candidate)
                                repair_max_steps = compute_repair_max_steps(task)
                                active_max_steps[0] = repair_max_steps
                                repair_timeout = compute_repair_timeout_seconds(task)
                                active_timeout_seconds[0] = repair_timeout
                                _capture_assertion_locals(task, run_dir, test_env=test_env)
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
                                try:
                                    if (run_dir / 'test.log').is_file():
                                        shutil.copy2(run_dir / 'test.log',
                                                     run_dir / 'test_before_repair.log')
                                except OSError:
                                    pass
                                hashes_before_repair = _fh(task['files'])
                                first_test_log_text = ''
                                repaired_files: list[str] = []
                                if proxy is not None and hasattr(proxy, 'reset_session'):
                                    proxy.reset_session()
                                try:
                                    event(run_dir, 'micro_repair_started', session_id=sid2)
                                    try:
                                        report_herdr_task(task['task_id'], 'working',
                                                          model=task.get('model'),
                                                          step=result.get('steps', 0),
                                                          max_steps=task['max_steps'],
                                                          cost=result.get('cost') if result.get('cost') is not None else result.get('observed_cost'),
                                                          budget=budget, phase='micro_repair',
                                                          session_id=sid2)
                                    except Exception:
                                        pass
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
                                            if salvage_candidate is not None:
                                                raise GuardError(salvage_candidate) from None
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
                                    repaired_files = [f for f in task['files']
                                                      if _fh([f]).get(f) != hashes_before_repair.get(f)]
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
                                        test_env = build_test_env_overrides(task)
                                        rc_fast = execute(_seatbelt_test_argv(task, run_dir, targeted_argv),
                                                          task['directory'],
                                                          min(30, task['test_timeout_seconds']),
                                                          run_dir / 'test.log',
                                                          env_overrides=test_env)
                                        if rc_fast not in (0, 4):
                                            result['test_return_code'] = rc_fast
                                            event(run_dir, 'micro_repair_fast_fail',
                                                  return_code=rc_fast)
                                        else:
                                            result['test_return_code'] = execute(
                                                _seatbelt_test_argv(task, run_dir, task['test_argv']),
                                                task['directory'],
                                                task['test_timeout_seconds'], run_dir / 'test.log',
                                                env_overrides=test_env)
                                    else:
                                        result['test_return_code'] = execute(
                                            _seatbelt_test_argv(task, run_dir, task['test_argv']),
                                            task['directory'],
                                            task['test_timeout_seconds'], run_dir / 'test.log',
                                            env_overrides=build_test_env_overrides(task))
                            if result['test_return_code']:
                                raise GuardError(salvage_candidate or repair_salvage_reason or 'validation_failed')
                            post_critic_issues = _evaluate_diff_critic()
                            if post_critic_issues:
                                with private_log(run_dir / 'test.log') as f:
                                    f.write('[DIFF_CRITIC_FAILURE]\n' + '\n'.join(post_critic_issues) + '\n')
                                event(run_dir, 'diff_critic_rejected', issues=post_critic_issues)
                                raise GuardError('reward_hacking_detected')
                            if salvage_candidate is not None:
                                result['salvaged_from'] = f'salvage_repaired_{salvage_candidate}'
                            elif repair_salvage_reason is not None:
                                result['salvaged_from'] = f'micro_repair_{repair_salvage_reason}'
                            try:
                                if result.get('test_return_code') == 0 and repaired_files:
                                    record_repair_lesson(
                                        state, task, first_test_log_text, repaired_files,
                                        patch_text=(
                                            (run_dir / 'changes.patch').read_text(
                                                encoding='utf-8', errors='replace')
                                            if (run_dir / 'changes.patch').is_file() else ''
                                        ),
                                    )
                            except Exception:
                                pass
                        if ro_files:
                            ro_hashes_after = _fh(ro_files)
                            modified_ro = [
                                p for p in ro_files
                                if ro_hashes_after.get(p) != ro_hashes_before.get(p)
                            ]
                            if modified_ro:
                                explicit_ro = set(task.get('explicit_read_only_files', ()))
                                has_non_explicit = any(p not in explicit_ro for p in modified_ro)
                                concurrent_owned = (
                                    _concurrent_sibling_editable_files(
                                        state, task['task_id'], started_at, time.time()
                                    )
                                    if has_non_explicit else set()
                                )
                                for ro_path in modified_ro:
                                    if ro_path in explicit_ro or ro_path not in concurrent_owned:
                                        raise GuardError('read_only_file_modified')
                                for ro_path in modified_ro:
                                    event(run_dir, 'concurrent_sibling_read_only_change_ignored',
                                          file=ro_path)
                        if task.get('origin_files'):
                            if _fh(task['origin_files']) != task['origin_hashes']:
                                raise GuardError('worktree_origin_modified_concurrently')
                            for src, dst in zip(task['files'], task['origin_files']):
                                if Path(src).exists():
                                    Path(dst).parent.mkdir(parents=True, exist_ok=True)
                                    Path(dst).write_bytes(Path(src).read_bytes())
                                elif Path(dst).exists():
                                    Path(dst).unlink()
                        if salvage_candidate is not None and not result.get('salvaged_from'):
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
                            result['test_return_code'] = execute(
                                _seatbelt_test_argv(task, run_dir, task['test_argv']),
                                task['directory'],
                                task['test_timeout_seconds'], run_dir / 'test.log',
                                env_overrides=build_test_env_overrides(task))
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
    try:
        _herdr_state = 'done' if result.get('status') in (
            'validation_passed_review_required', 'patch_ready_review_required',
            'configuration_verified_no_model_call') else 'blocked'
        _cost = (reconciled_cost if reconciled_cost is not None
                 else (result.get('cost') if result.get('cost') is not None
                       else result.get('observed_cost')))
        report_herdr_task(result.get('task_id') or 'unknown', _herdr_state,
                          model=result.get('model'), step=result.get('steps'),
                          max_steps=(result.get('limits') or {}).get('max_steps'),
                          cost=_cost,
                          budget=(result.get('limits') or {}).get('available_cost_usd'),
                          reason=result.get('reason') or result.get('salvaged_from') or None,
                          session_id=result.get('session_id'),
                          phase='salvaged' if result.get('salvaged_from') else None)
    except Exception:
        pass
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


_ROLLOUT_PASS_STATUSES = frozenset({
    'pass',
    'passed',
    'success',
    'succeeded',
    'ok',
    'complete',
    'completed',
    'done',
    '0',
    'true',
    'validation_passed_review_required',
    'patch_ready_review_required',
    'validated',
})


def restore_files(snapshot):
    """Restore exact file bytes from a {path: bytes | None} snapshot or a run_dir."""
    if isinstance(snapshot, (str, Path)):
        run_dir = Path(snapshot)
        before_path = run_dir / 'files.before.json'
        if not before_path.exists():
            raise GuardError('missing_files_snapshot')
        raw = json.loads(before_path.read_text(encoding='utf-8'))
        snapshot = {
            name: base64.b64decode(item['base64']) if item.get('base64') is not None else None
            for name, item in raw.items()
        }
    for name, data in snapshot.items():
        p = Path(name)
        if data is None:
            if p.exists():
                p.unlink()
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)


def _snapshot_file_bytes(files):
    return {
        name: (Path(name).read_bytes() if Path(name).exists() else None)
        for name in files
    }


def _diff_file_snapshots(before, after):
    diff = []
    for name in before:
        old_bytes = before[name]
        new_bytes = after.get(name)
        if old_bytes == new_bytes:
            continue
        try:
            old_text = (old_bytes or b'').decode('utf-8')
            new_text = (new_bytes or b'').decode('utf-8')
        except UnicodeDecodeError:
            continue
        diff.extend(
            difflib.unified_diff(
                old_text.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile=name,
                tofile=name,
            )
        )
    return ''.join(
        line if line.endswith('\n') else line + '\n\\ No newline at end of file\n'
        for line in diff
    )


def run_group_rollout(
    task: dict,
    *,
    group_size: int = 4,
    models: list[str] | None = None,
    state_dir: str | Path | None = None,
    runner=None,
) -> dict:
    """Execute ``group_size`` isolated rollouts on ``task`` and compute GRPO advantages."""
    if isinstance(group_size, bool) or not isinstance(group_size, int) or not (1 <= group_size <= 16):
        raise GuardError('invalid_group_size')
    validated = validate_task(dict(task))
    resolved_state = Path(state_dir) if state_dir is not None else STATE
    model_cycle = [m for m in (models or []) if m]
    tracked = list(validated['files']) + (
        [validated['patch_file']] if validated['mode'] == 'patch' else []
    )
    originals = _snapshot_file_bytes(tracked)

    def _default_runner(rollout_task):
        fd, tmp_path = tempfile.mkstemp(prefix='rollout_', suffix='.json')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as fh:
                json.dump(rollout_task, fh)
            return run_task(tmp_path, state=resolved_state, _quiet=True, _return_result=True)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    active_runner = _default_runner if runner is None else runner
    rollouts = []
    snapshots = []

    for rollout_index in range(group_size):
        restore_files(originals)
        rollout_model = (
            model_cycle[rollout_index % len(model_cycle)]
            if model_cycle
            else validated.get('model')
        )
        rollout_task = dict(validated)
        rollout_task['task_id'] = validated['task_id']
        rollout_task['rollout_index'] = rollout_index
        rollout_task['strategy'] = 'single'
        rollout_task['auto_escalate'] = False
        rollout_task.pop('candidate_models', None)
        if rollout_model is not None and rollout_task['mode'] != 'command':
            rollout_task['model'] = rollout_model

        try:
            res = active_runner(rollout_task)
            if not isinstance(res, dict):
                res = {'status': 'blocked', 'reason': 'invalid_runner_result'}
        except Exception as exc:  # noqa: BLE001
            res = {
                'status': 'blocked',
                'reason': str(exc) if isinstance(exc, GuardError) else type(exc).__name__,
            }

        after_snap = _snapshot_file_bytes(tracked)
        snapshots.append(after_snap)
        patch = _diff_file_snapshots(originals, after_snap)
        artifact_path = res.get('artifact')
        if not patch and artifact_path:
            patch_file = Path(artifact_path) / 'changes.patch'
            if patch_file.is_file():
                patch = patch_file.read_text(encoding='utf-8', errors='replace')

        status_str = str(res.get('status') or '').strip().lower()
        passed = res.get('passed') is True or status_str in _ROLLOUT_PASS_STATUSES
        raw_steps = res.get('steps')
        steps = int(raw_steps) if raw_steps is not None else (1 if passed else 0)
        guard_interventions = int(res.get('guard_interventions') or 0)
        sql_or_egress_violations = max(
            int(res.get('sql_or_egress_violations') or 0),
            int(res.get('sql_violations') or 0) + int(res.get('egress_violations') or 0),
        )
        r_process = max(0.0, 1.0 - 0.25 * guard_interventions - 0.5 * sql_or_egress_violations)
        evidence_ok = (
            bool(patch.strip())
            or res.get('test_return_code') == 0
            or bool(res.get('evidence_ok'))
        )
        r_evidence = 1.0 if steps > 0 and evidence_ok else 0.0
        r_outcome = 1.0 if passed else 0.0
        r_total = round(r_outcome * r_evidence * r_process, 6)
        run_id = (
            str(res['run_id'])
            if res.get('run_id')
            else (
                Path(artifact_path).name
                if artifact_path
                else f"{validated['task_id']}-rollout-{rollout_index}"
            )
        )

        rollouts.append({
            'task_id': validated['task_id'],
            'rollout_index': rollout_index,
            'run_id': run_id,
            'model': rollout_model,
            'status': res.get('status'),
            'passed': passed,
            'steps': steps,
            'cost': res.get('cost'),
            'salvaged_from': res.get('salvaged_from'),
            'artifact': artifact_path,
            'patch': patch,
            'r_outcome': r_outcome,
            'r_evidence': r_evidence,
            'r_process': r_process,
            'r_total': r_total,
        })

    rewards = [entry['r_total'] for entry in rollouts]
    mean_r = sum(rewards) / len(rewards)
    std_r = math.sqrt(sum((r - mean_r) ** 2 for r in rewards) / len(rewards))
    advantages = [(r - mean_r) / (std_r + 1e-6) for r in rewards]
    for entry, adv in zip(rollouts, advantages):
        entry['advantage'] = round(adv, 6)

    passing = [
        (entry, snap)
        for entry, snap in zip(rollouts, snapshots)
        if entry['passed']
    ]
    if passing:
        best_entry, best_snap = min(
            passing,
            key=lambda pair: (
                -pair[0]['r_total'],
                1 if pair[0].get('salvaged_from') else 0,
                pair[0]['steps'],
                float('inf') if pair[0].get('cost') is None else pair[0]['cost'],
                pair[0]['rollout_index'],
            ),
        )
        restore_files(best_snap)
        best_run_id = best_entry['run_id']
    else:
        restore_files(originals)
        best_run_id = None

    pass_count = len(passing)
    return {
        'task_id': validated['task_id'],
        'group_size': group_size,
        'pass_count': pass_count,
        'pass_rate': round(pass_count / group_size, 6),
        'mean_reward': round(mean_r, 6),
        'std_reward': round(std_r, 6),
        'best_run_id': best_run_id,
        'rollouts': rollouts,
    }

def validate_plan(raw_plan):
    """H-66: validate and normalize a parallel worktree DAG plan."""
    if not isinstance(raw_plan, dict):
        raise GuardError('plan_must_be_an_object')
    plan_id = raw_plan.get('plan_id')
    if not isinstance(plan_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', plan_id):
        raise GuardError('invalid_plan_id')
    directory = raw_plan.get('directory')
    if not isinstance(directory, str) or not Path(directory).is_absolute() \
            or not Path(directory).is_dir():
        raise GuardError('invalid_working_directory')
    directory = str(Path(directory).resolve())
    max_parallel = raw_plan.get('max_parallel', 2)
    if isinstance(max_parallel, bool) or not isinstance(max_parallel, int) \
            or not (1 <= max_parallel <= 4):
        raise GuardError('invalid_max_parallel')
    final_test_argv = raw_plan.get('final_test_argv')
    if final_test_argv is not None:
        final_test_argv = executable_argv(final_test_argv)
    nodes = raw_plan.get('nodes')
    if not isinstance(nodes, list) or not nodes or len(nodes) > 8 \
            or not all(isinstance(n, dict) for n in nodes):
        raise GuardError('invalid_plan_nodes')
    normalized = []
    seen_ids = set()
    for node in nodes:
        node_copy = dict(node)
        node_copy.setdefault('directory', directory)
        depends_on = node_copy.get('depends_on', [])
        if not isinstance(depends_on, list) or not all(isinstance(d, str) for d in depends_on):
            raise GuardError('invalid_depends_on')
        node_copy['depends_on'] = list(depends_on)
        validate_task(node_copy)
        task_id = node_copy.get('task_id')
        if task_id in seen_ids:
            raise GuardError('duplicate_plan_task_id')
        seen_ids.add(task_id)
        normalized.append(node_copy)
    node_map = {n['task_id']: n for n in normalized}
    for node in normalized:
        for dep in node['depends_on']:
            if dep not in node_map:
                raise GuardError('unknown_plan_dependency')
    remaining = {n['task_id']: set(n['depends_on']) for n in normalized}
    waves = []
    placed = set()
    by_id = {n['task_id']: n for n in normalized}
    while remaining:
        ready = [tid for tid, deps in remaining.items() if deps <= placed]
        if not ready:
            raise GuardError('cyclic_plan_dependency')
        waves.append([by_id[tid] for tid in ready])
        placed.update(ready)
        for tid in ready:
            del remaining[tid]
    if len(waves) < len(nodes) and sum(len(w) for w in waves) < len(nodes):
        raise GuardError('cyclic_plan_dependency')
    for wave in waves:
        if len(wave) > 1:
            for i in range(len(wave)):
                for j in range(i + 1, len(wave)):
                    shared = set(wave[i]['files']) & set(wave[j]['files'])
                    if shared:
                        raise GuardError('parallel_nodes_share_editable_files')
    return {
        'plan_id': plan_id,
        'directory': directory,
        'nodes': normalized,
        'waves': waves,
        'max_parallel': max_parallel,
        'final_test_argv': final_test_argv,
    }

_PLAN_SUCCESS_STATUSES = ('validation_passed_review_required',
                          'patch_ready_review_required', 'succeeded')

def _sync_plan_artifacts(node_id, res, plan_dir, node_files=()):
    """Sync un-applied changes.patch / after/ files from a worktree run into the plan directory.

    ``res['artifact']`` may be either:
    - a dict with a ``changes.patch`` text key and an ``after`` mapping of
      relative path -> content; or
    - a str/Path directory containing a ``changes.patch`` file and an
      ``after/`` directory whose files are matched to ``node_files`` by
      relative path or filename.

    Returns None on success or a reason string when git apply reports a conflict.
    """
    artifact = res.get('artifact')
    after_items = []
    if isinstance(artifact, (str, Path)):
        artifact_dir = Path(artifact)
        patch_path = artifact_dir / 'changes.patch'
        patch = patch_path.read_text(encoding='utf-8', errors='replace') \
            if patch_path.is_file() else None
        after_dir = artifact_dir / 'after'
        if after_dir.is_dir():
            node_lookup = {}
            for nf in node_files or ():
                p = Path(nf)
                node_lookup[p.name] = p
                node_lookup[str(p)] = p
                try:
                    node_lookup[str(p.resolve())] = p
                except OSError:
                    pass
            for candidate in sorted(after_dir.rglob('*')):
                if not candidate.is_file():
                    continue
                rel = candidate.relative_to(after_dir)
                target = node_lookup.get(str(rel)) or node_lookup.get(rel.name)
                if target is None and len(set(node_files or ())) == 1:
                    target = Path(list(dict.fromkeys(node_files or ()))[0])
                if target is None:
                    target = Path(plan_dir) / str(rel)
                after_items.append((str(target),
                                    candidate.read_text(encoding='utf-8',
                                                        errors='replace')))
    else:
        artifact = artifact or {}
        patch = artifact.get('changes.patch')
        after = artifact.get('after') or {}
        for path, content in after.items():
            after_items.append((str(Path(plan_dir) / path), content))
    for path, content in after_items:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and target.read_text() == content:
            continue
        target.write_text(content)
    if patch:
        proc = subprocess.run(['git', 'apply', '--check', '-'],
                              input=patch, cwd=plan_dir, capture_output=True,
                              text=True)
        if proc.returncode == 0:
            proc = subprocess.run(['git', 'apply', '-'],
                                  input=patch, cwd=plan_dir, capture_output=True,
                                  text=True)
            if proc.returncode != 0:
                return 'worktree_patch_conflict'
        else:
            reverse = subprocess.run(['git', 'apply', '--reverse', '--check', '-'],
                                     input=patch, cwd=plan_dir, capture_output=True,
                                     text=True)
            if reverse.returncode != 0:
                return 'worktree_patch_conflict'
    return None

def execute_plan(raw_plan, *, runner=execute, state=None):
    """H-66: execute a validated parallel worktree DAG plan wave by wave."""
    plan = validate_plan(raw_plan)
    status = 'succeeded'
    reason = None
    failed_node = None
    final_test_returncode = None
    node_results = []

    def _run_plan_node(node):
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(prefix='plan_node_', suffix='.json')
            with os.fdopen(fd, 'w') as fh:
                json.dump(node, fh)
            return run_task(tmp_path, state=state, _quiet=True, _return_result=True)
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    actual_runner = _run_plan_node if runner is execute else runner

    for wave_idx, wave in enumerate(plan['waves']):
        try:
            report_herdr_task(f"{plan['plan_id']}/wave-{wave_idx}", 'working',
                              step=wave_idx, max_steps=len(plan['waves']),
                              phase=f"wave {wave_idx + 1}/{len(plan['waves'])} ({len(wave)} nodes)")
        except Exception:
            pass
        results = {}
        if len(wave) == 1:
            try:
                results[wave[0]['task_id']] = actual_runner(dict(wave[0]))
            except Exception as exc:  # noqa: BLE001 - surfaced as blocked node
                results[wave[0]['task_id']] = {
                    'task_id': wave[0]['task_id'], 'status': 'blocked',
                    'reason': str(exc) if isinstance(exc, GuardError) else type(exc).__name__}
        else:
            parallel_nodes = []
            for node in wave:
                node_copy = dict(node)
                node_copy['worktree'] = True
                parallel_nodes.append(node_copy)
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=plan['max_parallel']) as pool:
                futures = {pool.submit(actual_runner, dict(n)): n
                           for n in parallel_nodes}
                for fut, node in futures.items():
                    try:
                        results[node['task_id']] = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        results[node['task_id']] = {
                            'task_id': node['task_id'], 'status': 'blocked',
                            'reason': str(exc) if isinstance(exc, GuardError)
                            else type(exc).__name__}
        for node in wave:
            res = dict(results.get(node['task_id']) or {})
            res.setdefault('task_id', node['task_id'])
            res.setdefault('wave', wave_idx)
            if 'wave' not in res:
                res['wave'] = wave_idx
            node_results.append(res)
        first_failed = None
        failed_node_id = None
        for node in wave:
            res = results.get(node['task_id']) or {}
            if res.get('status') == 'blocked' \
                    or res.get('status') not in _PLAN_SUCCESS_STATUSES:
                first_failed = res
                failed_node_id = node['task_id']
                break
        if first_failed is not None:
            status = 'blocked'
            failed_node = failed_node_id or first_failed.get('task_id')
            reason = first_failed.get('reason') or 'node_failed'
            for rem_idx, rem_wave in enumerate(plan['waves'][wave_idx + 1:], wave_idx + 1):
                for n in rem_wave:
                    node_results.append({'task_id': n['task_id'],
                                         'status': 'skipped_upstream_failure',
                                         'wave': rem_idx})
            break
        for node in wave:
            res = results.get(node['task_id']) or {}
            conflict = _sync_plan_artifacts(node['task_id'], res,
                                            plan['directory'],
                                            node_files=node.get('files', ()))
            if conflict:
                    status = 'blocked'
                    reason = conflict
                    failed_node = node['task_id']
                    for rem_idx, rem_wave in enumerate(plan['waves'][wave_idx + 1:],
                                                       wave_idx + 1):
                        for n in rem_wave:
                            node_results.append({'task_id': n['task_id'],
                                                 'status': 'skipped_upstream_failure',
                                                 'wave': rem_idx})
                    break
        if status == 'blocked':
            break

    if status == 'succeeded' and plan.get('final_test_argv') is not None:
        try:
            proc = subprocess.run(plan['final_test_argv'], cwd=plan['directory'],
                                  capture_output=True, text=True, timeout=120)
            final_test_returncode = proc.returncode
            if proc.returncode != 0:
                status = 'blocked'
                reason = 'final_validation_failed'
        except (OSError, subprocess.TimeoutExpired) as exc:
            final_test_returncode = None
            status = 'blocked'
            reason = 'final_validation_failed'
            del exc

    total_cost = round(sum((r.get('cost') or 0.0) for r in node_results), 8)
    total_steps = sum((r.get('steps') or 0) for r in node_results)
    try:
        report_herdr_task(plan['plan_id'], 'done' if status == 'succeeded' else 'blocked',
                          step=len(plan['waves']), max_steps=len(plan['waves']),
                          reason=reason)
    except Exception:
        pass
    return {
        'plan_id': plan['plan_id'],
        'status': status,
        'reason': reason,
        'failed_node': failed_node,
        'waves': [[n['task_id'] for n in w] for w in plan['waves']],
        'node_results': node_results,
        'total_cost': total_cost,
        'total_steps': total_steps,
        'final_test_returncode': final_test_returncode,
    }

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
