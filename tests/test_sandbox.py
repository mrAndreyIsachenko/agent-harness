import os
import sys
import unittest

from harness.sandbox import (config_for, infer_read_only_companion_files,
                             is_seatbelt_available, render_guard_plugin,
                             stage_guard_plugin, verify_permissions, worker_env)


def make_task(directory, files):
    return {'task_id': 't1', 'directory': str(directory), 'files': [str(f) for f in files],
            'mode': 'code', 'model': 'openrouter/z-ai/glm-5.3-flash',
            'max_steps': 16, 'prompt': 'do work'}


class IsolationSandboxTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.run_dir = os.path.realpath(self._td.name)

    def test_sandbox_root_path_is_short(self):
        from harness.sandbox import sandbox_root_for
        root = sandbox_root_for(self.run_dir)
        self.assertLess(len(str(root)), 30)
        self.assertTrue(str(root).startswith('/tmp/hb-'))

    def test_stage_creates_dirs_and_env(self):
        from harness.sandbox import stage_isolation_sandbox
        env = stage_isolation_sandbox(self.run_dir)
        for key, sub in (('TMPDIR', 'tmp'), ('PYTHONUSERBASE', 'python-user'),
                         ('NPM_CONFIG_PREFIX', 'npm-global')):
            self.assertIn(f'/{sub}', env[key])
            self.assertTrue(os.path.isdir(env[key]))

    def test_cleanup_removes_sandbox(self):
        from harness.sandbox import cleanup_isolation_sandbox, sandbox_root_for, stage_isolation_sandbox
        stage_isolation_sandbox(self.run_dir)
        self.assertTrue(sandbox_root_for(self.run_dir).is_dir())
        cleanup_isolation_sandbox(self.run_dir)
        self.assertFalse(sandbox_root_for(self.run_dir).exists())


def rules_for(task):
    return config_for(task)['agents']['guarded_worker']['permissions']


def effective_effect(rules, action, resource):
    matching = [r for r in rules
                if r.get('action') in (action, '*')
                and _fnmatch(resource, r.get('resource', ''))]
    if not matching:
        return None
    return matching[-1].get('effect')


def _fnmatch(resource, pattern):
    import fnmatch
    return fnmatch.fnmatchcase(resource, pattern)


class ConfigForTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.task_dir = os.path.realpath(self._td.name)
        self.main_file = os.path.join(self.task_dir, 'src.py')
        with open(self.main_file, 'w') as fh:
            fh.write('x = 1\n')
        self.task = make_task(self.task_dir, [self.main_file])

    def test_sibling_files_readable_but_edit_restricted(self):
        sibling = os.path.join(self.task_dir, 'sibling.txt')
        rules = rules_for(self.task)
        self.assertEqual(effective_effect(rules, 'read', sibling), 'allow')
        self.assertEqual(effective_effect(rules, 'read', self.task_dir), 'allow')
        self.assertEqual(effective_effect(rules, 'edit', sibling), 'deny')
        self.assertEqual(effective_effect(rules, 'edit', self.main_file), 'allow')

    def test_sensitive_files_denied_for_read_and_edit(self):
        rules = rules_for(self.task)
        names = ['.env', '.env.local', 'key.pem', 'priv.key',
                 'credentials.json', 'my_secret.txt']
        for name in names:
            direct = os.path.join(self.task_dir, name)
            nested = os.path.join(self.task_dir, 'sub', name)
            for path in (direct, nested):
                self.assertEqual(effective_effect(rules, 'read', path), 'deny',
                                 f'read should be denied for {path}')
                self.assertEqual(effective_effect(rules, 'edit', path), 'deny',
                                 f'edit should be denied for {path}')

    def test_safe_env_templates_readable(self):
        rules = rules_for(self.task)
        for name in ('.env.example', '.env.sample', '.env.template'):
            direct = os.path.join(self.task_dir, name)
            nested = os.path.join(self.task_dir, 'sub', name)
            for path in (direct, nested):
                self.assertEqual(effective_effect(rules, 'read', path), 'allow',
                                 f'read should be allowed for {path}')

    def test_snapshot_disabled(self):
        self.assertIs(config_for(self.task).get('snapshot'), False)

    def test_verify_permissions_accepts_generated_config(self):
        payload = {'data': [{'id': 'guarded_worker',
                             'steps': self.task['max_steps'],
                             'permissions': rules_for(self.task)}]}
        verify_permissions(payload, self.task)

    def test_config_for_allows_isolation_sandbox_reads(self):
        rules = rules_for(self.task)
        self.assertEqual(effective_effect(rules, 'read', '/tmp/hb-abc123def456/tmp/foo.txt'), 'allow')


class WorkerEnvTests(unittest.TestCase):
    def test_default_output_token_max(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            env = worker_env(td)
            self.assertEqual(env['OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX'], '64000')
            self.assertEqual(env['PWD'], str(td))

    def test_caller_override_preserved(self):
        import tempfile
        saved = os.environ.get('OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX')
        try:
            os.environ['OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX'] = '1234'
            with tempfile.TemporaryDirectory() as td:
                env = worker_env(td)
                self.assertEqual(env['OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX'], '1234')
        finally:
            if saved is None:
                os.environ.pop('OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX', None)
            else:
                os.environ['OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX'] = saved


class GuardPluginTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.task_dir = os.path.realpath(self._td.name)
        self.main_file = os.path.join(self.task_dir, 'src.py')
        with open(self.main_file, 'w') as fh:
            fh.write('x = 1\n')
        self.task = make_task(self.task_dir, [self.main_file])

    def test_render_guard_plugin_contents(self):
        src = render_guard_plugin()
        self.assertIn('tool.execute.after', src)
        self.assertIn('py_compile', src)
        self.assertIn('JSON.parse', src)
        self.assertIn('[HARNESS EDIT HINT]', src)
        self.assertIn('id: "harness.guard"', src)
        self.assertIn('server: harnessGuardPlugin', src)

    def test_stage_guard_plugin_creates_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path = stage_guard_plugin(td)
            self.assertEqual(path, __import__('pathlib').Path(td) / '.opencode' / 'plugin' / 'harness_guard.ts')
            self.assertTrue(path.is_file())
            self.assertIn('tool.execute.after', path.read_text())

    def test_config_for_with_run_dir_registers_plugin_uri(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            cfg = config_for(self.task, run_dir=Path(td))
            plugins = cfg.get('plugin')
            self.assertIsInstance(plugins, list)
            expected = str((Path(td) / '.opencode' / 'plugin').resolve())
            self.assertIn(expected, plugins)
            self.assertTrue((Path(td) / '.opencode' / 'plugin' / 'harness_guard.ts').is_file())

    def test_live_opencode_cli_loads_guard_plugin_without_load_error(self):
        import json
        import tempfile
        import time
        from pathlib import Path
        cli = '/Applications/OpenCode.app/Contents/Resources/opencode-cli'
        log_dir = Path.home() / '.local/share/opencode/log'
        if not Path(cli).is_file():
            self.skipTest('live opencode-cli not present')
        if sys.platform == 'darwin' and not is_seatbelt_available():
            self.skipTest('running inside outer Seatbelt sandbox; cannot write '
                          '~/.local/share/opencode/log')
        import subprocess
        sizes_before = (
            {p: p.stat().st_size for p in log_dir.glob('*.log')}
            if log_dir.is_dir() else {}
        )
        with tempfile.TemporaryDirectory() as td:
            td_real = Path(os.path.realpath(td))
            cfg = config_for(self.task, run_dir=td_real)
            cfg_path = td_real / 'opencode.json'
            cfg_path.write_text(
                json.dumps({'$schema': 'https://opencode.ai/config.json', 'plugin': cfg['plugin']}),
                encoding='utf-8')
            proc = subprocess.run([cli, 'debug', 'agents', '--standalone', '--print-logs'],
                                  cwd=str(td_real),
                                  env={**os.environ, 'OPENCODE_CONFIG': str(cfg_path)},
                                  capture_output=True, timeout=60)
            time.sleep(0.3)
            parts = []
            for stream in (proc.stdout, proc.stderr):
                if isinstance(stream, bytes):
                    parts.append(stream.decode('utf-8', errors='replace'))
                elif stream:
                    parts.append(stream)
            if log_dir.is_dir():
                for p in log_dir.glob('*.log'):
                    before = sizes_before.get(p, 0)
                    if p.stat().st_size > before:
                        parts.append(
                            p.read_bytes()[before:].decode('utf-8', errors='replace'))
            output = '\n'.join(parts)
            self.assertIn('harness_guard.ts', output)
            self.assertNotIn('PluginModule.LoadError', output)
            self.assertNotIn('configured plugin path must be a directory', output)

    def test_config_for_without_run_dir_backward_compatible(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            task = make_task(td, [td + '/src.py'])
            cfg = config_for(task)
            self.assertNotIn('plugin', cfg)
            self.assertIn('agents', cfg)


class InferReadOnlyCompanionTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.task_dir = os.path.realpath(self._td.name)

    def _write(self, rel):
        path = os.path.join(self.task_dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as fh:
            fh.write('\n')
        return path

    def test_conftest_ancestors_included_for_test_targets(self):
        test_file = self._write('tests/test_mod.py')
        conftest = self._write('tests/conftest.py')
        conftest_top = self._write('conftest.py')
        mod = self._write('mod.py')
        result = infer_read_only_companion_files([test_file], directory=self.task_dir)
        self.assertIn(conftest, result)
        self.assertIn(conftest_top, result)
        self.assertIn(mod, result)

    def test_conftest_respects_max_files_and_seen(self):
        test_file = self._write('tests/test_mod.py')
        conftest = self._write('tests/conftest.py')
        self._write('conftest.py')
        result = infer_read_only_companion_files([test_file], directory=self.task_dir, max_files=1)
        self.assertEqual(result, [conftest])

    def test_non_test_files_get_no_conftest(self):
        src = self._write('src.py')
        self._write('conftest.py')
        result = infer_read_only_companion_files([src], directory=self.task_dir)
        self.assertEqual(result, [])

    def test_conftest_outside_directory_excluded(self):
        import tempfile
        outer = os.path.realpath(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__('shutil').rmtree(outer, ignore_errors=True))
        inner = os.path.join(outer, 'proj')
        os.makedirs(os.path.join(inner, 'tests'), exist_ok=True)
        test_file = os.path.join(inner, 'tests', 'test_mod.py')
        with open(test_file, 'w') as fh:
            fh.write('\n')
        with open(os.path.join(inner, 'tests', 'conftest.py'), 'w') as fh:
            fh.write('\n')
        with open(os.path.join(outer, 'conftest.py'), 'w') as fh:
            fh.write('\n')
        result = infer_read_only_companion_files([test_file], directory=inner)
        self.assertIn(os.path.join(inner, 'tests', 'conftest.py'), result)
        self.assertNotIn(os.path.join(outer, 'conftest.py'), result)


class GuardPluginFilesHintTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.task_dir = os.path.realpath(self._td.name)
        self.main_file = os.path.join(self.task_dir, 'src.py')
        with open(self.main_file, 'w') as fh:
            fh.write('x = 1\n')

    def test_render_with_files_lists_editable_and_read_only(self):
        ro = os.path.join(self.task_dir, 'ro.py')
        src = render_guard_plugin(allowed_files=[self.main_file], read_only_files=[ro])
        self.assertIn('[HARNESS READ GUARD]', src)
        self.assertIn(self.main_file, src)
        self.assertIn(ro, src)
        self.assertIn('[HARNESS EDIT HINT]', src)
        self.assertIn('indexOf', src)

    def test_render_without_files_has_generic_hint(self):
        src = render_guard_plugin()
        self.assertIn('[HARNESS READ GUARD]', src)
        self.assertIn('Use only the allowed files specified in the prompt', src)
        self.assertIn('[HARNESS EDIT HINT]', src)

    def test_config_for_forwards_files_to_staged_plugin(self):
        ro = os.path.join(self.task_dir, 'ro.py')
        with open(ro, 'w') as fh:
            fh.write('\n')
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            task = make_task(self.task_dir, [self.main_file])
            task['read_only_files'] = [ro]
            cfg = config_for(task, run_dir=Path(td))
            staged = (Path(td) / '.opencode' / 'plugin' / 'harness_guard.ts').read_text()
            self.assertIn(self.main_file, staged)
            self.assertIn(ro, staged)
            self.assertIn('[HARNESS READ GUARD]', staged)
            self.assertIn('plugin', cfg)

    def test_render_includes_h85_h86_markers(self):
        src = render_guard_plugin(allowed_files=[self.main_file])
        self.assertIn('[HARNESS AUTO-APPLIED NORMALIZED EDIT]', src)
        self.assertIn('[HARNESS IN-LOOP SEMANTIC LINT]', src)

    def test_stage_guard_plugin_forwards_files(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            stage_guard_plugin(td, allowed_files=[self.main_file])
            staged = Path(td) / '.opencode' / 'plugin' / 'harness_guard.ts'
            self.assertIn(self.main_file, staged.read_text())


class CheckPythonUndefinedNamesTests(unittest.TestCase):
    def test_catches_undefined_name_missing_import_and_typo(self):
        from harness.guards import check_python_undefined_names
        self.assertTrue(any('os' in msg for msg in check_python_undefined_names('os.path.join("a", "b")\n')))
        src = 'def f():\n    return re.compile("x")\n'
        self.assertTrue(any('re' in msg for msg in check_python_undefined_names(src)))
        src = 'value = 1\nprint(valuee)\n'
        self.assertTrue(any('valuee' in msg for msg in check_python_undefined_names(src)))

    def test_allows_valid_names(self):
        from harness.guards import check_python_undefined_names
        valid = (
            # local + builtins
            'def f(x):\n    y = x + 1\n    return len(str(y))\n',
            # enclosing scope
            'def outer():\n    a = 1\n    def inner():\n        return a\n    return inner()\n',
            # plain import + from import
            'import json\nfrom pathlib import Path\nprint(json.dumps({}), Path("."))\n',
        )
        for src in valid:
            self.assertEqual(check_python_undefined_names(src), [], src)

    def test_ignores_type_checking_and_star_and_future_annotations(self):
        from harness.guards import check_python_undefined_names
        src = 'from __future__ import annotations\nfrom typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from collections import OrderedDict\n\ndef f(x: OrderedDict) -> None:\n    return None\n'
        self.assertEqual(check_python_undefined_names(src), [])
        src = 'from os import *\ndef f():\n    return getcwd()\n'
        self.assertEqual(check_python_undefined_names(src), [])


class ConcurrentReadOnlyCompanionTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.root = Path(os.path.realpath(self._td.name))
        self.repo = self.root / 'repo'
        self.state = self.root / 'state'
        self.repo.mkdir()
        (self.repo / 'tests').mkdir()
        self.state.mkdir()
        self.test_file = self.repo / 'tests' / 'test_mod.py'
        self.mod_file = self.repo / 'mod.py'
        self.test_file.write_text('def test_ok():\n    assert True\n', encoding='utf-8')
        self.mod_file.write_text('VALUE = 1\n', encoding='utf-8')

    def _write_sibling_run(self, run_name, *, task_id, status, files, started_at=None, duration_seconds=None):
        import json
        import time
        run_dir = self.state / 'runs' / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        res = {
            'task_id': task_id,
            'status': status,
            'started_at': time.time() if started_at is None else started_at,
        }
        if duration_seconds is not None:
            res['duration_seconds'] = duration_seconds
        (run_dir / 'result.json').write_text(json.dumps(res), encoding='utf-8')
        (run_dir / 'task.json').write_text(json.dumps({'task_id': task_id, 'files': [str(f) for f in files]}),
                                           encoding='utf-8')
        return run_dir

    def _run_model_task(self, task_dict, model_side_effect):
        import json
        import sys
        from contextlib import nullcontext
        from unittest import mock
        from harness.sandbox import run_task

        task_path = self.root / 'task.json'
        task_path.write_text(json.dumps(task_dict), encoding='utf-8')
        with mock.patch('harness.sandbox.preflight'), \
                mock.patch('harness.sandbox.verify_db'), \
                mock.patch('harness.sandbox.openrouter_proxy', return_value=nullcontext(None)), \
                mock.patch('harness.sandbox.snapshot', return_value=None), \
                mock.patch('harness.sandbox.model_run', side_effect=model_side_effect):
            return run_task(
                task_path,
                cli='/usr/bin/true',
                db=self.root / 'db.sqlite',
                state=self.state,
                _quiet=True,
                _return_result=True,
            )

    def test_validate_task_preserves_explicit_read_only_files(self):
        from harness.sandbox import validate_task
        explicit = self.repo / 'explicit.py'
        explicit.write_text('E = 1\n', encoding='utf-8')
        task = validate_task({
            'task_id': 't_explicit',
            'directory': str(self.repo),
            'files': [str(self.test_file)],
            'read_only_files': [str(explicit)],
            'test_argv': ['/usr/bin/true'],
            'mode': 'model',
            'prompt': 'edit test',
        })
        self.assertEqual(task['explicit_read_only_files'], [str(explicit)])
        self.assertIn(str(self.mod_file), task['read_only_files'])
        self.assertIn(str(explicit), task['read_only_files'])

    def test_active_pending_sibling_editable_file_excluded_from_auto_inferred_ro(self):
        import json
        from pathlib import Path
        self._write_sibling_run(
            'sib_pending',
            task_id='t_sib',
            status='pending',
            files=[self.mod_file],
        )

        def fake_model(cli, db, task, run_dir, result, budget, **kwargs):
            Path(task['files'][0]).write_text('def test_ok():\n    assert 1 == 1\n', encoding='utf-8')
            result['cost'] = 0.0
            result['steps'] = 1

        res = self._run_model_task({
            'task_id': 't_main',
            'directory': str(self.repo),
            'files': [str(self.test_file)],
            'test_argv': ['/usr/bin/true'],
            'mode': 'model',
            'prompt': 'edit test',
        }, fake_model)
        self.assertEqual(res['status'], 'validation_passed_review_required')
        recorded_task = json.loads((Path(res['artifact']) / 'task.json').read_text(encoding='utf-8'))
        self.assertNotIn(str(self.mod_file), recorded_task['read_only_files'])

    def test_auto_inferred_ro_modified_by_concurrent_sibling_is_allowed(self):
        import json
        import time
        from pathlib import Path

        def fake_model(cli, db, task, run_dir, result, budget, **kwargs):
            Path(task['files'][0]).write_text('def test_ok():\n    assert 2 == 2\n', encoding='utf-8')
            # Simulate a concurrent sibling run starting mid-run and editing mod.py
            self._write_sibling_run(
                'sib_concurrent',
                task_id='t_sib_concurrent',
                status='pending',
                files=[self.mod_file],
                started_at=time.time(),
            )
            self.mod_file.write_text('VALUE = 2\n', encoding='utf-8')
            result['cost'] = 0.0
            result['steps'] = 1

        res = self._run_model_task({
            'task_id': 't_main_concurrent',
            'directory': str(self.repo),
            'files': [str(self.test_file)],
            'test_argv': ['/usr/bin/true'],
            'mode': 'model',
            'prompt': 'edit test',
        }, fake_model)
        self.assertEqual(res['status'], 'validation_passed_review_required')
        self.assertEqual(self.mod_file.read_text(encoding='utf-8'), 'VALUE = 2\n')
        events_path = Path(res['artifact']) / 'events.jsonl'
        events = [json.loads(line) for line in events_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        self.assertTrue(
            any(ev.get('event') == 'concurrent_sibling_read_only_change_ignored'
                and ev.get('file') == str(self.mod_file) for ev in events),
            f'expected concurrent_sibling_read_only_change_ignored in {events}',
        )

    def test_explicit_ro_modified_during_run_still_blocked(self):
        import time
        from pathlib import Path

        def fake_model(cli, db, task, run_dir, result, budget, **kwargs):
            Path(task['files'][0]).write_text('def test_ok():\n    assert 3 == 3\n', encoding='utf-8')
            self._write_sibling_run(
                'sib_concurrent_explicit',
                task_id='t_sib_explicit',
                status='pending',
                files=[self.mod_file],
                started_at=time.time(),
            )
            self.mod_file.write_text('VALUE = 3\n', encoding='utf-8')
            result['cost'] = 0.0
            result['steps'] = 1

        res = self._run_model_task({
            'task_id': 't_main_explicit',
            'directory': str(self.repo),
            'files': [str(self.test_file)],
            'read_only_files': [str(self.mod_file)],
            'test_argv': ['/usr/bin/true'],
            'mode': 'model',
            'prompt': 'edit test',
        }, fake_model)
        self.assertEqual(res['status'], 'blocked')
        self.assertEqual(res['reason'], 'read_only_file_modified')

    def test_read_only_public_skeletons_for_test_file_task(self):
        import ast
        from harness.guards import build_read_only_public_skeletons

        filler = "\n".join(f"# filler line {i}" for i in range(150))
        self.mod_file.write_text(
            'from dataclasses import dataclass\n\n'
            'class RefModel:\n'
            '    name: str\n'
            '    value: int\n'
            '    _private: str\n\n'
            '    def __init__(self, name: str, value: int) -> None:\n'
            '        self.name = name\n\n'
            '    def public_method(self, x: int = 0) -> int:\n'
            '        return x\n\n'
            '    def _helper(self) -> None:\n'
            '        return None\n\n\n'
            'async def public_async(s: str) -> str:\n'
            '    return s\n\n'
            'def public_fn(a, b="x"):\n'
            '    return a\n\n'
            'def _private_fn(a):\n'
            '    return a\n\n'
            + filler + '\n',
            encoding='utf-8',
        )

        skeleton = build_read_only_public_skeletons([str(self.mod_file)])
        self.assertTrue(skeleton.startswith('[READ-ONLY PUBLIC SIGNATURES]'))
        self.assertIn(f'--- {self.mod_file} (', skeleton)
        self.assertIn('class RefModel(', skeleton)
        self.assertIn('    name: str', skeleton)
        self.assertIn('    value: int', skeleton)
        self.assertNotIn('_private: str', skeleton)
        self.assertIn('def __init__(self, name: str, value: int', skeleton)
        self.assertIn('def public_method(self, x: int', skeleton)
        self.assertNotIn('_helper', skeleton)
        self.assertIn('async def public_async(s: str)', skeleton)
        self.assertIn('def public_fn(a, b)', skeleton)
        self.assertNotIn('_private_fn', skeleton)

        prompts = []

        def fake_model(cli, db, task, run_dir, result, budget, **kwargs):
            prompts.append(task['prompt'])
            self.test_file.write_text(
                'def test_ok():\n    assert 1 == 1\n', encoding='utf-8'
            )
            result['cost'] = 0.0
            result['steps'] = 1

        self.test_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.test_file.exists():
            self.test_file.write_text('def test_placeholder(): pass\n', encoding='utf-8')
        res = self._run_model_task({
            'task_id': 't_ro_skeleton',
            'directory': str(self.repo),
            'files': [str(self.test_file)],
            'read_only_files': [str(self.mod_file)],
            'test_argv': ['/usr/bin/true'],
            'mode': 'model',
            'prompt': 'edit test only',
        }, fake_model)
        self.assertEqual(res['status'], 'validation_passed_review_required')
        self.assertEqual(len(prompts), 1)

    def test_build_read_only_public_skeletons_includes_return_annotations(self):
        from harness.guards import build_read_only_public_skeletons

        self.mod_file.write_text(
            'class WidgetSpec:\n'
            '    def names(self) -> list[str]:\n'
            '        return []\n\n'
            'def compute_widget(count: int) -> dict[str, int]:\n'
            '    return {}\n',
            encoding='utf-8',
        )

        skeleton = build_read_only_public_skeletons([str(self.mod_file)])
        self.assertTrue(skeleton.startswith('[READ-ONLY PUBLIC SIGNATURES]'))
        self.assertIn('def compute_widget(count: int) -> dict[str, int]', skeleton)
        self.assertIn('def names(self) -> list[str]', skeleton)

    def test_salvage_micro_repair_succeeds_after_step_limit(self):
        import json
        import sys
        from pathlib import Path
        from harness.sandbox import GuardError

        self.test_file.write_text(
            'from mod import VALUE\n'
            'def test_val():\n'
            '    assert VALUE == 2\n',
            encoding='utf-8',
        )
        calls = []

        def fake_model_run(cli, db, task, run_dir, result, budget, **kwargs):
            calls.append(task['prompt'])
            if len(calls) == 1:
                self.mod_file.write_text('VALUE = 99\n', encoding='utf-8')
                result['steps'] = 5
                result['cost'] = 0.01
                raise GuardError('step_limit')
            self.mod_file.write_text('VALUE = 2\n', encoding='utf-8')
            result['steps'] = 1
            result['cost'] = 0.005

        res = self._run_model_task({
            'task_id': 'h90-step-limit-pass',
            'mode': 'model',
            'directory': str(self.repo),
            'files': [str(self.mod_file)],
            'test_argv': [sys.executable, '-m', 'pytest', str(self.test_file)],
            'max_steps': 4,
            'budget_usd': 0.5,
            'prompt': 'set VALUE=2',
        }, fake_model_run)

        self.assertEqual(len(calls), 2)
        self.assertEqual(res['status'], 'validation_passed_review_required')
        self.assertEqual(res['salvaged_from'], 'salvage_repaired_step_limit')
        self.assertTrue(res.get('micro_repair_attempted'))
        self.assertEqual(res['test_return_code'], 0)
        events_path = Path(res['artifact']) / 'events.jsonl'
        events = [json.loads(line)['event'] for line in events_path.read_text(encoding='utf-8').splitlines()]
        self.assertIn('salvage_micro_repair_started', events)

    def test_salvage_micro_repair_preserves_original_reason_on_failure(self):
        import sys
        from harness.sandbox import GuardError

        self.test_file.write_text(
            'from mod import VALUE\n'
            'def test_val():\n'
            '    assert VALUE == 2\n',
            encoding='utf-8',
        )
        calls = []

        def fake_model_run(cli, db, task, run_dir, result, budget, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                self.mod_file.write_text('VALUE = 99\n', encoding='utf-8')
                result['steps'] = 2
                result['cost'] = 0.01
                raise GuardError('worker_timeout')
            self.mod_file.write_text('VALUE = 100\n', encoding='utf-8')
            result['steps'] = 1
            result['cost'] = 0.005

        res = self._run_model_task({
            'task_id': 'h90-timeout-fail',
            'mode': 'model',
            'directory': str(self.repo),
            'files': [str(self.mod_file)],
            'test_argv': [sys.executable, '-m', 'pytest', str(self.test_file)],
            'max_steps': 4,
            'budget_usd': 0.5,
            'prompt': 'set VALUE=2',
        }, fake_model_run)

        self.assertEqual(len(calls), 2)
        self.assertEqual(res['status'], 'blocked')
        self.assertEqual(res['reason'], 'worker_timeout')
        self.assertTrue(res.get('micro_repair_attempted'))

    def test_salvage_micro_repair_skipped_for_non_repairable_or_test_timeout(self):
        import sys
        from unittest import mock
        from harness.sandbox import GuardError

        self.test_file.write_text(
            'from mod import VALUE\n'
            'def test_val():\n'
            '    assert VALUE == 2\n',
            encoding='utf-8',
        )

        calls = []

        def fake_model_run_non_repairable(cli, db, task, run_dir, result, budget, **kwargs):
            calls.append(1)
            self.mod_file.write_text('VALUE = 99\n', encoding='utf-8')
            result['steps'] = 2
            result['cost'] = 0.01
            raise GuardError('repeated_tool_error')

        res1 = self._run_model_task({
            'task_id': 'h90-non-repairable',
            'mode': 'model',
            'directory': str(self.repo),
            'files': [str(self.mod_file)],
            'test_argv': [sys.executable, '-m', 'pytest', str(self.test_file)],
            'max_steps': 4,
            'budget_usd': 0.5,
            'prompt': 'set VALUE=2',
        }, fake_model_run_non_repairable)
        self.assertEqual(len(calls), 1)
        self.assertEqual(res1['status'], 'blocked')
        self.assertEqual(res1['reason'], 'repeated_tool_error')
        self.assertFalse(res1.get('micro_repair_attempted', False))

        self.mod_file.write_text('VALUE = 1\n', encoding='utf-8')
        calls.clear()

        def fake_model_run_timeout(cli, db, task, run_dir, result, budget, **kwargs):
            calls.append(1)
            self.mod_file.write_text('VALUE = 99\n', encoding='utf-8')
            result['steps'] = 2
            result['cost'] = 0.01
            raise GuardError('worker_timeout')

        with mock.patch('harness.sandbox.execute', return_value=124):
            res2 = self._run_model_task({
                'task_id': 'h90-test-timeout-124',
                'mode': 'model',
                'directory': str(self.repo),
                'files': [str(self.mod_file)],
                'test_argv': [sys.executable, '-m', 'pytest', str(self.test_file)],
                'max_steps': 4,
                'budget_usd': 0.5,
                'prompt': 'set VALUE=2',
            }, fake_model_run_timeout)
        self.assertEqual(len(calls), 1)
        self.assertEqual(res2['status'], 'blocked')
        self.assertEqual(res2['reason'], 'worker_timeout')
        self.assertFalse(res2.get('micro_repair_attempted', False))

    def test_micro_repair_second_round_fixes_test(self):
        import json
        import sys
        from pathlib import Path

        self.test_file.write_text(
            'from mod import VALUE\n'
            'def test_val():\n'
            '    assert VALUE == 2\n',
            encoding='utf-8',
        )
        calls = []

        def fake_model_run(cli, db, task, run_dir, result, budget, **kwargs):
            calls.append(task['prompt'])
            if len(calls) == 1:
                self.mod_file.write_text('VALUE = 98\n', encoding='utf-8')
            elif len(calls) == 2:
                self.mod_file.write_text('VALUE = 99\n', encoding='utf-8')
            else:
                self.mod_file.write_text('VALUE = 2\n', encoding='utf-8')
            result['steps'] = 1
            result['cost'] = 0.01

        res = self._run_model_task({
            'task_id': 'h90-two-rounds',
            'mode': 'model',
            'directory': str(self.repo),
            'files': [str(self.mod_file)],
            'test_argv': [sys.executable, '-m', 'pytest', str(self.test_file)],
            'max_steps': 4,
            'budget_usd': 0.5,
            'prompt': 'set VALUE=2',
        }, fake_model_run)

        self.assertEqual(len(calls), 3)
        self.assertEqual(res['status'], 'validation_passed_review_required')
        self.assertEqual(res.get('micro_repair_rounds'), 2)
        self.assertTrue(res.get('micro_repair_attempted'))
        self.assertEqual(res['test_return_code'], 0)
        events = [json.loads(line) for line
                  in (Path(res['artifact']) / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
        rounds = [ev.get('round') for ev in events if ev.get('event') == 'micro_repair_started']
        self.assertEqual(rounds, [1, 2])
        self.assertTrue((Path(res['artifact']) / 'test_after_repair_1.log').is_file())

    def test_micro_repair_multifile_two_rounds_accumulate_repaired_files(self):
        import sys
        from unittest import mock

        mod_a = self.repo / 'mod_a.py'
        mod_b = self.repo / 'mod_b.py'
        self.test_file.write_text(
            'from mod_a import VALUE_A\n'
            'from mod_b import VALUE_B\n'
            'def test_vals():\n'
            '    assert VALUE_A == 2\n'
            '    assert VALUE_B == 300\n',
            encoding='utf-8',
        )
        mod_a.write_text('VALUE_A = 1\n', encoding='utf-8')
        mod_b.write_text('VALUE_B = 1\n', encoding='utf-8')
        calls = []

        def fake_model_run(cli, db, task, run_dir, result, budget, **kwargs):
            calls.append(task['prompt'])
            if len(calls) == 1:
                mod_a.write_text('VALUE_A = 999\n', encoding='utf-8')
            elif len(calls) == 2:
                mod_a.write_text('VALUE_A = 2\n', encoding='utf-8')
            else:
                mod_b.write_text('VALUE_B = 300\n', encoding='utf-8')
            result['steps'] = 1
            result['cost'] = 0.01

        with mock.patch('harness.sandbox.record_repair_lesson') as lesson_mock:
            res = self._run_model_task({
                'task_id': 'h90-two-files-two-rounds',
                'mode': 'model',
                'directory': str(self.repo),
                'files': [str(mod_a), str(mod_b)],
                'test_argv': [sys.executable, '-m', 'pytest', str(self.test_file)],
                'max_steps': 4,
                'budget_usd': 0.5,
                'prompt': 'fix VALUE_A and VALUE_B',
            }, fake_model_run)

        self.assertEqual(len(calls), 3)
        self.assertEqual(res['status'], 'validation_passed_review_required')
        self.assertEqual(res.get('micro_repair_rounds'), 2)
        self.assertEqual(res['test_return_code'], 0)
        self.assertEqual(mod_a.read_text(encoding='utf-8'), 'VALUE_A = 2\n')
        self.assertEqual(mod_b.read_text(encoding='utf-8'), 'VALUE_B = 300\n')
        lesson_mock.assert_called_once()
        self.assertEqual(lesson_mock.call_args[0][3], [str(mod_a), str(mod_b)])

    def test_micro_repair_no_change_stops_after_round_1(self):
        import json
        import sys
        from pathlib import Path

        self.test_file.write_text(
            'from mod import VALUE\n'
            'def test_val():\n'
            '    assert VALUE == 2\n',
            encoding='utf-8',
        )
        calls = []

        def fake_model_run(cli, db, task, run_dir, result, budget, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                self.mod_file.write_text('VALUE = 99\n', encoding='utf-8')
            result['steps'] = 1
            result['cost'] = 0.01

        res = self._run_model_task({
            'task_id': 'h90-no-change',
            'mode': 'model',
            'directory': str(self.repo),
            'files': [str(self.mod_file)],
            'test_argv': [sys.executable, '-m', 'pytest', str(self.test_file)],
            'max_steps': 4,
            'budget_usd': 0.5,
            'prompt': 'set VALUE=2',
        }, fake_model_run)

        self.assertEqual(len(calls), 2)
        self.assertEqual(res['status'], 'blocked')
        self.assertEqual(res.get('micro_repair_rounds'), 1)
        events = [json.loads(line) for line
                  in (Path(res['artifact']) / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertIn('micro_repair_no_change', [ev.get('event') for ev in events])

    def test_micro_repair_spend_threshold_blocks_round_2(self):
        import sys

        self.test_file.write_text(
            'from mod import VALUE\n'
            'def test_val():\n'
            '    assert VALUE == 2\n',
            encoding='utf-8',
        )
        calls = []

        def fake_model_run(cli, db, task, run_dir, result, budget, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                self.mod_file.write_text('VALUE = 99\n', encoding='utf-8')
                result['cost'] = 0.01
            else:
                self.mod_file.write_text('VALUE = 98\n', encoding='utf-8')
                result['cost'] = 0.09
            result['steps'] = 1

        res = self._run_model_task({
            'task_id': 'h90-spend-cap',
            'mode': 'model',
            'directory': str(self.repo),
            'files': [str(self.mod_file)],
            'test_argv': [sys.executable, '-m', 'pytest', str(self.test_file)],
            'max_steps': 4,
            'budget_usd': 0.5,
            'prompt': 'set VALUE=2',
        }, fake_model_run)

        self.assertEqual(len(calls), 2)
        self.assertEqual(res['status'], 'blocked')
        self.assertEqual(res.get('micro_repair_rounds'), 1)


class TestEnvAndPreflightTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.repo = Path(os.path.realpath(self._td.name))

    def test_build_test_env_overrides_enriches_pythonpath_and_preserves_caller_first(self):
        from harness.sandbox import build_test_env_overrides

        ai_mcp = self.repo / 'ai' / 'mcp'
        ai_tools = self.repo / 'ai' / 'tools'
        ai_mcp.mkdir(parents=True)
        ai_tools.mkdir(parents=True)
        (ai_tools / 'agent_coding_policy.py').write_text('POLICY = 1\n', encoding='utf-8')
        test_file = ai_tools / 'test_capri_coding_config.py'
        test_file.write_text('import agent_coding_policy\n', encoding='utf-8')

        caller_dir = str(self.repo / 'custom_caller')
        saved = os.environ.get('PYTHONPATH')
        try:
            os.environ['PYTHONPATH'] = caller_dir
            env = build_test_env_overrides({
                'directory': str(self.repo),
                'files': [str(test_file)],
                'test_argv': ['/usr/bin/env', f'PYTHONPATH={caller_dir}', 'python', '-m', 'pytest', str(test_file)],
            })
        finally:
            if saved is None:
                os.environ.pop('PYTHONPATH', None)
            else:
                os.environ['PYTHONPATH'] = saved

        parts = env['PYTHONPATH'].split(os.pathsep)
        self.assertEqual(parts[0], caller_dir)
        self.assertEqual(parts.count(caller_dir), 1)
        self.assertIn(str(self.repo), parts)
        self.assertIn(str(ai_tools), parts)
        self.assertIn(str(ai_mcp), parts)

    def test_build_test_env_overrides_with_run_dir_sets_sandbox_tmp_env(self):
        import tempfile
        from pathlib import Path
        from harness.sandbox import build_test_env_overrides, cleanup_isolation_sandbox, stage_isolation_sandbox

        target = self.repo / 'mod.py'
        target.write_text('VALUE = 1\n', encoding='utf-8')
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        run_dir = Path(os.path.realpath(td.name))

        sb_env = stage_isolation_sandbox(run_dir)
        self.addCleanup(cleanup_isolation_sandbox, run_dir)
        sb_tmp = Path(sb_env['TMPDIR'])

        env = build_test_env_overrides(
            {'directory': str(self.repo), 'files': [str(target)], 'test_argv': ['/usr/bin/true']},
            run_dir=run_dir)
        self.assertEqual(env['TMPDIR'], str(sb_tmp))
        self.assertEqual(env['TEMP'], str(sb_tmp))
        self.assertEqual(env['TMP'], str(sb_tmp))
        self.assertEqual(env['DBT_LOG_PATH'], str(sb_tmp / 'dbt_logs'))
        self.assertEqual(env['DBT_TARGET_PATH'], str(sb_tmp / 'dbt_target'))
        self.assertTrue((sb_tmp / 'pytest_cache').is_dir())
        self.assertNotIn('PYTEST_ADDOPTS', env)

    def test_build_test_env_overrides_with_run_dir_sets_pytest_cache_dir(self):
        import tempfile
        from pathlib import Path
        from harness.sandbox import build_test_env_overrides, cleanup_isolation_sandbox, stage_isolation_sandbox

        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        run_dir = Path(os.path.realpath(td.name))

        sb_env = stage_isolation_sandbox(run_dir)
        self.addCleanup(cleanup_isolation_sandbox, run_dir)
        sb_tmp = Path(sb_env['TMPDIR'])
        expected = f'-o cache_dir={sb_tmp / "pytest_cache"}'

        env = build_test_env_overrides(
            {'directory': str(self.repo),
             'files': [],
             'test_argv': ['python', '-m', 'pytest', '-q', str(self.repo / 'test_x.py')]},
            run_dir=run_dir)
        self.assertEqual(env.get('PYTEST_ADDOPTS'), expected)

        saved = os.environ.get('PYTEST_ADDOPTS')
        try:
            os.environ['PYTEST_ADDOPTS'] = '-o cache_dir=/caller/own'
            preserved = build_test_env_overrides(
                {'directory': str(self.repo),
                 'files': [],
                 'test_argv': ['python', '-m', 'pytest']},
                run_dir=run_dir)
            self.assertNotIn('PYTEST_ADDOPTS', preserved)
        finally:
            if saved is None:
                os.environ.pop('PYTEST_ADDOPTS', None)
            else:
                os.environ['PYTEST_ADDOPTS'] = saved

    def test_preflight_test_argv_rejects_unsupported_flag_unless_script_editable(self):
        import sys
        from harness.sandbox import GuardError, preflight_test_argv

        validator = self.repo / 'validate_openspec.py'
        validator.write_text(
            'import argparse\n'
            'p = argparse.ArgumentParser()\n'
            'p.add_argument("--strict", action="store_true")\n'
            'p.parse_args()\n',
            encoding='utf-8',
        )

        # Non-editable validator with unsupported --change flag -> GuardError
        with self.assertRaises(GuardError) as ctx:
            preflight_test_argv({
                'directory': str(self.repo),
                'files': [str(self.repo / 'other.py')],
                'test_argv': [sys.executable, str(validator), '--change', 'abc'],
            }, self.repo)
        self.assertEqual(str(ctx.exception), 'invalid_test_argv_flags')

        # Supported flag -> no error
        preflight_test_argv({
            'directory': str(self.repo),
            'files': [str(self.repo / 'other.py')],
            'test_argv': [sys.executable, str(validator), '--strict'],
        }, self.repo)

        # Editable validator -> no error even with unsupported --change flag
        preflight_test_argv({
            'directory': str(self.repo),
            'files': [str(validator)],
            'test_argv': [sys.executable, str(validator), '--change', 'abc'],
        }, self.repo)

    def test_preflight_test_argv_pytest_missing_target_and_empty_collection(self):
        import sys
        from harness.sandbox import GuardError, preflight_test_argv

        mod_file = self.repo / 'mod.py'
        mod_file.write_text('VALUE = 1\n', encoding='utf-8')
        (self.repo / 'tests').mkdir(exist_ok=True)
        (self.repo / 'tests' / 'test_mod.py').write_text('def test_ok():\n    assert True\n', encoding='utf-8')

        # Missing non-editable pytest target -> missing_test_target
        with self.assertRaises(GuardError) as ctx:
            preflight_test_argv({
                'directory': str(self.repo),
                'files': [str(mod_file)],
                'test_argv': [sys.executable, '-m', 'pytest', 'tests/missing_test.py::test_x'],
            }, self.repo)
        self.assertEqual(str(ctx.exception), 'missing_test_target')

        # Missing target that IS in editable files -> allowed (worker may create it)
        preflight_test_argv({
            'directory': str(self.repo),
            'files': [str(self.repo / 'tests' / 'new_test.py')],
            'test_argv': [sys.executable, '-m', 'pytest', 'tests/new_test.py'],
        }, self.repo)

        # Existing test file with -k filter matching nothing -> empty_pytest_collection
        with self.assertRaises(GuardError) as ctx:
            preflight_test_argv({
                'directory': str(self.repo),
                'files': [str(mod_file)],
                'test_argv': [sys.executable, '-m', 'pytest', 'tests/test_mod.py', '-k', 'no_such_test_xyz'],
            }, self.repo)
        self.assertEqual(str(ctx.exception), 'empty_pytest_collection')

        # Existing test file with valid collection -> no error
        preflight_test_argv({
            'directory': str(self.repo),
            'files': [str(mod_file)],
            'test_argv': [sys.executable, '-m', 'pytest', 'tests/test_mod.py'],
        }, self.repo)

    def test_preflight_test_argv_script_missing_positional_args(self):
        import sys
        from harness.sandbox import GuardError, preflight_test_argv

        mod_file = self.repo / 'mod.py'
        mod_file.write_text('VALUE = 1\n', encoding='utf-8')
        req_script = self.repo / 'check_req.py'
        req_script.write_text(
            'import argparse\n'
            'p = argparse.ArgumentParser()\n'
            'p.add_argument("--strict", action="store_true")\n'
            'p.add_argument("target")\n'
            'p.parse_args()\n',
            encoding='utf-8',
        )

        # Zero args when positional target is required -> missing_test_script_args
        with self.assertRaises(GuardError) as ctx:
            preflight_test_argv({
                'directory': str(self.repo),
                'files': [str(mod_file)],
                'test_argv': [sys.executable, str(req_script)],
            }, self.repo)
        self.assertEqual(str(ctx.exception), 'missing_test_script_args')

        # Flag only, still missing positional -> missing_test_script_args
        with self.assertRaises(GuardError) as ctx:
            preflight_test_argv({
                'directory': str(self.repo),
                'files': [str(mod_file)],
                'test_argv': [sys.executable, str(req_script), '--strict'],
            }, self.repo)
        self.assertEqual(str(ctx.exception), 'missing_test_script_args')

        # Positional arg supplied -> no error
        preflight_test_argv({
            'directory': str(self.repo),
            'files': [str(mod_file)],
            'test_argv': [sys.executable, str(req_script), 'foo.txt'],
        }, self.repo)

        # Optional positional arg -> no error with zero args
        opt_script = self.repo / 'check_opt.py'
        opt_script.write_text(
            'import argparse\n'
            'p = argparse.ArgumentParser()\n'
            'p.add_argument("target", nargs="?")\n'
            'p.parse_args()\n',
            encoding='utf-8',
        )
        preflight_test_argv({
            'directory': str(self.repo),
            'files': [str(mod_file)],
            'test_argv': [sys.executable, str(opt_script)],
        }, self.repo)

    def test_check_yaml_and_jinja_integrity(self):
        from harness.guards import check_yaml_and_jinja_integrity

        # Valid dbt schema.yml with Jinja, block scalar, apostrophe, and list items
        valid_dbt = (
            "version: 2\n"
            "models:\n"
            "  - name: users\n"
            "    description: |\n"
            "      User table it's great\n"
            "      \ttab_inside_block_scalar_is_ok\n"
            "    columns:\n"
            "      - name: id\n"
            "        description: \"{{ doc('user_id') }}\"\n"
            "      - name: email\n"
            "        description: User's email\n"
            "        meta: {a: {b: 1}}\n"
        )
        self.assertEqual(check_yaml_and_jinja_integrity(valid_dbt, "schema.yml"), [])

        # Tab indentation outside block scalar
        tab_yaml = "models:\n\t- name: users\n"
        issues = check_yaml_and_jinja_integrity(tab_yaml, "schema.yml")
        self.assertTrue(any("tab indentation" in i for i in issues), issues)

        # Duplicate sibling mapping keys in same parent / list item
        dup_yaml = (
            "models:\n"
            "  - name: users\n"
            "    description: first\n"
            "    description: second\n"
        )
        issues = check_yaml_and_jinja_integrity(dup_yaml, "schema.yml")
        self.assertTrue(any("duplicate YAML mapping key 'description'" in i for i in issues), issues)

        # Unbalanced quote on single-line scalar
        quote_yaml = 'name: "unclosed\n'
        issues = check_yaml_and_jinja_integrity(quote_yaml, "config.yaml")
        self.assertTrue(any("unbalanced" in i for i in issues), issues)

        # Unclosed Jinja delimiters in .sql and .yml
        sql_bad = "SELECT * FROM {{ ref('users')\n{% if is_incremental()\n"
        issues = check_yaml_and_jinja_integrity(sql_bad, "model.sql")
        self.assertEqual(len(issues), 2, issues)
        self.assertTrue(any("'{{" in i for i in issues), issues)
        self.assertTrue(any("'{%" in i for i in issues), issues)

    def test_check_python_class_and_call_integrity(self):
        from harness.guards import check_python_class_and_call_integrity

        # Wrong positional arity, unknown kwarg, missing self._helper()
        bad_py = (
            "def add(a, b=1):\n"
            "    return a + b\n"
            "\n"
            "class Worker:\n"
            "    def run(self):\n"
            "        self._missing()\n"
            "        return add(1, 2, 3)\n"
            "\n"
            "x = add()\n"
            "y = add(1, bogus=2)\n"
        )
        issues = check_python_class_and_call_integrity(bad_py, "mod.py")
        self.assertTrue(any("has no method '_missing'" in i for i in issues), issues)
        self.assertTrue(any("too many positional arguments" in i for i in issues), issues)
        self.assertTrue(any("missing required argument(s) a" in i for i in issues), issues)
        self.assertTrue(any("unexpected keyword argument 'bogus'" in i for i in issues), issues)

        # Decorated, *args/**kwargs, inherited, __getattr__, and self.attr callable -> no false positives
        clean_py = (
            "def dec(f):\n"
            "    return f\n"
            "\n"
            "@dec\n"
            "def wrapped(a):\n"
            "    return a\n"
            "\n"
            "def variadic(*args, **kwargs):\n"
            "    return args\n"
            "\n"
            "class Child(Base):\n"
            "    def run(self):\n"
            "        return self.inherited_method()\n"
            "\n"
            "class Dyn:\n"
            "    def __getattr__(self, name):\n"
            "        return lambda: None\n"
            "    def go(self):\n"
            "        return self.anything()\n"
            "\n"
            "class WithCallback:\n"
            "    def __init__(self, fn):\n"
            "        self.fn = fn\n"
            "    def call_it(self):\n"
            "        return self.fn()\n"
            "\n"
            "wrapped(1, 2)\n"
            "variadic(1, 2, x=3)\n"
        )
        self.assertEqual(check_python_class_and_call_integrity(clean_py, "clean.py"), [])

    def test_render_guard_plugin_polyglot_hooks(self):
        from harness.sandbox import render_guard_plugin

        plugin = render_guard_plugin()
        for needle in (
            '"apply_patch"',
            '.mjs',
            '"node", "--check"',
            '.ts',
            'Bun.Transpiler',
            '.yaml',
            '.yml',
            '.sql',
            '[HARNESS IN-LOOP YAML/JINJA CHECK FAILED]',
            'Bun.which("ruff")',
            'has no method',
            'unexpected kwarg',
        ):
            self.assertIn(needle, plugin)


class PolyglotCompanionAndSymbolContextTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.task_dir = os.path.realpath(self._td.name)

    def _write(self, rel_path, content=''):
        path = os.path.join(self.task_dir, rel_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(content)
        return path

    def test_2hop_python_chain_ranks_models_and_skips_sensitive(self):
        service = self._write('service.py', 'from repo import load_user\n')
        repo = self._write('repo.py', 'from models import UserRecord\n\ndef load_user():\n    return UserRecord\n')
        models = self._write(
            'models.py',
            'from dataclasses import dataclass\n\n@dataclass\nclass UserRecord:\n    user_id: int\n',
        )
        secret_env = self._write('.env', 'SECRET=1\n')
        result = infer_read_only_companion_files(
            [service],
            directory=self.task_dir,
            prompt='Update `service.py` to handle `UserRecord` fields',
        )
        self.assertIn(repo, result)
        self.assertIn(models, result)
        self.assertNotIn(secret_env, result)

        from harness.guards import build_cross_file_symbol_context
        ctx = build_cross_file_symbol_context(
            [service],
            prompt='Handle `UserRecord` in `service.py`',
            directory=self.task_dir,
        )
        self.assertIn('class UserRecord:', ctx)
        self.assertIn('user_id: int', ctx)

    def test_mjs_relative_import_companion_and_symbol_context(self):
        main_mjs = self._write(
            'src/main.mjs',
            "import { computeBounds } from './bounds.mjs';\nexport function run() { return computeBounds(); }\n",
        )
        bounds_mjs = self._write(
            'src/bounds.mjs',
            'export function computeBounds(width, height) {\n  return width * height;\n}\n'
            'export const MAX_BOUNDS = 100;\n',
        )
        result = infer_read_only_companion_files([main_mjs], directory=self.task_dir)
        self.assertEqual(result, [bounds_mjs])

        from harness.guards import build_cross_file_symbol_context
        ctx = build_cross_file_symbol_context(
            [main_mjs],
            prompt='Fix `computeBounds` usage',
            directory=self.task_dir,
        )
        self.assertIn('src/bounds.mjs', ctx)
        self.assertIn('export function computeBounds(width, height)', ctx)

    def test_dbt_ref_companion_discovery_under_models(self):
        fct_sql = self._write(
            'models/marts/fct_orders.sql',
            "select * from {{ ref('stg_payments') }}\n",
        )
        stg_sql = self._write(
            'models/staging/stg_payments.sql',
            'select 1 as payment_id\n',
        )
        stg_yml = self._write(
            'models/staging/schema.yml',
            'version: 2\nmodels:\n  - name: stg_payments\n',
        )
        result = infer_read_only_companion_files([fct_sql], directory=self.task_dir)
        self.assertIn(stg_sql, result)
        self.assertIn(stg_yml, result)

    def test_nonexistent_test_file_infers_subpackage_target_and_2hop_contract(self):
        handler = self._write(
            'pkg/sub/handler.py',
            'from contracts import HandlerSchema\n\ndef run():\n    return HandlerSchema\n',
        )
        contracts = self._write(
            'pkg/contracts.py',
            'class HandlerSchema:\n    pass\n',
        )
        # The editable test file does not exist yet (brand-new task).
        missing_test = os.path.join(self.task_dir, 'pkg', 'tests', 'test_handler.py')
        result = infer_read_only_companion_files([missing_test], directory=self.task_dir)
        self.assertIn(handler, result)
        self.assertIn(contracts, result)


class RuntimeAssertionLocalsTests(unittest.TestCase):
    def test_extract_runtime_assertion_locals_parses_diff_strips_noise_redacts_and_bounds(self):
        from harness.guards import extract_runtime_assertion_locals

        long_str = 'x' * 300
        diag_log = (
            "=================================== FAILURES ===================================\n"
            "________________________________ test_dict_diff ________________________________\n"
            "tests/test_ex.py:4: in test_dict_diff\n"
            "    compute(actual)\n"
            "outer_only = 999\n"
            "src/mod.py:12: in compute\n"
            "    assert actual == expected\n"
            "E   AssertionError: assert {'a': 1, 'b': 2} == {'a': 1, 'b': 9}\n"
            "E     Full diff:\n"
            "E     - {'a': 1, 'b': 9}\n"
            "E     + {'a': 1, 'b': 2}\n"
            "actual     = {'a': 1, 'b': 2}\n"
            "expected   = {'a': 1, 'b': 9}\n"
            "__builtins__ = <builtins>\n"
            "mod        = <module 'mod' from '/tmp/mod.py'>\n"
            "helper     = <function helper at 0x104abc120>\n"
            "os_env     = environ({'PATH': '/usr/bin'})\n"
            "cfg        = <AppConfig object at 0x104abc999>\n"
            f"api_key    = '{'sk-or-' + 'v1-secret1234567890abcdef'}'\n"
            f"pem_key    = '{'-----BEGIN RSA ' + 'PRIVATE KEY-----'}\\nMIIEowIBAAKCAQEA123456789012345678901234567890'\n"
            f"long_val   = '{long_str}'\n"
            "=========================== short test summary info ============================\n"
            "FAILED tests/test_ex.py::test_dict_diff - AssertionError\n"
        )
        out = extract_runtime_assertion_locals(diag_log)
        self.assertIn("E     - {'a': 1, 'b': 9}", out)
        self.assertIn("actual = {'a': 1, 'b': 2}", out)
        self.assertIn("expected = {'a': 1, 'b': 9}", out)
        self.assertNotIn("outer_only", out)
        self.assertNotIn("__builtins__", out)
        self.assertNotIn("<module", out)
        self.assertNotIn("<function", out)
        self.assertNotIn("environ({", out)
        self.assertNotIn("<AppConfig", out)
        self.assertNotIn("sk-or-" + "v1-secret", out)
        self.assertNotIn("MIIEowIBAAKCAQEA", out)
        self.assertIn("[REDACTED]", out)
        long_lines = [line for line in out.splitlines() if line.startswith("long_val = ")]
        self.assertEqual(len(long_lines), 1)
        self.assertLessEqual(len(long_lines[0]), len("long_val = ") + 240)

        bounded = extract_runtime_assertion_locals(diag_log, max_chars=120)
        self.assertLessEqual(len(bounded), 120)
        self.assertTrue(bounded)

    def test_build_micro_repair_prompt_injects_runtime_assertion_locals(self):
        import tempfile
        from pathlib import Path
        from harness.sandbox import build_micro_repair_prompt

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mod_file = tmp_path / 'mod.py'
            mod_file.write_text('def val():\n    return 1\n', encoding='utf-8')
            run_dir = tmp_path / 'run'
            run_dir.mkdir()
            (run_dir / 'changes.patch').write_text('--- a/mod.py\n+++ b/mod.py\n', encoding='utf-8')
            (run_dir / 'test.log').write_text(
                'FAILED test_mod.py::test_val - AssertionError: assert 1 == 2\n',
                encoding='utf-8',
            )
            (run_dir / 'test_assertion_locals.log').write_text(
                "test_mod.py:3: in test_val\n"
                "    assert actual == expected\n"
                "E   AssertionError: assert {'k': 1} == {'k': 2}\n"
                "actual     = {'k': 1}\n"
                "expected   = {'k': 2}\n",
                encoding='utf-8',
            )
            task = {
                'prompt': 'fix val()',
                'directory': str(tmp_path),
                'files': [str(mod_file)],
                'read_only_files': [],
            }
            prompt = build_micro_repair_prompt(task, run_dir)
            self.assertIn('\n[RUNTIME_ASSERTION_LOCALS]\n', prompt)
            self.assertIn("actual = {'k': 1}", prompt)
            self.assertLess(
                prompt.index('[TEST_ARGV FAILURE'),
                prompt.index('[RUNTIME_ASSERTION_LOCALS]'),
            )

    def test_capture_assertion_locals_strips_quiet_and_swallows_timeout(self):
        import sys
        import tempfile
        from pathlib import Path
        from unittest import mock
        from harness.sandbox import GuardError, _capture_assertion_locals

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / 'test.log').write_text(
                'FAILED tests/test_mod.py::test_a - AssertionError: assert 1 == 2\n'
                'E   AssertionError: assert 1 == 2\n',
                encoding='utf-8',
            )
            recorded = {}

            def fake_execute(argv, directory, timeout, path, *, env_overrides=None):
                recorded['argv'] = argv
                recorded['timeout'] = timeout
                recorded['path'] = path
                raise GuardError('command_timeout')

            task = {
                'directory': tmp,
                'test_argv': [sys.executable, '-m', 'pytest', '-q', str(Path(tmp) / 'tests')],
                'test_timeout_seconds': 60,
            }
            with mock.patch('harness.sandbox.execute', side_effect=fake_execute):
                _capture_assertion_locals(task, run_dir)

            self.assertEqual(recorded['timeout'], 15)
            self.assertEqual(recorded['path'], run_dir / 'test_assertion_locals.log')
            self.assertNotIn('-q', recorded['argv'])
            self.assertIn('--showlocals', recorded['argv'])
            self.assertIn('-vv', recorded['argv'])
            self.assertIn('--tb=short', recorded['argv'])
            self.assertIn('--no-header', recorded['argv'])


from harness.sandbox import is_seatbelt_available


class SeatbeltSandboxTests(unittest.TestCase):
    def test_build_seatbelt_profile_canonicalizes_and_escapes(self):
        import os
        import tempfile
        from pathlib import Path
        from harness.sandbox import build_seatbelt_profile, wrap_argv_with_seatbelt

        with tempfile.TemporaryDirectory() as tmp:
            quoted_dir = Path(tmp) / 'dir"with"quotes'
            quoted_dir.mkdir()
            secret_dir = Path(tmp) / 'secret_creds'
            secret_dir.mkdir()

            profile = build_seatbelt_profile(
                writable_paths=['/tmp', quoted_dir],
                denied_read_paths=[secret_dir],
                allow_localhost_only=True,
                allowed_tcp_ports=[8080],
            )
            self.assertTrue(profile.startswith('(version 1)\n(allow default)\n(deny file-write*)\n'))
            self.assertIn(f'(allow file-write* (subpath "{os.path.realpath("/tmp")}"))', profile)
            self.assertIn('dir\\"with\\"quotes', profile)
            self.assertIn(f'(deny file-read* (subpath "{os.path.realpath(secret_dir)}"))', profile)
            self.assertIn('(allow network* (local unix))', profile)
            self.assertIn('(deny network-outbound (remote ip))', profile)
            self.assertIn('(allow network-outbound (remote ip "localhost:8080"))', profile)

            profile_localhost = build_seatbelt_profile(
                writable_paths=[quoted_dir],
                allow_localhost_only=True,
            )
            self.assertIn('(allow network-outbound (remote ip "localhost:*"))', profile_localhost)

            wrapped = wrap_argv_with_seatbelt(['python3', '-V'], Path(tmp) / 'p.sb')
            self.assertEqual(wrapped, ['/usr/bin/sandbox-exec', '-f', str(Path(tmp) / 'p.sb'), 'python3', '-V'])

    def test_validate_task_os_sandbox_modes(self):
        import os
        import tempfile
        from pathlib import Path
        from unittest import mock
        from harness.proxy import GuardError
        from harness.sandbox import validate_task

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'a.py'
            target.write_text('x = 1\n', encoding='utf-8')
            base = {
                'task_id': 't-sb',
                'directory': tmp,
                'files': [str(target)],
                'prompt': 'edit a.py',
                'test_argv': ['/usr/bin/true'],
            }
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop('HARNESS_OS_SANDBOX', None)
                os.environ.pop('HARNESS_DIFF_CRITIC', None)
                expected_sb = 'test_only' if is_seatbelt_available() else 'off'
                self.assertEqual(validate_task(dict(base))['os_sandbox'], expected_sb)
                self.assertFalse(validate_task(dict(base))['enable_diff_critic'])
            with mock.patch.dict(os.environ, {'HARNESS_OS_SANDBOX': 'off'}):
                self.assertEqual(validate_task(dict(base))['os_sandbox'], 'off')
            with mock.patch.dict(os.environ, {'HARNESS_OS_SANDBOX': 'test_only'}):
                self.assertEqual(validate_task(dict(base))['os_sandbox'], 'test_only')
            with mock.patch.dict(os.environ, {'HARNESS_DIFF_CRITIC': '0'}):
                self.assertFalse(validate_task(dict(base))['enable_diff_critic'])
            with mock.patch.dict(os.environ, {'HARNESS_DIFF_CRITIC': 'true'}):
                self.assertTrue(validate_task(dict(base))['enable_diff_critic'])
            self.assertTrue(validate_task({**base, 'enable_diff_critic': True})['enable_diff_critic'])
            with self.assertRaises(GuardError) as ctx_type:
                validate_task({**base, 'enable_diff_critic': 'yes'})
            self.assertEqual(str(ctx_type.exception), 'invalid_enable_diff_critic_flag')
            for mode in ('off', 'test_only', 'strict'):
                t = validate_task({**base, 'os_sandbox': mode})
                self.assertEqual(t['os_sandbox'], mode)
            with self.assertRaises(GuardError) as ctx:
                validate_task({**base, 'os_sandbox': 'bogus'})
            self.assertEqual(str(ctx.exception), 'invalid_os_sandbox')

    @unittest.skipUnless(is_seatbelt_available(), "macOS sandbox-exec required")
    def test_seatbelt_exec_enforces_write_confinement(self):
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        from harness.sandbox import build_seatbelt_profile, wrap_argv_with_seatbelt

        with tempfile.TemporaryDirectory() as allowed_tmp, tempfile.TemporaryDirectory() as outside_tmp:
            profile_path = Path(allowed_tmp) / 'test.sb'
            profile_path.write_text(
                build_seatbelt_profile(writable_paths=[allowed_tmp], allow_localhost_only=True),
                encoding='utf-8',
            )
            inside_file = Path(allowed_tmp) / 'ok.txt'
            ok_argv = wrap_argv_with_seatbelt(
                [sys.executable, '-c', f'from pathlib import Path; Path({str(inside_file)!r}).write_text("ok")'],
                profile_path,
            )
            ok_proc = subprocess.run(ok_argv, capture_output=True, text=True)
            if ok_proc.returncode == 71 and 'Operation not permitted' in ok_proc.stderr:
                self.skipTest('nested sandbox-exec denied in this environment')
            self.assertEqual(ok_proc.returncode, 0, ok_proc.stderr)
            self.assertEqual(inside_file.read_text(encoding='utf-8'), 'ok')

            outside_file = Path(outside_tmp) / 'blocked.txt'
            bad_argv = wrap_argv_with_seatbelt(
                [sys.executable, '-c', f'from pathlib import Path; Path({str(outside_file)!r}).write_text("no")'],
                profile_path,
            )
            bad_proc = subprocess.run(bad_argv, capture_output=True, text=True)
            self.assertNotEqual(bad_proc.returncode, 0)
            self.assertIn('PermissionError', bad_proc.stderr)
            self.assertFalse(outside_file.exists())

    def test_stage_test_seatbelt_profile_includes_enclosing_git_repo_root(self):
        import re
        import tempfile
        from pathlib import Path
        from harness.sandbox import _stage_test_seatbelt_profile

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp) / 'repo'
            task_dir = repo_root / 'services' / 'worker'
            task_dir.mkdir(parents=True)
            (repo_root / '.git').mkdir()
            run_dir = Path(tmp) / 'run'
            run_dir.mkdir()
            task = {
                'task_id': 'sb-repo-root',
                'directory': str(task_dir),
                'files': [str(task_dir / 'mod.py')],
                'mode': 'model',
            }
            profile_path = _stage_test_seatbelt_profile(task, run_dir)
            profile = profile_path.read_text(encoding='utf-8')
            canon = str(Path(repo_root).resolve())
            self.assertIn('(allow file-write* (regex #"^' + re.escape(canon), profile)

    def test_build_seatbelt_profile_bytecode_cache_roots(self):
        import re
        import tempfile
        from pathlib import Path
        from harness.sandbox import build_seatbelt_profile

        with tempfile.TemporaryDirectory() as tmp:
            repo_tmp = Path(tmp) / 'repo'
            (repo_tmp / 'tests').mkdir(parents=True)
            canon = str(Path(repo_tmp).resolve())
            profile = build_seatbelt_profile(bytecode_cache_roots=[repo_tmp])
            self.assertIn('(allow file-write* (regex #"^' + re.escape(canon), profile)
            self.assertIn('(.+/)?((__pycache__|\\.pytest_cache|\\.hypothesis|\\.ruff_cache|\\.mypy_cache|logs|generated)(/.*)?|pytest-cache-files-[^/]+(/.*)?)', profile)
            empty = build_seatbelt_profile()
            self.assertNotIn('(regex #', empty)

    def test_seatbelt_profile_allows_generated_logs_and_pytest_cache_dirs(self):
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        from harness.sandbox import build_seatbelt_profile, wrap_argv_with_seatbelt

        with tempfile.TemporaryDirectory() as allowed_tmp, tempfile.TemporaryDirectory() as outside_tmp:
            profile_path = Path(allowed_tmp) / 'test.sb'
            profile = build_seatbelt_profile(
                writable_paths=[allowed_tmp],
                bytecode_cache_roots=[allowed_tmp],
                allow_localhost_only=True,
            )
            for marker in ('\\.mypy_cache', '|logs', '|generated', 'pytest-cache-files-'):
                self.assertIn(marker, profile)
            profile_path.write_text(profile, encoding='utf-8')

            writes = {
                'logs/dbt.log': 'log',
                'generated/out.txt': 'gen',
                '.mypy_cache/x': 'my',
                'pytest-cache-files-abc/tmp': 'pf',
            }
            script = (
                'from pathlib import Path\n'
                + ''.join(
                    f'Path({str(Path(allowed_tmp) / rel)!r}).parent.mkdir(parents=True, exist_ok=True);'
                    f'Path({str(Path(allowed_tmp) / rel)!r}).write_text({content!r})\n'
                    for rel, content in writes.items())
            )
            ok_argv = wrap_argv_with_seatbelt([sys.executable, '-c', script], profile_path)
            ok_proc = subprocess.run(ok_argv, capture_output=True, text=True)
            if ok_proc.returncode == 71 and 'Operation not permitted' in ok_proc.stderr:
                self.skipTest('nested sandbox-exec denied in this environment')
            self.assertEqual(ok_proc.returncode, 0, ok_proc.stderr)
            for rel, content in writes.items():
                self.assertEqual((Path(allowed_tmp) / rel).read_text(encoding='utf-8'), content)

            outside_file = Path(outside_tmp) / 'blocked.py'
            bad_argv = wrap_argv_with_seatbelt(
                [sys.executable, '-c', f'from pathlib import Path; Path({str(outside_file)!r}).write_text("no")'],
                profile_path,
            )
            bad_proc = subprocess.run(bad_argv, capture_output=True, text=True)
            self.assertNotEqual(bad_proc.returncode, 0)
            self.assertIn('PermissionError', bad_proc.stderr)
            self.assertFalse(outside_file.exists())

    @unittest.skipUnless(is_seatbelt_available(), "macOS sandbox-exec required")
    def test_seatbelt_exec_bytecode_cache_allowed_source_blocked(self):
        import subprocess
        import sys
        import tempfile
        from pathlib import Path
        from harness.sandbox import build_seatbelt_profile, wrap_argv_with_seatbelt

        with tempfile.TemporaryDirectory() as tmp:
            repo_tmp = Path(tmp) / 'repo'
            tests_dir = repo_tmp / 'tests'
            tests_dir.mkdir(parents=True)
            profile_path = Path(tmp) / 'test.sb'
            profile_path.write_text(
                build_seatbelt_profile(bytecode_cache_roots=[repo_tmp], allow_localhost_only=True),
                encoding='utf-8',
            )
            pyc_path = tests_dir / '__pycache__' / 'x.pyc'
            ok_argv = wrap_argv_with_seatbelt(
                [sys.executable, '-c',
                 f'from pathlib import Path; p = Path({str(pyc_path)!r}); p.parent.mkdir(exist_ok=True); p.write_bytes(b"pyc")'],
                profile_path,
            )
            ok_proc = subprocess.run(ok_argv, capture_output=True, text=True)
            if ok_proc.returncode == 71 and 'Operation not permitted' in ok_proc.stderr:
                self.skipTest('nested sandbox-exec denied in this environment')
            self.assertEqual(ok_proc.returncode, 0, ok_proc.stderr)
            self.assertEqual(pyc_path.read_bytes(), b'pyc')

            src_path = tests_dir / 'test_x.py'
            bad_argv = wrap_argv_with_seatbelt(
                [sys.executable, '-c', f'from pathlib import Path; Path({str(src_path)!r}).write_text("no")'],
                profile_path,
            )
            bad_proc = subprocess.run(bad_argv, capture_output=True, text=True)
            self.assertNotEqual(bad_proc.returncode, 0)
            self.assertFalse(src_path.exists())

    def test_run_semantic_diff_critic_auth_bearer_and_openrouter_prefix(self):
        import json as _json
        import os
        from unittest import mock
        from harness.sandbox import run_semantic_diff_critic

        task = {'model': 'openrouter/z-ai/glm-5.3-flash', 'prompt': 'edit'}
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured['url'] = req.full_url
            captured['auth'] = req.headers.get('Authorization')
            captured['model'] = _json.loads(req.data.decode('utf-8'))['model']
            body = _json.dumps(
                {'choices': [{'message': {'content': '{"verdict": "pass", "issues": []}'}}]}
            ).encode('utf-8')
            return mock.MagicMock(**{'__enter__.return_value.read.return_value': body})

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('OPENROUTER_API_KEY', None)
            with mock.patch('harness.sandbox.urllib.request.urlopen', side_effect=fake_urlopen):
                out = run_semantic_diff_critic(
                    task, '- a\n+ b\n',
                    base_url='https://openrouter.ai/api/v1/chat/completions',
                    auth_bearer='test-bearer-token',
                )
        self.assertEqual(out['verdict'], 'pass')
        self.assertEqual(captured['auth'], 'Bearer test-bearer-token')
        self.assertEqual(captured['model'], 'z-ai/glm-5.3-flash')
        self.assertIn('openrouter.ai', captured['url'])

    def test_run_semantic_diff_critic_skips_openrouter_without_key(self):
        import os
        from unittest import mock
        from harness.sandbox import run_semantic_diff_critic

        task = {'model': 'openrouter/z-ai/glm-5.3-flash', 'prompt': 'edit'}
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('OPENROUTER_API_KEY', None)
            with mock.patch('harness.sandbox.urllib.request.urlopen') as urlopen_mock:
                out = run_semantic_diff_critic(
                    task, '- a\n+ b\n', base_url='https://openrouter.ai/api/v1/chat/completions'
                )
        self.assertEqual(out, {'verdict': 'pass', 'issues': [], 'skipped': True})
        urlopen_mock.assert_not_called()


if __name__ == '__main__':
    unittest.main()
