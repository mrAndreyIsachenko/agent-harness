import os
import unittest

from harness.sandbox import (config_for, infer_read_only_companion_files, render_guard_plugin,
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
            expected = (Path(td) / '.opencode' / 'plugin' / 'harness_guard.ts').resolve().as_uri()
            self.assertIn(expected, plugins)
            self.assertTrue((Path(td) / '.opencode' / 'plugin' / 'harness_guard.ts').is_file())

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


if __name__ == '__main__':
    unittest.main()
