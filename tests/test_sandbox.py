import os
import unittest

from harness.sandbox import config_for, render_guard_plugin, stage_guard_plugin, verify_permissions, worker_env


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


if __name__ == '__main__':
    unittest.main()
