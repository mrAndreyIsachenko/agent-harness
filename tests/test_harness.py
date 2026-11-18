"""Contract tests for the standalone ``harness`` package (H-1..H-8, H-10 + report/CLI).

Layout under test:
- ``harness.guards``  -- H-1/H-2/H-4/H-5/H-7/H-8 pure helpers (AST/SQL preflight,
  failure digest, duplicate tool-call detection, stale-read compaction, file
  skeletons).
- ``harness.routing`` -- H-3 sticky provider pinning + adaptive reasoning budgets.
- ``harness.proxy``   -- the local L7 MITM guard proxy wiring H-1/H-5 notices,
  provider-health circuit breaking and sticky pinning into the request path.
- ``harness.report``  -- benchmark run aggregation and Markdown rendering.
- ``harness.cli``     -- ``preflight`` / ``skeleton`` / ``bench-report`` subcommands.
"""

import contextlib
import http.server
import io
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from harness import OpenRouterGuardProxy
from harness.cli import main as cli_main
from harness.finops import (auto_reconcile_from_proxy,
                            build_unexposed_settlement_evidence,
                            can_settle_from_proxy_local, capture_requests,
                            connect, export_sml_cost_manifest,
                            fetch_generation_settled, locks as finops_locks,
                            metrics, number, prior_cost, reconcile_run,
                            request_records, resolved_cost, snapshot,
                            verify_db, violation)
from harness.guards import (FORBIDDEN_EGRESS_KEYS, _EDIT_FAILURE_HINT,
                            _message_text, build_file_skeletons,
                            check_allowed_files_syntax, check_egress_payload,
                            check_readonly_sql_tool_calls, check_sql_syntax,
                            collect_python_top_symbols, compact_context_window,
                            compact_stale_reads, detect_duplicate_tool_calls,
                            detect_edit_oscillation,
                            estimate_messages_tokens, extract_compact_test_failure,
                            inject_edit_failure_hints,
                            inject_read_guard_hints,
                            prune_noisy_tool_outputs)
from harness.report import (aggregate_benchmark_runs, format_benchmark_markdown,
                            format_replay_markdown, format_state_stats_markdown,
                            scan_state_runs, simulate_replay, summarize_run)
from harness.routing import (DEFAULT_MODEL, DEFAULT_PROVIDER_HEALTH_TTL_SECONDS, DEFAULT_REASONING_MAX_TOKENS,
                             ERROR_RECOVERY_MAX_TOKENS_CAP,
                             ERROR_RECOVERY_REASONING_MAX_TOKENS, LATE_TURN_REASONING_MAX_TOKENS,
                             LATE_TURN_START, QUIET_LATE_TURN_REASONING_MAX_TOKENS,
                             ProviderHealthStore, apply_provider_routing,
                             select_escalated_model)
from harness.sandbox import (GuardError, _snapshot_files_for_feedback,
                             build_prior_attempt_feedback, capture_files,
                             compute_repair_max_steps, config_for, execute,
                             file_hashes, isolated_worktree, locks,
                             main as sandbox_main, model_run, owned_process,
                             preflight, run_best_of_2, run_task, stop_group,
                             validate_task, verify_permissions)
from harness.telemetry import OPENINFERENCE_SPAN_KINDS, HarnessTracer, format_trace_tree

GLM = 'openrouter/z-ai/glm-5.3-flash'
DEEPSEEK = 'openrouter/deepseek/deepseek-v4.1-flash-1017'


def _scripted_upstream(steps):
    """Spin up an upstream stub; ``steps`` are consumed per request."""
    captured = []

    class _Upstream(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def do_POST(self):
            length = int(self.headers.get('Content-Length') or '0')
            raw = self.rfile.read(length)
            try:
                captured.append(json.loads(raw))
            except ValueError:
                captured.append(None)
            step = steps.pop(0) if steps else {'status': 200}
            payload = json.dumps({'choices': [{'message': {'role': 'assistant', 'content': 'ok'}}]}).encode()
            self.send_response(step.get('status', 200))
            self.send_header('Content-Type', 'application/json')
            if step.get('provider'):
                self.send_header('X-OpenRouter-Provider', step['provider'])
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), _Upstream)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, captured


def _post_chat(base_url, body):
    request = urllib.request.Request(base_url + '/chat/completions', data=body,
                                     headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(request, timeout=15) as resp:
            resp.read()
            return resp.status
    except urllib.error.HTTPError as exc:  # expected for upstream 4xx/5xx steps
        exc.read()
        return exc.code


class HarnessTestCase(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix='harness-test-')
        self._prev_otel_disabled = os.environ.get('HARNESS_OTEL_DISABLED')
        os.environ['HARNESS_OTEL_DISABLED'] = '1'

    def tearDown(self):
        if self._prev_otel_disabled is None:
            os.environ.pop('HARNESS_OTEL_DISABLED', None)
        else:
            os.environ['HARNESS_OTEL_DISABLED'] = self._prev_otel_disabled

    def path(self, name):
        import os
        return str(os.path.join(self.root, name))


class TestH1Preflight(HarnessTestCase):
    def test_py_syntax_error_and_json_decode_error(self):
        bad_py = self.path('bad.py')
        bad_json = self.path('bad.json')
        skipped = self.path('notes.txt')
        with open(bad_py, 'w') as fh:
            fh.write('def broken(:\n    pass\n')
        with open(bad_json, 'w') as fh:
            fh.write('{"a": 1,\n')
        with open(skipped, 'w') as fh:
            fh.write('def not even python(((\n')
        errors = check_allowed_files_syntax([bad_py, bad_json, skipped])
        self.assertIn('%s:1: invalid syntax' % bad_py, errors)
        self.assertIn("%s:2: Expecting property name enclosed in double quotes" % bad_json, errors)
        self.assertFalse(any(skipped in err for err in errors), errors)
        self.assertEqual(check_allowed_files_syntax([self.path('missing.py')]), [])

    def test_proxy_ast_preflight_error_injection_and_immediate_budget(self):
        bad_py = self.path('bad.py')
        with open(bad_py, 'w') as fh:
            fh.write('def broken(:\n')
        upstream, captured = _scripted_upstream([{'status': 200}] * 3)
        proxy = OpenRouterGuardProxy(lambda: None, upstream_url='http://127.0.0.1:%d' % upstream.server_port,
                                     allowed_files=[bad_py])
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
        try:
            body = json.dumps({'model': GLM, 'messages': [
                {'role': 'user', 'content': 'edit the file'},
                {'role': 'assistant', 'content': 'done'}]}).encode()
            for _ in range(3):
                _post_chat(proxy.base_url, body)
        finally:
            proxy.shutdown(); proxy.server_close()
            upstream.shutdown(); upstream.server_close()
        self.assertEqual(len(captured), 3)
        self.assertNotIn('[AST PREFLIGHT ERROR', json.dumps(captured[0]))  # turn 1: no injection
        notice = captured[1]['messages'][-1]['content']
        self.assertIn('[AST PREFLIGHT ERROR: %s:1: invalid syntax' % bad_py, notice)
        # Turn 2 carries the notice but is still below the late-turn threshold.
        self.assertEqual(captured[1]['reasoning']['max_tokens'], DEFAULT_REASONING_MAX_TOKENS)
        # From turn 3 the injected error signal immediately widens the CoT budget.
        self.assertEqual(captured[2]['reasoning']['max_tokens'], ERROR_RECOVERY_REASONING_MAX_TOKENS)
        self.assertIn('[AST PREFLIGHT ERROR', captured[2]['messages'][-1]['content'])


class TestH2CompactFailure(HarnessTestCase):
    STDOUT = ('Traceback (most recent call last):\n'
              '  File "/x/test_a.py", line 10, in run\n'
              '  File "/x/test_b.py", line 20, in helper\n'
              'E   AssertionError: expected fixed\n'
              'E   TypeError: bad type\n'
              'ImportError: cannot import name\n'
              'SyntaxError: invalid syntax\n'
              'some noise line\n'
              '1 failed, 2 passed in 0.1s\n'
              'FAILED tests/test_b.py::test_one - AssertionError\n')

    def test_extraction(self):
        out = extract_compact_test_failure(self.STDOUT, '', max_lines=25)
        self.assertIn('FAILED tests/test_b.py::test_one', out)
        self.assertIn('E   AssertionError', out)
        self.assertIn('E   TypeError', out)
        self.assertIn('SyntaxError', out)
        self.assertIn('ImportError', out)
        self.assertIn('File "/x/test_b.py", line 20', out)  # last traceback location only
        self.assertNotIn('File "/x/test_a.py"', out)
        self.assertNotIn('some noise', out)
        self.assertLessEqual(len(out.splitlines()), 25)

    def test_max_lines_truncation_and_stderr_fallback(self):
        self.assertEqual(extract_compact_test_failure(self.STDOUT, '', max_lines=2).count('\n'), 1)
        self.assertIn('E   ValueError: from stderr',
                      extract_compact_test_failure('', 'E   ValueError: from stderr\n', max_lines=25))
        tail = extract_compact_test_failure('', 'line one\nline two\n', max_lines=25)
        self.assertEqual(tail, 'line one\nline two')  # fallback: tail of non-empty output


class TestH3StickyRouting(HarnessTestCase):
    def _body(self, model, messages=None):
        return json.dumps({'model': model, 'messages': messages or [{'role': 'user', 'content': 'hi'}]}).encode()

    def test_sticky_pinning_explicit_order_and_deepseek_fp8(self):
        out = json.loads(apply_provider_routing(self._body(GLM), '/chat/completions',
                                                turn_index=1, sticky_provider='siliconflow'))
        self.assertEqual(out['provider']['order'][0], 'siliconflow')
        self.assertNotIn('siliconflow', out['provider']['order'][1:])
        out = json.loads(apply_provider_routing(self._body(DEEPSEEK), '/chat/completions',
                                                turn_index=1, sticky_provider='novita'))
        self.assertEqual(out['provider']['order'], ['novita'])
        self.assertNotIn('sort', out['provider'])

    def test_degraded_provider_is_never_pinned(self):
        out = json.loads(apply_provider_routing(self._body(GLM), '/chat/completions',
                                                turn_index=1, sticky_provider='siliconflow',
                                                degraded_providers={'siliconflow'}))
        self.assertEqual(out['provider']['order'][0], 'z-ai')
        self.assertEqual(out['provider']['order'][-1], 'siliconflow')

    def test_adaptive_reasoning_budget(self):
        err = self._body(GLM, [{'role': 'user', 'content': 'go'},
                               {'role': 'assistant', 'content': 'ok'},
                               {'role': 'tool', 'content': '[AST PREFLIGHT ERROR: bad syntax]'}])
        quiet = self._body(GLM, [{'role': 'user', 'content': 'go'},
                                 {'role': 'assistant', 'content': 'ok'},
                                 {'role': 'tool', 'content': 'all checks passed'}])
        cases = [(err, 3, ERROR_RECOVERY_REASONING_MAX_TOKENS),
                 (quiet, 4, QUIET_LATE_TURN_REASONING_MAX_TOKENS),
                 (quiet, 3, LATE_TURN_REASONING_MAX_TOKENS),
                 (self._body(GLM), 1, DEFAULT_REASONING_MAX_TOKENS)]
        for body, turn, expected in cases:
            out = json.loads(apply_provider_routing(body, '/chat/completions', turn_index=turn))
            self.assertEqual(out['reasoning']['max_tokens'], expected,
                             'turn %d expected %d' % (turn, expected))
            self.assertEqual(out['max_tokens'], 8192)  # request cap applied

    def test_vllm_upstream_mode_omits_openrouter_provider_but_keeps_caps(self):
        for model in (GLM, 'qwen/qwen3-coder'):
            out = json.loads(apply_provider_routing(
                self._body(model),
                '/chat/completions',
                turn_index=1,
                sticky_provider='siliconflow',
                upstream_mode='vllm',
            ))
            self.assertNotIn('provider', out)
            self.assertEqual(out['max_tokens'], 8192)
            self.assertEqual(out['reasoning']['max_tokens'], DEFAULT_REASONING_MAX_TOKENS)


class TestH4SemanticPreflight(HarnessTestCase):
    def test_duplicate_keys_conflict_markers_dropped_symbols(self):
        dup, conflicted, clean = self.path('dup.py'), self.path('query.sql'), self.path('clean.py')
        with open(dup, 'w') as fh:
            fh.write('config = {\n    "a": 1,\n    "a": 2,\n}\n')
        with open(conflicted, 'w') as fh:
            fh.write('select 1;\n<<<<<<< HEAD\nselect 2;\n=======\nselect 3;\n>>>>>>> other\n')
        with open(clean, 'w') as fh:
            fh.write('def keep_one():\n    return 1\n')
        errors = check_allowed_files_syntax([dup, conflicted, clean])
        self.assertIn("%s:3: duplicate dict key 'a'" % dup, errors)
        self.assertIn('%s:2: merge conflict marker left in file' % conflicted, errors)
        self.assertIn('%s:4: merge conflict marker left in file' % conflicted, errors)
        self.assertIn('%s:6: merge conflict marker left in file' % conflicted, errors)
        self.assertFalse(any(clean in err for err in errors), errors)

    def test_baseline_symbol_drift(self):
        clean, conflicted = self.path('clean.py'), self.path('query.sql')
        with open(clean, 'w') as fh:
            fh.write('def keep_one():\n    return 1\n')
        with open(conflicted, 'w') as fh:
            fh.write('<<<<<<< HEAD\n')
        baseline = collect_python_top_symbols([clean, conflicted])
        self.assertEqual(baseline, {clean: {'keep_one'}})  # .sql not inventoried
        self.assertEqual(check_allowed_files_syntax([clean], baseline_symbols=baseline), [])
        with open(clean, 'w') as fh:
            fh.write('def renamed():\n    return 2\n')
        errors = check_allowed_files_syntax([clean], baseline_symbols=baseline)
        self.assertIn('%s:1: dropped top-level symbol(s): keep_one' % clean, errors)

    def test_build_file_skeletons_range_map(self):
        big, small = self.path('big.py'), self.path('small.py')
        with open(big, 'w') as fh:
            fh.write('x = 0\n')
            for i in range(130):
                fh.write('def fn_%d():\n    return %d\n\n' % (i, i))
        with open(small, 'w') as fh:
            fh.write('a = 1\n')
        skeleton = build_file_skeletons([big, small], line_threshold=100)
        self.assertTrue(skeleton.startswith('[FILE SKELETON MAP -- read only target line ranges]'))
        self.assertIn('--- %s (391 lines)' % big, skeleton)
        self.assertIn('L5-L6: def fn_1()', skeleton)  # L<start>-L<end> map
        self.assertNotIn(small, skeleton)  # short files are skipped
        self.assertEqual(build_file_skeletons([small]), '')


class TestH7AndH8(HarnessTestCase):
    def test_sql_trailing_comma_and_parens(self):
        self.assertEqual(check_sql_syntax('select a, from t', 'q.sql'),
                         ['q.sql:1: trailing comma before FROM'])
        grouped = 'select id\nfrom t\ngroup by id,\norder by id\n'
        self.assertEqual(check_sql_syntax(grouped),
                         ['<sql>:3: trailing comma before ORDER BY'])
        self.assertEqual(check_sql_syntax('select (1 + (2\n'), ['<sql>:1: unbalanced parenthesis'])
        self.assertEqual(check_sql_syntax('select 1\n)\n'), ['<sql>:2: unbalanced parenthesis'])
        self.assertEqual(check_sql_syntax('select 1))\n'), ['<sql>:1: unbalanced parenthesis'])
        # Parens/commas inside strings, comments and Jinja never reach the checks.
        clean = ("select 'a,(b)' as s,\n"
                 "       b -- , from (comment\n"
                 "from {{ ref('m') }}\n"
                 "where a = 'it''s'\n")
        self.assertEqual(check_sql_syntax(clean), [])

    def test_sql_jinja_and_comment_string_balance(self):
        self.assertEqual(check_sql_syntax('{% if x %}\nselect 1\n'),
                         ['<sql>:1: unbalanced Jinja tag'])
        self.assertEqual(check_sql_syntax('{% for x in y %}\nselect 1\n{% endif %}\n'),
                         ['<sql>:3: unbalanced Jinja tag'])
        self.assertEqual(check_sql_syntax("select {{ ref('"), ['<sql>:1: unbalanced Jinja tag'])
        self.assertEqual(check_sql_syntax('select 1 %}'), ['<sql>:1: unbalanced Jinja tag'])
        self.assertEqual(check_sql_syntax("select 'abc\nfrom t"),
                         ['<sql>:1: unclosed SQL comment or string literal'])
        self.assertEqual(check_sql_syntax('select 1\n/* dangling\ntext'),
                         ['<sql>:2: unclosed SQL comment or string literal'])
        clean = ('{% if enabled %}\n'
                 "select * from {{ ref('model_x') }}\n"
                 "where ds = '{{ ds }}'  -- it''s (fine\n"
                 '{% endif %}\n')
        self.assertEqual(check_sql_syntax(clean), [])

    def test_broken_local_sibling_import(self):
        sibling, importer = self.path('helper.py'), self.path('importer.py')
        with open(sibling, 'w') as fh:
            fh.write('def real_fn():\n    return 1\n\n'
                     'class RealClass:\n    pass\n\n'
                     'CONST = 2\n__all__ = ["CONST"]\n')
        with open(importer, 'w') as fh:
            fh.write('from helper import real_fn, missing_name\n'
                     'from helper import CONST\n'
                     'from helper import *\n'
                     'from .helper import RealClass\n'
                     'import json\n')
        self.assertEqual(check_allowed_files_syntax([importer]),
                         ["%s:1: cannot import name 'missing_name' from 'helper'" % importer])
        # Unparseable sibling is skipped rather than guessed at.
        broken, bad_import = self.path('broken.py'), self.path('bad_import.py')
        with open(broken, 'w') as fh:
            fh.write('def broken(:\n')
        with open(bad_import, 'w') as fh:
            fh.write('from broken import anything\n')
        self.assertEqual(check_allowed_files_syntax([bad_import]), [])
        # Wired into the pre-flight for both .sql and .py.
        sql = self.path('wired.sql')
        with open(sql, 'w') as fh:
            fh.write('select a,\nfrom t\n')
        errors = check_allowed_files_syntax([sql, importer])
        self.assertIn('%s:1: trailing comma before FROM' % sql, errors)
        self.assertIn("%s:1: cannot import name 'missing_name' from 'helper'" % importer, errors)

    def test_compact_stale_reads(self):
        long_a, long_b = 'x' * 700, 'y' * 700
        msgs = [{'role': 'user', 'content': 'go'},
                {'role': 'assistant', 'tool_calls': [
                    {'id': 'r1', 'function': {'name': 'read_file',
                                              'arguments': json.dumps({'path': 'a.py'})}},
                    {'id': 'r2', 'function': {'name': 'view',
                                              'arguments': json.dumps({'file_path': 'b.py'})}}]},
                {'role': 'tool', 'tool_call_id': 'r1', 'content': long_a},
                {'role': 'tool', 'tool_call_id': 'r2', 'content': long_b},
                {'role': 'assistant', 'tool_calls': [
                    {'id': 'e1', 'function': {'name': 'multi_edit',
                                              'arguments': json.dumps({'path': 'a.py'})}},
                    {'id': 'e2', 'function': {'name': 'write',
                                              'arguments': json.dumps({'target_file': 'b.py'})}}]},
                {'role': 'tool', 'tool_call_id': 'e1', 'content': 'ok'},
                {'role': 'assistant', 'tool_calls': [
                    {'id': 'r3', 'function': {'name': 'read',
                                              'arguments': json.dumps({'path': 'a.py'})}}]},
                {'role': 'tool', 'tool_call_id': 'r3', 'content': 'z' * 500}]
        new, count = compact_stale_reads(msgs, turn_index=5)
        self.assertEqual(count, 2)
        self.assertIsNot(new, msgs)
        self.assertIn('[STALE READ COMPACTED: a.py was modified in a later turn', new[2]['content'])
        self.assertIn('[STALE READ COMPACTED: b.py was modified in a later turn', new[3]['content'])
        self.assertNotIn(long_a, new[2]['content'])
        self.assertEqual(msgs[2]['content'], long_a)  # original list untouched
        # Reads after the edit, edit results and short reads stay intact.
        self.assertTrue(new[7] is msgs[7] and new[7]['content'] == 'z' * 500)
        self.assertTrue(new[5] is msgs[5] and new[5]['content'] == 'ok')
        self.assertEqual(compact_stale_reads(msgs, turn_index=0), (msgs, 0))
        self.assertEqual(compact_stale_reads('nope', turn_index=5), ('nope', 0))
        # H-30: the trailing keep_recent_turns window is never rewritten.
        self.assertEqual(compact_stale_reads(msgs, turn_index=5, keep_recent_turns=9),
                         (msgs, 0))
        # H-30: below min_total_saved_chars nothing is mutated (prefix kept).
        new, count = compact_stale_reads(msgs, turn_index=5, min_total_saved_chars=2400)
        self.assertEqual(count, 0)
        self.assertIs(new, msgs)
        big = [{'role': 'user', 'content': 'go'},
               {'role': 'assistant', 'tool_calls': [
                   {'id': 'r1', 'function': {'name': 'read_file',
                                             'arguments': json.dumps({'path': 'a.py'})}},
                   {'id': 'r2', 'function': {'name': 'view',
                                             'arguments': json.dumps({'file_path': 'b.py'})}}]},
               {'role': 'tool', 'tool_call_id': 'r1', 'content': 'x' * 3000},
               {'role': 'tool', 'tool_call_id': 'r2', 'content': 'y' * 3000},
               {'role': 'assistant', 'tool_calls': [
                   {'id': 'e1', 'function': {'name': 'edit',
                                             'arguments': json.dumps({'path': 'a.py'})}},
                   {'id': 'e2', 'function': {'name': 'edit',
                                             'arguments': json.dumps({'path': 'b.py'})}}]},
               {'role': 'tool', 'tool_call_id': 'e1', 'content': 'ok'},
               {'role': 'tool', 'tool_call_id': 'e2', 'content': 'ok'},
               {'role': 'user', 'content': 'go on'},
               {'role': 'assistant', 'content': 'ok'},
               {'role': 'user', 'content': 'done'},
               {'role': 'assistant', 'content': 'finished'}]
        new, count = compact_stale_reads(big, turn_index=6, min_total_saved_chars=2400)
        self.assertEqual(count, 2)
        self.assertIn('[STALE READ COMPACTED: a.py', new[2]['content'])
        self.assertIn('[STALE READ COMPACTED: b.py', new[3]['content'])
        # Anthropic-style tool_result content blocks are compacted per block.
        blocks = [{'role': 'user', 'content': 'go'},
                  {'role': 'assistant', 'content': [
                      {'type': 'tool_use', 'id': 'u1', 'name': 'read_file',
                       'input': {'path': 'c.py'}}]},
                  {'role': 'user', 'content': [
                      {'type': 'tool_result', 'tool_use_id': 'u1', 'content': 'w' * 700},
                      {'type': 'text', 'text': 'summary'}]},
                  {'role': 'assistant', 'content': [
                      {'type': 'tool_use', 'id': 'u2', 'name': 'edit',
                       'input': {'path': 'c.py'}}]},
                  {'role': 'user', 'content': [
                      {'type': 'tool_result', 'tool_use_id': 'u2', 'content': 'done'}]},
                  {'role': 'assistant', 'content': 'continue'},
                  {'role': 'user', 'content': 'more'},
                  {'role': 'assistant', 'content': 'ok'},
                  {'role': 'user', 'content': 'done'}]
        new_blocks, count = compact_stale_reads(blocks, turn_index=6)
        self.assertEqual(count, 1)
        self.assertIn('[STALE READ COMPACTED: c.py', new_blocks[2]['content'][0]['content'])
        self.assertEqual(new_blocks[2]['content'][1]['text'], 'summary')
        self.assertEqual(new_blocks[4]['content'][0]['content'], 'done')


class TestH5LoopGuard(HarnessTestCase):
    def test_detect_duplicate_tool_calls(self):
        msgs = [{'role': 'user', 'content': 'fix it'},
                {'role': 'assistant', 'tool_calls': [
                    {'function': {'name': 'edit',
                                  'arguments': '{"path": "a.py", "old": "x"}'}}]},
                {'role': 'tool', 'content': 'ok'},
                {'role': 'assistant', 'tool_calls': [
                    {'function': {'name': 'edit',
                                  'arguments': '{ "old" : "x" , "path":"a.py" }'}}]}]
        self.assertEqual(detect_duplicate_tool_calls(msgs),
                         ['duplicate tool call edit repeated with identical arguments'])
        msgs[3]['tool_calls'][0]['function']['arguments'] = '{"path": "a.py", "old": "y"}'
        self.assertEqual(detect_duplicate_tool_calls(msgs), [])
        spaced = [{'role': 'user', 'content': 'go'},
                  {'role': 'assistant', 'tool_calls': [
                      {'function': {'name': 'edit', 'arguments': '{"path": "a.py"}'}}]},
                  {'role': 'tool', 'content': 'ok'},
                  {'role': 'assistant', 'tool_calls': [
                      {'function': {'name': 'read', 'arguments': '{"path": "a.py"}'}}]},
                  {'role': 'tool', 'content': 'ok'},
                  {'role': 'assistant', 'tool_calls': [
                      {'function': {'name': 'edit', 'arguments': '{"path": "a.py"}'}}]}]
        self.assertEqual(detect_duplicate_tool_calls(spaced), [])

    def test_detect_edit_oscillation_revert_cycle(self):
        def edit(path, old, new):
            return {'role': 'assistant', 'tool_calls': [
                {'function': {'name': 'edit',
                              'arguments': json.dumps({'filePath': path,
                                                       'oldString': old,
                                                       'newString': new})}}]}

        msgs = [{'role': 'user', 'content': 'fix it'},
                edit('a.py', 'x = 1', 'x = 2'),
                {'role': 'tool', 'content': 'ok'},
                edit('a.py', 'x = 2', 'x = 1')]
        self.assertEqual(detect_edit_oscillation(msgs),
                         {'oscillating': True, 'file': 'a.py', 'reason': 'revert_cycle'})
        # A non-reversing second edit is not oscillation.
        msgs[-1] = edit('a.py', 'x = 2', 'x = 3')
        self.assertIsNone(detect_edit_oscillation(msgs))
        # Reversal on a different file is not oscillation either.
        msgs[-1] = edit('b.py', 'x = 2', 'x = 1')
        self.assertIsNone(detect_edit_oscillation(msgs))

    def test_detect_edit_oscillation_untested_edit_streak(self):
        def edit(path, old, new):
            return {'role': 'assistant', 'tool_calls': [
                {'function': {'name': 'edit',
                              'arguments': json.dumps({'filePath': path,
                                                       'oldString': old,
                                                       'newString': new})}}]}

        msgs = [{'role': 'user', 'content': 'fix it'},
                edit('a.py', 'a', 'b'),
                edit('a.py', 'b', 'c'),
                edit('a.py', 'c', 'd')]
        self.assertIsNone(detect_edit_oscillation(msgs))
        msgs.append(edit('a.py', 'd', 'e'))
        self.assertEqual(detect_edit_oscillation(msgs),
                         {'oscillating': True, 'file': 'a.py', 'reason': 'untested_edit_streak'})
        # A test run resets the streak.
        tested = msgs[:4] + [{'role': 'assistant', 'tool_calls': [
            {'function': {'name': 'bash', 'arguments': json.dumps({'command': 'pytest -q'})}}]},
            edit('a.py', 'e', 'f')]
        self.assertIsNone(detect_edit_oscillation(tested))

    def test_proxy_injects_plateau_guard_notice_and_span(self):
        upstream, captured = _scripted_upstream([{'status': 200}])
        proxy = OpenRouterGuardProxy(lambda: None, upstream_url='http://127.0.0.1:%d' % upstream.server_port)
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
        msgs = [{'role': 'user', 'content': 'fix it'},
                {'role': 'assistant', 'tool_calls': [
                    {'function': {'name': 'edit',
                                  'arguments': json.dumps({'path': 'a.py', 'old': 'x = 1',
                                                           'new': 'x = 2'})}}]},
                {'role': 'tool', 'content': 'ok'},
                {'role': 'assistant', 'tool_calls': [
                    {'function': {'name': 'edit',
                                  'arguments': json.dumps({'path': 'a.py', 'old': 'x = 2',
                                                           'new': 'x = 1'})}}]}]
        body = json.dumps({'model': GLM, 'messages': msgs}).encode()
        try:
            _post_chat(proxy.base_url, body)
        finally:
            proxy.shutdown(); proxy.server_close()
            upstream.shutdown(); upstream.server_close()
        self.assertEqual(len(captured), 1)
        forwarded = captured[0]['messages'][-1]['content']
        self.assertIn('[HARNESS PLATEAU GUARD: you are oscillating on a.py (revert_cycle)', forwarded)
        spans = [s for s in proxy.summary()['spans']
                 if s.get('name') == 'guard.edit_oscillation']
        self.assertEqual(len(spans), 1)
        span = spans[0]
        attrs = span.get('attributes', {})
        self.assertTrue(attrs.get('guardrail.triggered'))
        self.assertEqual(attrs.get('guardrail.file'), 'a.py')
        self.assertEqual(attrs.get('guardrail.reason'), 'revert_cycle')

    def test_proxy_injects_guard_loop_warning(self):
        upstream, captured = _scripted_upstream([{'status': 200}] * 2)
        proxy = OpenRouterGuardProxy(lambda: None, upstream_url='http://127.0.0.1:%d' % upstream.server_port)
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
        try:
            dup_msgs = [{'role': 'user', 'content': 'fix it'},
                        {'role': 'assistant', 'tool_calls': [
                            {'function': {'name': 'edit', 'arguments': '{"path": "a.py"}'}}]},
                        {'role': 'tool', 'content': 'ok'},
                        {'role': 'assistant', 'tool_calls': [
                            {'function': {'name': 'edit', 'arguments': '{ "path" : "a.py" }'}}]}]
            body = json.dumps({'model': GLM, 'messages': dup_msgs}).encode()
            for _ in range(2):
                _post_chat(proxy.base_url, body)
        finally:
            proxy.shutdown(); proxy.server_close()
            upstream.shutdown(); upstream.server_close()
        self.assertEqual(len(captured), 2)
        self.assertNotIn('GUARD LOOP WARNING', json.dumps(captured[0]))
        warning = captured[1]['messages'][-1]['content']
        self.assertIn('[GUARD LOOP WARNING: duplicate tool call edit repeated with identical arguments', warning)


class TestProxyCircuitBreaker(HarnessTestCase):
    def test_sticky_reset_and_demotion_on_upstream_500(self):
        upstream, captured = _scripted_upstream([
            {'status': 200, 'provider': 'z-ai'},
            {'status': 500, 'provider': 'z-ai'},
            {'status': 200, 'provider': 'z-ai'},
        ])
        proxy = OpenRouterGuardProxy(lambda: None, upstream_url='http://127.0.0.1:%d' % upstream.server_port)
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
        try:
            body = json.dumps({'model': GLM, 'messages': [{'role': 'user', 'content': 'hi'}]}).encode()
            _post_chat(proxy.base_url, body)          # healthy turn -> provider pinned
            self.assertEqual(proxy.sticky_provider, 'z-ai')
            _post_chat(proxy.base_url, body)          # 500 -> circuit breaker demotes
            self.assertEqual(proxy.degraded_providers, {'z-ai'})
            self.assertIsNone(proxy.sticky_provider)  # degradation resets the pin
            _post_chat(proxy.base_url, body)          # next request routes around it
        finally:
            proxy.shutdown(); proxy.server_close()
            upstream.shutdown(); upstream.server_close()
        self.assertEqual(captured[2]['provider']['order'][-1], 'z-ai')  # demoted to last
        self.assertEqual(captured[2]['provider']['order'][0], 'novita')


class TestReportAndCLI(HarnessTestCase):
    RUNS = [{'task_id': 't1', 'status': 'succeeded', 'model': GLM, 'steps': 3, 'cost': 0.02},
            {'task_id': 't2', 'status': 'succeeded', 'model': GLM, 'steps': 4, 'cost': 0.03,
             'micro_repair_used': True},
            {'task_id': 't3', 'status': 'validation_failed', 'model': DEEPSEEK, 'steps': 2,
             'cost_usd': 0.01}]

    def test_aggregate_benchmark_runs(self):
        report = aggregate_benchmark_runs(self.RUNS)
        self.assertEqual(report['total_runs'], 3)
        self.assertEqual(report['pass_count'], 2)
        self.assertAlmostEqual(report['pass_rate'], 2 / 3)
        self.assertAlmostEqual(report['total_cost_usd'], 0.06)
        self.assertEqual(report['median_steps'], 3.0)
        self.assertEqual(report['micro_repair_count'], 1)
        normalized = summarize_run({'status': 'Succeeded', 'cost': '0.02', 'steps': '3'})
        self.assertEqual(normalized['status'], 'succeeded')
        self.assertEqual(normalized['cost_usd'], 0.02)

    def test_format_benchmark_markdown(self):
        md = format_benchmark_markdown([], self.RUNS)
        self.assertIn('# Benchmark Comparison: Baseline vs Guarded Harness', md)
        self.assertIn('| Pass@1 (%) | 0.0 | 66.7 |', md)
        self.assertIn('| Total Cost (USD) | 0.0000 | 0.0600 |', md)
        self.assertIn('Runs compared: baseline 0, guarded 3.', md)

    def test_cli_preflight(self):
        clean, bad = self.path('ok.py'), self.path('nope.py')
        with open(clean, 'w') as fh:
            fh.write('a = 1\n')
        with open(bad, 'w') as fh:
            fh.write('def x(:\n')
        self.assertEqual(cli_main(['preflight', clean]), 0)
        self.assertEqual(cli_main(['preflight', bad]), 1)

    def test_cli_skeleton(self):
        big = self.path('big.py')
        with open(big, 'w') as fh:
            fh.write('x = 0\n')
            for i in range(20):
                fh.write('def fn_%d():\n    return %d\n\n' % (i, i))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(cli_main(['skeleton', big, '--threshold', '10']), 0)
        self.assertIn('FILE SKELETON MAP', buf.getvalue())
        self.assertIn('L5-L6: def fn_1()', buf.getvalue())

    def test_cli_bench_report(self):
        baseline = self.path('baseline.json')
        guarded = self.path('guarded.json')
        with open(baseline, 'w') as fh:
            json.dump(self.RUNS, fh)
        with open(guarded, 'w') as fh:
            json.dump(self.RUNS[:1], fh)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(cli_main(['bench-report', '--baseline', baseline, '--guarded', guarded]), 0)
        self.assertIn('# Benchmark Comparison: Baseline vs Guarded Harness', buf.getvalue())


class TestH6ProviderHealthStore(HarnessTestCase):
    def test_store_persistence_ttl_and_custom_clock(self):
        self.assertEqual(DEFAULT_PROVIDER_HEALTH_TTL_SECONDS, 600.0)
        path = self.path('provider_health.json')
        clock = {'now': 1000.0}
        store = ProviderHealthStore(path, ttl_seconds=300)
        store.set_sticky('z-ai', now=clock['now'])
        self.assertEqual(store.get_sticky(now=clock['now']), 'z-ai')
        store.mark_degraded('z-ai', now=clock['now'])
        self.assertEqual(store.active_degraded(now=clock['now']), {'z-ai'})
        self.assertIsNone(store.get_sticky(now=clock['now']))  # degraded is never pinned
        # A second instance over the same file inherits the persisted state.
        reloaded = ProviderHealthStore(path, ttl_seconds=300)
        self.assertEqual(reloaded.active_degraded(now=clock['now']), {'z-ai'})
        self.assertIsNone(reloaded.get_sticky(now=clock['now']))
        # TTL expiration driven by custom now timestamps.
        clock['now'] = 1301.0  # 301s after mark_degraded(1000) > ttl 300
        self.assertEqual(reloaded.active_degraded(now=clock['now']), set())
        self.assertIsNone(reloaded.get_sticky(now=clock['now']))
        reloaded.set_sticky('novita', now=clock['now'])
        self.assertEqual(reloaded.get_sticky(now=clock['now']), 'novita')
        clock['now'] = 1602.0  # 301s after set_sticky(1301): sticky expired too
        self.assertIsNone(reloaded.get_sticky(now=clock['now']))
        self.assertEqual(reloaded.active_degraded(now=clock['now']), set())

    def test_second_proxy_instance_inherits_store_state(self):
        path = self.path('provider_health.json')
        upstream, captured = _scripted_upstream([
            {'status': 200, 'provider': 'z-ai'},
            {'status': 500, 'provider': 'z-ai'},
            {'status': 200, 'provider': 'novita'},
        ])
        url = 'http://127.0.0.1:%d' % upstream.server_port
        body = json.dumps({'model': GLM, 'messages': [{'role': 'user', 'content': 'hi'}]}).encode()

        def serve(proxy):
            threading.Thread(target=proxy.serve_forever, daemon=True).start()

        first = OpenRouterGuardProxy(lambda: None, upstream_url=url,
                                     health_store=ProviderHealthStore(path))
        serve(first)
        try:
            _post_chat(first.base_url, body)  # healthy turn -> pinned in memory and store
            self.assertEqual(first.sticky_provider, 'z-ai')
            self.assertEqual(ProviderHealthStore(path).get_sticky(), 'z-ai')
            _post_chat(first.base_url, body)  # 500 -> demoted and persisted
            self.assertEqual(first.degraded_providers, {'z-ai'})
            peek = ProviderHealthStore(path)
            self.assertEqual(peek.active_degraded(), {'z-ai'})
            self.assertIsNone(peek.get_sticky())  # degradation cleared the pin on disk too
        finally:
            first.shutdown(); first.server_close()

        second = OpenRouterGuardProxy(lambda: None, upstream_url=url,
                                      health_store=ProviderHealthStore(path))
        serve(second)
        try:
            # The second instance hydrates degraded + sticky state from the store.
            self.assertEqual(second.degraded_providers, {'z-ai'})
            self.assertIsNone(second.sticky_provider)
            _post_chat(second.base_url, body)  # inherited degradation steers routing
            self.assertEqual(captured[2]['provider']['order'][0], 'novita')
            self.assertEqual(captured[2]['provider']['order'][-1], 'z-ai')
            self.assertEqual(second.sticky_provider, 'novita')  # healthy fallback gets pinned
        finally:
            second.shutdown(); second.server_close()
            upstream.shutdown(); upstream.server_close()

        # A third instance inherits BOTH persisted facts: degraded z-ai and the
        # sticky novita pin recorded while z-ai was demoted.
        third_store = ProviderHealthStore(path)
        self.assertEqual(third_store.active_degraded(), {'z-ai'})
        self.assertEqual(third_store.get_sticky(), 'novita')
        third = OpenRouterGuardProxy(lambda: None, upstream_url=url, health_store=third_store)
        self.assertEqual(third.degraded_providers, {'z-ai'})
        self.assertEqual(third.sticky_provider, 'novita')
        third.server_close()


class TestH8ProxyStaleReadCompaction(HarnessTestCase):
    def test_proxy_compacts_stale_reads_on_epoch_turns(self):
        upstream, captured = _scripted_upstream([{'status': 200}] * 7)
        proxy = OpenRouterGuardProxy(lambda: None, upstream_url='http://127.0.0.1:%d' % upstream.server_port)
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
        long_read = 'x' * 3000
        msgs = [{'role': 'user', 'content': 'update the file'},
                {'role': 'assistant', 'tool_calls': [
                    {'id': 'r1', 'function': {'name': 'read_file',
                                              'arguments': json.dumps({'path': 'a.py'})}}]},
                {'role': 'tool', 'tool_call_id': 'r1', 'content': long_read},
                {'role': 'assistant', 'tool_calls': [
                    {'id': 'e1', 'function': {'name': 'edit',
                                              'arguments': json.dumps({'path': 'a.py'})}}]},
                {'role': 'tool', 'tool_call_id': 'e1', 'content': 'ok'},
                {'role': 'user', 'content': 'keep going'},
                {'role': 'assistant', 'content': 'working'},
                {'role': 'user', 'content': 'status?'},
                {'role': 'assistant', 'content': 'done'}]
        body = json.dumps({'model': GLM, 'messages': msgs}).encode()
        try:
            for _ in range(7):
                _post_chat(proxy.base_url, body)
        finally:
            proxy.shutdown(); proxy.server_close()
            upstream.shutdown(); upstream.server_close()
        self.assertEqual(len(captured), 7)
        # H-30: only the epoch turn (turn_index >= 6 and turn_index % 3 == 0)
        # busts the KV-cache prefix; turns 1-5 (captured[0..4]) and non-epoch
        # turn 7 (captured[6]) forward the original body untouched.
        self.assertEqual(proxy.summary()['compacted_reads'], 1)
        self.assertNotIn('STALE READ COMPACTED', json.dumps(captured[2]))  # turn 3 (< 6) untouched
        self.assertNotIn('[STALE READ COMPACTED', json.dumps(captured[0]))  # turn 1 untouched
        self.assertNotIn('[STALE READ COMPACTED', json.dumps(captured[4]))  # turn 5 untouched
        forwarded = captured[5]['messages'][2]['content']  # turn 6 (epoch) compacted
        self.assertIn('[STALE READ COMPACTED: a.py was modified in a later turn', forwarded)
        self.assertNotIn(long_read, json.dumps(captured[5]))
        self.assertNotIn('[STALE READ COMPACTED', json.dumps(captured[6]))  # turn 7 untouched
        # Later messages (edit result, recent turns) pass through untouched.
        self.assertEqual(captured[5]['messages'][4]['content'], 'ok')
        self.assertEqual(captured[5]['messages'][8]['content'], 'done')


class TestTelemetry(HarnessTestCase):
    def test_proxy_records_openinference_and_genai_spans(self):
        self.assertIn('LLM', OPENINFERENCE_SPAN_KINDS)
        self.assertIn('GUARDRAIL', OPENINFERENCE_SPAN_KINDS)
        bad_py = self.path('bad.py')
        with open(bad_py, 'w') as fh:
            fh.write('def broken(:\n')
        upstream, captured = _scripted_upstream([{'status': 200}] * 2)
        tracer = HarnessTracer(service_name='harness-tests')
        proxy = OpenRouterGuardProxy(lambda: None, upstream_url='http://127.0.0.1:%d' % upstream.server_port,
                                     allowed_files=[bad_py], tracer=tracer)
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
        try:
            body = json.dumps({'model': GLM, 'messages': [{'role': 'user', 'content': 'go'}]}).encode()
            for _ in range(2):
                _post_chat(proxy.base_url, body)
        finally:
            proxy.shutdown(); proxy.server_close()
            upstream.shutdown(); upstream.server_close()
        spans = proxy.summary()['spans']
        kinds = {span['attributes']['openinference.span.kind'] for span in spans}
        self.assertEqual(kinds, {'LLM', 'GUARDRAIL'})
        llm = next(s for s in spans if s['attributes']['openinference.span.kind'] == 'LLM')
        self.assertEqual(llm['attributes']['gen_ai.system'], 'openrouter')
        self.assertEqual(llm['attributes']['gen_ai.request.model'], GLM)
        self.assertIn('gen_ai.response.id', llm['attributes'])
        guard = next(s for s in spans if s['attributes']['openinference.span.kind'] == 'GUARDRAIL')
        self.assertEqual(guard['attributes']['guardrail.name'], 'ast_preflight')
        # The recorded spans convert into valid OTLP/HTTP JSON resourceSpans.
        payload = tracer.to_otlp_payload()
        json.dumps(payload)  # must be JSON-serializable end to end
        resource = payload['resourceSpans']
        self.assertEqual(len(resource), 1)
        otlp_spans = resource[0]['scopeSpans'][0]['spans']
        self.assertEqual(len(otlp_spans), len(spans))
        for otlp_span in otlp_spans:
            self.assertEqual(len(otlp_span['traceId']), 32)
            self.assertEqual(len(otlp_span['spanId']), 16)
            self.assertIsInstance(otlp_span['startTimeUnixNano'], str)
        by_name = {s['name']: {a['key']: next(iter(a['value'].values()))
                               for a in s['attributes']} for s in otlp_spans}
        self.assertEqual(by_name['gen_ai.chat']['openinference.span.kind'], 'LLM')
        self.assertEqual(by_name['gen_ai.chat']['gen_ai.request.model'], GLM)
        tree = format_trace_tree(tracer.spans)
        self.assertIn('trace %s' % tracer.trace_id, tree)
        self.assertIn('[LLM] gen_ai.chat', tree)
        self.assertIn('[GUARDRAIL] guard.ast_preflight', tree)

    def test_flush_otlp_sends_arize_headers_to_stub_collector(self):
        received = {}

        class _Collector(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                return

            def do_POST(self):
                length = int(self.headers.get('Content-Length') or '0')
                received['body'] = json.loads(self.rfile.read(length))
                received['api_key'] = self.headers.get('api_key')
                received['space_id'] = self.headers.get('space_id')
                payload = b'{}'
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), _Collector)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        tracer = HarnessTracer(service_name='harness-tests', arize_api_key='k-test',
                               arize_space_id='s-test')
        tracer.record_span('gen_ai.chat', kind='LLM', attributes={'gen_ai.request.model': GLM})
        try:
            ok = tracer.flush_otlp(endpoint='http://127.0.0.1:%d' % server.server_port)
        finally:
            server.shutdown(); server.server_close()
        self.assertTrue(ok)
        self.assertEqual(received['api_key'], 'k-test')
        self.assertEqual(received['space_id'], 's-test')
        spans = received['body']['resourceSpans'][0]['scopeSpans'][0]['spans']
        self.assertEqual(spans[0]['name'], 'gen_ai.chat')

    def test_flush_otlp_sends_protobuf_by_default_and_falls_back_on_415(self):
        seen = []

        class _ProtobufCollector(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                return

            def do_POST(self):
                length = int(self.headers.get('Content-Length') or '0')
                seen.append((self.headers.get('Content-Type'), self.rfile.read(length)))
                self.send_response(200)
                self.end_headers()

        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), _ProtobufCollector)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        tracer = HarnessTracer(service_name='pb-test', trace_id='01' * 16)
        tracer.record_span('harness.task', kind='AGENT', start_time_ns=1000, end_time_ns=5000,
                           status='ERROR', attributes={'llm.cost.total': 0.05, 'harness.steps': 3})
        try:
            ok = tracer.flush_otlp(endpoint='http://127.0.0.1:%d/v1/traces' % server.server_port)
        finally:
            server.shutdown()
            server.server_close()
        self.assertTrue(ok)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][0], 'application/x-protobuf')
        self.assertIn(bytes.fromhex('01' * 16), seen[0][1])
        self.assertIn(b'harness.task', seen[0][1])

        # Verify 415 fallback to application/json.
        seen_fb = []

        class _JsonOnlyCollector(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                return

            def do_POST(self):
                length = int(self.headers.get('Content-Length') or '0')
                ctype = self.headers.get('Content-Type')
                body = self.rfile.read(length)
                seen_fb.append(ctype)
                if ctype == 'application/x-protobuf':
                    self.send_response(415)
                    self.end_headers()
                    return
                json.loads(body)
                self.send_response(200)
                self.end_headers()

        srv2 = http.server.ThreadingHTTPServer(('127.0.0.1', 0), _JsonOnlyCollector)
        threading.Thread(target=srv2.serve_forever, daemon=True).start()
        try:
            ok2 = tracer.flush_otlp(endpoint='http://127.0.0.1:%d/v1/traces' % srv2.server_port)
        finally:
            srv2.shutdown()
            srv2.server_close()
        self.assertTrue(ok2)
        self.assertEqual(seen_fb, ['application/x-protobuf', 'application/json'])

    def test_synthesize_spans_openinference_status_tokens_cost_and_resynthesis(self):
        from pathlib import Path
        from harness.telemetry import synthesize_spans_from_run_dir

        run_dir = Path(self.path('synth-run'))
        run_dir.mkdir()
        (run_dir / 'task.json').write_text(json.dumps({
            'task_id': 't-synth', 'model': GLM, 'files': ['a.py'], 'prompt': 'Fix a.py',
        }))
        (run_dir / 'result.json').write_text(json.dumps({
            'task_id': 't-synth', 'status': 'blocked', 'reason': 'step_limit',
            'model': GLM, 'cost': 0.012, 'steps': 2, 'test_return_code': 1,
        }))
        (run_dir / 'events.jsonl').write_text(
            json.dumps({'time': 1700000000.0, 'event': 'run_started'}) + '\n'
            + json.dumps({'time': 1700000010.5, 'event': 'run_finished'}) + '\n'
        )
        (run_dir / 'requests.json').write_text(json.dumps({
            'requests': [{
                'ordinal': 1,
                'time': {'created': 1700000001000, 'completed': 1700000004000},
                'state': 'finished',
                'finish': 'stop',
                'cost': 0.012,
                'tokens': {'input': 100, 'output': 20, 'reasoning': 5, 'cache': {'read': 50, 'write': 0}},
                'provider_request_ids': ['gen-1'],
            }]
        }))
        # Seed an old incomplete traces.jsonl to verify automatic re-synthesis.
        (run_dir / 'traces.jsonl').write_text(json.dumps({
            'name': 'harness.task', 'span_id': 'a' * 16, 'trace_id': 'b' * 32,
            'parent_span_id': None, 'start_time_ns': 100, 'end_time_ns': 100,
            'status': 'OK', 'attributes': {'openinference.span.kind': 'AGENT'},
        }) + '\n')

        spans = synthesize_spans_from_run_dir(run_dir)
        self.assertEqual(len(spans), 2)
        root, child = spans[0], spans[1]
        self.assertEqual(root['name'], 'harness.task')
        self.assertEqual(root['status'], 'ERROR')
        self.assertEqual(root['status_message'], 'step_limit')
        self.assertEqual(root['end_time_ns'] - root['start_time_ns'], int(10.5 * 1e9))
        self.assertEqual(root['attributes']['otel.status_code'], 'ERROR')
        self.assertEqual(root['attributes']['otel.status_description'], 'step_limit')
        self.assertEqual(root['attributes']['llm.cost.total'], 0.012)
        self.assertEqual(root['attributes']['llm.token_count.prompt'], 150)
        self.assertEqual(root['attributes']['llm.token_count.completion'], 25)
        self.assertEqual(root['attributes']['llm.token_count.total'], 175)
        self.assertIn('Fix a.py', root['attributes']['input.value'])
        self.assertIn('step_limit', root['attributes']['output.value'])

        self.assertEqual(child['name'], 'gen_ai.chat')
        self.assertEqual(child['end_time_ns'] - child['start_time_ns'], 3_000_000_000)
        self.assertEqual(child['attributes']['llm.token_count.total'], 175)
        self.assertEqual(child['attributes']['llm.cost.total'], 0.012)

    def test_flush_otlp_disabled_env(self):
        seen = []

        class _Collector(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                return

            def do_POST(self):
                length = int(self.headers.get('Content-Length') or '0')
                self.rfile.read(length)
                seen.append(1)
                self.send_response(200)
                self.end_headers()

        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), _Collector)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        tracer = HarnessTracer(service_name='otel-disabled-test')
        tracer.record_span('gen_ai.chat', kind='LLM', attributes={'gen_ai.request.model': GLM})
        try:
            blocked = tracer.flush_otlp(endpoint=None)
            self.assertFalse(blocked)
            self.assertEqual(seen, [])
            explicit = tracer.flush_otlp(endpoint='http://127.0.0.1:%d' % server.server_port)
        finally:
            server.shutdown()
            server.server_close()
        self.assertTrue(explicit)
        self.assertEqual(len(seen), 1)

    def test_sync_state_traces_ledger_base_and_disabled_skip(self):
        from pathlib import Path
        from harness.telemetry import sync_state_traces_to_phoenix

        state_dir = Path(self.path('state'))
        seen = []

        class _Collector(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                return

            def do_POST(self):
                length = int(self.headers.get('Content-Length') or '0')
                self.rfile.read(length)
                seen.append(1)
                self.send_response(200)
                self.end_headers()

        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), _Collector)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        endpoint = 'http://127.0.0.1:%d' % server.server_port

        def make_run(run_id):
            run_dir = state_dir / 'runs' / run_id
            run_dir.mkdir(parents=True)
            (run_dir / 'task.json').write_text(json.dumps({
                'task_id': run_id, 'model': GLM, 'files': ['a.py'], 'prompt': 'Fix a.py',
            }))
            (run_dir / 'result.json').write_text(json.dumps({
                'task_id': run_id, 'status': 'succeeded', 'model': GLM,
                'cost': 0.01, 'steps': 1,
            }))
            (run_dir / 'events.jsonl').write_text(
                json.dumps({'time': 1700000000.0, 'event': 'run_started'}) + '\n'
                + json.dumps({'time': 1700000010.0, 'event': 'run_finished'}) + '\n'
            )
            return run_dir

        make_run('run-sync-1')
        try:
            result = sync_state_traces_to_phoenix(state_dir / 'runs' / 'run-sync-1',
                                                  endpoint=endpoint)
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(result['newly_synced'], 1)
        self.assertEqual(len(seen), 1)
        # The ledger lands next to `runs/`, not inside the run directory.
        ledger = state_dir / 'phoenix-sync-state.json'
        self.assertTrue(ledger.is_file())
        synced = json.loads(ledger.read_text())['synced']
        self.assertIn('run-sync-1', synced)
        self.assertFalse((state_dir / 'runs' / 'run-sync-1'
                          / 'phoenix-sync-state.json').exists())

        # HARNESS_OTEL_DISABLED=1 (set in setUp) skips the network entirely.
        make_run('run-sync-2')
        blocked = sync_state_traces_to_phoenix(state_dir, endpoint=None)
        self.assertEqual(blocked['newly_synced'], 0)
        self.assertEqual(blocked['pending_unsynced'], 1)
        self.assertFalse(blocked['collector_reachable'])
        self.assertEqual(len(seen), 1)  # no additional POST
        synced2 = json.loads(ledger.read_text())['synced']
        self.assertNotIn('run-sync-2', synced2)

    def test_record_tool_span_and_format_trace_tree(self):
        tracer = HarnessTracer(service_name='tool-test', trace_id='02' * 16)
        tracer.record_span('harness.task', kind='AGENT', start_time_ns=1_000_000, end_time_ns=5_000_000)
        long_arg = 'x' * 2500
        long_out = 'y' * 2500
        ok_span = tracer.record_tool_span(
            'edit',
            arguments={'file': 'a.py', 'data': long_arg},
            output=long_out,
            status='OK',
            start_time_ns=1_500_000,
            end_time_ns=2_500_000,
            step_index=1,
        )
        err_span = tracer.record_tool_span(
            'bash',
            arguments='pytest -q',
            output='exit code 1',
            status='ERROR',
            start_time_ns=2_600_000,
            end_time_ns=3_000_000,
        )
        self.assertEqual(ok_span['parent_span_id'], tracer.root_span_id)
        self.assertEqual(ok_span['attributes']['openinference.span.kind'], 'TOOL')
        self.assertEqual(ok_span['attributes']['tool.name'], 'edit')
        self.assertEqual(len(ok_span['attributes']['tool.parameters']), 2000)
        self.assertEqual(len(ok_span['attributes']['output.value']), 2000)
        self.assertEqual(ok_span['attributes']['harness.turn_index'], 1)
        self.assertEqual(err_span['status'], 'ERROR')

        # Parent tool spans to the recorded root span for tree rendering.
        root = tracer.spans[0]
        ok_span['parent_span_id'] = root['span_id']
        err_span['parent_span_id'] = root['span_id']
        tree = format_trace_tree(tracer.spans)
        self.assertIn('[TOOL] edit (OK)', tree)
        self.assertIn('[TOOL] bash (ERROR)', tree)

    def test_synthesize_spans_from_run_dir_with_tool_spans(self):
        from pathlib import Path
        from harness.telemetry import _deterministic_hex, synthesize_spans_from_run_dir

        run_dir = Path(self.path('synth-tools-run'))
        run_dir.mkdir()
        (run_dir / 'task.json').write_text(json.dumps({
            'task_id': 't-tools', 'model': GLM, 'files': ['a.py'], 'prompt': 'Edit a.py',
        }))
        (run_dir / 'result.json').write_text(json.dumps({
            'task_id': 't-tools', 'status': 'succeeded', 'model': GLM, 'cost': 0.01, 'steps': 1,
        }))
        (run_dir / 'requests.json').write_text(json.dumps({
            'requests': [{
                'ordinal': 1,
                'time': {'created': 1700000000000, 'completed': 1700000005000},
                'state': 'finished',
                'cost': 0.01,
                'tokens': {'input': 50, 'output': 10, 'reasoning': 0, 'cache': {'read': 0, 'write': 0}},
            }],
            'messages': [
                {
                    'role': 'assistant',
                    'parts': [
                        {
                            'type': 'tool',
                            'tool': 'edit',
                            'state': {
                                'status': 'completed',
                                'input': {'filePath': 'a.py'},
                                'output': 'edited a.py',
                                'time': {'start': 1700000001000, 'end': 1700000002000},
                            },
                        },
                        {
                            'type': 'tool',
                            'tool': 'bash',
                            'state': {
                                'status': 'error',
                                'input': {'command': 'pytest'},
                                'error': 'failed',
                                'time': {'start': 1700000002500, 'end': 1700000003500},
                            },
                        },
                    ],
                },
                {
                    'role': 'assistant',
                    'tool_calls': [
                        {
                            'id': 'call_3',
                            'type': 'function',
                            'function': {'name': 'read', 'arguments': '{"path": "a.py"}'},
                        },
                    ],
                },
            ],
        }))

        spans = synthesize_spans_from_run_dir(run_dir)
        tool_spans = [s for s in spans if (s.get('attributes') or {}).get('openinference.span.kind') == 'TOOL']
        llm_spans = [s for s in spans if (s.get('attributes') or {}).get('openinference.span.kind') == 'LLM']
        self.assertEqual(len(llm_spans), 1)
        self.assertEqual(len(tool_spans), 3)
        self.assertEqual(tool_spans[0]['span_id'], _deterministic_hex('synth-tools-run:tool:0', 16))
        self.assertEqual(tool_spans[1]['span_id'], _deterministic_hex('synth-tools-run:tool:1', 16))
        self.assertEqual(tool_spans[2]['span_id'], _deterministic_hex('synth-tools-run:tool:2', 16))
        self.assertEqual(tool_spans[0]['attributes']['tool.name'], 'edit')
        self.assertEqual(tool_spans[0]['status'], 'OK')
        self.assertIn('filePath', tool_spans[0]['attributes']['tool.parameters'])
        self.assertEqual(tool_spans[0]['attributes']['output.value'], 'edited a.py')
        self.assertEqual(tool_spans[1]['attributes']['tool.name'], 'bash')
        self.assertEqual(tool_spans[1]['status'], 'ERROR')
        self.assertEqual(tool_spans[2]['attributes']['tool.name'], 'read')

        tree = format_trace_tree(spans)
        self.assertIn('[TOOL] edit (OK)', tree)
        self.assertIn('[TOOL] bash (ERROR)', tree)
        self.assertIn('[TOOL] read (OK)', tree)

    def test_worker_log_tool_spans_resynthesis_and_replay(self):
        from pathlib import Path

        from harness.report import simulate_replay
        from harness.telemetry import _needs_resynthesis, synthesize_spans_from_run_dir

        # --- synthesize_spans_from_run_dir extracts TOOL spans from worker.log ---
        run_dir = Path(self.path('worker-log-run'))
        run_dir.mkdir()
        (run_dir / 'task.json').write_text(json.dumps({
            'task_id': 't-wlog', 'model': GLM, 'files': ['a.py'], 'prompt': 'Edit a.py',
        }))
        (run_dir / 'result.json').write_text(json.dumps({
            'task_id': 't-wlog', 'status': 'succeeded', 'model': GLM, 'cost': 0.02, 'steps': 1,
        }))
        (run_dir / 'requests.json').write_text(json.dumps({
            'requests': [{
                'ordinal': 1,
                'time': {'created': 1700000000000, 'completed': 1700000005000},
                'state': 'finished',
                'cost': 0.02,
                'tokens': {'input': 40, 'output': 8, 'reasoning': 0,
                           'cache': {'read': 0, 'write': 0}},
            }],
        }))
        tool_line = {
            'type': 'tool_use', 'timestamp': 1700000001000,
            'part': {'type': 'tool', 'tool': 'read',
                     'state': {'status': 'completed', 'input': {'path': 'a.py'},
                               'output': 'contents of a.py'}},
        }
        (run_dir / 'worker.log').write_text(json.dumps(tool_line) + '\n' + 'not json\n')

        # No TOOL spans + tool events in worker.log -> resynthesis required.
        self.assertTrue(_needs_resynthesis([], run_dir))
        spans = synthesize_spans_from_run_dir(run_dir)
        tool_spans = [s for s in spans
                      if (s.get('attributes') or {}).get('openinference.span.kind') == 'TOOL']
        self.assertEqual(len(tool_spans), 1)
        self.assertEqual(tool_spans[0]['attributes']['tool.name'], 'read')
        self.assertEqual(tool_spans[0]['attributes']['output.value'], 'contents of a.py')

        cached_llm_only = [{
            'name': 'gen_ai.chat', 'span_id': 'a' * 16, 'trace_id': 'b' * 32,
            'parent_span_id': None, 'start_time_ns': 1000, 'end_time_ns': 2000,
            'status': 'OK',
            'attributes': {'openinference.span.kind': 'LLM', 'gen_ai.request.model': GLM},
        }]
        self.assertTrue(_needs_resynthesis(cached_llm_only, run_dir))
        self.assertFalse(_needs_resynthesis(spans, run_dir))

        # --- simulate_replay loads episodes from worker.log in a state_dir ---
        state = self.path('replay-state')
        os.makedirs(os.path.join(state, 'runs', 'r1'))
        with open(os.path.join(state, 'runs', 'r1', 'result.json'), 'w') as fh:
            json.dump({'task_id': 'r1', 'status': 'succeeded', 'model': GLM}, fh)
        tool_args = {'command': 'SELECT 1'}
        with open(os.path.join(state, 'runs', 'r1', 'worker.log'), 'w') as fh:
            fh.write(json.dumps({
                'type': 'tool_use', 'timestamp': 1700000001000,
                'part': {'type': 'tool', 'tool': 'bash',
                         'state': {'status': 'completed', 'input': tool_args,
                                   'output': '1 row'}},
            }) + '\n')
        summary = simulate_replay(state)
        self.assertEqual(summary['total_episodes'], 1)
        self.assertGreater(summary['tokens_before'], 0)

        # --- simulate_replay over list/JSONL with loops, unsafe SQL, escalation & CLI ---
        big_rows = '\n'.join(f'row_{i}: val_{i}' for i in range(120))
        episodes = [
            {
                'task_id': 'ep-loop-sql',
                'model': GLM,
                'messages': [
                    {'role': 'system', 'content': 'diagnose warehouse'},
                    {'role': 'assistant', 'tool_calls': [
                        {'id': 'c1', 'function': {'name': 'query_warehouse', 'arguments': json.dumps({'sql': 'DROP TABLE users'})}}
                    ]},
                    {'role': 'tool', 'tool_call_id': 'c1', 'content': f'[SQL GUARD ERROR]\n{big_rows}'},
                    {'role': 'assistant', 'tool_calls': [
                        {'id': 'c2', 'function': {'name': 'query_warehouse', 'arguments': json.dumps({'sql': 'DROP TABLE users'})}}
                    ]},
                    {'role': 'tool', 'tool_call_id': 'c2', 'content': '[AST PREFLIGHT ERROR]'},
                    {'role': 'user', 'content': 'retry'},
                    {'role': 'assistant', 'content': 'done'},
                    {'role': 'user', 'content': 'check'},
                    {'role': 'assistant', 'content': 'ok'},
                ],
            }
        ]
        jsonl_file = self.path('episodes.jsonl')
        with open(jsonl_file, 'w') as fh:
            for ep in episodes:
                fh.write(json.dumps(ep) + '\n')
        rep = simulate_replay(
            jsonl_file,
            max_context_tokens=120,
            fallback_teacher_model='openrouter/qwen/qwen3-coder',
            upstream_mode='vllm',
        )
        self.assertEqual(rep['total_episodes'], 1)
        self.assertEqual(rep['loop_warnings_triggered'], 1)
        self.assertEqual(rep['sql_violations_caught'], 2)
        self.assertEqual(rep['escalated_to_teacher_count'], 1)
        self.assertGreater(rep['tokens_saved'], 0)
        self.assertGreater(rep['token_reduction_pct'], 0.0)
        md_rep = format_replay_markdown(rep)
        self.assertIn('# Offline Replay Simulation', md_rep)
        self.assertIn('Loop Warnings Triggered', md_rep)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cli_main([
                'replay', '--input', jsonl_file,
                '--max-context-tokens', '120',
                '--fallback-teacher-model', 'openrouter/qwen/qwen3-coder',
                '--upstream-mode', 'vllm',
            ])
        self.assertEqual(rc, 0)
        self.assertIn('# Offline Replay Simulation', buf.getvalue())

        buf_json = io.StringIO()
        with contextlib.redirect_stdout(buf_json):
            rc_json = cli_main(['replay', '--input', jsonl_file, '--json'])
        self.assertEqual(rc_json, 0)
        parsed_json = json.loads(buf_json.getvalue())
        self.assertEqual(parsed_json['total_episodes'], 1)


class TestH9StateStatsAndCLI(HarnessTestCase):
    def _make_state_dir(self, name):
        state = self.path(name)
        run1 = os.path.join(state, 'runs', 'r1')
        run2 = os.path.join(state, 'runs', 'r2')
        os.makedirs(run1)
        os.makedirs(run2)
        with open(os.path.join(run1, 'result.json'), 'w') as fh:
            json.dump({'task_id': 'r1', 'status': 'succeeded', 'model': GLM,
                       'steps': 3, 'cost': 0.02}, fh)
        with open(os.path.join(run2, 'result.json'), 'w') as fh:
            json.dump({'task_id': 'r2', 'status': 'validation_failed', 'model': DEEPSEEK,
                       'steps': 2, 'cost_usd': 0.01, 'micro_repair_used': True}, fh)
        return state, run1, run2

    def test_scan_state_runs_and_stats_markdown(self):
        state, _, _ = self._make_state_dir('state')
        scanned = scan_state_runs(state)
        self.assertEqual(len(scanned), 2)
        self.assertEqual({run['task_id'] for run in scanned}, {'r1', 'r2'})
        first = next(run for run in scanned if run['task_id'] == 'r1')
        self.assertEqual(first['status'], 'succeeded')
        self.assertEqual(first['cost_usd'], 0.02)
        md = format_state_stats_markdown(state)
        self.assertIn('# Live State Directory Stats', md)
        self.assertIn(GLM, md)
        self.assertIn(DEEPSEEK, md)
        self.assertIn('| ALL | 2 | 50.0 |', md)
        self.assertEqual(format_state_stats_markdown(self.path('missing')), 'No runs found.')

    def test_cli_stats_and_trace_on_temp_state(self):
        state, run1, _ = self._make_state_dir('state-cli')
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(cli_main(['stats', '--state-dir', state]), 0)
        self.assertIn('# Live State Directory Stats', buf.getvalue())
        self.assertIn('| ALL | 2 | 50.0 |', buf.getvalue())

        trace_id = 'b' * 32
        spans = [
            {'name': 'gen_ai.chat', 'span_id': 'a' * 16, 'trace_id': trace_id,
             'parent_span_id': None, 'start_time_ns': 1000, 'end_time_ns': 2500,
             'status': 'OK',
             'attributes': {'openinference.span.kind': 'LLM',
                            'gen_ai.request.model': GLM}},
            {'name': 'guard.ast_preflight', 'span_id': 'c' * 16, 'trace_id': trace_id,
             'parent_span_id': 'a' * 16, 'start_time_ns': 1200, 'end_time_ns': 1400,
             'status': 'ERROR',
             'attributes': {'openinference.span.kind': 'GUARDRAIL',
                            'guardrail.name': 'ast_preflight'}},
        ]
        traces_path = os.path.join(run1, 'traces.jsonl')
        with open(traces_path, 'w') as fh:
            for span in spans:
                fh.write(json.dumps(span) + '\n')
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(cli_main(['trace', run1]), 0)  # run-dir form finds traces.jsonl
        self.assertIn('trace %s' % trace_id, buf.getvalue())
        self.assertIn('[LLM] gen_ai.chat', buf.getvalue())
        self.assertIn('[GUARDRAIL] guard.ast_preflight', buf.getvalue())
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(cli_main(['trace', '--no-sync', traces_path]), 0)  # explicit file form
        self.assertIn('[LLM] gen_ai.chat', buf.getvalue())

        # Verify automatic Phoenix backfill sync on `harness trace` against a stub collector.
        received = []

        class _PhoenixStub(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                return

            def do_POST(self):
                length = int(self.headers.get('Content-Length') or '0')
                received.append(json.loads(self.rfile.read(length)))
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{}')

        collector = http.server.ThreadingHTTPServer(('127.0.0.1', 0), _PhoenixStub)
        threading.Thread(target=collector.serve_forever, daemon=True).start()
        endpoint = 'http://127.0.0.1:%d/v1/traces' % collector.server_port
        try:
            buf1 = io.StringIO()
            with contextlib.redirect_stdout(buf1):
                self.assertEqual(cli_main(['trace', '--state-dir', state, '--endpoint', endpoint]), 0)
            self.assertIn('newly_synced=2 already_synced=0 pending=0', buf1.getvalue())
            self.assertEqual(len(received), 2)

            buf2 = io.StringIO()
            with contextlib.redirect_stdout(buf2):
                self.assertEqual(cli_main(['trace', '--state-dir', state, '--endpoint', endpoint]), 0)
            self.assertIn('newly_synced=0 already_synced=2 pending=0', buf2.getvalue())
            self.assertEqual(len(received), 2)
        finally:
            collector.shutdown()
            collector.server_close()


class TestH10SandboxAndFinops(HarnessTestCase):
    """H-10: sandbox validation/locks/hashes + finops budget gate & settlement."""

    def _task(self, **overrides):
        task = {
            'task_id': 'h10',
            'directory': self.root,
            'files': [self.path('target.py')],
            'prompt': 'Change target.py to the required content.',
            'test_argv': [sys.executable, '-c', 'pass'],
        }
        task.update(overrides)
        return task

    # ---- finops ----------------------------------------------------

    def test_finops_number_validation(self):
        self.assertEqual(number(16, 'max_steps'), 16)
        self.assertEqual(number(0.15, 'max_cost_usd'), 0.15)
        for bad in ('0.5', None, True, float('nan'), float('inf'), -0.01):
            with self.assertRaisesRegex(GuardError, 'invalid_max_cost_usd'):
                number(bad, 'max_cost_usd')

    def test_finops_violation_step_cost_and_loop_checks(self):
        def step(**changes):
            message = {'finish': 'tool-calls', 'content': []}
            message.update(changes)
            return message

        def snap(cost, *messages):
            return {'cost': cost, 'messages': [('assistant', m) for m in messages]}

        self.assertIsNone(violation(metrics(snap(0.01, step(), step())), 0.15, 16))
        self.assertEqual(violation(metrics(snap(0.20, step())), 0.15, 16), 'cost_limit')
        self.assertEqual(violation(metrics(snap(0.01, step(), step(), step())), 0.15, 2),
                         'step_limit')
        # cost is judged before steps: a spent budget is the primary signal
        self.assertEqual(violation(metrics(snap(0.20, step(), step(), step())), 0.15, 2),
                         'cost_limit')
        self.assertEqual(violation(metrics(snap(0.01, step(error={'type': 'transport'}))),
                                   0.15, 16), 'model_error')
        self.assertEqual(violation(metrics({'cost': 0.01, 'messages': [('compaction', {})]}),
                                   0.15, 16), 'compaction')

    def test_finops_local_settlement_and_prior_cost(self):
        snap_ok = {
            'cost': 0.05,
            'messages': [
                ('assistant', {'finish': 'stop', 'cost': 0.05, 'content': []}),
                ('assistant', {'error': {'type': 'guard'}, 'content': []}),
            ],
        }
        proxy_ok = {
            'blocked_count': 1,
            'dispatched': [
                {'completed': True, 'http_status': 200, 'error': None, 'generation_id': 'gen-1'},
            ],
        }
        self.assertTrue(can_settle_from_proxy_local(snap_ok, proxy_ok))
        self.assertFalse(can_settle_from_proxy_local(snap_ok, {'blocked_count': 0, 'dispatched': []}))
        self.assertFalse(can_settle_from_proxy_local(None, proxy_ok))

        state = self.path('state-finops')
        for run_name, payload in [
            ('r1', {'task_id': 't1', 'cost': 0.02}),
            ('r2', {'task_id': 't1', 'cost': 0.03}),
            ('r3', {'task_id': 't2', 'cost': 0.10}),
        ]:
            run_dir = os.path.join(state, 'runs', run_name)
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, 'result.json'), 'w') as fh:
                json.dump(payload, fh)
        self.assertAlmostEqual(prior_cost(state, 't1'), 0.05, places=6)
        self.assertAlmostEqual(prior_cost(state, 't2'), 0.10, places=6)
        self.assertEqual(prior_cost(state, 'missing-task'), 0.0)

    # ---- sandbox ---------------------------------------------------

    def test_sandbox_validate_task_schema_and_boundary(self):
        validated = validate_task(self._task())
        self.assertEqual(validated['task_id'], 'h10')
        missing = self._task()
        del missing['prompt']
        with self.assertRaisesRegex(GuardError, 'prompt_required'):
            validate_task(missing)
        with self.assertRaisesRegex(GuardError, 'exactly_one_to_three_files_required'):
            validate_task(self._task(files=[self.path(f'{i}.py') for i in range(4)]))
        with self.assertRaisesRegex(GuardError, 'file_paths_must_be_absolute'):
            validate_task(self._task(files=['a.py']))
        with self.assertRaisesRegex(GuardError, 'executable_must_be_an_existing_absolute_path'):
            validate_task(self._task(test_argv=['pytest', '-q']))
        with self.assertRaisesRegex(GuardError, 'worktree_files_must_be_under_directory'):
            validate_task(self._task(worktree=True, files=['/tmp/outside_harness_test.py']))

    def test_sandbox_locks_exclusive_and_conflict(self):
        target = self.path('a.py')
        other = self.path('b.py')
        state = self.path('lock-state')
        with locks(state, [target], 'task-1'):
            with self.assertRaisesRegex(GuardError, 'files_or_task_already_running'):
                with locks(state, [target], 'task-2'):
                    pass
            with self.assertRaisesRegex(GuardError, 'files_or_task_already_running'):
                with locks(state, [other], 'task-1'):
                    pass
            with locks(state, [other], 'task-2'):
                pass
        with locks(state, [target], 'task-1'):
            pass

    def test_sandbox_file_hashes_digest_map(self):
        alpha = self.path('alpha.py')
        beta = self.path('beta.py')
        with open(alpha, 'w') as fh:
            fh.write('value = 1\n')
        with open(beta, 'w') as fh:
            fh.write('value = 2\n')
        missing = self.path('never-created.py')
        self.assertEqual(file_hashes([alpha, beta, missing]), {
            alpha: hashlib.sha256(b'value = 1\n').hexdigest(),
            beta: hashlib.sha256(b'value = 2\n').hexdigest(),
            missing: None,
        })

    def test_sandbox_reexports_finops_single_source_of_truth(self):
        import harness.finops as finops_mod
        import harness.sandbox as sandbox_mod

        shared = (
            'number', 'private_write', 'private_log', 'event',
            'request_records', 'capture_requests', 'resolved_cost',
            'fetch_generation', 'fetch_generation_settled', '_reconcile_locked',
            'reconcile_run', 'build_unexposed_settlement_evidence',
            'auto_reconcile_from_proxy', 'locks', 'prior_cost',
            'connect', 'verify_db', 'snapshot', 'metrics',
            'violation', 'can_settle_from_proxy_local',
        )
        for name in shared:
            self.assertIs(getattr(sandbox_mod, name), getattr(finops_mod, name), name)
        self.assertIs(sandbox_mod.reconcile_run, finops_mod.reconcile_run)
        self.assertIs(sandbox_mod.violation, finops_mod.violation)

    def test_env_var_configurable_paths(self):
        import importlib
        import harness.finops as finops_mod
        import harness.sandbox as sandbox_mod

        custom_state = self.path('custom-state')
        custom_db = self.path('custom.db')
        custom_cli = self.path('custom-opencode-cli')
        saved = {k: os.environ.get(k) for k in (
            'HARNESS_STATE_DIR', 'HARNESS_DB', 'OPENCODE_DB', 'HARNESS_CLI', 'OPENCODE_CLI'
        )}
        try:
            os.environ['HARNESS_STATE_DIR'] = custom_state
            os.environ['HARNESS_DB'] = custom_db
            os.environ['HARNESS_CLI'] = custom_cli
            importlib.reload(finops_mod)
            importlib.reload(sandbox_mod)
            self.assertEqual(str(finops_mod.STATE), custom_state)
            self.assertEqual(str(sandbox_mod.STATE), custom_state)
            self.assertEqual(str(finops_mod.DB), custom_db)
            self.assertEqual(str(sandbox_mod.DB), custom_db)
            self.assertEqual(sandbox_mod.CLI, custom_cli)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            importlib.reload(finops_mod)
            importlib.reload(sandbox_mod)


class TestFinopsAndSandboxUnit(HarnessTestCase):
    """Unit tests for harness.finops and harness.sandbox using temp dirs and synthetic SQLite DBs."""

    def _init_db(self, db_path, sessions=(), messages=()):
        import sqlite3
        conn = sqlite3.connect(db_path)
        try:
            conn.execute('CREATE TABLE session_v2 (id TEXT PRIMARY KEY, cost REAL)')
            conn.execute('CREATE TABLE session_message (session_id TEXT, type TEXT, data TEXT)')
            conn.executemany('INSERT INTO session_v2 (id, cost) VALUES (?, ?)', sessions)
            conn.executemany(
                'INSERT INTO session_message (session_id, type, data) VALUES (?, ?, ?)',
                [(sid, kind, json.dumps(data)) for sid, kind, data in messages],
            )
            conn.commit()
        finally:
            conn.close()

    def _valid_task(self, root, **overrides):
        target = os.path.join(root, 'target.py')
        task = {
            'task_id': 'unit_task-1',
            'directory': root,
            'files': [target],
            'prompt': 'Update target.py cleanly.',
            'test_argv': [sys.executable, '-c', 'pass'],
        }
        task.update(overrides)
        return task

    def test_finops_sqlite_connect_verify_snapshot_metrics_violation_and_settle(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as tmp:
            bad_db = os.path.join(tmp, 'empty.db')
            sqlite3.connect(bad_db).close()
            with self.assertRaises(sqlite3.OperationalError):
                verify_db(bad_db)

            db_path = os.path.join(tmp, 'opencode.db')
            self._init_db(
                db_path,
                sessions=[('ses_1', 0.05)],
                messages=[
                    ('ses_1', 'assistant', {
                        'id': 'gen-abc123',
                        'finish': 'tool-calls',
                        'cost': 0.02,
                        'content': [
                            {'type': 'tool', 'name': 'read', 'state': {
                                'status': 'completed',
                                'input': {'path': '/a.py'},
                                'content': 'x = 1',
                            }},
                        ],
                    }),
                    ('ses_1', 'assistant', {'finish': 'stop', 'cost': 0.03, 'content': []}),
                ],
            )
            verify_db(db_path)
            conn = connect(db_path)
            try:
                row = conn.execute('SELECT cost FROM session_v2 WHERE id=?', ('ses_1',)).fetchone()
                self.assertAlmostEqual(row[0], 0.05)
            finally:
                conn.close()

            self.assertIsNone(snapshot(db_path, 'missing_sid'))
            snap = snapshot(db_path, 'ses_1')
            self.assertIsNotNone(snap)
            self.assertAlmostEqual(snap['cost'], 0.05)
            m = metrics(snap)
            self.assertEqual(m['steps'], 2)
            self.assertFalse(m['compact'])
            self.assertFalse(m['model_error'])
            self.assertFalse(m['unfinished'])
            self.assertEqual(m['reads'], 1)
            self.assertEqual(m['file_reads'], 1)

            self.assertIsNone(violation(m, 0.15, 16))
            self.assertEqual(violation(m, 0.05, 16), 'cost_limit')
            self.assertEqual(violation(m, 0.15, 1), 'step_limit')

            err_snap = {
                'cost': 0.01,
                'messages': [
                    ('assistant', {'finish': 'tool-calls', 'content': [
                        {'type': 'tool', 'name': 'bash', 'state': {'status': 'error', 'error': 'boom'}},
                        {'type': 'tool', 'name': 'bash', 'state': {'status': 'error', 'error': 'boom'}},
                    ]}),
                ],
            }
            self.assertEqual(violation(metrics(err_snap), 0.15, 16), 'repeated_tool_error')

            read_snap = {
                'cost': 0.01,
                'messages': [
                    ('assistant', {'finish': 'tool-calls', 'content': [
                        {'type': 'tool', 'name': 'read', 'state': {'status': 'completed', 'input': {'path': '/a.py'}, 'content': 'same'}},
                        {'type': 'tool', 'name': 'read', 'state': {'status': 'completed', 'input': {'path': '/a.py'}, 'content': 'same'}},
                        {'type': 'tool', 'name': 'read', 'state': {'status': 'completed', 'input': {'path': '/a.py'}, 'content': 'same'}},
                    ]}),
                ],
            }
            self.assertEqual(violation(metrics(read_snap), 0.15, 16), 'repeated_identical_read')

            snap_blocked = {
                'cost': 0.05,
                'messages': snap['messages'] + [('assistant', {'error': {'type': 'guard'}, 'content': []})],
            }
            proxy_ok = {
                'blocked_count': 1,
                'dispatched': [
                    {'completed': True, 'http_status': 200, 'error': None, 'generation_id': 'gen-abc123'},
                    {'completed': True, 'http_status': 200, 'error': None, 'generation_id': 'gen-def456'},
                ],
            }
            self.assertTrue(can_settle_from_proxy_local(snap_blocked, proxy_ok))

    def test_finops_request_records_capture_resolved_and_prior_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = os.path.join(tmp, 'runs', 'r1')
            os.makedirs(run_dir)
            snap = {
                'cost': 0.03,
                'messages': [
                    ('assistant', {'id': 'gen-local1', 'finish': 'tool-calls', 'cost': 0.02, 'tokens': {'input': 10}}),
                    ('assistant', {'finish': 'stop', 'cost': 0.01}),
                    ('assistant', {'error': {'type': 'guard'}}),
                ],
            }
            proxy_summary = {
                'blocked_count': 1,
                'dispatched': [
                    {'completed': True, 'http_status': 200, 'generation_id': 'gen-local1'},
                    {'completed': True, 'http_status': 200, 'generation_id': 'gen-proxy2'},
                ],
            }
            recs = request_records(snap, proxy_summary)
            self.assertEqual(len(recs), 3)
            self.assertEqual(recs[0]['identity_status'], 'recorded')
            self.assertEqual(recs[1]['identity_status'], 'recorded_by_proxy')
            self.assertEqual(recs[1]['provider_request_ids'], ['gen-proxy2'])
            self.assertEqual(recs[2]['identity_status'], 'blocked_before_dispatch')
            self.assertEqual(recs[2]['state'], 'blocked_locally')

            capture_requests(run_dir, 'ses_cap', snap, proxy_summary)
            req_path = os.path.join(run_dir, 'requests.json')
            events_path = os.path.join(run_dir, 'events.jsonl')
            self.assertTrue(os.path.exists(req_path))
            self.assertTrue(os.path.exists(events_path))
            with open(events_path) as fh:
                lines_before = fh.readlines()
            capture_requests(run_dir, 'ses_cap', snap, proxy_summary)
            with open(events_path) as fh:
                self.assertEqual(len(fh.readlines()), len(lines_before))

            res_path = os.path.join(run_dir, 'result.json')
            with open(res_path, 'w') as fh:
                json.dump({'task_id': 't-cost', 'session_id': 'ses_cap', 'cost': 0.03}, fh)
            self.assertAlmostEqual(resolved_cost(res_path, {'task_id': 't-cost', 'session_id': 'ses_cap', 'cost': 0.03}), 0.03)

            with open(res_path, 'w') as fh:
                json.dump({'task_id': 't-cost', 'session_id': 'ses_cap', 'cost': None, 'observed_cost': 0.03}, fh)
            with self.assertRaisesRegex(GuardError, 'prior_cost_unknown_reconcile_before_retry'):
                resolved_cost(res_path, {'task_id': 't-cost', 'session_id': 'ses_cap', 'cost': None})

            with open(res_path, 'rb') as fh:
                res_sha = hashlib.sha256(fh.read()).hexdigest()
            recon_path = os.path.join(run_dir, 'billing-reconciliation.json')
            with open(recon_path, 'w') as fh:
                json.dump({
                    'schema': 'verified_billing.v1',
                    'task_id': 't-cost',
                    'session_id': 'ses_cap',
                    'result_sha256': res_sha,
                    'cost': 0.042,
                }, fh)
            self.assertAlmostEqual(prior_cost(tmp, 't-cost'), 0.042)

    def test_finops_reconcile_run_and_auto_reconcile_from_proxy(self):
        def stub_fetch(gid):
            return {
                'id': gid,
                'finish_reason': 'stop',
                'total_cost': 0.042,
                'native_tokens_prompt': 100,
                'native_tokens_completion': 50,
                'tokens_prompt': 100,
                'tokens_completion': 50,
            }

        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, 'opencode.db')
            self._init_db(db_path)

            run1 = os.path.join(tmp, 'runs', 'run-1')
            os.makedirs(run1)
            res1 = {'task_id': 't-rec', 'session_id': 'ses_r1', 'status': 'blocked', 'cost': None, 'observed_cost': 0.0}
            with open(os.path.join(run1, 'result.json'), 'w') as fh:
                json.dump(res1, fh)
            with open(os.path.join(run1, 'requests.json'), 'w') as fh:
                json.dump({
                    'session_id': 'ses_r1',
                    'requests': [{
                        'ordinal': 1,
                        'state': 'finished',
                        'cost': None,
                        'provider_request_ids': ['gen-stub1'],
                    }],
                }, fh)

            rec1 = reconcile_run(
                run1,
                {'session_id': 'ses_r1', 'requests': [{'ordinal': 1, 'generation_id': 'gen-stub1'}]},
                fetch=stub_fetch,
                state=tmp,
                db=db_path,
            )
            self.assertEqual(rec1['schema'], 'verified_billing.v1')
            self.assertAlmostEqual(rec1['cost'], 0.042)
            self.assertTrue(os.path.exists(os.path.join(run1, 'billing-reconciliation.json')))
            with self.assertRaisesRegex(GuardError, 'already_reconciled'):
                reconcile_run(
                    run1,
                    {'session_id': 'ses_r1', 'requests': [{'ordinal': 1, 'generation_id': 'gen-stub1'}]},
                    fetch=stub_fetch,
                    state=tmp,
                    db=db_path,
                )

            run2 = os.path.join(tmp, 'runs', 'run-2')
            os.makedirs(run2)
            res2_path = os.path.join(run2, 'result.json')
            res2 = {'task_id': 't-auto', 'session_id': 'ses_r2', 'status': 'blocked', 'cost': None, 'observed_cost': 0.0}
            with open(res2_path, 'w') as fh:
                json.dump(res2, fh)
            with open(os.path.join(run2, 'requests.json'), 'w') as fh:
                json.dump({
                    'session_id': 'ses_r2',
                    'requests': [{
                        'ordinal': 1,
                        'state': 'finished',
                        'cost': None,
                        'provider_request_ids': ['gen-stub2'],
                    }],
                    'proxy': {
                        'dispatched': [{'generation_id': 'gen-stub2', 'completed': True}],
                    },
                }, fh)

            rec2 = auto_reconcile_from_proxy(run2, res2_path, res2, fetch=stub_fetch, db=db_path)
            self.assertIsNotNone(rec2)
            self.assertAlmostEqual(rec2['cost'], 0.042)
            self.assertAlmostEqual(resolved_cost(res2_path, res2), 0.042)

    def test_finops_locks_exclusive_and_contention(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, 'a.py')
            with finops_locks(tmp, [target], 'task-lock'):
                with self.assertRaisesRegex(GuardError, 'files_or_task_already_running'):
                    with finops_locks(tmp, [target], 'task-other'):
                        pass
                with self.assertRaisesRegex(GuardError, 'files_or_task_already_running'):
                    with finops_locks(tmp, [], 'task-lock'):
                        pass

    def test_sandbox_validate_task_bounds_and_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            ok = validate_task(self._valid_task(tmp))
            self.assertEqual(ok['task_id'], 'unit_task-1')
            self.assertEqual(ok['directory'], os.path.realpath(tmp))

            for bad_id in ('', 'bad/id', '../escape', 'a' * 101):
                with self.assertRaisesRegex(GuardError, 'invalid_task_id'):
                    validate_task(self._valid_task(tmp, task_id=bad_id))

            for bad_steps in (0, -1, 1.5, 26):
                with self.assertRaisesRegex(GuardError, 'invalid_max_steps'):
                    validate_task(self._valid_task(tmp, max_steps=bad_steps, limit_reason='ok'))
            with self.assertRaisesRegex(GuardError, 'limit_increase_requires_written_reason'):
                validate_task(self._valid_task(tmp, max_steps=20))
            self.assertEqual(validate_task(self._valid_task(tmp, max_steps=20, limit_reason='extra'))['max_steps'], 20)

            for bad_cost in (-0.01, 0, 0.31):
                with self.assertRaisesRegex(GuardError, 'invalid_max_cost_usd'):
                    validate_task(self._valid_task(tmp, max_cost_usd=bad_cost, limit_reason='ok'))
            with self.assertRaisesRegex(GuardError, 'limit_increase_requires_written_reason'):
                validate_task(self._valid_task(tmp, max_cost_usd=0.20))

            with self.assertRaisesRegex(GuardError, 'file_paths_must_be_absolute'):
                validate_task(self._valid_task(tmp, files=['../outside.py']))
            with self.assertRaisesRegex(GuardError, 'invalid_file_target'):
                validate_task(self._valid_task(tmp, files=[os.path.join(tmp, '*.py')]))
            with self.assertRaisesRegex(GuardError, 'worktree_files_must_be_under_directory'):
                outside = os.path.join(tmp, 'sub', '..', 'outside.py')
                sub = os.path.join(tmp, 'sub')
                os.makedirs(sub)
                validate_task(self._valid_task(sub, worktree=True, files=[outside]))

    def test_validate_task_and_config_for_read_only_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            ro_path = os.path.join(tmp, 'ref.py')
            with open(ro_path, 'w', encoding='utf-8') as fh:
                fh.write('REF = 0\n')
            target = os.path.join(tmp, 'target.py')
            with open(target, 'w', encoding='utf-8') as fh:
                fh.write('VALUE = 0\n')

            base = self._valid_task(tmp)
            base['files'] = [target]
            base['test_argv'] = [sys.executable, '-c', 'raise SystemExit(0)']

            task = dict(base, read_only_files=[ro_path, ro_path])
            t = validate_task(task)
            self.assertEqual(t['read_only_files'], [os.path.realpath(ro_path)])

            cfg = config_for(t)
            perms = cfg['agents']['guarded_worker']['permissions']
            ro_rules = [r for r in perms if r['resource'] == os.path.realpath(ro_path)]
            self.assertEqual(ro_rules, [{'action': 'read', 'resource': os.path.realpath(ro_path), 'effect': 'allow'}])
            self.assertFalse(any(r['resource'] == os.path.realpath(ro_path) and r['action'] != 'read' for r in perms))

            with self.assertRaisesRegex(GuardError, 'read_only_files_must_be_a_list_of_up_to_five_files'):
                validate_task(dict(base, read_only_files=['a', 'b', 'c', 'd', 'e', 'f']))
            with self.assertRaisesRegex(GuardError, 'invalid_file_target'):
                validate_task(dict(base, read_only_files=[os.path.join(tmp, 'nope.py')]))
            with self.assertRaisesRegex(GuardError, 'read_only_file_overlaps_editable_files'):
                validate_task(dict(base, read_only_files=[target]))
            with self.assertRaisesRegex(GuardError, 'file_paths_must_be_absolute'):
                validate_task(dict(base, read_only_files=['relative/ref.py']))
            repo = os.path.join(tmp, 'repo')
            os.makedirs(repo)
            outside = os.path.join(tmp, 'outside.py')
            with open(outside, 'w', encoding='utf-8') as fh:
                fh.write('X = 1\n')
            in_repo = os.path.join(repo, 'target.py')
            with open(in_repo, 'w', encoding='utf-8') as fh:
                fh.write('VALUE = 0\n')
            wt_base = self._valid_task(tmp)
            wt_base['directory'] = repo
            wt_base['files'] = [in_repo]
            wt_base['worktree'] = True
            with self.assertRaisesRegex(GuardError, 'worktree_files_must_be_under_directory'):
                validate_task(dict(wt_base, read_only_files=[outside]))

    def test_sandbox_config_for_and_verify_permissions(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = validate_task(self._valid_task(tmp))
            cfg = config_for(task, base_url='http://127.0.0.1:9999')
            self.assertEqual(cfg['providers']['openrouter']['settings']['baseURL'], 'http://127.0.0.1:9999')
            perms = cfg['agents']['guarded_worker']['permissions']
            payload = {'data': [{'id': 'guarded_worker', 'steps': task['max_steps'], 'permissions': perms}]}
            verify_permissions(payload, task)

            patch_task = validate_task(self._valid_task(
                tmp, mode='patch', patch_file=os.path.join(tmp, 'fix.patch'),
            ))
            patch_cfg = config_for(patch_task)
            verify_permissions({'data': [{
                'id': 'guarded_worker',
                'steps': patch_task['max_steps'],
                'permissions': patch_cfg['agents']['guarded_worker']['permissions'],
            }]}, patch_task)

            with self.assertRaisesRegex(GuardError, 'unsupported_agent_model'):
                verify_permissions({'data': [{'id': 'guarded_worker', 'model': {'providerID': 'bad', 'id': 'm'}}]}, task)
            with self.assertRaisesRegex(GuardError, 'effective_agent_or_steps_mismatch'):
                verify_permissions({'data': [{'id': 'guarded_worker', 'steps': 99, 'permissions': perms}]}, task)
            with self.assertRaisesRegex(GuardError, 'effective_catch_all_deny_missing'):
                verify_permissions({'data': [{'id': 'guarded_worker', 'steps': task['max_steps'], 'permissions': perms[1:]}]}, task)
            with self.assertRaisesRegex(GuardError, 'effective_permissions_too_broad'):
                too_broad = perms + [{'action': 'bash', 'resource': '*', 'effect': 'allow'}]
                verify_permissions({'data': [{'id': 'guarded_worker', 'steps': task['max_steps'], 'permissions': too_broad}]}, task)
            with self.assertRaisesRegex(GuardError, 'effective_file_permissions_missing'):
                verify_permissions({'data': [{'id': 'guarded_worker', 'steps': task['max_steps'], 'permissions': perms[:1]}]}, task)

    def test_sandbox_file_hashes_capture_files_and_execute(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, 'mod.py')
            missing = os.path.join(tmp, 'missing.py')
            with open(target, 'w') as fh:
                fh.write('x = 1\n')
            self.assertEqual(file_hashes([target, missing]), {
                target: hashlib.sha256(b'x = 1\n').hexdigest(),
                missing: None,
            })

            run_dir = os.path.join(tmp, 'run')
            os.makedirs(run_dir)
            capture_files(run_dir, [target])
            self.assertTrue(os.path.exists(os.path.join(run_dir, 'files.before.json')))
            self.assertTrue(os.path.exists(os.path.join(run_dir, 'files.latest.json')))

            with open(target, 'w') as fh:
                fh.write('x = 2\n')
            capture_files(run_dir, [target])
            with open(os.path.join(run_dir, 'changes.patch')) as fh:
                patch_text = fh.read()
            self.assertIn('-x = 1', patch_text)
            self.assertIn('+x = 2', patch_text)
            self.assertTrue(os.path.exists(os.path.join(run_dir, 'events.jsonl')))

            log_path = os.path.join(tmp, 'exec.log')
            rc = execute([sys.executable, '-c', "print('ok')"], tmp, 10, log_path)
            self.assertEqual(rc, 0)
            with open(log_path) as fh:
                self.assertEqual(fh.read().strip(), 'ok')

            with self.assertRaisesRegex(GuardError, 'command_timeout'):
                execute([sys.executable, '-c', 'import time; time.sleep(5)'], tmp, 0.1, log_path)

    def test_cli_serve_and_run_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            url_file = os.path.join(tmp, 'proxy_url.json')
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = cli_main(['serve', '--duration-seconds', '0.05', '--print-url-file', url_file])
            self.assertEqual(rc, 0)
            info = json.loads(buf.getvalue().strip())
            self.assertTrue(info['base_url'].startswith('http://127.0.0.1:'))
            with open(url_file, 'r', encoding='utf-8') as fh:
                self.assertEqual(json.load(fh), info)

            target = os.path.join(tmp, 'target.py')
            with open(target, 'w', encoding='utf-8') as fh:
                fh.write('x = 1\n')
            manifest_path = os.path.join(tmp, 'task.json')
            state_dir = os.path.join(tmp, 'state')
            with open(manifest_path, 'w', encoding='utf-8') as fh:
                json.dump({
                    'task_id': 'cli-check-1',
                    'directory': tmp,
                    'files': [target],
                    'prompt': 'Check target.py',
                    'test_argv': [sys.executable, '-c', 'pass'],
                }, fh)
            run_buf = io.StringIO()
            with contextlib.redirect_stdout(run_buf):
                rc_run = cli_main(['run', '--check', '--state-dir', state_dir, manifest_path])
            self.assertEqual(rc_run, 0)
            run_out = json.loads(run_buf.getvalue())
            self.assertIn(run_out['status'], ('passed', 'validated', 'configuration_verified_no_model_call'))


class TestEgressAndSqlGuards(HarnessTestCase):
    def test_forbidden_egress_keys_and_check_egress_payload(self):
        self.assertIn('raw_rows', FORBIDDEN_EGRESS_KEYS)
        self.assertIn('duckdb_rows', FORBIDDEN_EGRESS_KEYS)
        self.assertIn('connection_string', FORBIDDEN_EGRESS_KEYS)

        clean_payload = {
            'model': GLM,
            'messages': [{'role': 'user', 'content': 'summarize metrics'}],
            'metadata': {'run_id': 'r1'},
        }
        self.assertEqual(check_egress_payload(clean_payload, allowed_models=[GLM]), [])

        bad_key_payload = {
            'model': GLM,
            'messages': [{'role': 'user', 'content': 'hi'}],
            'metadata': {'raw_rows': [{'id': 1}]},
        }
        violations = check_egress_payload(bad_key_payload)
        self.assertTrue(any('forbidden egress key: body.metadata.raw_rows' in v for v in violations), violations)

        secret_payload = {
            'model': GLM,
            'messages': [
                {'role': 'user', 'content': 'token is sk-or-v1-abcdef123456'},
                {'role': 'assistant', 'content': [{'type': 'text', 'text': '-----BEGIN RSA PRIVATE KEY-----'}]},
            ],
        }
        secret_violations = check_egress_payload(secret_payload)
        self.assertTrue(any('sk-or-v1-' in v for v in secret_violations), secret_violations)
        self.assertTrue(any('PRIVATE KEY-----' in v for v in secret_violations), secret_violations)

        model_violations = check_egress_payload(clean_payload, allowed_models=[DEEPSEEK])
        self.assertTrue(any('disallowed model' in v for v in model_violations), model_violations)

        oversized_payload = {
            'model': GLM,
            'messages': [{'role': 'user', 'content': 'x' * 300}],
        }
        size_violations = check_egress_payload(oversized_payload, max_payload_bytes=100)
        self.assertTrue(any('exceeds max_payload_bytes' in v for v in size_violations), size_violations)

    def test_check_readonly_sql_tool_calls(self):
        safe_msgs = [
            {'role': 'assistant', 'tool_calls': [
                {'id': 'q1', 'function': {'name': 'query_warehouse',
                                          'arguments': json.dumps({'sql': "SELECT id FROM orders WHERE status != 'DROP';"})}},
            ]},
        ]
        self.assertEqual(check_readonly_sql_tool_calls(safe_msgs), [])

        unsafe_msgs = [
            {'role': 'assistant', 'tool_calls': [
                {'id': 'q2', 'function': {'name': 'execute_sql',
                                          'arguments': json.dumps({'sql': 'SELECT 1; DROP TABLE users'})}},
                {'id': 'q3', 'function': {'name': 'run_sql',
                                          'arguments': json.dumps({'query': 'SELECT a, FROM orders'})}},
            ]},
        ]
        violations = check_readonly_sql_tool_calls(unsafe_msgs)
        self.assertTrue(any('multi-statement SQL injection' in v for v in violations), violations)
        self.assertTrue(any('unsafe SQL keyword DROP' in v for v in violations), violations)
        self.assertTrue(any('trailing comma before FROM' in v for v in violations), violations)

    def test_semantic_oscillation_in_detect_duplicate_tool_calls(self):
        table_oscillation = [
            {'role': 'user', 'content': 'inspect orders'},
            {'role': 'assistant', 'tool_calls': [
                {'id': 'c1', 'function': {'name': 'query_warehouse',
                                          'arguments': json.dumps({'sql': 'SELECT count(*) FROM orders'})}}]},
            {'role': 'tool', 'tool_call_id': 'c1', 'content': '10'},
            {'role': 'assistant', 'tool_calls': [
                {'id': 'c2', 'function': {'name': 'query_warehouse',
                                          'arguments': json.dumps({'sql': 'SELECT sum(total) FROM orders'})}}]},
            {'role': 'tool', 'tool_call_id': 'c2', 'content': '100'},
            {'role': 'assistant', 'tool_calls': [
                {'id': 'c3', 'function': {'name': 'query_warehouse',
                                          'arguments': json.dumps({'sql': 'SELECT max(id) FROM orders'})}}]},
            {'role': 'tool', 'tool_call_id': 'c3', 'content': '50'},
        ]
        warnings = detect_duplicate_tool_calls(table_oscillation)
        self.assertTrue(any('semantic tool oscillation: query_warehouse' in w for w in warnings), warnings)

        error_oscillation = [
            {'role': 'user', 'content': 'run tests'},
            {'role': 'assistant', 'tool_calls': [
                {'id': 'b1', 'function': {'name': 'bash',
                                          'arguments': json.dumps({'command': 'pytest tests/a.py'})}}]},
            {'role': 'tool', 'tool_call_id': 'b1', 'content': 'Error: module foo not found\ndetails 1'},
            {'role': 'assistant', 'tool_calls': [
                {'id': 'b2', 'function': {'name': 'bash',
                                          'arguments': json.dumps({'command': 'pytest tests/b.py'})}}]},
            {'role': 'tool', 'tool_call_id': 'b2', 'content': 'Error: module foo not found\ndetails 2'},
            {'role': 'assistant', 'tool_calls': [
                {'id': 'b3', 'function': {'name': 'bash',
                                          'arguments': json.dumps({'command': 'pytest tests/c.py'})}}]},
            {'role': 'tool', 'tool_call_id': 'b3', 'content': 'Error: module foo not found\ndetails 3'},
        ]
        err_warnings = detect_duplicate_tool_calls(error_oscillation)
        self.assertTrue(any('semantic tool oscillation: bash failed 3 times' in w for w in err_warnings), err_warnings)

    def test_proxy_enforces_egress_guard_and_readonly_sql(self):
        upstream, captured = _scripted_upstream([{'status': 200}] * 2)
        proxy = OpenRouterGuardProxy(
            lambda: None,
            upstream_url='http://127.0.0.1:%d' % upstream.server_port,
            allowed_models=[GLM],
            enforce_egress_guard=True,
            enforce_readonly_sql=True,
        )
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
        try:
            blocked_body = json.dumps({
                'model': GLM,
                'messages': [{'role': 'user', 'content': 'hi'}],
                'extra': {'raw_rows': [1, 2, 3]},
            }).encode()
            status_blocked = _post_chat(proxy.base_url, blocked_body)

            sql_body = json.dumps({
                'model': GLM,
                'messages': [
                    {'role': 'user', 'content': 'clean up'},
                    {'role': 'assistant', 'tool_calls': [
                        {'id': 's1', 'function': {'name': 'sql_query',
                                                  'arguments': json.dumps({'sql': 'DELETE FROM orders'})}}]},
                    {'role': 'tool', 'tool_call_id': 's1', 'content': 'ok'},
                ],
            }).encode()
            status_sql = _post_chat(proxy.base_url, sql_body)
        finally:
            proxy.shutdown(); proxy.server_close()
            upstream.shutdown(); upstream.server_close()

        self.assertEqual(status_blocked, 400)
        self.assertEqual(proxy.blocked_reason, 'egress_guard_blocked')
        self.assertEqual(status_sql, 200)
        self.assertEqual(len(captured), 1)
        self.assertIn('[SQL GUARD ERROR: unsafe SQL keyword DELETE in sql_query',
                      captured[0]['messages'][-1]['content'])


class TestSmlCostManifestAndEscalation(HarnessTestCase):
    def test_export_sml_cost_manifest_settled_and_atomic_write(self):
        run_dir = self.path('run-1')
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, 'result.json'), 'w') as fh:
            json.dump({
                'task_id': 't1',
                'session_id': 's1',
                'cost': 0.012345,
                'wall_clock_seconds': 42.5,
            }, fh)
        with open(os.path.join(run_dir, 'requests.json'), 'w') as fh:
            json.dump({
                'session_id': 's1',
                'observed_cost': 0.012,
                'requests': [
                    {'ordinal': 1, 'tokens': {'input': 120, 'output': 30}},
                    {'ordinal': 2, 'tokens': {'prompt_tokens': 80, 'completion_tokens': 20}},
                ],
            }, fh)
        out_path = self.path('results/run-1/cost.json')
        manifest = export_sml_cost_manifest(
            run_dir,
            out_path,
            gpu_hours=0.5,
            gpu_hourly_rate_usd=1.25,
        )
        expected = {
            'wall_clock_seconds': 42.5,
            'gpu_hours': 0.5,
            'HourlyRateUSD': 1.25,
            'hourly_rate_usd': 1.25,
            'compute_cost_usd': 0.625,
            'api_cost_usd': 0.012345,
            'total_cost_usd': 0.637345,
            'prompt_tokens': 200,
            'completion_tokens': 50,
            'request_count': 2,
            'cost_accounting_status': 'settled',
        }
        self.assertEqual(manifest, expected)
        self.assertTrue(os.path.exists(out_path))
        with open(out_path) as fh:
            self.assertEqual(json.load(fh), expected)

    def test_export_sml_cost_manifest_proxy_requests_fallback(self):
        run_dir = self.path('run-2')
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, 'result.json'), 'w') as fh:
            json.dump({
                'task_id': 't2',
                'session_id': 's2',
                'cost': None,
                'started_at': 100.0,
                'finished_at': 115.0,
            }, fh)
        with open(os.path.join(run_dir, 'proxy-requests.json'), 'w') as fh:
            json.dump({
                'dispatched': [
                    {'stream_cost': 0.004, 'usage': {'prompt_tokens': 50, 'completion_tokens': 10}},
                    {'stream_cost': 0.006, 'usage': {'prompt_tokens': 70, 'completion_tokens': 15}},
                ],
            }, fh)
        manifest = export_sml_cost_manifest(run_dir)
        self.assertEqual(manifest['wall_clock_seconds'], 15.0)
        self.assertEqual(manifest['api_cost_usd'], 0.01)
        self.assertEqual(manifest['total_cost_usd'], 0.01)
        self.assertEqual(manifest['prompt_tokens'], 120)
        self.assertEqual(manifest['completion_tokens'], 25)
        self.assertEqual(manifest['request_count'], 2)
        self.assertEqual(manifest['cost_accounting_status'], 'estimated')

    def test_select_escalated_model_and_apply_provider_routing(self):
        msgs_one_err = [
            {'role': 'user', 'content': 'fix bug'},
            {'role': 'tool', 'content': 'SyntaxError: bad syntax'},
        ]
        self.assertEqual(
            select_escalated_model(
                msgs_one_err,
                current_model=GLM,
                fallback_teacher_model='openrouter/qwen/qwen3-coder',
                error_turn_threshold=2,
            ),
            GLM,
        )
        msgs_two_errs = [
            {'role': 'user', 'content': 'fix bug'},
            {'role': 'tool', 'content': '[AST PREFLIGHT ERROR: bad.py:1]'},
            {'role': 'tool', 'content': '[SQL GUARD ERROR]: unsafe SQL'},
        ]
        self.assertEqual(
            select_escalated_model(
                msgs_two_errs,
                current_model=GLM,
                fallback_teacher_model='openrouter/qwen/qwen3-coder',
                error_turn_threshold=2,
            ),
            'openrouter/qwen/qwen3-coder',
        )
        body = json.dumps({'model': GLM, 'messages': msgs_two_errs}).encode()
        routed = json.loads(
            apply_provider_routing(
                body,
                '/chat/completions',
                turn_index=3,
                fallback_teacher_model='openrouter/qwen/qwen3-coder',
            )
        )
        self.assertEqual(routed['model'], 'openrouter/qwen/qwen3-coder')
        self.assertNotIn('provider', routed)
        self.assertEqual(routed['reasoning']['max_tokens'], ERROR_RECOVERY_REASONING_MAX_TOKENS)
        self.assertEqual(routed['max_tokens'], 8192)

    def test_clear_session_busy_disarms_zombie_sessions(self):
        import sqlite3
        import tempfile
        from harness.finops import clear_session_busy
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, 'state.db')
            conn = sqlite3.connect(db_path)
            conn.execute('CREATE TABLE session_v2 (id TEXT PRIMARY KEY, time_busy REAL, idle_outcome TEXT)')
            conn.execute("INSERT INTO session_v2 (id, time_busy, idle_outcome) VALUES ('ses_1', 12345, NULL)")
            conn.commit()
            conn.close()
            self.assertTrue(clear_session_busy(db_path, 'ses_1'))
            conn = sqlite3.connect(db_path)
            try:
                row = conn.execute('SELECT time_busy, idle_outcome FROM session_v2 WHERE id=?', ('ses_1',)).fetchone()
            finally:
                conn.close()
            self.assertIsNone(row[0])
            self.assertEqual(row[1], 'failed')


class TestContextBudgetCompaction(HarnessTestCase):
    def test_estimate_messages_tokens(self):
        self.assertEqual(estimate_messages_tokens(None), 1)
        self.assertEqual(estimate_messages_tokens([]), 1)
        msgs = [
            {'role': 'user', 'content': 'a' * 400},
            {
                'role': 'assistant',
                'content': '',
                'tool_calls': [{'id': 'c1', 'function': {'name': 'run_sql', 'arguments': '{"q": "SELECT 1"}'}}],
            },
        ]
        tokens = estimate_messages_tokens(msgs)
        self.assertGreater(tokens, 100)
        self.assertEqual(estimate_messages_tokens([{'role': 'user', 'content': 'abcd' * 10}]), 10)

    def test_compact_context_window_preserves_headers_tail_and_recent(self):
        table_lines = ['col_a | col_b | col_c', '------+-------+------']
        table_lines.extend(f'row_{i:03d} | val_{i:03d} | extra_{i:03d}' for i in range(60))
        big_table = '\n'.join(table_lines)
        recent_table = '\n'.join(f'recent_{i:02d} | ok' for i in range(35))
        messages = [
            {'role': 'system', 'content': 'System instructions'},
            {'role': 'user', 'content': 'Query the table'},
            {'role': 'tool', 'tool_call_id': 'c1', 'content': big_table},
            {'role': 'assistant', 'content': 'Got it'},
            {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'c2', 'content': big_table}]},
            {'role': 'assistant', 'content': 'Next step'},
            {'role': 'tool', 'tool_call_id': 'c3', 'content': recent_table},
            {'role': 'user', 'content': 'Summarize'},
        ]

        # Under budget: no truncation.
        unchanged, stats_ok = compact_context_window(messages, max_context_tokens=10000)
        self.assertEqual(stats_ok['truncated_messages'], 0)
        self.assertEqual(unchanged[2]['content'], big_table)

        # Over budget: compacts oldest tool outputs first, keeps system & last 4 messages untouched.
        compacted, stats = compact_context_window(
            messages, max_context_tokens=450, keep_recent_messages=4, head_lines=4, tail_lines=3
        )
        self.assertGreater(stats['truncated_messages'], 0)
        self.assertLess(stats['tokens_after'], stats['tokens_before'])
        self.assertEqual(compacted[0]['content'], 'System instructions')
        old_tool = compacted[2]['content']
        self.assertIn('[HARNESS CONTEXT BUDGET: truncated', old_tool)
        self.assertIn('col_a | col_b | col_c', old_tool)
        self.assertIn('row_059 | val_059 | extra_059', old_tool)
        self.assertNotIn('row_030 | val_030 | extra_030', old_tool)
        # Recent messages (last 4) remain untouched.
        self.assertEqual(compacted[-2]['content'], recent_table)

    def test_proxy_context_budget_compaction_and_span(self):
        big_output = '\n'.join(['header_1 | header_2'] + [f'r{i} | data_{i}_padding_text' for i in range(80)])
        upstream, captured = _scripted_upstream([{'status': 200}])
        proxy = OpenRouterGuardProxy(
            lambda: None,
            upstream_url='http://127.0.0.1:%d' % upstream.server_port,
            max_context_tokens=300,
        )
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
        try:
            body = json.dumps({
                'model': GLM,
                'messages': [
                    {'role': 'system', 'content': 'sys'},
                    {'role': 'tool', 'tool_call_id': 't1', 'content': big_output},
                    {'role': 'assistant', 'content': 'a1'},
                    {'role': 'user', 'content': 'u2'},
                    {'role': 'assistant', 'content': 'a2'},
                    {'role': 'user', 'content': 'u3'},
                ],
            }).encode()
            status = _post_chat(proxy.base_url, body)
        finally:
            proxy.shutdown(); proxy.server_close()
            upstream.shutdown(); upstream.server_close()

        self.assertEqual(status, 200)
        self.assertEqual(len(captured), 1)
        forwarded_tool = captured[0]['messages'][1]['content']
        self.assertIn('[HARNESS CONTEXT BUDGET: truncated', forwarded_tool)
        self.assertIn('header_1 | header_2', forwarded_tool)
        summary = proxy.summary()
        self.assertEqual(summary['compacted_reads'], 1)
        budget_spans = [s for s in summary['spans'] if s.get('name') == 'guard.context_budget']
        self.assertEqual(len(budget_spans), 1)
        self.assertEqual(budget_spans[0]['attributes']['openinference.span.kind'], 'GUARDRAIL')


class TestSandboxExecutionAndFinopsSettlement(HarnessTestCase):
    """End-to-end sandbox and finops tests using temp git repos and synthetic CLI stubs."""

    def _init_db(self, db_path, sessions=(), messages=()):
        import sqlite3
        conn = sqlite3.connect(db_path)
        try:
            conn.execute('CREATE TABLE session_v2 (id TEXT PRIMARY KEY, cost REAL, time_busy REAL, idle_outcome TEXT)')
            conn.execute('CREATE TABLE session_message (session_id TEXT, type TEXT, data TEXT)')
            conn.executemany(
                'INSERT INTO session_v2 (id, cost, time_busy, idle_outcome) VALUES (?, ?, NULL, NULL)',
                sessions,
            )
            conn.executemany(
                'INSERT INTO session_message (session_id, type, data) VALUES (?, ?, ?)',
                [(sid, kind, json.dumps(data)) for sid, kind, data in messages],
            )
            conn.commit()
        finally:
            conn.close()

    def _init_git_repo(self, repo_dir, filename='target.py', content='VALUE = 0\n'):
        import subprocess
        os.makedirs(repo_dir, exist_ok=True)
        target = os.path.join(repo_dir, filename)
        with open(target, 'w', encoding='utf-8') as fh:
            fh.write(content)
        for cmd in (
            ['/usr/bin/git', 'init'],
            ['/usr/bin/git', 'config', 'user.email', 'test@example.com'],
            ['/usr/bin/git', 'config', 'user.name', 'Test User'],
            ['/usr/bin/git', 'add', filename],
            ['/usr/bin/git', 'commit', '-m', 'init'],
        ):
            subprocess.run(cmd, cwd=repo_dir, check=True, capture_output=True)
        return target

    def _write_stub_cli(self, directory):
        cli_path = os.path.join(directory, 'stub_opencode_cli.py')
        script = '''#!{python}
import json
import os
import sqlite3
import sys

argv = sys.argv[1:]
if argv[:4] == ['api', '--standalone', 'GET', '/api/agent']:
    if os.environ.get('STUB_CLI_BAD_CONFIG') == '1':
        sys.exit(1)
    if os.environ.get('STUB_CLI_BAD_JSON') == '1':
        print('not-json')
        sys.exit(0)
    if os.environ.get('STUB_CLI_BAD_PERMS') == '1':
        print(json.dumps({{'data': [{{'id': 'guarded_worker', 'steps': 1, 'permissions': []}}]}}))
        sys.exit(0)
    if os.environ.get('STUB_CLI_FALLBACK_AGENTS') == '1':
        print(json.dumps({{'data': []}}))
        sys.exit(0)
    with open(os.path.join(os.getcwd(), 'opencode.jsonc'), 'r', encoding='utf-8') as fh:
        cfg = json.load(fh)
    gw = cfg['agents']['guarded_worker']
    print(json.dumps({{'data': [{{'id': 'guarded_worker', 'steps': gw['steps'], 'permissions': gw['permissions']}}]}}))
    sys.exit(0)

if argv[:2] == ['debug', 'agents']:
    with open(os.path.join(os.getcwd(), 'opencode.jsonc'), 'r', encoding='utf-8') as fh:
        cfg = json.load(fh)
    gw = cfg['agents']['guarded_worker']
    print(json.dumps([{{'id': 'guarded_worker', 'steps': gw['steps'], 'permissions': gw['permissions']}}]))
    sys.exit(0)

if argv and argv[0] == 'run':
    sid = argv[argv.index('--session') + 1]
    prompt = argv[-1]
    if os.environ.get('STUB_CLI_PROMPT_DUMP'):
        with open(os.environ['STUB_CLI_PROMPT_DUMP'], 'w', encoding='utf-8') as fh:
            fh.write(prompt)
    files = []
    for line in prompt.splitlines():
        if line.startswith('['):
            try:
                parsed = json.loads(line)
                if isinstance(parsed, list) and parsed and all(isinstance(x, str) for x in parsed):
                    files = parsed
                    break
            except ValueError:
                pass
    db_path = os.environ['HARNESS_TEST_DB']
    cost = float(os.environ.get('STUB_CLI_COST', '0.015'))
    finish = os.environ.get('STUB_CLI_FINISH', 'stop')
    conn = sqlite3.connect(db_path)
    try:
        conn.execute('INSERT INTO session_v2 (id, cost, time_busy, idle_outcome) VALUES (?, ?, 1.0, NULL)', (sid, cost))
        msg = {{
            'id': 'gen-' + sid,
            'cost': cost,
            'tokens': {{'input': 80, 'output': 20}},
            'content': [],
        }}
        if finish:
            msg['finish'] = finish
        conn.execute('INSERT INTO session_message (session_id, type, data) VALUES (?, ?, ?)', (sid, 'assistant', json.dumps(msg)))
        conn.commit()
    finally:
        conn.close()
    if os.environ.get('STUB_CLI_PATCH_CONTENT') and os.environ.get('HARNESS_TEST_PATCH_FILE'):
        with open(os.environ['HARNESS_TEST_PATCH_FILE'], 'w', encoding='utf-8') as fh:
            fh.write(os.environ['STUB_CLI_PATCH_CONTENT'])
    elif os.environ.get('STUB_CLI_NO_CHANGE') != '1' and files:
        content = (
            os.environ.get('STUB_CLI_REPAIR_CONTENT', 'VALUE = 2\\n')
            if '[TEST_ARGV FAILURE' in prompt
            else os.environ.get('STUB_CLI_INITIAL_CONTENT', 'VALUE = 1\\n')
        )
        with open(files[0], 'w', encoding='utf-8') as fh:
            fh.write(content)
    sys.exit(int(os.environ.get('STUB_CLI_EXIT', '0')))

sys.exit(1)
'''.format(python=sys.executable)
        with open(cli_path, 'w', encoding='utf-8') as fh:
            fh.write(script)
        os.chmod(cli_path, 0o755)
        return cli_path

    def test_isolated_worktree_lifecycle_and_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            non_git = os.path.join(tmp, 'plain')
            os.makedirs(non_git)
            plain_file = os.path.join(non_git, 'a.py')
            with open(plain_file, 'w', encoding='utf-8') as fh:
                fh.write('x = 1\n')
            run_dir = os.path.join(tmp, 'state', 'runs', 'r-plain')
            os.makedirs(run_dir)
            plain_task = validate_task({
                'task_id': 'wt-plain',
                'directory': non_git,
                'files': [plain_file],
                'prompt': 'edit',
                'test_argv': [sys.executable, '-c', 'raise SystemExit(0)'],
                'worktree': True,
            })
            with self.assertRaisesRegex(GuardError, 'worktree_requires_git_repository'):
                with isolated_worktree(plain_task, run_dir):
                    pass

            repo = os.path.join(tmp, 'repo')
            target = self._init_git_repo(repo, 'target.py', 'VALUE = 0\n')
            task = validate_task({
                'task_id': 'wt-git',
                'directory': repo,
                'files': [target],
                'prompt': 'edit ' + target,
                'test_argv': [sys.executable, '-c', 'raise SystemExit(0)'],
                'worktree': True,
            })
            wt_run_dir = os.path.join(tmp, 'state', 'runs', 'r-wt')
            os.makedirs(wt_run_dir)
            wt_root = os.path.join(tmp, 'state', 'worktrees', 'r-wt')
            with isolated_worktree(task, wt_run_dir) as wt_task:
                self.assertTrue(os.path.isdir(wt_root))
                self.assertNotEqual(wt_task['files'][0], target)
                self.assertEqual(wt_task['origin_files'], [os.path.realpath(target)])
                with open(wt_task['files'][0], 'w', encoding='utf-8') as fh:
                    fh.write('VALUE = 99\n')
            self.assertFalse(os.path.exists(wt_root))

    def test_owned_process_stop_group_and_preflight(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            cli_path = self._write_stub_cli(tmp)
            target = os.path.join(tmp, 'target.py')
            with open(target, 'w', encoding='utf-8') as fh:
                fh.write('x = 1\n')
            run_dir = os.path.join(tmp, 'run')
            os.makedirs(run_dir)
            task = validate_task({
                'task_id': 'preflight-1',
                'directory': tmp,
                'files': [target],
                'prompt': 'edit target.py',
                'test_argv': [sys.executable, '-c', 'raise SystemExit(0)'],
            })

            from pathlib import Path
            preflight(cli_path, task, Path(run_dir))

            with mock.patch.dict(os.environ, {'STUB_CLI_FALLBACK_AGENTS': '1'}):
                preflight(cli_path, task, Path(run_dir))

            with mock.patch.dict(os.environ, {'STUB_CLI_BAD_CONFIG': '1'}):
                with self.assertRaisesRegex(GuardError, 'configuration_check_failed'):
                    preflight(cli_path, task, Path(run_dir))

            with mock.patch.dict(os.environ, {'STUB_CLI_BAD_JSON': '1'}):
                with self.assertRaisesRegex(GuardError, 'configuration_response_invalid'):
                    preflight(cli_path, task, Path(run_dir))

            with mock.patch.dict(os.environ, {'STUB_CLI_BAD_PERMS': '1'}):
                with self.assertRaisesRegex(GuardError, 'effective_agent_or_steps_mismatch'):
                    preflight(cli_path, task, Path(run_dir))

            log_path = os.path.join(tmp, 'proc.log')
            with open(log_path, 'w', encoding='utf-8') as log_fh:
                with owned_process([sys.executable, '-c', 'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(5)'], tmp, log_fh) as proc:
                    self.assertIsNone(proc.poll())
                self.assertIsNotNone(proc.poll())

    def test_run_task_check_full_worktree_sync_and_micro_repair(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            cli_path = self._write_stub_cli(tmp)
            db_path = os.path.join(tmp, 'opencode.db')
            self._init_db(db_path)
            state_dir = os.path.join(tmp, 'state')

            repo = os.path.join(tmp, 'repo')
            target = self._init_git_repo(repo, 'target.py', 'VALUE = 0\n')
            task_path = os.path.join(tmp, 'task.json')
            with open(task_path, 'w', encoding='utf-8') as fh:
                json.dump({
                    'task_id': 'run-task-ok',
                    'directory': repo,
                    'files': [target],
                    'prompt': 'Set VALUE = 1 in ' + target,
                    'test_argv': [sys.executable, '-c', 'raise SystemExit(0)'],
                    'worktree': True,
                }, fh)

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc_check = run_task(task_path, cli=cli_path, db=db_path, state=state_dir, check=True)
            self.assertEqual(rc_check, 0)
            self.assertEqual(json.loads(buf.getvalue())['status'], 'configuration_verified_no_model_call')

            buf = io.StringIO()
            with mock.patch.dict(os.environ, {'HARNESS_TEST_DB': db_path, 'STUB_CLI_INITIAL_CONTENT': 'VALUE = 1\n'}):
                with contextlib.redirect_stdout(buf):
                    rc_run = run_task(task_path, cli=cli_path, db=db_path, state=state_dir, check=False)
            self.assertEqual(rc_run, 0)
            out = json.loads(buf.getvalue())
            self.assertEqual(out['status'], 'validation_passed_review_required')
            self.assertAlmostEqual(out['cost'], 0.015)
            with open(target, 'r', encoding='utf-8') as fh:
                self.assertEqual(fh.read(), 'VALUE = 1\n')
            wt_dir = os.path.join(state_dir, 'worktrees')
            self.assertEqual(os.listdir(wt_dir), [])

            # Micro-repair path: first test_argv fails with assertion error, second passes after repair
            repo2 = os.path.join(tmp, 'repo2')
            target2 = self._init_git_repo(repo2, 'target.py', 'VALUE = 0\n')
            repair_task_path = os.path.join(tmp, 'task_repair.json')
            test_code = (
                "import pathlib, sys; "
                "txt = pathlib.Path('target.py').read_text(); "
                "print('E   AssertionError: expected VALUE = 2') if 'VALUE = 2' not in txt else None; "
                "sys.exit(0 if 'VALUE = 2' in txt else 1)"
            )
            with open(repair_task_path, 'w', encoding='utf-8') as fh:
                json.dump({
                    'task_id': 'run-task-repair',
                    'directory': repo2,
                    'files': [target2],
                    'prompt': 'Set VALUE = 2 in ' + target2,
                    'test_argv': [sys.executable, '-c', test_code],
                    'worktree': True,
                }, fh)
            buf = io.StringIO()
            with mock.patch.dict(os.environ, {
                'HARNESS_TEST_DB': db_path,
                'STUB_CLI_INITIAL_CONTENT': 'VALUE = 1\n',
                'STUB_CLI_REPAIR_CONTENT': 'VALUE = 2\n',
            }):
                with contextlib.redirect_stdout(buf):
                    rc_repair = run_task(repair_task_path, cli=cli_path, db=db_path, state=state_dir, check=False)
            self.assertEqual(rc_repair, 0)
            out_repair = json.loads(buf.getvalue())
            self.assertEqual(out_repair['status'], 'validation_passed_review_required')
            self.assertEqual(out_repair['steps'], 2)
            self.assertAlmostEqual(out_repair['cost'], 0.03)
            with open(os.path.join(out_repair['artifact'], 'result.json'), 'r', encoding='utf-8') as fh:
                res_repair = json.load(fh)
            self.assertTrue(res_repair['micro_repair_attempted'])
            with open(target2, 'r', encoding='utf-8') as fh:
                self.assertEqual(fh.read(), 'VALUE = 2\n')

    def test_run_task_command_and_patch_modes(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            cli_path = TestSandboxExecutionAndFinopsSettlement._write_stub_cli(self, tmp)
            db_path = os.path.join(tmp, 'opencode.db')
            TestSandboxExecutionAndFinopsSettlement._init_db(self, db_path)
            state_dir = os.path.join(tmp, 'state')
            repo = os.path.join(tmp, 'repo')
            target = TestSandboxExecutionAndFinopsSettlement._init_git_repo(self, repo, 'target.py', 'VALUE = 0\n')

            cmd_task_path = os.path.join(tmp, 'cmd_task.json')
            with open(cmd_task_path, 'w', encoding='utf-8') as fh:
                json.dump({
                    'task_id': 'cmd-task-1',
                    'directory': repo,
                    'files': [target],
                    'mode': 'command',
                    'command_argv': [sys.executable, '-c', "open('target.py', 'w').write('VALUE = 3\\n')"],
                    'test_argv': [sys.executable, '-c', 'raise SystemExit(0)'],
                }, fh)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc_cmd = run_task(cmd_task_path, cli=cli_path, db=db_path, state=state_dir, check=False)
            self.assertEqual(rc_cmd, 0)
            self.assertEqual(json.loads(buf.getvalue())['status'], 'validation_passed_review_required')

            patch_file = os.path.join(repo, 'proposed.patch')
            patch_task_path = os.path.join(tmp, 'patch_task.json')
            with open(patch_task_path, 'w', encoding='utf-8') as fh:
                json.dump({
                    'task_id': 'patch-task-1',
                    'directory': repo,
                    'files': [target],
                    'mode': 'patch',
                    'patch_file': patch_file,
                    'prompt': 'Propose patch',
                    'test_argv': [sys.executable, '-c', 'raise SystemExit(0)'],
                }, fh)
            patch_text = '--- a/target.py\n+++ b/target.py\n@@ -1 +1 @@\n-VALUE = 3\n+VALUE = 4\n'
            # Reset target.py to committed content so git apply --check succeeds
            with open(target, 'w', encoding='utf-8') as fh:
                fh.write('VALUE = 0\n')
            patch_text = '--- a/target.py\n+++ b/target.py\n@@ -1 +1 @@\n-VALUE = 0\n+VALUE = 4\n'
            buf = io.StringIO()
            with mock.patch.dict(os.environ, {
                'HARNESS_TEST_DB': db_path,
                'HARNESS_TEST_PATCH_FILE': patch_file,
                'STUB_CLI_PATCH_CONTENT': patch_text,
            }):
                with contextlib.redirect_stdout(buf):
                    rc_patch = run_task(patch_task_path, cli=cli_path, db=db_path, state=state_dir, check=False)
            self.assertEqual(rc_patch, 0)
            self.assertEqual(json.loads(buf.getvalue())['status'], 'patch_ready_review_required')

    def test_run_task_worktree_read_only_files_guard(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            cli_path = self._write_stub_cli(tmp)
            db_path = os.path.join(tmp, 'opencode.db')
            self._init_db(db_path)
            state_dir = os.path.join(tmp, 'state')

            repo = os.path.join(tmp, 'repo')
            target = self._init_git_repo(repo, 'target.py', 'VALUE = 0\n')
            ro_path = os.path.join(repo, 'ref.py')
            with open(ro_path, 'w', encoding='utf-8') as fh:
                fh.write('REF = 0\n')
            task_path = os.path.join(tmp, 'task_ro.json')
            with open(task_path, 'w', encoding='utf-8') as fh:
                json.dump({
                    'task_id': 'run-task-ro-ok',
                    'directory': repo,
                    'files': [target],
                    'read_only_files': [ro_path],
                    'prompt': 'Set VALUE = 1 in ' + target,
                    'test_argv': [sys.executable, '-c', 'raise SystemExit(0)'],
                    'worktree': True,
                }, fh)

            buf = io.StringIO()
            with mock.patch.dict(os.environ, {'HARNESS_TEST_DB': db_path, 'STUB_CLI_INITIAL_CONTENT': 'VALUE = 1\n'}):
                with contextlib.redirect_stdout(buf):
                    rc_run = run_task(task_path, cli=cli_path, db=db_path, state=state_dir, check=False)
            self.assertEqual(rc_run, 0)
            self.assertEqual(json.loads(buf.getvalue())['status'], 'validation_passed_review_required')
            with open(target, 'r', encoding='utf-8') as fh:
                self.assertEqual(fh.read(), 'VALUE = 1\n')
            with open(ro_path, 'r', encoding='utf-8') as fh:
                self.assertEqual(fh.read(), 'REF = 0\n')

            # A worker/command that touches a read-only file is blocked even when validation passes.
            cmd_task_path = os.path.join(tmp, 'task_ro_bad.json')
            with open(cmd_task_path, 'w', encoding='utf-8') as fh:
                json.dump({
                    'task_id': 'run-task-ro-bad',
                    'directory': repo,
                    'files': [target],
                    'mode': 'command',
                    'read_only_files': [ro_path],
                    'command_argv': [sys.executable, '-c',
                                     "open('target.py', 'w').write('VALUE = 3\\n');"
                                     "open('ref.py', 'w').write('REF = 1\\n')"],
                    'test_argv': [sys.executable, '-c', 'raise SystemExit(0)'],
                    'worktree': True,
                }, fh)
            buf = io.StringIO()
            with mock.patch.dict(os.environ, {'HARNESS_TEST_DB': db_path}):
                with contextlib.redirect_stdout(buf):
                    rc_bad = run_task(cmd_task_path, cli=cli_path, db=db_path, state=state_dir, check=False)
            self.assertNotEqual(rc_bad, 0)
            self.assertEqual(json.loads(buf.getvalue())['reason'], 'read_only_file_modified')
            with open(ro_path, 'r', encoding='utf-8') as fh:
                self.assertEqual(fh.read(), 'REF = 0\n')

    def test_validate_task_strategy_and_candidate_models(self):
        from harness import DEFAULT_MODEL, MODELS, TEACHER_MODELS
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, 'target.py')
            with open(target, 'w', encoding='utf-8') as fh:
                fh.write('x = 1\n')
            base = {
                'task_id': 'strategy-1',
                'directory': tmp,
                'files': [target],
                'prompt': 'edit ' + target,
                'test_argv': [sys.executable, '-c', 'raise SystemExit(0)'],
            }
            with self.assertRaisesRegex(GuardError, 'invalid_strategy'):
                validate_task(dict(base, strategy='best_of_3'))
            with self.assertRaisesRegex(GuardError, 'invalid_candidate_models'):
                validate_task(dict(base, candidate_models=['not-a-model']))
            with self.assertRaisesRegex(GuardError, 'invalid_candidate_models'):
                validate_task(dict(base, candidate_models=['a', 'b', 'c']))
            t = validate_task(dict(base, strategy='best_of_2'))
            self.assertEqual(t['candidate_models'],
                             [DEFAULT_MODEL, 'openrouter/deepseek/deepseek-v4.1-flash-1017'])
            good = sorted(MODELS | TEACHER_MODELS)[0]
            t2 = validate_task(dict(base, candidate_models=[good]))
            self.assertEqual(t2['candidate_models'], [good])
            t3 = validate_task(dict(base, strategy='single'))
            self.assertEqual(t3['strategy'], 'single')

    def test_run_task_best_of_2_early_exit(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            cli_path = self._write_stub_cli(tmp)
            db_path = os.path.join(tmp, 'opencode.db')
            self._init_db(db_path)
            state_dir = os.path.join(tmp, 'state')
            target = os.path.join(tmp, 'target.py')
            with open(target, 'w', encoding='utf-8') as fh:
                fh.write('VALUE = 0\n')
            task_path = os.path.join(tmp, 'task.json')
            with open(task_path, 'w', encoding='utf-8') as fh:
                json.dump({
                    'task_id': 'bo2-early',
                    'directory': tmp,
                    'files': [target],
                    'prompt': 'Set VALUE = 1 in ' + target,
                    'test_argv': [sys.executable, '-c', 'raise SystemExit(0)'],
                    'strategy': 'best_of_2',
                }, fh)
            buf = io.StringIO()
            with mock.patch.dict(os.environ, {'HARNESS_TEST_DB': db_path,
                                              'STUB_CLI_INITIAL_CONTENT': 'VALUE = 1\n'}):
                with contextlib.redirect_stdout(buf):
                    rc = run_task(task_path, cli=cli_path, db=db_path, state=state_dir)
            self.assertEqual(rc, 0)
            self.assertEqual(len(buf.getvalue().strip().splitlines()), 1)
            out = json.loads(buf.getvalue())
            self.assertEqual(out['status'], 'validation_passed_review_required')
            self.assertEqual(out['candidates_run'], 1)
            self.assertEqual(out['steps'], 1)
            self.assertEqual(out['winning_model'], DEFAULT_MODEL)
            with open(target, 'r', encoding='utf-8') as fh:
                self.assertEqual(fh.read(), 'VALUE = 1\n')

    def test_run_best_of_2_fallback_to_candidate_2(self):
        import harness.sandbox as sandbox_module
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            cli_path = self._write_stub_cli(tmp)
            db_path = os.path.join(tmp, 'opencode.db')
            self._init_db(db_path)
            state_dir = os.path.join(tmp, 'state')
            target = os.path.join(tmp, 'target.py')
            with open(target, 'w', encoding='utf-8') as fh:
                fh.write('VALUE = 0\n')
            task_path = os.path.join(tmp, 'task.json')
            with open(task_path, 'w', encoding='utf-8') as fh:
                json.dump({
                    'task_id': 'bo2-fallback',
                    'directory': tmp,
                    'files': [target],
                    'prompt': 'Set VALUE = 1 in ' + target,
                    'test_argv': [sys.executable, '-c', 'raise SystemExit(0)'],
                    'strategy': 'best_of_2',
                }, fh)
            real_run_task = sandbox_module.run_task
            calls = []

            def staged(path, **kwargs):
                calls.append(path)
                if len(calls) == 1:
                    os.environ['STUB_CLI_NO_CHANGE'] = '1'
                else:
                    os.environ.pop('STUB_CLI_NO_CHANGE', None)
                return real_run_task(path, **kwargs)

            buf = io.StringIO()
            with mock.patch.object(sandbox_module, 'run_task', side_effect=staged):
                with mock.patch.dict(os.environ, {'HARNESS_TEST_DB': db_path,
                                                  'STUB_CLI_INITIAL_CONTENT': 'VALUE = 1\n'}):
                    with contextlib.redirect_stdout(buf):
                        rc = run_best_of_2(task_path, cli=cli_path, db=db_path, state=state_dir)
            self.assertEqual(rc, 0)
            out = json.loads(buf.getvalue())
            self.assertEqual(out['status'], 'validation_passed_review_required')
            self.assertEqual(out['candidates_run'], 2)
            self.assertEqual(out['winning_model'], 'openrouter/deepseek/deepseek-v4.1-flash-1017')
            self.assertEqual(out['steps'], 2)
            self.assertAlmostEqual(out['cost'], 0.03)
            with open(target, 'r', encoding='utf-8') as fh:
                self.assertEqual(fh.read(), 'VALUE = 1\n')

    def test_finops_fetch_generation_settled_and_unexposed_settlement(self):
        from unittest import mock
        attempts = {'count': 0}

        class _Resp:
            def __init__(self, body):
                self._body = body

            def read(self):
                return self._body

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def fake_urlopen(req, timeout=15):
            attempts['count'] += 1
            if attempts['count'] == 1:
                return _Resp(b'{}')
            return _Resp(json.dumps({
                'data': {
                    'id': 'gen-retry-1',
                    'finish_reason': 'stop',
                    'total_cost': 0.015,
                    'native_tokens_prompt': 80,
                    'native_tokens_completion': 20,
                }
            }).encode())

        with mock.patch.dict(os.environ, {'OPENROUTER_API_KEY': 'sk-or-test'}):
            with mock.patch('urllib.request.urlopen', side_effect=fake_urlopen):
                settled = fetch_generation_settled('gen-retry-1', delays=(0.0,))
        self.assertEqual(attempts['count'], 2)
        self.assertAlmostEqual(settled['total_cost'], 0.015)
        self.assertEqual(settled['native_tokens_prompt'], 80)
        self.assertEqual(settled['native_tokens_completion'], 20)

        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, 'opencode.db')
            self._init_db(
                db_path,
                sessions=[('ses_unexp', 0.015)],
                messages=[
                    ('ses_unexp', 'assistant', {
                        'id': 'gen-u1',
                        'finish': 'stop',
                        'cost': 0.015,
                        'tokens': {'input': 80, 'output': 20},
                        'content': [],
                    }),
                ],
            )
            run_dir = os.path.join(tmp, 'runs', 'run-unexp')
            os.makedirs(run_dir)
            with open(os.path.join(run_dir, 'result.json'), 'w', encoding='utf-8') as fh:
                json.dump({
                    'task_id': 't-unexp',
                    'session_id': 'ses_unexp',
                    'status': 'blocked',
                    'cost': None,
                    'observed_cost': 0.015,
                    'limits': {'max_steps': 16},
                }, fh)
            with open(os.path.join(run_dir, 'requests.json'), 'w', encoding='utf-8') as fh:
                json.dump({
                    'session_id': 'ses_unexp',
                    'observed_cost': 0.015,
                    'requests': [{
                        'ordinal': 1,
                        'state': 'finished',
                        'cost': 0.015,
                        'provider_request_ids': [],
                    }],
                    'proxy': {
                        'dispatched': [],
                        'blocked': [],
                    },
                }, fh)

            evidence = build_unexposed_settlement_evidence(
                run_dir,
                verified_by='unit-test',
                mapping_source='synthetic local telemetry',
                db=db_path,
            )
            self.assertEqual(evidence['session_id'], 'ses_unexp')
            self.assertTrue(evidence['allow_proxy_stream_fallback'])
            self.assertEqual(len(evidence['requests']), 1)
            self.assertEqual(evidence['requests'][0]['verified_by'], 'unit-test')
            self.assertEqual(evidence['requests'][0]['mapping_source'], 'synthetic local telemetry')
            self.assertEqual(evidence['requests'][0]['resolution'], 'local_telemetry_finished')
            rec = reconcile_run(run_dir, evidence, state=tmp, db=db_path)
            self.assertEqual(rec['schema'], 'verified_billing.v1')
            self.assertAlmostEqual(rec['cost'], 0.015)

            # can_settle_from_proxy_local negative branches
            valid_snap = {'cost': 0.01, 'messages': [
                ('assistant', {'finish': 'stop', 'content': []}),
                ('assistant', {'error': {'type': 'guard'}, 'content': []}),
            ]}
            self.assertFalse(can_settle_from_proxy_local(valid_snap, None))
            self.assertFalse(can_settle_from_proxy_local(valid_snap, {'blocked_count': 0, 'dispatched': []}))
            self.assertFalse(can_settle_from_proxy_local(valid_snap, {'blocked_count': 1, 'dispatched': []}))
            self.assertFalse(can_settle_from_proxy_local(valid_snap, {
                'blocked_count': 1,
                'dispatched': [{'completed': False, 'http_status': 200, 'generation_id': 'g1'}],
            }))
            self.assertFalse(can_settle_from_proxy_local(valid_snap, {
                'blocked_count': 1,
                'dispatched': [{'completed': True, 'http_status': 500, 'generation_id': 'g1'}],
            }))
            self.assertFalse(can_settle_from_proxy_local(valid_snap, {
                'blocked_count': 1,
                'dispatched': [{'completed': True, 'http_status': 200, 'error': 'err', 'generation_id': 'g1'}],
            }))
            self.assertFalse(can_settle_from_proxy_local(valid_snap, {
                'blocked_count': 1,
                'dispatched': [{'completed': True, 'http_status': 200, 'error': None, 'generation_id': None}],
            }))

    def test_sandbox_main_and_validation_edge_cases(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, 'target.py')
            with open(target, 'w', encoding='utf-8') as fh:
                fh.write('x = 1\n')
            base = {
                'task_id': 'edge-1',
                'directory': tmp,
                'files': [target],
                'prompt': 'Edit target.py',
                'test_argv': [sys.executable, '-c', 'raise SystemExit(0)'],
            }
            for bad, pattern in (
                ('not-a-dict', 'task_must_be_an_object'),
                (dict(base, directory=os.path.join(tmp, 'missing')), 'invalid_working_directory'),
                (dict(base, files=[target, target]), 'duplicate_file_or_symlink_alias'),
                (dict(base, test_argv='not-a-list'), 'invalid_argv'),
                (dict(base, mode='unknown'), 'invalid_mode'),
                (dict(base, mode='command', command_argv=['relative']), 'executable_must_be_an_existing_absolute_path'),
                (dict(base, prompt='   '), 'prompt_required'),
                (dict(base, mode='patch', patch_file=target), 'patch_file_must_be_new_absolute_path'),
                (dict(base, files=[os.path.join(tmp, 'new_target.py')], mode='patch',
                      patch_file=os.path.join(tmp, 'new_target.py')), 'patch_cannot_replace_source'),
                (dict(base, model='openrouter/unknown/model'), 'unsupported_model'),
                (dict(base, reasoning_max_tokens=0), 'invalid_reasoning_max_tokens'),
                (dict(base, worktree='yes'), 'invalid_worktree_flag'),
                (dict(base, mode='patch', patch_file=os.path.join(tmp, 'p.patch'), worktree=True), 'worktree_not_supported_for_patch_mode'),
                (dict(base, auto_micro_repair='no'), 'invalid_auto_micro_repair_flag'),
            ):
                with self.assertRaisesRegex(GuardError, pattern):
                    validate_task(bad)

            bin_file = os.path.join(tmp, 'blob.bin')
            with open(bin_file, 'wb') as fh:
                fh.write(b'\xff\xfe\x00\x01')
            bin_run = os.path.join(tmp, 'bin-run')
            os.makedirs(bin_run)
            capture_files(bin_run, [bin_file])
            with open(bin_file, 'wb') as fh:
                fh.write(b'\xff\xfe\x00\x02')
            capture_files(bin_run, [bin_file])

            cli_path = self._write_stub_cli(tmp)
            db_path = os.path.join(tmp, 'opencode.db')
            self._init_db(db_path)
            state_dir = os.path.join(tmp, 'state')
            task_path = os.path.join(tmp, 'task.json')
            with open(task_path, 'w', encoding='utf-8') as fh:
                json.dump(base, fh)

            with mock.patch('harness.sandbox.CLI', cli_path), \
                 mock.patch('harness.sandbox.DB', db_path), \
                 mock.patch('harness.sandbox.STATE', state_dir):
                buf = io.StringIO()
                with mock.patch.object(sys, 'argv', ['opencode_guard', 'check', task_path]):
                    with contextlib.redirect_stdout(buf):
                        self.assertEqual(sandbox_main(), 0)
                self.assertEqual(json.loads(buf.getvalue())['status'], 'configuration_verified_no_model_call')

                buf = io.StringIO()
                with mock.patch.dict(os.environ, {'HARNESS_TEST_DB': db_path, 'STUB_CLI_NO_CHANGE': '1'}):
                    with contextlib.redirect_stdout(buf):
                        rc_no_change = run_task(task_path, cli=cli_path, db=db_path, state=state_dir, check=False)
                self.assertEqual(rc_no_change, 1)
                no_change_out = json.loads(buf.getvalue())
                self.assertEqual(no_change_out['reason'], 'no_change')

                buf = io.StringIO()
                with mock.patch.object(sys, 'argv', ['opencode_guard', 'inspect', no_change_out['artifact']]):
                    with contextlib.redirect_stdout(buf):
                        self.assertEqual(sandbox_main(), 0)
                self.assertEqual(json.loads(buf.getvalue())['task_id'], 'edge-1')

                ev_path = os.path.join(tmp, 'evidence.json')
                with open(ev_path, 'w', encoding='utf-8') as fh:
                    json.dump({'session_id': 'missing', 'requests': []}, fh)
                buf = io.StringIO()
                with mock.patch.object(sys, 'argv', ['opencode_guard', 'reconcile', no_change_out['artifact'], '--evidence', ev_path]):
                    with contextlib.redirect_stdout(buf):
                        self.assertEqual(sandbox_main(), 1)
                self.assertEqual(json.loads(buf.getvalue())['status'], 'blocked')

                buf = io.StringIO()
                with mock.patch.object(sys, 'argv', ['opencode_guard', 'reconcile', no_change_out['artifact']]):
                    with contextlib.redirect_stdout(buf):
                        self.assertEqual(sandbox_main(), 1)
                self.assertEqual(json.loads(buf.getvalue())['reason'], 'evidence_file_required')

                buf = io.StringIO()
                with mock.patch.object(sys, 'argv', [
                    'opencode_guard', 'reconcile', no_change_out['artifact'],
                    '--evidence', ev_path, '--settle-unexposed',
                ]):
                    with contextlib.redirect_stdout(buf):
                        self.assertEqual(sandbox_main(), 1)
                self.assertEqual(json.loads(buf.getvalue())['reason'], 'choose_either_evidence_or_settle_unexposed')

                # prepared_command_failed & validation_failed
                fail_cmd_path = os.path.join(tmp, 'fail_cmd.json')
                with open(fail_cmd_path, 'w', encoding='utf-8') as fh:
                    json.dump(dict(base, mode='command', command_argv=[sys.executable, '-c', 'raise SystemExit(2)']), fh)
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(run_task(fail_cmd_path, cli=cli_path, db=db_path, state=state_dir, check=False), 1)

                fail_val_path = os.path.join(tmp, 'fail_val.json')
                with open(fail_val_path, 'w', encoding='utf-8') as fh:
                    json.dump(dict(
                        base,
                        auto_micro_repair=False,
                        test_argv=[sys.executable, '-c', 'raise SystemExit(3)'],
                    ), fh)
                with mock.patch.dict(os.environ, {'HARNESS_TEST_DB': db_path, 'STUB_CLI_INITIAL_CONTENT': 'x = 2\n'}):
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(run_task(fail_val_path, cli=cli_path, db=db_path, state=state_dir, check=False), 1)

                # run_task with upstream_url + auto_reconcile_from_proxy when finish='tool-calls'
                upstream, _ = _scripted_upstream([{'status': 200}])
                try:
                    with open(target, 'w', encoding='utf-8') as fh:
                        fh.write('x = 1\n')
                    with mock.patch.dict(os.environ, {
                        'HARNESS_TEST_DB': db_path,
                        'STUB_CLI_INITIAL_CONTENT': 'x = 9\n',
                        'STUB_CLI_FINISH': '',
                        'STUB_CLI_EXIT': '1',
                    }):
                        buf = io.StringIO()
                        with contextlib.redirect_stdout(buf):
                            rc_proxy = run_task(
                                task_path,
                                cli=cli_path,
                                db=db_path,
                                state=state_dir,
                                check=False,
                                upstream_url='http://127.0.0.1:%d' % upstream.server_port,
                                fetch=lambda gid: {
                                    'id': gid,
                                    'finish_reason': 'stop',
                                    'total_cost': 0.015,
                                    'native_tokens_prompt': 80,
                                    'native_tokens_completion': 20,
                                },
                            )
                    self.assertEqual(rc_proxy, 1)
                finally:
                    upstream.shutdown()
                    upstream.server_close()

    def test_finops_reconcile_proxy_fallback_and_manifest_edge_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, 'opencode.db')
            self._init_db(
                db_path,
                sessions=[('ses_f1', 0.02), ('ses_f2', 0.0)],
                messages=[
                    ('ses_f1', 'assistant', {
                        'id': 'gen-f1',
                        'finish': 'tool-calls',
                        'cost': 0.02,
                        'tokens': {'input': 40, 'output': 10},
                        'content': [],
                    }),
                ],
            )

            # 1. reconcile_run with proxy stream fallback + blocked_locally
            run1 = os.path.join(tmp, 'runs', 'run-f1')
            os.makedirs(run1)
            with open(os.path.join(run1, 'result.json'), 'w', encoding='utf-8') as fh:
                json.dump({'task_id': 't-f1', 'session_id': 'ses_f1', 'cost': None, 'observed_cost': 0.02}, fh)
            with open(os.path.join(run1, 'requests.json'), 'w', encoding='utf-8') as fh:
                json.dump({
                    'session_id': 'ses_f1',
                    'requests': [
                        {'ordinal': 1, 'state': 'finished', 'cost': 0.02, 'provider_request_ids': []},
                    ],
                    'proxy': {
                        'dispatched': [{'generation_id': 'gen-stream-1', 'completed': True, 'stream_cost': 0.02}],
                        'blocked': [{'reason': 'loop'}],
                    },
                }, fh)
            rec1 = reconcile_run(
                run1,
                {
                    'session_id': 'ses_f1',
                    'allow_proxy_stream_fallback': True,
                    'requests': [{'ordinal': 1, 'resolution': 'local_telemetry_finished', 'verified_by': 'u', 'mapping_source': 'm'}],
                },
                state=tmp,
                db=db_path,
            )
            self.assertAlmostEqual(rec1['cost'], 0.02)

            # 2. auto_reconcile_from_proxy when blocked_count > 0 and dispatched == 0 -> cost 0.0
            run2 = os.path.join(tmp, 'runs', 'run-f2')
            os.makedirs(run2)
            res2_path = os.path.join(run2, 'result.json')
            res2 = {'task_id': 't-f2', 'session_id': 'ses_f2', 'cost': None, 'observed_cost': 0.0}
            with open(res2_path, 'w', encoding='utf-8') as fh:
                json.dump(res2, fh)
            with open(os.path.join(run2, 'requests.json'), 'w', encoding='utf-8') as fh:
                json.dump({
                    'session_id': 'ses_f2',
                    'requests': [{'ordinal': 1, 'state': 'blocked_locally', 'cost': 0.0, 'provider_request_ids': []}],
                    'proxy': {
                        'dispatched_count': 0,
                        'blocked_count': 1,
                        'dispatched': [],
                        'blocked': [{'reason': 'egress_guard_blocked'}],
                    },
                }, fh)
            rec2 = auto_reconcile_from_proxy(run2, res2_path, res2, fetch=lambda gid: {}, db=db_path)
            if rec2 is not None:
                self.assertEqual(rec2['cost'], 0.0)

            # 3. auto_reconcile_from_proxy with multiple dispatched generation_ids
            run3 = os.path.join(tmp, 'runs', 'run-f3')
            os.makedirs(run3)
            res3_path = os.path.join(run3, 'result.json')
            res3 = {'task_id': 't-f3', 'session_id': 'ses_f3', 'cost': None, 'observed_cost': 0.0}
            with open(res3_path, 'w', encoding='utf-8') as fh:
                json.dump(res3, fh)
            with open(os.path.join(run3, 'requests.json'), 'w', encoding='utf-8') as fh:
                json.dump({
                    'session_id': 'ses_f3',
                    'requests': [
                        {'ordinal': 1, 'state': 'finished', 'cost': None, 'provider_request_ids': ['gen-a']},
                        {'ordinal': 2, 'state': 'finished', 'cost': None, 'provider_request_ids': ['gen-b']},
                    ],
                    'proxy': {
                        'dispatched': [
                            {'generation_id': 'gen-a', 'completed': True},
                            {'generation_id': 'gen-b', 'completed': True},
                        ],
                    },
                }, fh)
            rec3 = auto_reconcile_from_proxy(
                run3,
                res3_path,
                res3,
                fetch=lambda gid: {
                    'id': gid,
                    'finish_reason': 'stop',
                    'total_cost': 0.01,
                    'native_tokens_prompt': 10,
                    'native_tokens_completion': 5,
                },
                db=db_path,
            )
            self.assertIsNotNone(rec3)
            self.assertAlmostEqual(rec3['cost'], 0.02)

            # 4. export_sml_cost_manifest with unknown cost (no proxy-requests.json)
            run4 = os.path.join(tmp, 'runs', 'run-f4')
            os.makedirs(run4)
            with open(os.path.join(run4, 'result.json'), 'w', encoding='utf-8') as fh:
                json.dump({'task_id': 't-f4', 'session_id': 'ses_f4', 'cost': None}, fh)
            manifest4 = export_sml_cost_manifest(run4)
            self.assertEqual(manifest4['api_cost_usd'], 0.0)
            self.assertEqual(manifest4['cost_accounting_status'], 'unreconciled')

            # 5. build_unexposed_settlement_evidence with dispatched generation_id & blocked
            run5 = os.path.join(tmp, 'runs', 'run-f5')
            os.makedirs(run5)
            with open(os.path.join(run5, 'result.json'), 'w', encoding='utf-8') as fh:
                json.dump({'task_id': 't-f5', 'session_id': 'ses_f1', 'cost': None, 'limits': {'max_steps': 16}}, fh)
            with open(os.path.join(run5, 'requests.json'), 'w', encoding='utf-8') as fh:
                json.dump({
                    'session_id': 'ses_f1',
                    'requests': [
                        {'ordinal': 1, 'state': 'finished', 'cost': None, 'provider_request_ids': ['gen-p1']},
                        {'ordinal': 2, 'state': 'blocked_locally', 'cost': 0.0, 'provider_request_ids': []},
                    ],
                    'proxy': {
                        'dispatched': [{'generation_id': 'gen-p1', 'completed': True, 'stream_cost': 0.01}],
                        'blocked': [{'reason': 'loop'}],
                    },
                }, fh)
            ev1 = build_unexposed_settlement_evidence(
                run5,
                verified_by='unit-test',
                mapping_source='proxy-stream',
                db=db_path,
            )
            self.assertEqual(ev1['session_id'], 'ses_f1')
            self.assertGreaterEqual(len(ev1['requests']), 1)

            # 5b. reconcile_run rejects unexposed resolution on exposed request, then succeeds with generation_id
            with self.assertRaisesRegex(GuardError, 'finished_request_must_use_actual_cost'):
                reconcile_run(run5, ev1, fetch=lambda gid: {}, state=tmp, db=db_path)

            run5b = os.path.join(tmp, 'runs', 'run-f5b')
            os.makedirs(run5b)
            with open(os.path.join(run5b, 'result.json'), 'w', encoding='utf-8') as fh:
                json.dump({'task_id': 't-f5b', 'session_id': 'ses_f1', 'cost': None, 'limits': {'max_steps': 16}}, fh)
            with open(os.path.join(run5b, 'requests.json'), 'w', encoding='utf-8') as fh:
                json.dump({
                    'session_id': 'ses_f1',
                    'requests': [{'ordinal': 1, 'state': 'finished', 'cost': None, 'provider_request_ids': ['gen-p1']}],
                }, fh)
            rec5 = reconcile_run(
                run5b,
                {'session_id': 'ses_f1', 'requests': [{'ordinal': 1, 'generation_id': 'gen-p1'}]},
                fetch=lambda gid: {
                    'id': gid,
                    'finish_reason': 'stop',
                    'total_cost': 0.01,
                    'native_tokens_prompt': 20,
                    'native_tokens_completion': 10,
                },
                state=tmp,
                db=db_path,
            )
            self.assertAlmostEqual(rec5['cost'], 0.01)

            # 5c. resolved_cost via billing-reconciliation.json & prior_cost across runs
            self.assertAlmostEqual(
                resolved_cost(os.path.join(run5b, 'result.json'), {'task_id': 't-f5b', 'session_id': 'ses_f1', 'cost': None}),
                0.01,
            )
            self.assertAlmostEqual(prior_cost(tmp, 't-f5b'), 0.01)

            # 5d. clear_session_busy False cases + metrics/violation edge cases
            from harness.finops import clear_session_busy, fetch_generation
            self.assertFalse(clear_session_busy(db_path, 'ses_f1'))
            self.assertFalse(clear_session_busy(os.path.join(tmp, 'no-such.db'), 'ses_f1'))
            m_err = metrics({'cost': 0.01, 'messages': [
                ('assistant', {'error': {'type': 'api'}, 'finish': 'stop', 'content': []}),
            ]})
            self.assertTrue(m_err['model_error'])
            self.assertEqual(violation(m_err, 0.15, 16), 'model_error')
            m_unfin = metrics({'cost': 0.01, 'messages': [
                ('assistant', {'content': []}),
            ]})
            self.assertTrue(m_unfin['unfinished'])

            # 5e. fetch_generation missing key / HTTPError / exhausted retries
            from unittest import mock
            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(GuardError):
                    fetch_generation('gen-nokey')
            with mock.patch.dict(os.environ, {'OPENROUTER_API_KEY': 'sk-or-test'}):
                err = urllib.error.HTTPError('http://x', 502, 'bad', {}, None)
                with mock.patch('urllib.request.urlopen', side_effect=err):
                    with self.assertRaises(GuardError):
                        fetch_generation('gen-http-err')
                    with self.assertRaises(GuardError):
                        fetch_generation_settled('gen-exhaust', attempts=2, delays=(0.0,))

            # 5f. resolved_cost tampered reconciliation & prior_cost with multiple runs
            recon_path = os.path.join(run5b, 'billing-reconciliation.json')
            with open(recon_path, 'r', encoding='utf-8') as fh:
                recon_obj = json.load(fh)
            recon_obj['result_sha256'] = '0' * 64
            with open(recon_path, 'w', encoding='utf-8') as fh:
                json.dump(recon_obj, fh)
            with self.assertRaises(GuardError):
                resolved_cost(os.path.join(run5b, 'result.json'), {'task_id': 't-f5b', 'session_id': 'ses_f1', 'cost': None})

            run_paid = os.path.join(tmp, 'runs', 'run-paid')
            os.makedirs(run_paid)
            with open(os.path.join(run_paid, 'result.json'), 'w', encoding='utf-8') as fh:
                json.dump({'task_id': 't-multi', 'session_id': 'ses_m1', 'cost': 0.025}, fh)
            run_other = os.path.join(tmp, 'runs', 'run-other')
            os.makedirs(run_other)
            with open(os.path.join(run_other, 'result.json'), 'w', encoding='utf-8') as fh:
                json.dump({'task_id': 't-other', 'session_id': 'ses_o1', 'cost': 0.05}, fh)
            self.assertAlmostEqual(prior_cost(tmp, 't-multi'), 0.025)

            # 5g. auto_reconcile_from_proxy early returns & reconcile_run edge errors
            self.assertIsNone(auto_reconcile_from_proxy(run_paid, os.path.join(run_paid, 'result.json'), {'cost': 0.025}, db=db_path))
            self.assertIsNone(auto_reconcile_from_proxy(run4, os.path.join(run4, 'result.json'), {'cost': None}, db=db_path))
            with self.assertRaises(GuardError):
                reconcile_run(run_paid, {'session_id': 'ses_m1', 'requests': []}, state=tmp, db=db_path)
            with self.assertRaises(GuardError):
                reconcile_run(run4, {'session_id': 'ses_f4', 'requests': 'bad'}, state=tmp, db=db_path)
            with self.assertRaises(GuardError):
                reconcile_run(run4, {'session_id': 'ses_f4', 'requests': [{'ordinal': 'x'}]}, state=tmp, db=db_path)

            # 6. reconcile_run validation / error branches
            with self.assertRaises(GuardError):
                reconcile_run(os.path.join(tmp, 'missing-run'), {'session_id': 'x', 'requests': []}, state=tmp, db=db_path)
            with self.assertRaises(GuardError):
                reconcile_run(run4, {'session_id': 'wrong', 'requests': []}, state=tmp, db=db_path)
            with self.assertRaises(GuardError):
                reconcile_run(run4, 'not-a-dict', state=tmp, db=db_path)
            with open(os.path.join(run4, 'requests.json'), 'w', encoding='utf-8') as fh:
                json.dump({
                    'session_id': 'ses_f4',
                    'requests': [{'ordinal': 1, 'state': 'finished', 'cost': None, 'provider_request_ids': []}],
                }, fh)
            with self.assertRaises(GuardError):
                reconcile_run(run4, {'session_id': 'ses_f4', 'requests': []}, state=tmp, db=db_path)

            # 7. export_sml_cost_manifest with billing-reconciliation.json & missing result.json
            with open(os.path.join(run1, 'billing-reconciliation.json'), 'r', encoding='utf-8') as fh:
                recon_data = json.load(fh)
            self.assertIn('cost', recon_data)
            m1 = export_sml_cost_manifest(run1, gpu_hours=1.0, gpu_hourly_rate_usd=2.0)
            self.assertEqual(m1['cost_accounting_status'], 'settled')
            self.assertAlmostEqual(m1['compute_cost_usd'], 2.0)

            empty_run = os.path.join(tmp, 'runs', 'empty-run')
            os.makedirs(empty_run)
            m_empty = export_sml_cost_manifest(empty_run)
            self.assertEqual(m_empty['cost_accounting_status'], 'unreconciled')

            # 8. Additional finops coverage: request_records, reconcile_run validation, fetch_generation non-dict
            recs = request_records(
                {
                    'cost': 0.01,
                    'messages': [
                        ('assistant', {'finish': 'tool-calls', 'cost': 0.01, 'tokens': {'prompt_tokens': 10, 'completion_tokens': 5}}),
                    ],
                },
                {'dispatched': [{'completed': True, 'http_status': 200, 'generation_id': None}], 'blocked_count': 0},
            )
            self.assertEqual(len(recs), 1)

            with self.assertRaises(FileNotFoundError):
                reconcile_run(empty_run, {'session_id': 'x', 'requests': []}, state=tmp, db=db_path)
            with self.assertRaises(GuardError):
                reconcile_run(run4, {'session_id': 'ses_f4', 'requests': [{'ordinal': 1, 'generation_id': 'g1'}, {'ordinal': 1, 'generation_id': 'g2'}]}, state=tmp, db=db_path)
            with self.assertRaises(GuardError):
                reconcile_run(run4, {'session_id': 'ses_f4', 'requests': [{'ordinal': 1}]}, state=tmp, db=db_path)
            with self.assertRaises(GuardError):
                reconcile_run(run4, {'session_id': 'ses_f4', 'requests': [{'ordinal': 99, 'generation_id': 'g1'}]}, state=tmp, db=db_path)

            # 10. auto_reconcile_from_proxy when fetch fails -> returns None; prior_cost rejects corrupt run
            corrupt_state = os.path.join(tmp, 'corrupt-state')
            corrupt_run = os.path.join(corrupt_state, 'runs', 'run-corrupt')
            os.makedirs(corrupt_run)
            with open(os.path.join(corrupt_run, 'result.json'), 'w', encoding='utf-8') as fh:
                fh.write('{bad json')
            with self.assertRaisesRegex(GuardError, 'unreadable_cost_journal'):
                prior_cost(corrupt_state, 't-multi')

            run_fail_fetch = os.path.join(tmp, 'runs', 'run-fail-fetch')
            os.makedirs(run_fail_fetch)
            res_ff_path = os.path.join(run_fail_fetch, 'result.json')
            res_ff = {'task_id': 't-ff', 'session_id': 'ses_ff', 'cost': None, 'observed_cost': 0.0}
            with open(res_ff_path, 'w', encoding='utf-8') as fh:
                json.dump(res_ff, fh)
            with open(os.path.join(run_fail_fetch, 'requests.json'), 'w', encoding='utf-8') as fh:
                json.dump({
                    'session_id': 'ses_ff',
                    'requests': [{'ordinal': 1, 'state': 'finished', 'cost': None, 'provider_request_ids': ['gen-bad']}],
                    'proxy': {'dispatched': [{'generation_id': 'gen-bad', 'completed': True}]},
                }, fh)
            with self.assertRaisesRegex(GuardError, 'billing_not_final_or_wrong_identity'):
                auto_reconcile_from_proxy(run_fail_fetch, res_ff_path, res_ff, fetch=lambda gid: {}, db=db_path)

            # 11. number() validation, snapshot with NULL cost, and manifest fallback branches
            for bad_num in (True, float('nan'), 'abc'):
                with self.assertRaises(GuardError):
                    number(bad_num, 'max_cost_usd')

            import sqlite3
            conn_null = sqlite3.connect(db_path)
            try:
                conn_null.execute("INSERT INTO session_v2 (id, cost, time_busy, idle_outcome) VALUES ('ses_null', NULL, NULL, NULL)")
                conn_null.commit()
            finally:
                conn_null.close()
            with self.assertRaisesRegex(GuardError, 'invalid_telemetry_cost'):
                snapshot(db_path, 'ses_null')

            run_proxy_only = os.path.join(tmp, 'runs', 'run-proxy-only')
            os.makedirs(run_proxy_only)
            with open(os.path.join(run_proxy_only, 'result.json'), 'w', encoding='utf-8') as fh:
                json.dump({'task_id': 't-po', 'session_id': 'ses_po', 'cost': None, 'started_at': 10.0}, fh)
            with open(os.path.join(run_proxy_only, 'proxy-requests.json'), 'w', encoding='utf-8') as fh:
                json.dump({'dispatched': [{'stream_cost': 0.005, 'usage': {'input': 10, 'output': 5}}]}, fh)
            m_po = export_sml_cost_manifest(run_proxy_only)
            self.assertAlmostEqual(m_po['api_cost_usd'], 0.005)
            self.assertEqual(m_po['prompt_tokens'], 10)
            self.assertEqual(m_po['completion_tokens'], 5)

            # 12. request_records with non-gen id + error+tokens
            recs_err = request_records(
                {'cost': 0.01, 'messages': [
                    ('assistant', {'id': 'msg_local_1', 'finish': 'stop', 'tokens': {'input': 5, 'output': 2}}),
                    ('assistant', {'error': {'type': 'api'}, 'tokens': {'input': 5, 'output': 0}}),
                ]},
                {'dispatched': [{'completed': True, 'http_status': 200, 'generation_id': 'gen-from-proxy'}], 'blocked_count': 1},
            )
            self.assertEqual(len(recs_err), 2)
            self.assertEqual(recs_err[0]['provider_request_ids'], ['gen-from-proxy'])

            # 13. build_unexposed_settlement_evidence error cases
            with self.assertRaisesRegex(GuardError, 'verified_by_required'):
                build_unexposed_settlement_evidence(run5, verified_by='', mapping_source='m', db=db_path)
            with self.assertRaises(GuardError):
                build_unexposed_settlement_evidence(run5, verified_by='u', mapping_source='', db=db_path)

            # 13b. reconcile_run invalid generation_id, wrong-identity fetch & blocked_locally-only run
            with self.assertRaisesRegex(GuardError, 'invalid_or_duplicate_generation_id'):
                reconcile_run(
                    run4,
                    {'session_id': 'ses_f4', 'requests': [{'ordinal': 1, 'generation_id': 'bad-id'}]},
                    fetch=lambda gid: {},
                    state=tmp,
                    db=db_path,
                )
            with self.assertRaisesRegex(GuardError, 'mapping_evidence_required'):
                reconcile_run(
                    run4,
                    {'session_id': 'ses_f4', 'requests': [{'ordinal': 1, 'generation_id': 'gen-want'}]},
                    fetch=lambda gid: {'id': 'gen-other', 'finish_reason': 'stop', 'total_cost': 0.01},
                    state=tmp,
                    db=db_path,
                )
            run_wrong = os.path.join(tmp, 'runs', 'run-wrong')
            os.makedirs(run_wrong)
            with open(os.path.join(run_wrong, 'result.json'), 'w', encoding='utf-8') as fh:
                json.dump({'task_id': 't-w', 'session_id': 'ses_w', 'cost': None}, fh)
            with open(os.path.join(run_wrong, 'requests.json'), 'w', encoding='utf-8') as fh:
                json.dump({
                    'session_id': 'ses_w',
                    'requests': [{'ordinal': 1, 'state': 'finished', 'cost': None, 'provider_request_ids': ['gen-want']}],
                }, fh)
            with self.assertRaisesRegex(GuardError, 'billing_not_final_or_wrong_identity'):
                reconcile_run(
                    run_wrong,
                    {'session_id': 'ses_w', 'requests': [{'ordinal': 1, 'generation_id': 'gen-want'}]},
                    fetch=lambda gid: {'id': 'gen-other', 'finish_reason': 'stop', 'total_cost': 0.01},
                    state=tmp,
                    db=db_path,
                )
            run_blk = os.path.join(tmp, 'runs', 'run-blk')
            os.makedirs(run_blk)
            with open(os.path.join(run_blk, 'result.json'), 'w', encoding='utf-8') as fh:
                json.dump({'task_id': 't-blk', 'session_id': 'ses_blk', 'cost': None}, fh)
            with open(os.path.join(run_blk, 'requests.json'), 'w', encoding='utf-8') as fh:
                json.dump({
                    'session_id': 'ses_blk',
                    'requests': [{'ordinal': 1, 'state': 'finished', 'cost': 0.0, 'provider_request_ids': []}],
                    'proxy': {'dispatched': [], 'blocked': [{'reason': 'loop'}]},
                }, fh)
            ev_blk = build_unexposed_settlement_evidence(run_blk, verified_by='u', mapping_source='m', db=db_path)
            rec_blk = reconcile_run(run_blk, ev_blk, state=tmp, db=db_path)
            self.assertEqual(rec_blk['cost'], 0.0)

            # 13c. metrics with tool_use/compact/read & prior_cost with unknown cost run
            m_reads = metrics({'cost': 0.01, 'messages': [
                ('assistant', {'finish': 'stop', 'content': [
                    {'type': 'tool', 'name': 'bash', 'state': {'status': 'completed'}},
                ]}),
            ]})
            self.assertEqual(m_reads['reads'], 0)
            with self.assertRaisesRegex(GuardError, 'prior_cost_unknown_reconcile_before_retry'):
                prior_cost(tmp, 't-f4')

            # 13. build_unexposed_settlement_evidence when result.json has no session_id -> GuardError
            run_nosid = os.path.join(tmp, 'runs', 'run-nosid')
            os.makedirs(run_nosid)
            with open(os.path.join(run_nosid, 'result.json'), 'w', encoding='utf-8') as fh:
                json.dump({'task_id': 't-ns', 'session_id': None}, fh)
            with self.assertRaises(GuardError):
                build_unexposed_settlement_evidence(run_nosid, verified_by='u', mapping_source='m', db=db_path)

            # 14. auto_reconcile_from_proxy with auth_bearer + requests with missing provider_request_ids
            run_noid = os.path.join(tmp, 'runs', 'run-noid')
            os.makedirs(run_noid)
            res_noid_path = os.path.join(run_noid, 'result.json')
            res_noid = {'task_id': 't-noid', 'session_id': 'ses_noid', 'cost': None}
            with open(res_noid_path, 'w', encoding='utf-8') as fh:
                json.dump(res_noid, fh)
            with open(os.path.join(run_noid, 'requests.json'), 'w', encoding='utf-8') as fh:
                json.dump({
                    'session_id': 'ses_noid',
                    'requests': [{'ordinal': 1, 'state': 'finished', 'cost': None, 'provider_request_ids': []}],
                    'proxy': {'dispatched': []},
                }, fh)
            rec_noid = auto_reconcile_from_proxy(run_noid, res_noid_path, res_noid, fetch=lambda gid: {}, db=db_path)
            self.assertIsNotNone(rec_noid)
            self.assertAlmostEqual(rec_noid['cost'], 0.01)

            run_dur = os.path.join(tmp, 'runs', 'run-dur')
            os.makedirs(run_dur)
            with open(os.path.join(run_dur, 'result.json'), 'w', encoding='utf-8') as fh:
                json.dump({'task_id': 't-dur', 'session_id': 'ses_d', 'cost': 0.01, 'duration_seconds': 12.5}, fh)
            with open(os.path.join(run_dur, 'requests.json'), 'w', encoding='utf-8') as fh:
                json.dump({'requests': [{'ordinal': 1, 'tokens': {'prompt_tokens': 30, 'completion_tokens': 12}}]}, fh)
            m_dur = export_sml_cost_manifest(run_dur)
            self.assertEqual(m_dur['wall_clock_seconds'], 12.5)
            self.assertEqual(m_dur['prompt_tokens'], 30)
            self.assertEqual(m_dur['completion_tokens'], 12)

            # 9. fetch_generation returning non-dict JSON & urllib.error.URLError
            class _RespJson:
                def __init__(self, body):
                    self._body = body
                def read(self):
                    return self._body
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    return False

            with mock.patch.dict(os.environ, {'OPENROUTER_API_KEY': 'sk-or-test'}):
                with mock.patch('urllib.request.urlopen', return_value=_RespJson(b'[1, 2]')):
                    with self.assertRaises(TypeError):
                        fetch_generation('gen-list')
                with mock.patch('urllib.request.urlopen', side_effect=urllib.error.URLError('down')):
                    with self.assertRaises(GuardError):
                        fetch_generation('gen-url-err')

            # 15. reconcile_run with conservative_step_cap
            run_cap = os.path.join(tmp, 'runs', 'run-cap')
            os.makedirs(run_cap)
            with open(os.path.join(run_cap, 'result.json'), 'w', encoding='utf-8') as fh:
                json.dump({'task_id': 't-cap', 'session_id': 'ses_cap2', 'cost': None, 'limits': {'max_steps': 16}}, fh)
            with open(os.path.join(run_cap, 'requests.json'), 'w', encoding='utf-8') as fh:
                json.dump({
                    'session_id': 'ses_cap2',
                    'requests': [{'ordinal': 1, 'state': 'dispatched', 'cost': None, 'provider_request_ids': []}],
                }, fh)
            rec_cap = reconcile_run(
                run_cap,
                {
                    'session_id': 'ses_cap2',
                    'requests': [{
                        'ordinal': 1,
                        'resolution': 'conservative_step_cap',
                        'capped_cost': 0.01,
                        'verified_by': 'u',
                        'mapping_source': 'm',
                    }],
                },
                state=tmp,
                db=db_path,
            )
            self.assertAlmostEqual(rec_cap['cost'], 0.01)

            # 16. build_unexposed_settlement_evidence with unfinished/dispatched request + stream_cost
            run_unfin = os.path.join(tmp, 'runs', 'run-unfin')
            os.makedirs(run_unfin)
            with open(os.path.join(run_unfin, 'result.json'), 'w', encoding='utf-8') as fh:
                json.dump({'task_id': 't-unfin', 'session_id': 'ses_unfin', 'cost': None, 'limits': {'max_steps': 16}}, fh)
            with open(os.path.join(run_unfin, 'requests.json'), 'w', encoding='utf-8') as fh:
                json.dump({
                    'session_id': 'ses_unfin',
                    'requests': [{'ordinal': 1, 'state': 'dispatched', 'cost': None, 'provider_request_ids': []}],
                    'proxy': {'dispatched': [{'completed': True, 'stream_cost': 0.007, 'generation_id': None}], 'blocked': []},
                }, fh)
            ev_unfin = build_unexposed_settlement_evidence(run_unfin, verified_by='u', mapping_source='m', db=db_path)
            self.assertEqual(ev_unfin['session_id'], 'ses_unfin')
            rec_unfin = reconcile_run(run_unfin, ev_unfin, state=tmp, db=db_path)
            self.assertGreater(rec_unfin['cost'], 0.0)

            # 17. export_sml_cost_manifest with requests.json usage + out_path
            run_usage = os.path.join(tmp, 'runs', 'run-usage')
            os.makedirs(run_usage)
            with open(os.path.join(run_usage, 'result.json'), 'w', encoding='utf-8') as fh:
                json.dump({'task_id': 't-u', 'session_id': 'ses_u', 'cost': 0.02, 'wall_clock_seconds': 5.0}, fh)
            with open(os.path.join(run_usage, 'requests.json'), 'w', encoding='utf-8') as fh:
                json.dump({'requests': [{'ordinal': 1, 'tokens': {'input_tokens': 11, 'output_tokens': 7}}]}, fh)
            m_u = export_sml_cost_manifest(run_usage, os.path.join(run_usage, 'cost.json'))
            self.assertEqual(m_u['prompt_tokens'], 11)
            self.assertEqual(m_u['completion_tokens'], 7)

            # 18. clear_session_busy with outcome='succeeded', old-schema DB, and can_settle_from_proxy_local branches
            from harness.finops import clear_session_busy as _csb
            conn_busy = sqlite3.connect(db_path)
            try:
                conn_busy.execute("INSERT INTO session_v2 (id, cost, time_busy, idle_outcome) VALUES ('ses_busy2', 0.01, 9.0, NULL)")
                conn_busy.commit()
            finally:
                conn_busy.close()
            self.assertTrue(_csb(db_path, 'ses_busy2', outcome='succeeded'))

            old_db = os.path.join(tmp, 'old.db')
            c_old = sqlite3.connect(old_db)
            c_old.execute('CREATE TABLE session_v2 (id TEXT PRIMARY KEY, cost REAL)')
            c_old.execute("INSERT INTO session_v2 VALUES ('ses_old', 0.01)")
            c_old.commit()
            c_old.close()
            self.assertFalse(_csb(old_db, 'ses_old'))

            self.assertFalse(can_settle_from_proxy_local(
                {'cost': 0.01, 'messages': [('assistant', {'finish': 'stop', 'content': []})]},
                {'blocked_count': 1, 'dispatched': [{'completed': True, 'http_status': 200, 'generation_id': 'gen-1'}]},
            ))
            self.assertFalse(can_settle_from_proxy_local(
                {'cost': 0.01, 'messages': [
                    ('assistant', {'finish': 'stop', 'content': []}),
                    ('assistant', {'error': {'type': 'guard'}, 'content': []}),
                ]},
                {'blocked_count': 2, 'dispatched': [{'completed': True, 'http_status': 200, 'generation_id': 'gen-1'}]},
            ))


class TestProxyPerTaskSmlManifests(HarnessTestCase):
    def test_proxy_per_task_id_summaries_and_auto_flush_sml_manifests(self):
        from harness.proxy import openrouter_proxy

        class _UsageUpstream(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                return

            def do_POST(self):
                length = int(self.headers.get('Content-Length') or '0')
                self.rfile.read(length)
                req_idx = self.server.req_idx
                self.server.req_idx += 1
                if req_idx == 0:
                    payload = json.dumps({
                        'id': 'gen-ep1-a',
                        'choices': [{'message': {'role': 'assistant', 'content': 'ok'}}],
                        'usage': {'prompt_tokens': 100, 'completion_tokens': 25, 'cost': 0.004},
                    }).encode()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                elif req_idx == 1:
                    chunks = (
                        b'data: {"id":"gen-ep1-b","choices":[{"delta":{"content":"hi"}}]}\n\n'
                        b'data: {"id":"gen-ep1-b","choices":[],"usage":{"prompt_tokens":60,"completion_tokens":15,"cost":0.002}}\n\n'
                        b'data: [DONE]\n\n'
                    )
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Content-Length', str(len(chunks)))
                    self.end_headers()
                    self.wfile.write(chunks)
                else:
                    payload = json.dumps({
                        'id': 'gen-ep2-a',
                        'choices': [{'message': {'role': 'assistant', 'content': 'ok'}}],
                        'usage': {'prompt_tokens': 210, 'completion_tokens': 45, 'cost': 0.009},
                    }).encode()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)

        upstream = http.server.ThreadingHTTPServer(('127.0.0.1', 0), _UsageUpstream)
        upstream.req_idx = 0
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        results_dir = self.path('sml-results')

        def _send(base_url, task_id):
            body = json.dumps({
                'model': GLM,
                'messages': [{'role': 'user', 'content': 'hi'}],
            }).encode()
            req = urllib.request.Request(
                base_url + '/chat/completions',
                data=body,
                headers={
                    'Content-Type': 'application/json',
                    'X-Harness-Task-Id': task_id,
                },
                method='POST',
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                resp.read()
                return resp.status

        try:
            with openrouter_proxy(
                lambda: None,
                upstream_url=f'http://127.0.0.1:{upstream.server_port}',
                sml_results_dir=results_dir,
            ) as proxy:
                self.assertEqual(_send(proxy.base_url, 'sml-ep-01'), 200)
                self.assertEqual(_send(proxy.base_url, 'sml-ep-01'), 200)
                self.assertEqual(_send(proxy.base_url, 'sml-ep-02'), 200)
                summaries = proxy.task_summaries()
                spans = list(proxy.tracer.spans)
        finally:
            upstream.shutdown()
            upstream.server_close()

        self.assertEqual(summaries['sml-ep-01']['request_count'], 2)
        self.assertEqual(summaries['sml-ep-01']['prompt_tokens'], 160)
        self.assertEqual(summaries['sml-ep-01']['completion_tokens'], 40)
        self.assertEqual(summaries['sml-ep-01']['api_cost_usd'], 0.006)
        self.assertEqual(summaries['sml-ep-02']['request_count'], 1)
        self.assertEqual(summaries['sml-ep-02']['prompt_tokens'], 210)
        self.assertEqual(summaries['sml-ep-02']['completion_tokens'], 45)
        self.assertEqual(summaries['sml-ep-02']['api_cost_usd'], 0.009)
        self.assertEqual(
            [s['attributes'].get('harness.task_id') for s in spans if s['name'] == 'gen_ai.chat'],
            ['sml-ep-01', 'sml-ep-01', 'sml-ep-02'],
        )

        ep1_path = os.path.join(results_dir, 'sml-ep-01', 'cost.json')
        ep2_path = os.path.join(results_dir, 'sml-ep-02', 'cost.json')
        self.assertTrue(os.path.exists(ep1_path))
        self.assertTrue(os.path.exists(ep2_path))
        with open(ep1_path) as fh:
            ep1 = json.load(fh)
        with open(ep2_path) as fh:
            ep2 = json.load(fh)
        self.assertEqual(ep1['request_count'], 2)
        self.assertEqual(ep1['prompt_tokens'], 160)
        self.assertEqual(ep1['completion_tokens'], 40)
        self.assertEqual(ep1['api_cost_usd'], 0.006)
        self.assertEqual(ep1['total_cost_usd'], 0.006)
        self.assertEqual(ep1['cost_accounting_status'], 'estimated')
        self.assertEqual(ep2['request_count'], 1)
        self.assertEqual(ep2['prompt_tokens'], 210)
        self.assertEqual(ep2['completion_tokens'], 45)
        self.assertEqual(ep2['api_cost_usd'], 0.009)
        self.assertEqual(ep2['total_cost_usd'], 0.009)
        self.assertEqual(ep2['cost_accounting_status'], 'estimated')


class TestLastStepPatchSalvage(TestSandboxExecutionAndFinopsSettlement):
    def test_step_limit_salvage_passes_when_validation_passes(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            cli_path = self._write_stub_cli(tmp)
            db_path = os.path.join(tmp, 'opencode.db')
            self._init_db(db_path)
            state_dir = os.path.join(tmp, 'state')

            repo = os.path.join(tmp, 'repo')
            target = self._init_git_repo(repo, 'target.py', 'VALUE = 0\n')
            task_path = os.path.join(tmp, 'task_salvage_ok.json')
            test_code = (
                "import pathlib, sys; "
                "sys.exit(0 if 'VALUE = 1' in pathlib.Path('target.py').read_text() else 1)"
            )
            with open(task_path, 'w', encoding='utf-8') as fh:
                json.dump({
                    'task_id': 'salvage-pass',
                    'directory': repo,
                    'files': [target],
                    'prompt': 'Set VALUE = 1 in ' + target,
                    'test_argv': [sys.executable, '-c', test_code],
                    'worktree': True,
                }, fh)

            def _fake_model_run(cli, db, task, run_dir, result, budget, proxy=None, started_ref=None):
                with open(task['files'][0], 'w', encoding='utf-8') as fh:
                    fh.write('VALUE = 1\n')
                raise GuardError('step_limit')

            buf = io.StringIO()
            with mock.patch('harness.sandbox.model_run', side_effect=_fake_model_run):
                with contextlib.redirect_stdout(buf):
                    rc = run_task(task_path, cli=cli_path, db=db_path, state=state_dir, check=False)
            self.assertEqual(rc, 0)
            out = json.loads(buf.getvalue())
            self.assertEqual(out['status'], 'validation_passed_review_required')
            with open(os.path.join(out['artifact'], 'result.json'), 'r', encoding='utf-8') as fh:
                res = json.load(fh)
            self.assertEqual(res['status'], 'validation_passed_review_required')
            self.assertEqual(res['salvaged_from'], 'step_limit')
            with open(target, 'r', encoding='utf-8') as fh:
                self.assertEqual(fh.read(), 'VALUE = 1\n')

    def test_step_limit_salvage_blocks_and_restores_when_validation_fails(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            cli_path = self._write_stub_cli(tmp)
            db_path = os.path.join(tmp, 'opencode.db')
            self._init_db(db_path)
            state_dir = os.path.join(tmp, 'state')

            repo = os.path.join(tmp, 'repo')
            target = self._init_git_repo(repo, 'target.py', 'VALUE = 0\n')
            task_path = os.path.join(tmp, 'task_salvage_fail.json')
            test_code = (
                "import pathlib, sys; "
                "sys.exit(0 if 'VALUE = 2' in pathlib.Path('target.py').read_text() else 1)"
            )
            with open(task_path, 'w', encoding='utf-8') as fh:
                json.dump({
                    'task_id': 'salvage-fail',
                    'directory': repo,
                    'files': [target],
                    'prompt': 'Set VALUE = 2 in ' + target,
                    'test_argv': [sys.executable, '-c', test_code],
                    'worktree': True,
                }, fh)

            def _fake_model_run(cli, db, task, run_dir, result, budget, proxy=None, started_ref=None):
                with open(task['files'][0], 'w', encoding='utf-8') as fh:
                    fh.write('VALUE = 1\n')
                raise GuardError('step_limit')

            buf = io.StringIO()
            with mock.patch('harness.sandbox.model_run', side_effect=_fake_model_run):
                with contextlib.redirect_stdout(buf):
                    rc = run_task(task_path, cli=cli_path, db=db_path, state=state_dir, check=False)
            self.assertEqual(rc, 1)
            out = json.loads(buf.getvalue())
            self.assertEqual(out['status'], 'blocked')
            self.assertEqual(out['reason'], 'step_limit')
            with open(os.path.join(out['artifact'], 'result.json'), 'r', encoding='utf-8') as fh:
                res = json.load(fh)
            self.assertEqual(res['status'], 'blocked')
            self.assertEqual(res['reason'], 'step_limit')
            self.assertIsNone(res.get('salvaged_from'))
            self.assertFalse(res.get('micro_repair_attempted', False))
            with open(target, 'r', encoding='utf-8') as fh:
                self.assertEqual(fh.read(), 'VALUE = 0\n')

    def test_validate_task_adaptive_steps_floor(self):
        with tempfile.TemporaryDirectory() as tmp:
            f1 = os.path.join(tmp, 'a.py')
            f2 = os.path.join(tmp, 'b.py')
            f3 = os.path.join(tmp, 'c.py')
            for p in (f1, f2, f3):
                with open(p, 'w', encoding='utf-8') as fh:
                    fh.write('x = 1\n')
            base = {
                'task_id': 'adaptive-1',
                'directory': tmp,
                'prompt': 'edit files',
                'test_argv': [sys.executable, '-c', 'pass'],
            }
            self.assertEqual(validate_task(dict(base, files=[f1], max_steps=6))['max_steps'], 7)
            self.assertEqual(validate_task(dict(base, files=[f1], max_steps=10))['max_steps'], 10)
            self.assertEqual(validate_task(dict(base, files=[f1, f2], max_steps=6))['max_steps'], 10)
            self.assertEqual(validate_task(dict(base, files=[f1, f2, f3], max_steps=6))['max_steps'], 13)
            self.assertEqual(
                validate_task(dict(base, files=[f1, f2, f3], max_steps=6, adaptive_steps=False))['max_steps'],
                6,
            )
            self.assertEqual(
                validate_task(dict(
                    base,
                    files=[f1, f2, f3],
                    max_steps=6,
                    mode='patch',
                    patch_file=os.path.join(tmp, 'out.patch'),
                ))['max_steps'],
                6,
            )

    def test_auto_micro_repair_triggers_above_8_steps_with_current_patch(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            cli_path = self._write_stub_cli(tmp)
            db_path = os.path.join(tmp, 'opencode.db')
            self._init_db(db_path)
            state_dir = os.path.join(tmp, 'state')

            repo = os.path.join(tmp, 'repo')
            target = self._init_git_repo(repo, 'target.py', 'VALUE = 0\n')
            helper = os.path.join(repo, 'helper.py')
            with open(helper, 'w', encoding='utf-8') as fh:
                fh.write('HELPER = 1\n')
            task_path = os.path.join(tmp, 'task_repair.json')
            test_code = (
                "import pathlib, sys; "
                "sys.exit(0 if 'VALUE = 2' in pathlib.Path('target.py').read_text() else 1)"
            )
            with open(task_path, 'w', encoding='utf-8') as fh:
                json.dump({
                    'task_id': 'repair-10-steps',
                    'directory': repo,
                    'files': [target, helper],
                    'prompt': 'Set VALUE = 2 in ' + target,
                    'test_argv': [sys.executable, '-c', test_code],
                    'max_steps': 12,
                    'max_cost_usd': 0.10,
                }, fh)

            seen_repair_tasks = []

            def _fake_model_run(cli, db, task, run_dir, result, budget, proxy=None, started_ref=None):
                if len(seen_repair_tasks) == 0 and not result.get('micro_repair_attempted'):
                    with open(target, 'w', encoding='utf-8') as fh:
                        fh.write('VALUE = 1\n')
                    result['steps'] = 10
                    result['cost'] = 0.05
                    return
                seen_repair_tasks.append(dict(task))
                with open(target, 'w', encoding='utf-8') as fh:
                    fh.write('VALUE = 2\n')
                result['steps'] = 2
                result['cost'] = 0.01

            buf = io.StringIO()
            with mock.patch('harness.sandbox.model_run', side_effect=_fake_model_run):
                with contextlib.redirect_stdout(buf):
                    rc = run_task(task_path, cli=cli_path, db=db_path, state=state_dir, check=False)
            self.assertEqual(rc, 0)
            self.assertEqual(len(seen_repair_tasks), 1)
            repair_task = seen_repair_tasks[0]
            self.assertEqual(repair_task['max_steps'], 8)
            self.assertIn('Apply the minimal targeted edit directly to fix the failure above', repair_task['prompt'])
            self.assertIn('[CURRENT_PATCH]', repair_task['prompt'])
            self.assertIn('+VALUE = 1', repair_task['prompt'])
            out = json.loads(buf.getvalue())
            with open(os.path.join(out['artifact'], 'result.json'), 'r', encoding='utf-8') as fh:
                res = json.load(fh)
            self.assertTrue(res['micro_repair_attempted'])
            self.assertEqual(res['steps'], 12)
            self.assertAlmostEqual(res['cost'], 0.06)

    def test_inject_edit_failure_hints_and_idempotency(self):
        # Empty / non-list inputs pass through cleanly.
        self.assertEqual(inject_edit_failure_hints([]), ([], 0))

        # Non-trailing tool error is ignored.
        non_trailing = [
            {'role': 'tool', 'tool_call_id': 'e0', 'content': 'oldString not found in file'},
            {'role': 'assistant', 'content': 'retrying'},
            {'role': 'tool', 'tool_call_id': 'r1', 'content': 'ok'},
        ]
        out, count = inject_edit_failure_hints(non_trailing)
        self.assertEqual(count, 0)
        self.assertIs(out, non_trailing)

        # All four mismatch patterns trigger on trailing tool messages (case-insensitive).
        for phrase in (
            'Error: oldString not found in content',
            'old_string not found',
            'FAILED TO MATCH target block',
            'No exact match found',
        ):
            msgs = [
                {'role': 'user', 'content': 'edit it'},
                {'role': 'tool', 'tool_call_id': 'e1', 'content': phrase},
            ]
            snapshot = json.dumps(msgs)
            updated, count = inject_edit_failure_hints(msgs)
            self.assertEqual(count, 1)
            self.assertEqual(json.dumps(msgs), snapshot)  # pure: input untouched
            self.assertIn('[HARNESS_EDIT_HINT:', updated[1]['content'])
            # Idempotent on re-run.
            again, count2 = inject_edit_failure_hints(updated)
            self.assertEqual(count2, 0)
            self.assertIs(again, updated)

        # Trailing tool_result block in user message is also handled idempotently.
        block_msgs = [
            {
                'role': 'user',
                'content': [
                    {'type': 'tool_result', 'tool_use_id': 'e2', 'content': 'oldString not found'},
                ],
            }
        ]
        updated_blocks, bcount = inject_edit_failure_hints(block_msgs)
        self.assertEqual(bcount, 1)
        self.assertIn('[HARNESS_EDIT_HINT:', updated_blocks[0]['content'][0]['content'])
        again_blocks, bcount2 = inject_edit_failure_hints(updated_blocks)
        self.assertEqual(bcount2, 0)
        self.assertIs(again_blocks, updated_blocks)

    def test_inject_read_guard_hints_and_idempotency(self):
        # Empty / non-list inputs pass through cleanly.
        self.assertEqual(inject_read_guard_hints([]), ([], 0))

        # a) Permission-denied read hint on a trailing tool message (pure, idempotent).
        msgs = [
            {'role': 'user', 'content': 'read it'},
            {'role': 'tool', 'tool_call_id': 'r1', 'content': 'Permission denied: read'},
        ]
        snapshot = json.dumps(msgs)
        updated, _count = inject_read_guard_hints(msgs)
        self.assertEqual(json.dumps(msgs), snapshot)  # pure: input untouched
        self.assertIn('[HARNESS_READ_HINT:', updated[1]['content'])
        again, count2 = inject_read_guard_hints(updated)
        self.assertEqual(count2, 0)
        self.assertEqual(again, updated)

        # Non-trailing permission error is ignored (assistant turn in between).
        non_trailing = [
            {'role': 'tool', 'tool_call_id': 'r0', 'content': 'Permission denied: read'},
            {'role': 'assistant', 'content': 'retrying'},
            {'role': 'tool', 'tool_call_id': 'r1', 'content': 'ok'},
        ]
        out, count = inject_read_guard_hints(non_trailing)
        self.assertEqual(count, 0)
        self.assertEqual(out, non_trailing)

        # a) tool_result block in the last user message is also handled.
        block_msgs = [
            {
                'role': 'user',
                'content': [
                    {'type': 'tool_result', 'tool_use_id': 'r2',
                     'content': 'Permission denied: read'},
                ],
            }
        ]
        updated_blocks, _bcount = inject_read_guard_hints(block_msgs)
        self.assertIn('[HARNESS_READ_HINT:', updated_blocks[0]['content'][0]['content'])
        again_blocks, bcount2 = inject_read_guard_hints(updated_blocks)
        self.assertEqual(bcount2, 0)
        self.assertEqual(again_blocks, updated_blocks)

        def _read_call(rid, path):
            return {'role': 'assistant', 'tool_calls': [
                {'id': rid, 'function': {'name': 'read',
                                         'arguments': json.dumps({'path': path})}}]}

        # b) >=3 consecutive reads -> budget warning on the last tool message.
        budget_msgs = [
            {'role': 'user', 'content': 'read files'},
            _read_call('r1', 'a.py'),
            {'role': 'tool', 'tool_call_id': 'r1', 'content': 'file a'},
            _read_call('r2', 'b.py'),
            {'role': 'tool', 'tool_call_id': 'r2', 'content': 'file b'},
            _read_call('r3', 'c.py'),
            {'role': 'tool', 'tool_call_id': 'r3', 'content': 'file c'},
        ]
        updated, _count = inject_read_guard_hints(budget_msgs)
        self.assertIn('[HARNESS_READ_BUDGET_WARNING:', updated[-1]['content'])
        again, count2 = inject_read_guard_hints(updated)
        self.assertEqual(count2, 0)
        self.assertEqual(again, updated)

        # b) Two consecutive reads are below the threshold.
        two_msgs = [
            {'role': 'user', 'content': 'read files'},
            _read_call('r1', 'a.py'),
            {'role': 'tool', 'tool_call_id': 'r1', 'content': 'file a'},
            _read_call('r2', 'b.py'),
            {'role': 'tool', 'tool_call_id': 'r2', 'content': 'file b'},
        ]
        out, count = inject_read_guard_hints(two_msgs)
        self.assertEqual(count, 0)
        self.assertEqual(out, two_msgs)

        # b) An edit resets the streak; a later >=3 read streak re-triggers
        #    on the fresh trailing tool message.
        reset_msgs = [
            _read_call('r1', 'a.py'),
            {'role': 'tool', 'tool_call_id': 'r1', 'content': 'file a'},
            _read_call('r2', 'b.py'),
            {'role': 'tool', 'tool_call_id': 'r2', 'content': 'file b'},
            {'role': 'assistant', 'tool_calls': [
                {'id': 'e1', 'function': {'name': 'edit',
                                          'arguments': json.dumps({'path': 'a.py'})}}]},
            {'role': 'tool', 'tool_call_id': 'e1', 'content': 'edited'},
            _read_call('r3', 'c.py'),
            {'role': 'tool', 'tool_call_id': 'r3', 'content': 'file c'},
        ]
        out, count = inject_read_guard_hints(reset_msgs)
        self.assertEqual(count, 0)
        self.assertEqual(out, reset_msgs)
        reset_msgs.append(_read_call('r4', 'd.py'))
        reset_msgs.append({'role': 'tool', 'tool_call_id': 'r4', 'content': 'file d'})
        reset_msgs.append(_read_call('r5', 'e.py'))
        reset_msgs.append({'role': 'tool', 'tool_call_id': 'r5', 'content': 'file e'})
        updated, _count = inject_read_guard_hints(reset_msgs)
        self.assertIn('[HARNESS_READ_BUDGET_WARNING:', updated[-1]['content'])

        # b) `write` and `apply_patch` also reset the streak.
        for reset_tool in ('write', 'apply_patch'):
            reset_case = [
                _read_call('r1', 'a.py'),
                {'role': 'tool', 'tool_call_id': 'r1', 'content': 'file a'},
                _read_call('r2', 'b.py'),
                {'role': 'tool', 'tool_call_id': 'r2', 'content': 'file b'},
                {'role': 'assistant', 'tool_calls': [
                    {'id': 'w1', 'function': {'name': reset_tool,
                                              'arguments': json.dumps({})}}]},
                {'role': 'tool', 'tool_call_id': 'w1', 'content': 'wrote'},
                _read_call('r3', 'c.py'),
                {'role': 'tool', 'tool_call_id': 'r3', 'content': 'file c'},
            ]
            out, count = inject_read_guard_hints(reset_case)
            self.assertEqual(count, 0)
            self.assertEqual(out, reset_case)

        # b) Anthropic-style tool_use blocks are counted too.
        anthropic_msgs = [
            {'role': 'user', 'content': 'read files'},
            {'role': 'assistant', 'content': [
                {'type': 'tool_use', 'id': 't1', 'name': 'read',
                 'input': {'path': 'a.py'}}]},
            {'role': 'user', 'content': [
                {'type': 'tool_result', 'tool_use_id': 't1', 'content': 'file a'}]},
            {'role': 'assistant', 'content': [
                {'type': 'tool_use', 'id': 't2', 'name': 'read',
                 'input': {'path': 'b.py'}}]},
            {'role': 'user', 'content': [
                {'type': 'tool_result', 'tool_use_id': 't2', 'content': 'file b'}]},
            {'role': 'assistant', 'content': [
                {'type': 'tool_use', 'id': 't3', 'name': 'read',
                 'input': {'path': 'c.py'}}]},
            {'role': 'user', 'content': [
                {'type': 'tool_result', 'tool_use_id': 't3', 'content': 'file c'}]},
        ]
        updated, _count = inject_read_guard_hints(anthropic_msgs)
        self.assertIn('[HARNESS_READ_BUDGET_WARNING:',
                      updated[-1]['content'][-1]['content'])
        again, count2 = inject_read_guard_hints(updated)
        self.assertEqual(count2, 0)
        self.assertEqual(again, updated)

    def test_proxy_injects_read_guard_hints(self):
        upstream, captured = _scripted_upstream([{'status': 200}])
        proxy = OpenRouterGuardProxy(lambda: None, upstream_url='http://127.0.0.1:%d' % upstream.server_port)
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
        msgs = [
            {'role': 'user', 'content': 'read files'},
            {'role': 'assistant', 'tool_calls': [
                {'id': 'r1', 'function': {'name': 'read',
                                          'arguments': json.dumps({'path': 'a.py'})}}]},
            {'role': 'tool', 'tool_call_id': 'r1', 'content': 'file a'},
            {'role': 'assistant', 'tool_calls': [
                {'id': 'r2', 'function': {'name': 'read',
                                          'arguments': json.dumps({'path': 'b.py'})}}]},
            {'role': 'tool', 'tool_call_id': 'r2', 'content': 'file b'},
            {'role': 'assistant', 'tool_calls': [
                {'id': 'r3', 'function': {'name': 'read',
                                          'arguments': json.dumps({'path': 'c.py'})}}]},
            {'role': 'tool', 'tool_call_id': 'r3', 'content': 'file c'},
        ]
        body = json.dumps({'model': GLM, 'messages': msgs}).encode()
        try:
            _post_chat(proxy.base_url, body)
        finally:
            proxy.shutdown(); proxy.server_close()
            upstream.shutdown(); upstream.server_close()
        self.assertEqual(len(captured), 1)
        forwarded = captured[0]['messages'][-1]['content']
        self.assertIn('[HARNESS_READ_BUDGET_WARNING:', forwarded)
        summary = proxy.summary()
        read_spans = [s for s in summary['spans']
                      if s.get('name') == 'guard.read_guard_hint']
        self.assertEqual(len(read_spans), 1)
        span = read_spans[0]
        kind = self._span_kind(span)
        if kind is not None:
            self.assertEqual(kind, 'GUARDRAIL')
        self.assertTrue(span.get('attributes', {}).get('guardrail.triggered'))
        self.assertEqual(
            span.get('attributes', {}).get('harness.read_hints_injected'), 1)

    def _span_kind(self, span):
        return (span.get('kind')
                or span.get('attributes', {}).get('openinference.span.kind'))

    def test_proxy_injects_read_permission_hint_span(self):
        upstream, captured = _scripted_upstream([{'status': 200}])
        proxy = OpenRouterGuardProxy(lambda: None, upstream_url='http://127.0.0.1:%d' % upstream.server_port)
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
        msgs = [
            {'role': 'user', 'content': 'read it'},
            {'role': 'assistant', 'tool_calls': [
                {'id': 'r1', 'function': {'name': 'read',
                                          'arguments': json.dumps({'path': 'a.py'})}}]},
            {'role': 'tool', 'tool_call_id': 'r1', 'content': 'Permission denied: read'},
        ]
        body = json.dumps({'model': GLM, 'messages': msgs}).encode()
        try:
            _post_chat(proxy.base_url, body)
        finally:
            proxy.shutdown(); proxy.server_close()
            upstream.shutdown(); upstream.server_close()
        self.assertEqual(len(captured), 1)
        forwarded = captured[0]['messages'][2]['content']
        self.assertIn('[HARNESS_READ_HINT:', forwarded)
        summary = proxy.summary()
        read_spans = [s for s in summary['spans']
                      if s.get('name') == 'guard.read_guard_hint']
        self.assertEqual(len(read_spans), 1)
        span = read_spans[0]
        kind = self._span_kind(span)
        if kind is not None:
            self.assertEqual(kind, 'GUARDRAIL')
        self.assertTrue(span.get('attributes', {}).get('guardrail.triggered'))
        self.assertEqual(
            span.get('attributes', {}).get('harness.read_hints_injected'), 1)

    def test_proxy_injects_edit_failure_hints(self):
        upstream, captured = _scripted_upstream([{'status': 200}])
        proxy = OpenRouterGuardProxy(lambda: None, upstream_url='http://127.0.0.1:%d' % upstream.server_port)
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
        msgs = [
            {'role': 'user', 'content': 'fix function'},
            {'role': 'assistant', 'tool_calls': [
                {'id': 'e1', 'function': {'name': 'edit', 'arguments': json.dumps({'path': 'a.py'})}}]},
            {'role': 'tool', 'tool_call_id': 'e1', 'content': 'Error: oldString not found in file'},
        ]
        body = json.dumps({'model': GLM, 'messages': msgs}).encode()
        try:
            _post_chat(proxy.base_url, body)
        finally:
            proxy.shutdown(); proxy.server_close()
            upstream.shutdown(); upstream.server_close()
        self.assertEqual(len(captured), 1)
        forwarded = captured[0]['messages'][2]['content']
        self.assertIn('[HARNESS_EDIT_HINT:', forwarded)
        summary = proxy.summary()
        self.assertEqual(summary['edit_hints_injected'], 1)
        self.assertTrue(any(s.get('name') == 'guard.edit_failure_hint' for s in summary['spans']))


        self.assertTrue(any(s.get('name') == 'guard.edit_failure_hint' for s in summary['spans']))

    def test_validate_task_symlink_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            real = os.path.join(tmp, 'real')
            os.makedirs(real)
            with open(os.path.join(real, 'a.py'), 'w', encoding='utf-8') as fh:
                fh.write('x = 1\n')
            with open(os.path.join(real, 'ro.py'), 'w', encoding='utf-8') as fh:
                fh.write('y = 2\n')
            link = os.path.join(tmp, 'link')
            os.symlink(real, link)
            t = validate_task({
                'task_id': 'symlink-alias',
                'directory': link,
                'files': [os.path.join(link, 'a.py')],
                'read_only_files': [os.path.join(link, 'ro.py')],
                'prompt': 'edit ' + os.path.join(link, 'a.py') + ' under ' + link,
                'test_argv': [sys.executable, '-c', 'raise SystemExit(0)'],
            })
            real_dir = os.path.realpath(link)
            self.assertEqual(t['directory'], real_dir)
            self.assertEqual(t['files'], [os.path.join(real_dir, 'a.py')])
            self.assertEqual(t['read_only_files'], [os.path.join(real_dir, 'ro.py')])
            self.assertEqual(t['prompt'], 'edit ' + os.path.join(real_dir, 'a.py') + ' under ' + real_dir)
            self.assertIn(os.path.join(link, 'a.py'), t['file_aliases'])
            self.assertIn(os.path.join(link, 'ro.py'), t['read_only_file_aliases'])
            cfg = config_for(t)
            perms = cfg['agents']['guarded_worker']['permissions']
            resources = [p['resource'] for p in perms if p['effect'] == 'allow']
            self.assertIn(os.path.join(real_dir, 'a.py'), resources)
            self.assertIn(os.path.join(link, 'a.py'), resources)
            self.assertIn(os.path.join(link, 'ro.py'), resources)

    def test_compute_repair_max_steps(self):
        self.assertEqual(compute_repair_max_steps({'files': ['a.py']}), 6)
        self.assertEqual(compute_repair_max_steps({'files': ['a.py', 'b.py']}), 8)
        self.assertEqual(compute_repair_max_steps({'files': ['a.py', 'b.py', 'c.py']}), 10)
        self.assertEqual(compute_repair_max_steps({'files': []}), 6)

    def test_micro_repair_adaptive_steps(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            cli_path = TestSandboxExecutionAndFinopsSettlement._write_stub_cli(self, tmp)
            db_path = os.path.join(tmp, 'opencode.db')
            TestSandboxExecutionAndFinopsSettlement._init_db(self, db_path)
            state_dir = os.path.join(tmp, 'state')
            repo = os.path.join(tmp, 'repo')
            target = TestSandboxExecutionAndFinopsSettlement._init_git_repo(self, repo, 'target.py', 'VALUE = 0\n')
            second = os.path.join(repo, 'b.py')
            with open(second, 'w', encoding='utf-8') as fh:
                fh.write('b = 1\n')
            test_code = (
                "import pathlib, sys; "
                "txt = pathlib.Path('target.py').read_text(); "
                "print('E   AssertionError: expected VALUE = 2') if 'VALUE = 2' not in txt else None; "
                "sys.exit(0 if 'VALUE = 2' in txt else 1)"
            )
            task_path = os.path.join(tmp, 'task.json')
            with open(task_path, 'w', encoding='utf-8') as fh:
                json.dump({
                    'task_id': 'repair-adaptive',
                    'directory': repo,
                    'files': [target, second],
                    'prompt': 'Set VALUE = 2 in ' + target,
                    'test_argv': [sys.executable, '-c', test_code],
                }, fh)
            seen_steps = []

            def fake_model_run(cli, db, task, run_dir, result, budget, **kwargs):
                seen_steps.append(task['max_steps'])
                content = 'VALUE = 2\n' if '[TEST_ARGV FAILURE' in task['prompt'] else 'VALUE = 1\n'
                with open(target, 'w', encoding='utf-8') as fh:
                    fh.write(content)
                result['steps'] = 1
                result['cost'] = 0.01
                result['observed_cost'] = 0.01

            buf = io.StringIO()
            with mock.patch.dict(os.environ, {'HARNESS_TEST_DB': db_path}):
                with mock.patch('harness.sandbox.model_run', side_effect=fake_model_run):
                    with contextlib.redirect_stdout(buf):
                        rc = run_task(task_path, cli=cli_path,
                                      db=db_path, state=state_dir, check=False)
            self.assertEqual(rc, 0)
            self.assertEqual(len(seen_steps), 2)
            self.assertEqual(seen_steps[-1], 8)
            self.assertNotEqual(seen_steps[0], 8)
            with open(target, 'r', encoding='utf-8') as fh:
                self.assertEqual(fh.read(), 'VALUE = 2\n')

    def test_inject_edit_failure_hints_opencode_errors(self):
        errors = [
            'Could not find oldString in /path/file.py',
            'Found 2 matches for oldString, but expected exactly one.',
            'No changes to apply: oldString and newString are identical.',
            'The tool input could not be parsed as JSON, so the call was not executed.',
            'Invalid arguments for tool "edit": missing oldString',
        ]
        for text in errors:
            messages = [{'role': 'tool', 'content': text}]
            updated, count = inject_edit_failure_hints(messages)
            self.assertEqual(count, 1)
            final_text = updated[-1]['content']
            self.assertIn('[HARNESS_EDIT_HINT:', final_text)
            if ('could not be parsed as JSON' in text
                    or 'Invalid arguments for tool' in text):
                self.assertIn('Keep oldString/newString small', final_text)
            elif 'Found 2 matches for oldString' in text:
                self.assertIn('Multiple matches found', final_text)
            elif 'oldString and newString are identical' in text:
                self.assertIn('Target change is already present on disk', final_text)
            else:
                self.assertIn('Exact target text did not match file contents', final_text)

    def test_routing_boosts_max_tokens_on_json_truncation(self):
        body = json.dumps({
            'model': 'openrouter/z-ai/glm-5.3-flash',
            'max_tokens': 8192,
            'messages': [
                {'role': 'user', 'content': 'hi'},
                {'role': 'tool',
                 'content': 'The tool input could not be parsed as JSON, '
                            'so the call was not executed.'},
            ],
        }).encode('utf-8')
        out = json.loads(apply_provider_routing(
            body, '/v1/chat/completions', turn_index=3))
        self.assertEqual(out['max_tokens'], ERROR_RECOVERY_MAX_TOKENS_CAP)

    def test_h26_build_preloaded_file_excerpts_short_and_ranges(self):
        from harness import build_preloaded_file_excerpts as _build

        def _excerpts(prompt, files, **kw):
            # Be tolerant about positional order between the prompt and the
            # file list across harness versions.
            for order in ((prompt, files), (files, prompt)):
                try:
                    result = _build(*order, **kw)
                except TypeError:
                    continue
                if isinstance(result, str):
                    return result
            return ''

        with tempfile.TemporaryDirectory() as tmp:
            short = os.path.join(tmp, 'short.py')
            with open(short, 'w', encoding='utf-8') as fh:
                fh.write('\n'.join('line %d body' % i for i in range(1, 21)))
            long_path = os.path.join(tmp, 'long.py')
            with open(long_path, 'w', encoding='utf-8') as fh:
                fh.write('\n'.join('row %d body' % i for i in range(1, 201)))
            # Short file (<=150 lines): full inclusion with line-number prefix.
            out = _excerpts('please fix line 3', [short])
            self.assertIn('1: ', out)
            self.assertIn('line 3 body', out)
            self.assertIn('20: ', out)
            # Long file with an explicit range in three notations (+-5 context).
            for marker in ('lines 80-90', 'L80-L90', 'строки 80-90'):
                out = _excerpts('see %s for details' % marker, [long_path])
                self.assertIn('75: ', out, marker)
                self.assertIn('row 80 body', out)
                self.assertIn('row 90 body', out)
                self.assertIn('95: ', out, marker)
            # max_total_chars caps the result (the harness may add a small
            # per-file header on top of the budget, so compare sized outputs).
            unbounded = _excerpts('lines 80-90', [long_path])
            capped = _excerpts('lines 80-90', [long_path], max_total_chars=50)
            self.assertTrue(capped)
            self.assertLess(len(capped), len(unbounded))
            # No relevant match -> empty string. (`short` is <=150 lines so it
            # is always preloaded; only the long file can yield nothing.)
            self.assertEqual(
                _excerpts('totally unrelated xyzzy query', [long_path]), '')

    def test_h26_model_run_preloaded_excerpts_and_read_only_files(self):
        import sys
        from pathlib import Path
        from unittest import mock
        from harness.sandbox import model_run

        with tempfile.TemporaryDirectory() as tmp:
            argv_dump = os.path.join(tmp, 'argv.txt')
            stub = os.path.join(tmp, 'stub_cli.py')
            with open(stub, 'w', encoding='utf-8') as fh:
                fh.write(
                    '#!/usr/bin/env python3\n'
                    'import sys\n'
                    'with open(%r, "w", encoding="utf-8") as out:\n'
                    '    out.write(sys.argv[-1] if len(sys.argv) > 1 else "")\n'
                    % argv_dump)
            os.chmod(stub, 0o755)
            target_file = os.path.join(tmp, 'target.py')
            with open(target_file, 'w', encoding='utf-8') as fh:
                fh.write('\n'.join('line %d body' % i for i in range(1, 201)))
            readonly = os.path.join(tmp, 'ref.py')
            with open(readonly, 'w', encoding='utf-8') as fh:
                fh.write('\n'.join('ref_%d = %d' % (i, i) for i in range(1, 41)))
            run_dir = os.path.join(tmp, 'state', 'runs', 'r1')
            os.makedirs(run_dir)
            result = {'session_id': 'ses_test_h26', 'cost': None}
            budget = 0.15
            task = {
                'task_id': 'h26',
                'mode': 'model',
                'model': 'openrouter/z-ai/glm-5.3-flash',
                'files': [target_file],
                'read_only_files': [readonly],
                'test_argv': ['/usr/bin/true'],
                'timeout_seconds': 30,
                'max_steps': 8,
                'prompt': 'Implement the change.',
            }
            with mock.patch('harness.sandbox.snapshot',
                            return_value={'id': 'ses_test_h26'}), \
                    mock.patch('harness.sandbox.metrics',
                               return_value={'steps': 1, 'cost': 0.001,
                                             'unfinished': False,
                                             'model_error': False}), \
                    mock.patch('harness.sandbox.violation',
                               lambda *a, **k: None), \
                    mock.patch('harness.sandbox.capture_requests',
                               lambda *a, **k: None), \
                    mock.patch('harness.sandbox.capture_files',
                               lambda *a, **k: None):
                model_run(Path(stub), Path(tmp) / 'opencode.db', task,
                          Path(run_dir), result, budget)
            with open(argv_dump, encoding='utf-8') as fh:
                prompt = fh.read()
            self.assertIn('Read-only reference files (do NOT edit):', prompt)
            self.assertIn('[PRELOADED_FILE_EXCERPTS', prompt)
            # A task that already carries a current patch skips preloading.
            task['prompt'] = 'Implement the change.\n[CURRENT_PATCH]\n--- a\n+++ b\n'
            with mock.patch('harness.sandbox.snapshot',
                            return_value={'id': 'ses_test_h26'}), \
                    mock.patch('harness.sandbox.metrics',
                               return_value={'steps': 1, 'cost': 0.001,
                                             'unfinished': False,
                                             'model_error': False}), \
                    mock.patch('harness.sandbox.violation',
                               lambda *a, **k: None), \
                    mock.patch('harness.sandbox.capture_requests',
                               lambda *a, **k: None), \
                    mock.patch('harness.sandbox.capture_files',
                               lambda *a, **k: None):
                model_run(Path(stub), Path(tmp) / 'opencode.db', task,
                          Path(run_dir), result, budget)
            with open(argv_dump, encoding='utf-8') as fh:
                prompt_with_patch = fh.read()
            self.assertIn('[CURRENT_PATCH]', prompt_with_patch)
            self.assertNotIn('[PRELOADED_FILE_EXCERPTS', prompt_with_patch)

    def test_h27_build_prior_attempt_feedback_and_run_task_injection(self):
        import time
        from unittest import mock
        from harness.sandbox import build_prior_attempt_feedback, run_task

        def _feedback(fn, candidates):
            for args in candidates:
                try:
                    result = fn(*args)
                except TypeError:
                    continue
                if result:
                    return result, args
            return '', candidates[0]

        with tempfile.TemporaryDirectory() as tmp:
            state = os.path.join(tmp, 'state')
            prev = os.path.join(state, 'runs', 'run_prev')
            os.makedirs(prev)
            with open(os.path.join(prev, 'result.json'), 'w', encoding='utf-8') as fh:
                json.dump({'status': 'blocked', 'reason': 'tests_failed',
                           'model': 'm1', 'task_id': 't1'}, fh)
            with open(os.path.join(prev, 'test.log'), 'w', encoding='utf-8') as fh:
                fh.write('FAILED tests/test_x.py::test_y\nE   AssertionError: boom\n')
            with open(os.path.join(prev, 'changes.patch'), 'w', encoding='utf-8') as fh:
                fh.write('--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n')
            feedback, winning = _feedback(
                build_prior_attempt_feedback,
                [(state, 't1'), ('t1', state), (prev,), (state,)])
            self.assertTrue(feedback)
            self.assertIn('[PRIOR_ATTEMPT_FEEDBACK', feedback)
            self.assertIn('tests_failed', feedback)
            self.assertIn('m1', feedback)
            self.assertIn('boom', feedback)
            self.assertIn('--- a/x.py', feedback)
            # A newer successful attempt suppresses the feedback.
            newer = os.path.join(state, 'runs', 'run_new')
            os.makedirs(newer)
            with open(os.path.join(newer, 'result.json'), 'w', encoding='utf-8') as fh:
                json.dump({'status': 'succeeded', 'task_id': 't1',
                           'model': 'm1'}, fh)
            os.utime(os.path.join(newer, 'result.json'),
                     (time.time() + 10, time.time() + 10))
            self.assertEqual(build_prior_attempt_feedback(*winning), '')

            # run_task must inject the feedback into task['prompt'] and log it.
            state2 = os.path.join(tmp, 'state2')
            blocked2 = os.path.join(state2, 'runs', 'run_prev2')
            os.makedirs(blocked2)
            with open(os.path.join(blocked2, 'result.json'), 'w', encoding='utf-8') as fh:
                json.dump({'status': 'blocked', 'reason': 'tests_failed',
                           'model': 'm1', 'task_id': 't2', 'cost': 0.01}, fh)
            with open(os.path.join(blocked2, 'test.log'), 'w', encoding='utf-8') as fh:
                fh.write('E   AssertionError: boom2\n')
            from pathlib import Path
            from unittest import mock

            captured = {}

            def _fake_model_run(*args, **kwargs):
                for value in list(args) + list(kwargs.values()):
                    if isinstance(value, dict) and 'prompt' in value:
                        captured['task'] = value
                return {'status': 'succeeded', 'steps': 1}

            target_file = os.path.join(tmp, 'target2.py')
            with open(target_file, 'w', encoding='utf-8') as fh:
                fh.write('x = 1\n')
            task = {
                'task_id': 't2',
                'directory': tmp,
                'files': [target_file],
                'test_argv': ['/usr/bin/true'],
                'prompt': 'Implement the change.',
                'worktree': False,
            }
            task_path = os.path.join(tmp, 'task.json')
            with open(task_path, 'w', encoding='utf-8') as fh:
                json.dump(task, fh)
            with mock.patch('harness.sandbox.model_run', _fake_model_run), \
                    mock.patch('harness.sandbox.preflight',
                               lambda *a, **k: None), \
                    mock.patch('harness.sandbox.verify_db',
                               lambda *a, **k: None), \
                    mock.patch('harness.sandbox.snapshot', return_value=None), \
                    mock.patch('harness.sandbox.execute', return_value=0), \
                    mock.patch('harness.sandbox.file_hashes',
                               side_effect=lambda files: {'f': 'h2' if captured else 'h1'}):
                run_task(task_path, cli=Path('/usr/bin/true'),
                         db=Path(tmp) / 'opencode.db', state=Path(state2))
            injected = captured.get('task', {}).get('prompt', '')
            self.assertIn('[PRIOR_ATTEMPT_FEEDBACK', injected)
            events = []
            for root, _dirs, names in os.walk(state2):
                if 'events.jsonl' in names:
                    with open(os.path.join(root, 'events.jsonl'), encoding='utf-8') as fh:
                        events.append(fh.read())
            self.assertTrue(any('prior_attempt_feedback_injected' in chunk
                                for chunk in events))

    def test_h28_compute_state_insights_and_cli_stats_insights(self):
        from harness import compute_state_insights, format_state_insights_markdown
        from harness.cli import main as cli_main

        with tempfile.TemporaryDirectory() as tmp:
            runs = os.path.join(tmp, 'runs')
            layouts = {
                'first': {'task_id': 'first', 'status': 'succeeded',
                          'model': 'm1', 'steps': 2},
                'repair': {'task_id': 'repair', 'status': 'succeeded',
                           'model': 'm1', 'steps': 4, 'micro_repair_used': True},
                'salvaged': {'task_id': 'salvaged',
                             'status': 'validation_passed_review_required',
                             'model': 'm1', 'steps': 8, 'salvaged': True},
                'escalated': {'task_id': 'escalated', 'status': 'succeeded',
                              'model': 'm2', 'steps': 5,
                              'escalated_from': 'm1'},
                'blocked': {'task_id': 'blocked', 'status': 'blocked',
                            'reason': 'tests_failed', 'model': 'm1'},
                'cached': {'task_id': 'cached',
                           'status': 'configuration_verified_no_model_call',
                           'model': 'm1'},
            }
            for name, payload in layouts.items():
                run_dir = os.path.join(runs, name)
                os.makedirs(run_dir)
                with open(os.path.join(run_dir, 'result.json'), 'w',
                          encoding='utf-8') as fh:
                    json.dump(payload, fh)
                if name == 'blocked':
                    spans = [
                        {'name': 'tool.read', 'span_id': 'a' * 16,
                         'trace_id': 'b' * 32, 'parent_span_id': None,
                         'start_time_ns': 1, 'end_time_ns': 2,
                         'status': 'ERROR',
                         'attributes': {'openinference.span.kind': 'TOOL',
                                        'tool.name': 'read'}},
                        {'name': 'tool.edit', 'span_id': 'c' * 16,
                         'trace_id': 'b' * 32, 'parent_span_id': 'a' * 16,
                         'start_time_ns': 3, 'end_time_ns': 4, 'status': 'OK',
                         'attributes': {'openinference.span.kind': 'TOOL',
                                        'tool.name': 'edit'}},
                    ]
                    with open(os.path.join(run_dir, 'traces.jsonl'), 'w',
                              encoding='utf-8') as fh:
                        for span in spans:
                            fh.write(json.dumps(span) + '\n')

            insights = compute_state_insights(tmp)
            if isinstance(insights, str):
                text = insights
            else:
                try:
                    text = json.dumps(insights, ensure_ascii=False)
                except TypeError:
                    text = str(insights)
            try:
                markdown = format_state_insights_markdown(tmp)
            except Exception:
                markdown = format_state_insights_markdown(insights)
            blob = text + '\n' + markdown
            self.assertIn('tests_failed', blob)
            self.assertIn('blocked', blob)
            self.assertIn('read', blob)
            self.assertTrue('first_try' in blob or 'first-try' in blob)
            self.assertTrue('micro_repair' in blob or 'micro-repair' in blob)

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(
                    cli_main(['stats', '--insights', '--state-dir', tmp]), 0)
            cli_out = buf.getvalue()
            self.assertTrue(cli_out)
            self.assertIn('tests_failed', cli_out)
            self.assertIn('Average reads before first edit', cli_out)


class TestH35RepoMemoryHint(HarnessTestCase):
    def _make_run(self, state, name, result, spans=(), task=None):
        run_dir = os.path.join(state, 'runs', name)
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, 'result.json'), 'w') as fh:
            json.dump(result, fh)
        if spans:
            with open(os.path.join(run_dir, 'traces.jsonl'), 'w') as fh:
                for span in spans:
                    fh.write(json.dumps(span) + '\n')
        if task is not None:
            with open(os.path.join(run_dir, 'task.json'), 'w') as fh:
                json.dump(task, fh)
        return run_dir

    @staticmethod
    def _error_span(tool, message, start_ns):
        return {'name': tool, 'span_id': 'd' * 16, 'trace_id': 'e' * 32,
                'parent_span_id': None, 'start_time_ns': start_ns,
                'end_time_ns': start_ns + 100, 'status': 'ERROR',
                'attributes': {'openinference.span.kind': 'TOOL',
                               'tool.name': tool, 'output.value': message}}

    def test_hint_empty_below_min_occurrences(self):
        from harness.report import build_repo_memory_hint
        state = self.path('state-empty')
        self._make_run(state, 'r1', {'task_id': 'r1', 'status': 'blocked', 'reason': 'task_budget_exhausted'})
        self._make_run(state, 'r2', {'task_id': 'r2', 'status': 'succeeded'})
        self.assertEqual(build_repo_memory_hint(state), '')
        self.assertEqual(build_repo_memory_hint(self.path('nonexistent'), min_occurrences=1), '')

    def test_hint_formats_recurring_block_and_tool_error(self):
        from harness.report import build_repo_memory_hint
        state = self.path('state-hint')
        self._make_run(state, 'r1', {'task_id': 'r1', 'status': 'blocked', 'reason': 'task_budget_exhausted'},
                       spans=[self._error_span('edit_file', 'invalid_edit\nmore', 1000)])
        self._make_run(state, 'r2', {'task_id': 'r2', 'status': 'blocked', 'reason': 'task_budget_exhausted'},
                       spans=[self._error_span('edit_file', 'invalid_edit\nmore', 2000)])
        hint = build_repo_memory_hint(state)
        self.assertIn('[REPO MEMORY HINT', hint)
        self.assertIn('- Recurring block reason: task_budget_exhausted (2x)', hint)
        self.assertIn('- Recurring tool error in edit_file: invalid_edit (2x).', hint)
        self.assertIn('avoid repeating this failure mode', hint)
        # Directory filtering: no prior task.json matches this directory -> falls back to all runs.
        hint_dir = build_repo_memory_hint(state, self.path('nowhere'), min_occurrences=1)
        self.assertIn('[REPO MEMORY HINT', hint_dir)
        self.assertLessEqual(len(build_repo_memory_hint(state, max_chars=50)), 50)

    def test_run_task_injects_repo_memory_hint_and_logs_event(self):
        from harness.report import build_repo_memory_hint  # noqa: F401 (import parity)
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            cli_path = TestSandboxExecutionAndFinopsSettlement._write_stub_cli(self, tmp)
            db_path = os.path.join(tmp, 'opencode.db')
            TestSandboxExecutionAndFinopsSettlement._init_db(self, db_path)
            state_dir = os.path.join(tmp, 'state')
            repo = os.path.join(tmp, 'repo')
            target = TestSandboxExecutionAndFinopsSettlement._init_git_repo(self, repo, 'target.py', 'VALUE = 0\n')
            # Two prior blocked runs for the same directory -> recurring block reason.
            for name in ('prior-1', 'prior-2'):
                self._make_run(state_dir, name,
                               {'task_id': name, 'status': 'blocked', 'reason': 'task_budget_exhausted'},
                               task={'task_id': name, 'directory': os.path.realpath(repo)})
            task_path = os.path.join(tmp, 'task.json')
            with open(task_path, 'w', encoding='utf-8') as fh:
                json.dump({
                    'task_id': 'repo-hint-1',
                    'directory': repo,
                    'files': [target],
                    'prompt': 'Set VALUE = 1 in ' + target,
                    'test_argv': [sys.executable, '-c', 'raise SystemExit(0)'],
                    'worktree': True,
                    'include_repo_memory': True,
                }, fh)
            prompt_dump = os.path.join(tmp, 'prompt.txt')
            buf = io.StringIO()
            with mock.patch.dict(os.environ, {
                'HARNESS_TEST_DB': db_path,
                'STUB_CLI_INITIAL_CONTENT': 'VALUE = 1\n',
                'STUB_CLI_PROMPT_DUMP': prompt_dump,
            }):
                with contextlib.redirect_stdout(buf):
                    rc = run_task(task_path, cli=cli_path, db=db_path, state=state_dir, check=False)
            self.assertEqual(rc, 0)
            with open(prompt_dump, 'r', encoding='utf-8') as fh:
                prompt = fh.read()
            self.assertIn('[REPO MEMORY HINT', prompt)
            self.assertIn('Recurring block reason: task_budget_exhausted', prompt)
            injected = False
            for run_name in sorted(os.listdir(os.path.join(state_dir, 'runs'))):
                run_dir = os.path.join(state_dir, 'runs', run_name)
                for fname in os.listdir(run_dir):
                    try:
                        with open(os.path.join(run_dir, fname), 'r', encoding='utf-8', errors='ignore') as fh:
                            if 'repo_memory_hint_injected' in fh.read():
                                injected = True
                    except OSError:
                        continue
            self.assertTrue(injected)


class TestH31SwePruner(unittest.TestCase):
    def _pytest_output(self, chars=6000):
        filler = 'assertions/details line\n'
        lines = ['=== FAILURES ===', '_ test_broken _', 'Traceback (most recent call last):',
                 '  File "test_x.py", line 10, in test_broken', 'E   AssertionError: boom']
        body = '\n'.join(lines)
        while len(body) < chars:
            body += filler
        body += '\n=== short test summary info ===\nFAILED test_x.py::test_broken'
        return body

    def test_prune_compacts_bloated_pytest_output(self):
        original = {'role': 'tool', 'tool_call_id': 'c1', 'content': self._pytest_output()}
        messages = [{'role': 'user', 'content': 'run tests'}, original,
                    {'role': 'assistant', 'content': 'ok'}]
        new_messages, pruned = prune_noisy_tool_outputs(messages)
        self.assertEqual(pruned, 1)
        self.assertIsNot(new_messages[1], original)
        # input is never mutated
        self.assertGreater(len(original['content']), 4000)
        pruned_content = new_messages[1]['content']
        self.assertIn('[HARNESS SWE-PRUNER: noisy test output compacted]', pruned_content)
        self.assertLess(len(pruned_content), len(original['content']))
        self.assertLessEqual(len(pruned_content), 4000)
        self.assertIn('FAILED test_x.py::test_broken', pruned_content)
        self.assertIn('AssertionError: boom', pruned_content)
        # non-tool messages pass through untouched
        self.assertIs(new_messages[0], messages[0])
        self.assertIs(new_messages[2], messages[2])

    def test_prune_leaves_non_test_output_and_already_pruned_intact(self):
        big_dump = 'x' * 6000  # large but not test output
        already_pruned = 'FAILED test\n[HARNESS SWE-PRUNER: noisy test output compacted]'
        messages = [
            {'role': 'tool', 'tool_call_id': 'a', 'content': big_dump},
            {'role': 'tool', 'tool_call_id': 'b', 'content': already_pruned + 'y' * 6000},
            {'role': 'assistant', 'content': 'x' * 6000},
        ]
        snapshot = [m['content'] for m in messages]
        new_messages, pruned = prune_noisy_tool_outputs(messages)
        self.assertEqual(pruned, 0)
        self.assertEqual([m['content'] for m in new_messages], snapshot)
        self.assertIsNot(new_messages, messages)

    def test_prune_respects_custom_max_tool_chars_and_threshold(self):
        small = '=== FAILURES ===\nFAILED t\n' + 'z' * 100  # under threshold
        messages = [{'role': 'tool', 'tool_call_id': 'c', 'content': small},
                    {'role': 'tool', 'tool_call_id': 'd', 'content': self._pytest_output(800)}]
        new_messages, pruned = prune_noisy_tool_outputs(messages, max_tool_chars=400)
        self.assertEqual(pruned, 1)
        self.assertEqual(new_messages[0]['content'], small)
        self.assertIn('[HARNESS SWE-PRUNER:', new_messages[1]['content'])
        self.assertLessEqual(len(new_messages[1]['content']), 400)


class TestH29ValidateAndProxyWiring(HarnessTestCase):
    """H-29: TEACHER_MODELS accepted by validate_task and new proxy wiring in run_task."""

    def _task(self, root, **over):
        wd = os.path.join(root, 'wd')
        os.makedirs(wd, exist_ok=True)
        target = os.path.join(wd, 'a.py')
        with open(target, 'w', encoding='utf-8') as fh:
            fh.write('VALUE = 0\n')
        task = {
            'task_id': 'h29-validate',
            'directory': wd,
            'files': [target],
            'prompt': 'Set VALUE = 1 in ' + target,
            'test_argv': [sys.executable, '-c', 'raise SystemExit(0)'],
        }
        task.update(over)
        return task

    def test_validate_task_teacher_models_and_new_fields(self):
        from harness.routing import MODELS, TEACHER_MODELS
        teacher = sorted(TEACHER_MODELS)[0]
        with tempfile.TemporaryDirectory() as tmp:
            base = self._task(tmp)
            for model in sorted(MODELS | TEACHER_MODELS):
                validated = validate_task({**base, 'model': model})
                self.assertEqual(validated['model'], model)
            validated = validate_task(dict(base))
            self.assertEqual(validated['max_context_tokens'], 32000)
            self.assertIsNone(validated['fallback_teacher_model'])
            self.assertTrue(validated['enforce_egress_guard'])
            self.assertFalse(validated['enforce_readonly_sql'])
            self.assertEqual(validated['upstream_mode'], 'openrouter')
            validated = validate_task(dict(
                base, max_context_tokens=0, fallback_teacher_model=teacher,
                enforce_egress_guard=False, enforce_readonly_sql=True,
                upstream_mode='vllm'))
            self.assertIsNone(validated['max_context_tokens'])
            self.assertEqual(validated['fallback_teacher_model'], teacher)
            self.assertFalse(validated['enforce_egress_guard'])
            self.assertTrue(validated['enforce_readonly_sql'])
            self.assertEqual(validated['upstream_mode'], 'vllm')
            for bad in ({'model': 'openrouter/not-a-model'},
                        {'max_context_tokens': -1},
                        {'max_context_tokens': 'big'},
                        {'fallback_teacher_model': 'nope'},
                        {'enforce_egress_guard': 'yes'},
                        {'enforce_readonly_sql': 1},
                        {'upstream_mode': 'nope'}):
                with self.assertRaises(GuardError):
                    validate_task(dict(base, **bad))

    def test_run_task_wires_health_store_and_proxy_options(self):
        from unittest import mock
        import harness.sandbox as sandbox_mod
        from harness.routing import ProviderHealthStore
        with tempfile.TemporaryDirectory() as tmp:
            base = self._task(tmp, enforce_egress_guard=False, enforce_readonly_sql=True,
                              upstream_mode='vllm')
            task_path = os.path.join(tmp, 'task.json')
            with open(task_path, 'w', encoding='utf-8') as fh:
                json.dump(base, fh)
            cli_path = os.path.join(tmp, 'cli.py')
            with open(cli_path, 'w', encoding='utf-8') as fh:
                fh.write('print("ok")\n')
            state_dir = os.path.join(tmp, 'state')
            captured = {}

            def fake_proxy(*args, **kwargs):
                captured.update(kwargs)
                stub = mock.MagicMock()
                stub.base_url = None
                stub.__enter__.return_value = stub
                return stub

            with mock.patch.object(sandbox_mod, 'openrouter_proxy', fake_proxy), \
                    mock.patch.object(sandbox_mod, 'model_run'), \
                    mock.patch.object(sandbox_mod, 'verify_db'), \
                    mock.patch.object(sandbox_mod, 'snapshot', return_value=None):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf), contextlib.suppress(Exception):
                    run_task(task_path, cli=cli_path, db=os.path.join(tmp, 'opencode.db'),
                             state=state_dir, check=False, upstream_url='http://127.0.0.1:9')
            self.assertIn('health_store', captured)
            self.assertIsInstance(captured['health_store'], ProviderHealthStore)
            self.assertEqual(captured['max_context_tokens'], 32000)
            self.assertIsNone(captured['fallback_teacher_model'])
            self.assertFalse(captured['enforce_egress_guard'])
            self.assertTrue(captured['enforce_readonly_sql'])
            self.assertEqual(captured['upstream_mode'], 'vllm')


class TestH33RetryFeedbackDistillation(HarnessTestCase):
    def test_feedback_lists_symbol_diff_and_respects_max_chars(self):
        tmp = tempfile.mkdtemp()
        state = os.path.join(tmp, 'state')
        run_dir = Path(state) / 'runs' / 'abc123'
        (run_dir / 'before').mkdir(parents=True)
        (run_dir / 'after').mkdir(parents=True)
        before_src = 'def foo():\n    return 1\n\nclass Bar:\n    pass\n\ndef gone():\n    return 3\n'
        after_src = 'def foo():\n    return 2\n\nclass Bar:\n    pass\n\ndef fresh():\n    return 4\n'
        (run_dir / 'before' / 'mod.py').write_text(before_src)
        (run_dir / 'after' / 'mod.py').write_text(after_src)
        (run_dir / 'events.jsonl').write_text(
            json.dumps({'event': 'salvage_validation_started'}) + '\n'
            + json.dumps({'event': 'run_finished'}) + '\n'
            + 'not-json\n')
        (run_dir / 'result.json').write_text(json.dumps(
            {'task_id': 't33', 'status': 'blocked', 'reason': 'validation_failed',
             'model': 'm', 'started_at': 1.0}))
        feedback = build_prior_attempt_feedback(state, 't33')
        self.assertIn('Modified symbols: foo()', feedback)
        self.assertIn('Added symbols: fresh()', feedback)
        self.assertIn('Dropped symbols: gone', feedback)
        self.assertNotIn('Bar', feedback)
        self.assertIn('salvage_validation_started', feedback)
        self.assertNotIn('run_finished', feedback)
        self.assertIn('validation_failed', feedback)
        long_dir = Path(state) / 'runs' / 'def456'
        (long_dir / 'before').mkdir(parents=True)
        (long_dir / 'after').mkdir(parents=True)
        big = 'def foo():\n    return 1\n\n' + '\n'.join(
            f'def f{i}():\n    return {i}\n' for i in range(200))
        (long_dir / 'before' / 'mod.py').write_text(big)
        (long_dir / 'after' / 'mod.py').write_text(big.replace('return 1', 'return 2'))
        (long_dir / 'result.json').write_text(json.dumps(
            {'task_id': 't33b', 'status': 'blocked', 'reason': 'validation_failed',
             'model': 'm', 'started_at': 1.0}))
        capped = build_prior_attempt_feedback(state, 't33b', max_chars=1200)
        self.assertLessEqual(len(capped), 1200)

    def test_snapshot_files_for_feedback_writes_python_snapshots(self):
        tmp = tempfile.mkdtemp()
        src = Path(tmp) / 'a.py'
        src.write_text('def foo():\n    return 1\n')
        txt = Path(tmp) / 'b.txt'
        txt.write_text('ignored')
        _snapshot_files_for_feedback(Path(tmp) / 'run', [str(src), str(txt)], 'after')
        self.assertEqual((Path(tmp) / 'run' / 'after' / str(src).replace('/', '__')).read_text(),
                         'def foo():\n    return 1\n')
        self.assertFalse((Path(tmp) / 'run' / 'after' / str(txt).replace('/', '__')).exists())


class TestH36TargetedEditAndToolHints(HarnessTestCase):
    def _run(self, messages):
        from harness.guards import inject_edit_failure_hints

        return inject_edit_failure_hints(messages)

    def test_noop_edit_hint(self):
        from harness.guards import _EDIT_NOOP_HINT

        messages = [{'role': 'tool', 'content':
                     'Error: oldString and NewString are identical'}]
        new_messages, injected = self._run(messages)
        self.assertEqual(injected, 1)
        self.assertIn(_EDIT_NOOP_HINT, new_messages[0]['content'])
        self.assertIn('[HARNESS_EDIT_HINT', new_messages[0]['content'])

    def test_noop_edit_idempotent(self):
        messages, injected = self._run(
            [{'role': 'tool', 'content':
              'oldString and newString are identical'}])
        self.assertEqual(injected, 1)
        again, injected2 = self._run(messages)
        self.assertEqual(injected2, 0)
        self.assertEqual(again, messages)

    def test_ambiguous_match_with_file_on_disk(self):
        from harness.guards import _EDIT_HINT_MARKER

        tmp = tempfile.mkdtemp()
        target = Path(tmp) / 'code.py'
        target.write_text('def a():\n    return 1\n\nmarker\n\ndef b():\n'
                          '    return 2\n\nmarker\n')
        messages = [
            {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'call-1', 'type': 'function', 'function': {
                    'name': 'edit',
                    'arguments': json.dumps({
                        'filePath': str(target),
                        'oldString': 'marker',
                        'newString': 'replacement',
                    }),
                }},
            ]},
            {'role': 'tool', 'tool_call_id': 'call-1',
             'content': 'Error: Found 2 matches for oldString'},
        ]
        new_messages, injected = self._run(messages)
        self.assertEqual(injected, 1)
        text = new_messages[1]['content']
        self.assertIn(_EDIT_HINT_MARKER, text)
        self.assertIn('(at lines L4, L9)', text)
        self.assertIn('3-5 unique surrounding context lines', text)

    def test_ambiguous_match_caps_at_five_occurrences(self):
        tmp = tempfile.mkdtemp()
        target = Path(tmp) / 'code.py'
        target.write_text('\n'.join(['marker'] * 7) + '\n')
        messages = [
            {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'call-1', 'type': 'function', 'function': {
                    'name': 'edit',
                    'arguments': json.dumps({
                        'filePath': str(target),
                        'oldString': 'marker',
                        'newString': 'replacement',
                    }),
                }},
            ]},
            {'role': 'tool', 'tool_call_id': 'call-1',
             'content': 'Error: Found 7 matches for oldString'},
        ]
        new_messages, injected = self._run(messages)
        self.assertEqual(injected, 1)
        text = new_messages[1]['content']
        self.assertIn('(at lines L1, L2, L3, L4, L5)', text)
        self.assertNotIn('L6', text)

    def test_ambiguous_match_without_file_on_disk(self):
        from harness.guards import _EDIT_HINT_MARKER

        messages = [
            {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'call-1', 'type': 'function', 'function': {
                    'name': 'edit',
                    'arguments': json.dumps({
                        'filePath': '/nonexistent/missing.py',
                        'oldString': 'marker',
                        'newString': 'replacement',
                    }),
                }},
            ]},
            {'role': 'tool', 'tool_call_id': 'call-1',
             'content': 'Error: Found 2 matches for oldString'},
        ]
        new_messages, injected = self._run(messages)
        self.assertEqual(injected, 1)
        text = new_messages[1]['content']
        self.assertIn(_EDIT_HINT_MARKER, text)
        self.assertIn('Multiple matches found', text)
        self.assertNotIn('(at lines', text)

    def test_ambiguous_match_no_line_numbers_when_file_changed(self):
        tmp = tempfile.mkdtemp()
        target = Path(tmp) / 'code.py'
        target.write_text('other content entirely\n')
        messages = [
            {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'call-1', 'type': 'function', 'function': {
                    'name': 'edit',
                    'arguments': json.dumps({
                        'filePath': str(target),
                        'oldString': 'marker',
                        'newString': 'replacement',
                    }),
                }},
            ]},
            {'role': 'tool', 'tool_call_id': 'call-1',
             'content': 'Error: Found 2 matches for oldString'},
        ]
        new_messages, injected = self._run(messages)
        self.assertEqual(injected, 1)
        self.assertNotIn('(at lines', new_messages[1]['content'])

    def test_unknown_tool_hint(self):
        from harness.guards import _EDIT_UNKNOWN_TOOL_HINT

        for tool in ('execute', 'run_command', 'shell', 'cmd'):
            messages = [{'role': 'tool', 'content':
                         'Error: No tool named "%s" is currently available' % tool}]
            new_messages, injected = self._run(messages)
            self.assertEqual(injected, 1)
            self.assertIn(_EDIT_UNKNOWN_TOOL_HINT, new_messages[0]['content'])

    def test_tool_result_block_unknown_tool_hint(self):
        from harness.guards import _EDIT_UNKNOWN_TOOL_HINT

        messages = [{'role': 'assistant', 'content': [
            {'type': 'tool_result', 'tool_use_id': 'tu-1',
             'content': 'No tool named "shell" is currently available'},
        ]}]
        new_messages, injected = self._run(messages)
        self.assertEqual(injected, 1)
        self.assertIn(_EDIT_UNKNOWN_TOOL_HINT,
                      new_messages[0]['content'][0]['content'])

    def test_mismatch_and_json_truncation_hints_still_work(self):
        from harness.guards import _EDIT_FAILURE_HINT, _EDIT_JSON_TRUNCATION_HINT

        messages = [{'role': 'tool', 'content': 'Error: oldString not found'}]
        new_messages, injected = self._run(messages)
        self.assertEqual(injected, 1)
        self.assertIn(_EDIT_FAILURE_HINT, new_messages[0]['content'])

        messages = [{'role': 'tool', 'content':
                     'Arguments could not be parsed as json: {oops'}]
        new_messages, injected = self._run(messages)
        self.assertEqual(injected, 1)
        self.assertIn(_EDIT_JSON_TRUNCATION_HINT, new_messages[0]['content'])

    def test_ambiguous_match_tool_result_block_with_tool_use_id(self):
        tmp = tempfile.mkdtemp()
        target = Path(tmp) / 'code.py'
        target.write_text('x\nmarker\ny\nmarker\n')
        messages = [
            {'role': 'assistant', 'content': [
                {'type': 'tool_use', 'id': 'tu-9', 'name': 'edit', 'input': {
                    'path': str(target),
                    'oldString': 'marker',
                    'newString': 'replacement',
                }},
            ]},
            {'role': 'assistant', 'content': [
                {'type': 'tool_result', 'tool_use_id': 'tu-9',
                 'content': 'Error: Found 2 matches for oldString'},
            ]},
        ]
        new_messages, injected = self._run(messages)
        self.assertEqual(injected, 1)
        self.assertIn('(at lines L2, L4)',
                      new_messages[1]['content'][0]['content'])


class TestH37ReadOnlyPreloadAndPermissionGuard(unittest.TestCase):
    """H-37: read-only excerpt preloading and permission-denied guard."""

    def test_readonly_header_marked(self):
        import re as _re
        import tempfile, unittest
        from harness.guards import build_preloaded_file_excerpts
        tmp = tempfile.mkdtemp()
        path = Path(tmp) / 'ref.py'
        path.write_text('x = 1\n' * 5)
        out = build_preloaded_file_excerpts([str(path)], '', read_only_files=[str(path)])
        self.assertIn('READ-ONLY — preloaded below; do NOT call read or edit on this path', out)
        plain = build_preloaded_file_excerpts([str(path)], '')
        self.assertNotIn('READ-ONLY', plain)

    def test_long_readonly_symbol_windows_capped_at_80(self):
        import re as _re
        import tempfile
        from harness.guards import build_preloaded_file_excerpts
        tmp = tempfile.mkdtemp()
        path = Path(tmp) / 'big.py'
        lines = ['filler line\n'] * 300
        for lineno in (20, 60, 100, 140):
            lines[lineno - 1] = 'def helper_x():  # at line %d\n' % lineno
        for lineno in (200, 250):
            lines[lineno - 1] = 'x = `alpha`-like\n'
        path.write_text(''.join(lines))
        prompt = 'fix `helper_x` and `alpha` in def helper_x'
        out = build_preloaded_file_excerpts([str(path)], prompt, read_only_files=[str(path)])
        included_linenos = [int(m.group(1)) for m in _re.finditer(r'^(\d+): ', out, _re.M)]
        self.assertTrue(included_linenos)
        self.assertLessEqual(len(included_linenos), 80)
        self.assertIn(20, included_linenos)
        self.assertIn(60, included_linenos)
        # The 80-line cap is consumed by the earliest merged windows, so
        # matches near the end of the file (line 250) are capped out.
        self.assertNotIn(250, included_linenos)
        self.assertNotIn(1, included_linenos)  # not the whole file
        # Non-read-only long file without ranges still yields nothing.
        empty = build_preloaded_file_excerpts([str(path)], prompt)
        self.assertEqual(empty, '')

    def test_inject_preloaded_permission_hint_tool_role(self):
        import json
        from harness.guards import inject_read_guard_hints
        path = '/allowed/ref.py'
        messages = [
            {'role': 'user', 'content': '[PRELOADED_FILE_EXCERPTS — exact lines]\n--- '
                                        + path + ' (5 lines, READ-ONLY)'},
            {'role': 'assistant', 'tool_calls': [
                {'id': 'c1', 'function': {'name': 'read',
                                          'arguments': json.dumps({'path': path})}}]},
            {'role': 'tool', 'tool_call_id': 'c1',
             'content': 'Permission denied: read ' + path},
        ]
        new_messages, injected = inject_read_guard_hints(messages)
        self.assertEqual(injected, 1)
        text = new_messages[2]['content']
        self.assertIn('[HARNESS_READ_GUARD:', text)
        self.assertIn('ALREADY preloaded', text)
        # Idempotent.
        self.assertEqual(inject_read_guard_hints(new_messages)[1], 0)

    def test_inject_preloaded_permission_hint_tool_result_block(self):
        import json
        from harness.guards import inject_read_guard_hints
        path = '/allowed/ref2.py'
        messages = [
            {'role': 'user', 'content': '[FILE_SKELETONS]\n' + path},
            {'role': 'assistant', 'content': [
                {'type': 'tool_use', 'id': 'tu1', 'name': 'read',
                 'input': {'file_path': path}}]},
            {'role': 'user', 'content': [
                {'type': 'tool_result', 'tool_use_id': 'tu1',
                 'content': 'Permission denied: read'}]},
        ]
        new_messages, injected = inject_read_guard_hints(messages)
        self.assertEqual(injected, 1)
        self.assertIn('[HARNESS_READ_GUARD:',
                      new_messages[2]['content'][0]['content'])
        self.assertEqual(inject_read_guard_hints(new_messages)[1], 0)

    def test_inject_fallback_hint_when_not_preloaded(self):
        import json
        from harness.guards import inject_read_guard_hints
        path = '/other/secret.py'
        messages = [
            {'role': 'user', 'content': '[PRELOADED_FILE_EXCERPTS]\n/allowed/ref.py'},
            {'role': 'assistant', 'tool_calls': [
                {'id': 'c2', 'function': {'name': 'read',
                                          'arguments': json.dumps({'filePath': path})}}]},
            {'role': 'tool', 'tool_call_id': 'c2',
             'content': 'Permission denied: read'},
        ]
        new_messages, injected = inject_read_guard_hints(messages)
        self.assertEqual(injected, 1)
        self.assertIn('[HARNESS_READ_HINT:', new_messages[2]['content'])
        self.assertNotIn('[HARNESS_READ_GUARD:', new_messages[2]['content'])

    def test_model_run_passes_read_only_files(self):
        import inspect
        import harness.sandbox as sandbox
        source = inspect.getsource(sandbox.model_run)
        self.assertIn("read_only_files=task.get('read_only_files', [])", source)


class TestH38ProxyTimeoutFailover(HarnessTestCase):
    """H-38: a read timeout on the pinned provider marks it degraded, retries
    the next healthy provider inline, and a double timeout surfaces 502."""

    def _post_status(self, base_url, body):
        import http.client
        import urllib.parse
        parsed = urllib.parse.urlparse(base_url)
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=10)
        try:
            conn.request('POST', '/v1/chat/completions', body=body,
                         headers={'Content-Type': 'application/json'})
            resp = conn.getresponse()
            resp.read()
            return resp.status
        finally:
            conn.close()

    def _new_health_store(self, tmp):
        import importlib
        for mod_name in ('harness.proxy', 'harness.sandbox', 'harness.health'):
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, 'ProviderHealthStore', None)
            if cls is not None:
                try:
                    return cls()
                except TypeError:
                    try:
                        return cls(state_path=str(tmp / 'health.json'))
                    except Exception:
                        return None
        return None

    def _start_proxy(self, tmp):
        from harness.proxy import OpenRouterGuardProxy
        hs = self._new_health_store(tmp)
        try:
            proxy = OpenRouterGuardProxy(lambda: None,
                                         upstream_url='http://127.0.0.1:9',
                                         health_store=hs)
        except TypeError:
            proxy = OpenRouterGuardProxy(lambda: None,
                                         upstream_url='http://127.0.0.1:9')
            hs = None
        threading.Thread(target=proxy.serve_forever, daemon=True).start()
        return proxy, hs

    def test_timeout_marks_degraded_and_retries_next_provider(self):
        calls = []

        class _FakeResp:
            status = 200

            def __init__(self, chunks):
                self._chunks = list(chunks)
                self.headers = {'Content-Type': 'application/json'}

            def read(self, size=-1):
                return self._chunks.pop(0) if self._chunks else b''

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def _fake_urlopen(req, timeout=None):
            calls.append(req)
            if len(calls) == 1:
                raise TimeoutError('timed out')
            return _FakeResp([b'{"id":"gen-1","choices":[]}', b''])

        proxy, hs = self._start_proxy(self.path('h38-health'))
        try:
            body = json.dumps({'model': GLM,
                               'messages': [{'role': 'user', 'content': 'hi'}]}).encode()
            import unittest.mock as _mock
            with _mock.patch('urllib.request.urlopen', side_effect=_fake_urlopen):
                status = self._post_status(proxy.base_url, body)
            self.assertEqual(status, 200)
            self.assertEqual(len(calls), 2)
            first = json.loads(calls[0].data.decode())
            second = json.loads(calls[1].data.decode())
            first_provider = first['provider']['order'][0]
            self.assertIn(first_provider, proxy.degraded_providers)
            if hs is not None:
                self.assertIn(first_provider, hs.active_degraded())
            self.assertNotEqual(second['provider']['order'][0], first_provider)
        finally:
            proxy.shutdown(); proxy.server_close()

    def test_double_timeout_returns_502(self):
        proxy, _hs = self._start_proxy(self.path('h38-health-2'))
        try:
            body = json.dumps({'model': GLM,
                               'messages': [{'role': 'user', 'content': 'hi'}]}).encode()
            import unittest.mock as _mock
            with _mock.patch('urllib.request.urlopen',
                             side_effect=TimeoutError('timed out')):
                status = self._post_status(proxy.base_url, body)
            self.assertEqual(status, 502)
        finally:
            proxy.shutdown(); proxy.server_close()


class TestH39BestOf2CascadeHandoff(HarnessTestCase):
    """H-39: best_of_2 fills the default candidate cascade and hands compact
    TraJSpec feedback (modified symbols + test failure) to candidate 2 after
    restoring the original file bytes."""

    def test_validate_task_best_of_2_default_cascade(self):
        import os
        from harness.sandbox import validate_task
        d = self.path('h39a')
        os.makedirs(d, exist_ok=True)
        f = os.path.join(d, 'a.py')
        with open(f, 'w', encoding='utf-8') as fh:
            fh.write('VALUE = 0\n')
        task = validate_task({
            'task_id': 'cascade-default',
            'directory': d,
            'files': [f],
            'prompt': 'Fix a.py',
            'test_argv': [sys.executable, '-c', 'pass'],
            'strategy': 'best_of_2',
            'model': GLM,
        })
        self.assertEqual(task['candidate_models'], [GLM, DEEPSEEK])

    def test_run_best_of_2_handoffs_trajspec_feedback_to_candidate_2(self):
        import os
        import tempfile
        from pathlib import Path
        from harness.sandbox import run_best_of_2

        original_src = 'def target_symbol():\n    return 1\n'
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp).resolve()
            py_path = tmp_path / 'mod_under_test.py'
            py_path.write_text(original_src)
            artifact_dir = tmp_path / 'artifact'
            task_path = tmp_path / 'task.json'
            task_path.write_text(json.dumps({
                'task_id': 'h39-handoff',
                'directory': str(tmp_path),
                'files': [str(py_path)],
                'prompt': 'Fix target_symbol in mod_under_test.py',
                'test_argv': [sys.executable, '-c', 'pass'],
                'strategy': 'best_of_2',
                'model': GLM,
            }))

            def fake_run_task(path, **kwargs):
                calls.append(json.loads(Path(path).read_text()))
                if len(calls) == 1:
                    py_path.write_text('def target_symbol():\n    return 2\n')
                    artifact_dir.mkdir()
                    (artifact_dir / 'test.log').write_text(
                        'FAILED tests/test_mod.py::test_target_symbol\n'
                        'E   AssertionError: assert 1 == 2\n')
                    return {'status': 'blocked', 'reason': 'validation_failed',
                            'model': GLM, 'artifact': str(artifact_dir),
                            'steps': 3, 'cost': 0.01}
                prompt = calls[1]['prompt']
                self.assertIn('[CANDIDATE_1_TRAJSPEC_FEEDBACK]', prompt)
                self.assertIn('target_symbol', prompt)
                self.assertIn('validation_failed', prompt)
                self.assertIn('E   AssertionError', prompt)
                self.assertEqual(py_path.read_text(), original_src)
                return {'status': 'validation_passed_review_required',
                        'model': DEEPSEEK, 'steps': 2, 'cost': 0.01}

            old_cwd = os.getcwd()
            os.chdir(tmp_path)
            try:
                import unittest.mock as _mock
                with _mock.patch('harness.sandbox.run_task', side_effect=fake_run_task):
                    rc = run_best_of_2(str(task_path))
            finally:
                os.chdir(old_cwd)
            self.assertEqual(rc, 0)
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0]['model'], GLM)
            self.assertEqual(calls[1]['model'], DEEPSEEK)


if __name__ == '__main__':
    unittest.main()

