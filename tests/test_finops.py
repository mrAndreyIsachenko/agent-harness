"""Unit tests for finops window-read tracking and recoverable edit errors."""

import unittest

from harness.finops import metrics, violation


def _snap(messages):
    return {'cost': 0.0, 'messages': messages}


def _msg(parts):
    return ('assistant', {'cost': 0.0, 'finish': 'stop', 'content': parts})


def _read(path, offset, limit=50, content=None):
    state = {'status': 'completed', 'input': {'path': path, 'offset': offset, 'limit': limit}}
    if content is not None:
        state['content'] = content
    return {'type': 'tool', 'name': 'read', 'state': state}


def _edit(path, status='completed', error=None):
    return {
        'type': 'tool', 'name': 'edit',
        'state': {'status': status, 'input': {'path': path, 'oldString': 'a', 'newString': 'b'},
                  **({'error': error} if error else {})},
    }


class TestWindowReads(unittest.TestCase):
    def test_paging_distinct_windows_not_blocked(self):
        # 8 distinct (offset, limit) chunks across turns: legitimate paging.
        msgs = [_msg([_read('/a.py', o)]) for o in (1, 100, 200, 300, 400, 500, 600, 700)]
        m = metrics(_snap(msgs))
        self.assertEqual(m['file_reads'], 8)
        self.assertEqual(m['window_reads'], 1)  # each distinct window seen once
        self.assertIsNone(violation(m, 1e9, 25))

    def test_same_window_repeated_blocked(self):
        msgs = [_msg([_read('/a.py', 1, 50, content=f'v{i}')]) for i in range(4)]
        m = metrics(_snap(msgs))
        self.assertEqual(m['window_reads'], 4)
        self.assertEqual(violation(m, 1e9, 25), 'repeated_file_read_without_write')

    def test_edit_resets_read_counters_for_path(self):
        msgs = [_msg([_read('/a.py', 1), _read('/a.py', 100)]) for _ in range(3)]
        msgs.append(_msg([_edit('/a.py')]))
        m = metrics(_snap(msgs))
        self.assertEqual(m['file_reads'], 0)
        self.assertEqual(m['window_reads'], 0)


class TestRecoverableEditErrors(unittest.TestCase):
    def _snap_with(self, parts):
        return _snap([_msg(parts)])

    def test_recoverable_then_read_then_edit_no_block(self):
        mid = [_edit('/a.py', status='error', error='Could not find oldString'),
               _edit('/a.py', status='error', error='Found 2 matches'),
               _read('/a.py', 1)]
        m = metrics(self._snap_with(mid))
        # 2 recoverable errors do not trip 'repeated_tool_error' (threshold 3).
        self.assertEqual(m['recoverable_errors'], 2)
        self.assertIsNone(violation(m, 1e9, 25))
        full = mid + [_edit('/a.py')]
        m = metrics(self._snap_with(full))
        self.assertEqual(m['recoverable_errors'], 0)
        self.assertEqual(m['window_reads'], 0)

    def test_three_recoverable_errors_block(self):
        parts = [_edit('/a.py', status='error', error='Could not find oldString') for _ in range(3)]
        m = metrics(self._snap_with(parts))
        self.assertEqual(m['recoverable_errors'], 3)
        self.assertEqual(violation(m, 1e9, 25), 'repeated_tool_error')

    def test_two_non_recoverable_errors_block(self):
        parts = [_edit('/a.py', status='error', error='Permission denied'),
                 _edit('/a.py', status='error', error='Permission denied')]
        m = metrics(self._snap_with(parts))
        self.assertEqual(m['errors'], 2)
        self.assertEqual(m['recoverable_errors'], 0)
        self.assertEqual(violation(m, 1e9, 25), 'repeated_tool_error')

class TestNewStallCounters(unittest.TestCase):
    def test_write_edit_calls_and_steps_since_last_edit(self):
        msgs = [
            _msg([_read('/a.py', 1)]),          # turn 1: first completed read -> baseline 0
            _msg([_read('/a.py', 100)]),        # turn 2: non-edit turn -> 1
            _msg([_edit('/a.py')]),             # turn 3: edit -> reset to 0
            _msg([_read('/a.py', 200)]),        # turn 4 -> 1
            _msg([]),                           # turn 5: no tools -> 2
        ]
        m = metrics(_snap(msgs))
        self.assertEqual(m['write_edit_calls'], 1)
        self.assertEqual(m['steps_since_last_edit'], 2)
        self.assertEqual(m['reasoning_only_streak'], 1)

    def test_no_edit_steps_since_last_edit_tracks_reads(self):
        msgs = [_msg([_read('/a.py', o)]) for o in (1, 100, 200)]
        msgs.append(_msg([]))
        m = metrics(_snap(msgs))
        self.assertEqual(m['write_edit_calls'], 0)
        # first read turn starts at 0; 3 subsequent turns (2 reads + 1 empty) increment
        self.assertEqual(m['steps_since_last_edit'], 3)

    def test_multiple_write_edit_calls_counted(self):
        msgs = [_msg([_edit('/a.py')]), _msg([_edit('/b.py')]), _msg([_edit('/c.py')])]
        m = metrics(_snap(msgs))
        self.assertEqual(m['write_edit_calls'], 3)
        self.assertEqual(m['steps_since_last_edit'], 0)

    def test_four_reasoning_turns_block(self):
        msgs = [_msg([]) for _ in range(4)]
        m = metrics(_snap(msgs))
        self.assertEqual(m['reasoning_only_streak'], 4)
        self.assertEqual(violation(m, 1e9, 25), 'reasoning_stall_no_tool_call')

    def test_tool_turn_resets_reasoning_streak(self):
        msgs = [_msg([]) for _ in range(3)]
        msgs.append(_msg([_read('/a.py', 1)]))
        msgs.extend(_msg([]) for _ in range(3))
        m = metrics(_snap(msgs))
        self.assertEqual(m['reasoning_only_streak'], 3)
        self.assertIsNone(violation(m, 1e9, 25))


if __name__ == '__main__':
    unittest.main()
