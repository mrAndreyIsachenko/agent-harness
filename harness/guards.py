"""Guard helpers ported from the opencode_guard supervisor.

The module is stdlib-only (``ast``, ``json``, ``re``, ``pathlib``) so it can be
imported from any Python >= 3.10 environment without installing dependencies.

Public functions:

- :func:`check_allowed_files_syntax` -- zero-shell AST/JSON/SQL syntax pre-flight.
- :func:`check_sql_syntax` -- SQL/Jinja template pre-flight (H-7).
- :func:`collect_python_top_symbols` -- top-level Python symbol inventory.
- :func:`detect_duplicate_tool_calls` -- repeated identical tool-call detection (H-5).
- :func:`compact_stale_reads` -- stale read-result compaction (H-8).
- :func:`estimate_messages_tokens` -- deterministic message token estimator.
- :func:`compact_context_window` -- context-budget compaction for older tool outputs.
- :func:`build_file_skeletons` -- compact line-range maps for large files.
- :func:`extract_compact_test_failure` -- bounded digest of a failing test run.

Private helpers (``_message_text``, ``_sql_skeleton_lines``) support the public
API and are not re-exported.
"""

from __future__ import annotations

import ast
import json
import re
from bisect import bisect_right
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "FORBIDDEN_EGRESS_KEYS",
    "build_file_skeletons",
    "check_allowed_files_syntax",
    "check_egress_payload",
    "check_readonly_sql_tool_calls",
    "check_sql_syntax",
    "collect_python_top_symbols",
    "compact_context_window",
    "compact_stale_reads",
    "detect_duplicate_tool_calls",
    "estimate_messages_tokens",
    "extract_compact_test_failure",
    "inject_edit_failure_hints",
]

#: Forbidden keys that must never leave the local harness in an LLM egress payload.
FORBIDDEN_EGRESS_KEYS: frozenset[str] = frozenset(
    {
        "rows",
        "raw_rows",
        "records",
        "dataframe",
        "table_dump",
        "query_results",
        "duckdb_rows",
        "credentials",
        "secret",
        "api_key",
        "password",
        "private_key",
        "access_token",
        "connection_string",
    }
)

#: Standard OpenAI/OpenRouter top-level request envelope keys exempt from key-name flagging.
_STANDARD_ENVELOPE_KEYS: frozenset[str] = frozenset(
    {
        "model",
        "messages",
        "tools",
        "tool_choice",
        "temperature",
        "top_p",
        "max_tokens",
        "stream",
        "stream_options",
        "reasoning",
        "provider",
        "response_format",
        "metadata",
        "user",
        "stop",
        "seed",
        "n",
        "frequency_penalty",
        "presence_penalty",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "parallel_tool_calls",
        "store",
        "modalities",
        "audio",
        "prediction",
        "service_tier",
        "input",
        "instructions",
        "max_output_tokens",
        "previous_response_id",
        "truncation",
    }
)

#: High-confidence leaked credential patterns scanned in message text content.
_CREDENTIAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("sk-or-v1-...", re.compile(r"sk-or-v1-[A-Za-z0-9_-]+")),
    ("PRIVATE KEY-----", re.compile(r"PRIVATE KEY-----")),
)

#: SQL tool names subject to read-only enforcement.
_SQL_TOOL_NAMES: frozenset[str] = frozenset(
    {"query_warehouse", "execute_sql", "run_sql", "sql_query"}
)

#: Mutating or unsafe SQL keywords blocked in read-only SQL tool calls.
_UNSAFE_SQL_KEYWORD_RE = re.compile(
    r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|ATTACH|COPY|PRAGMA|INSTALL|LOAD)\b",
    re.IGNORECASE,
)

#: Tools monitored for semantic oscillation across recent assistant turns.
_OSCILLATION_TOOLS: frozenset[str] = frozenset(
    {"query_warehouse", "get_metric_lineage", "bash", "edit"}
)

#: Argument keys used to extract target table/metric/command/file for oscillation detection.
_SQL_TARGET_KEYS: tuple[str, ...] = (
    "table",
    "table_name",
    "model",
    "metric",
    "metric_name",
    "target",
)
_SQL_TABLE_REF_RE = re.compile(r"\b(?:FROM|JOIN)\s+([A-Za-z_][\w.]*)", re.IGNORECASE)
_ERROR_RESULT_RE = re.compile(
    r"\b(error|failed|exception|traceback|syntax|denied|not found|invalid)\b",
    re.IGNORECASE,
)

#: File suffixes eligible for syntax pre-flight in :func:`check_allowed_files_syntax`.
_PARSEABLE_SUFFIXES: tuple[str, ...] = (".py", ".json", ".sql")

#: Git merge-conflict markers rejected by the pre-flight.
_CONFLICT_MARKERS: tuple[str, ...] = ("<<<<<<< ", "=======", ">>>>>>> ")

#: AST node classes counted as top-level definitions.
_DEF_NODES: tuple[type[ast.stmt], ...] = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
)

#: Major SQL clause keywords that must not directly follow a trailing comma.
_TRAILING_COMMA_RE = re.compile(
    r",\s*\b(FROM|WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|UNION)\b", re.IGNORECASE
)

#: Jinja block tags tracked as open/close pairs by :func:`check_sql_syntax`.
_JINJA_OPENERS: dict[str, str] = {"if": "if", "for": "for", "macro": "macro"}
_JINJA_CLOSERS: dict[str, str] = {"endif": "if", "endfor": "for", "endmacro": "macro"}

#: Tool-name substrings classifying calls for :func:`compact_stale_reads` (H-8).
_STALE_EDIT_MARKERS: tuple[str, ...] = ("edit", "write", "patch")
_STALE_READ_MARKERS: tuple[str, ...] = ("read", "view")

#: Argument keys inspected for a file path in tool-call arguments.
_STALE_PATH_KEYS: tuple[str, ...] = ("path", "file_path", "filePath", "target_file")

#: Replacement text injected into a compacted stale read result.
_STALE_READ_NOTICE = (
    "[STALE READ COMPACTED: {path} was modified in a later turn — "
    "re-read target lines if needed]"
)


def check_allowed_files_syntax(
    files: Iterable[str] | None,
    *,
    baseline_symbols: dict[str, set[str]] | None = None,
) -> list[str]:
    """Zero-shell AST/JSON syntax pre-flight over allowed files.

    Parses each ``.py``/``.json``/``.sql`` path without spawning any process and
    returns ``path:line: message`` error strings; an empty list means the
    pre-flight passed. Per Python file it also flags duplicate ``dict`` literal
    keys and, when a baseline map is supplied, dropped top-level symbols.
    Non-parseable suffixes, missing paths, and unreadable files are skipped.

    Args:
        files: Paths to inspect.
        baseline_symbols: Optional ``{path: top-level symbol set}`` baseline as
            produced by :func:`collect_python_top_symbols`. For any path present
            here, removing one of its top-level function/class symbols is an error.
    """
    errors: list[str] = []
    for path in files or []:
        try:
            if not path.endswith(_PARSEABLE_SUFFIXES) or not Path(path).is_file():
                continue
            source = Path(path).read_text(encoding="utf-8")
            if path.endswith(".py"):
                try:
                    tree = ast.parse(source, filename=path)
                except SyntaxError as exc:
                    errors.append(f"{path}:{exc.lineno or 1}: {exc.msg}")
                else:
                    current_top_symbols = {
                        node.name for node in tree.body if isinstance(node, _DEF_NODES)
                    }
                    if baseline_symbols is not None and path in baseline_symbols:
                        missing = sorted(baseline_symbols[path] - current_top_symbols)
                        if missing:
                            errors.append(
                                f"{path}:1: dropped top-level symbol(s): {', '.join(missing)}"
                            )
                    for node in ast.walk(tree):
                        if not isinstance(node, ast.Dict):
                            continue
                        seen: set[tuple[type, Any]] = set()
                        for key in node.keys:
                            if isinstance(key, ast.Constant) and (
                                isinstance(key.value, (str, int, float, bool)) or key.value is None
                            ):
                                identity = (type(key.value), key.value)
                                if identity in seen:
                                    errors.append(
                                        f"{path}:{key.lineno}: duplicate dict key {key.value!r}"
                                    )
                                seen.add(identity)
                    errors.extend(_sibling_import_errors(path, tree))
            elif path.endswith(".json"):
                try:
                    json.loads(source)
                except json.JSONDecodeError as exc:
                    errors.append(f"{path}:{exc.lineno}: {exc.msg}")
            elif path.endswith(".sql"):
                errors.extend(check_sql_syntax(source, path))
            for lineno, line in enumerate(source.splitlines(), 1):
                if line.startswith(_CONFLICT_MARKERS):
                    errors.append(f"{path}:{lineno}: merge conflict marker left in file")
        except (OSError, UnicodeDecodeError):
            continue
    return errors


def collect_python_top_symbols(files: Iterable[str] | None) -> dict[str, set[str]]:
    """Parse existing ``.py`` files into ``{path: set of top-level function/class names}``.

    Files with a non-``.py`` suffix, missing paths, unreadable files, and files
    that fail to parse are silently omitted, so callers can build a baseline
    from whatever currently parses. Feed the result back as
    ``check_allowed_files_syntax(..., baseline_symbols=...)`` to reject edits
    that drop top-level symbols.
    """
    symbols: dict[str, set[str]] = {}
    for path in files or []:
        try:
            if not path.endswith(".py") or not Path(path).is_file():
                continue
            tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=path)
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        symbols[path] = {node.name for node in tree.body if isinstance(node, _DEF_NODES)}
    return symbols


def _sibling_defined_names(tree: ast.Module) -> set[str]:
    """Names defined, assigned, imported, or exported via ``__all__`` in ``tree``."""
    known: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, _DEF_NODES):
            known.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                known.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                for name_node in ast.walk(target):
                    if isinstance(name_node, ast.Name):
                        known.add(name_node.id)
            if any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
                for element in ast.walk(node.value):
                    if isinstance(element, ast.Constant) and isinstance(element.value, str):
                        known.add(element.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            known.add(node.target.id)
    return known


def _sibling_import_errors(path: str, tree: ast.Module) -> list[str]:
    """H-7: flag ``from <sibling> import <missing>`` without executing anything.

    For each ``ImportFrom`` naming a simple sibling module (no dots, ``level``
    0 or 1) whose ``<module>.py`` exists next to ``path`` and parses cleanly,
    every non-``*`` imported alias must match a name collected by
    :func:`_sibling_defined_names`. Unparseable siblings are skipped rather
    than guessed at.
    """
    errors: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        if not module or "." in module or node.level not in (0, 1):
            continue
        sibling = Path(path).parent / f"{module}.py"
        if not sibling.is_file():
            continue
        try:
            sibling_tree = ast.parse(sibling.read_text(encoding="utf-8"), filename=str(sibling))
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        known = _sibling_defined_names(sibling_tree)
        for alias in node.names:
            if alias.name != "*" and alias.name not in known:
                errors.append(
                    f"{path}:{node.lineno}: cannot import name {alias.name!r} from {module!r}"
                )
    return errors


def _check_jinja_statement(
    content: str,
    stmt_line: int,
    jinja_stack: list[tuple[str, int]],
    filename: str,
    errors: list[str],
) -> None:
    """Classify one ``{% ... %}`` statement body and update the open-tag stack."""
    words = content.strip().strip("-").strip().split()
    if not words:
        return
    tag = words[0].lower()
    if tag in _JINJA_OPENERS:
        jinja_stack.append((tag, stmt_line))
    elif tag in _JINJA_CLOSERS:
        if not jinja_stack or jinja_stack[-1][0] != _JINJA_CLOSERS[tag]:
            errors.append(f"{filename}:{stmt_line}: unbalanced Jinja tag")
        if jinja_stack:
            jinja_stack.pop()


def check_sql_syntax(text: str, filename: str = "<sql>") -> list[str]:
    """H-7: pre-flight a SQL/Jinja template without spawning any process.

    Scans ``text`` while stripping ``--`` line comments, ``/* ... */`` block
    comments, single-quoted string literals (``''`` escapes handled), and Jinja
    ``{{ ... }}`` / ``{% ... %}`` regions, then checks the remaining SQL surface
    for balanced parentheses and trailing commas before major clause keywords.
    Jinja statement tags are balance-checked as ``if``/``endif``,
    ``for``/``endfor`` and ``macro``/``endmacro`` pairs. Returns
    ``filename:line: message`` error strings; an empty list means the check
    passed.
    """
    errors: list[str] = []
    if not isinstance(text, str):
        return errors
    line_starts = [0]
    for offset, char in enumerate(text):
        if char == "\n":
            line_starts.append(offset + 1)

    def line_of(pos: int) -> int:
        return bisect_right(line_starts, pos)

    chars = list(text)
    n = len(text)
    i = 0
    in_block_comment = False
    block_comment_line = 0
    in_string = False
    string_line = 0
    in_expr = False
    expr_line = 0
    in_stmt = False
    stmt_line = 0
    stmt_chars: list[str] = []
    jinja_stack: list[tuple[str, int]] = []
    while i < n:
        char = text[i]
        if in_block_comment:
            if char != "\n":
                chars[i] = " "
            if text.startswith("*/", i):
                chars[i + 1] = " "
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue
        if in_string:
            if char != "\n":
                chars[i] = " "
            if char == "'":
                if text.startswith("''", i):
                    chars[i + 1] = " "
                    i += 2
                else:
                    in_string = False
                    i += 1
            else:
                i += 1
            continue
        if in_expr:
            if char != "\n":
                chars[i] = " "
            if text.startswith("}}", i):
                chars[i + 1] = " "
                in_expr = False
                i += 2
            else:
                i += 1
            continue
        if in_stmt:
            if text.startswith("%}", i):
                chars[i] = " "
                chars[i + 1] = " "
                in_stmt = False
                _check_jinja_statement("".join(stmt_chars), stmt_line, jinja_stack, filename, errors)
                i += 2
            else:
                if char != "\n":
                    chars[i] = " "
                stmt_chars.append(char)
                i += 1
            continue
        if text.startswith("--", i):
            end = text.find("\n", i)
            end = n if end == -1 else end
            for index in range(i, end):
                chars[index] = " "
            i = end
            continue
        if text.startswith("/*", i):
            chars[i] = " "
            chars[i + 1] = " "
            in_block_comment = True
            block_comment_line = line_of(i)
            i += 2
            continue
        if text.startswith("{{", i):
            chars[i] = " "
            chars[i + 1] = " "
            in_expr = True
            expr_line = line_of(i)
            i += 2
            continue
        if text.startswith("}}", i):
            chars[i] = " "
            chars[i + 1] = " "
            errors.append(f"{filename}:{line_of(i)}: unbalanced Jinja tag")
            i += 2
            continue
        if text.startswith("{%", i):
            chars[i] = " "
            chars[i + 1] = " "
            in_stmt = True
            stmt_line = line_of(i)
            stmt_chars = []
            i += 2
            continue
        if text.startswith("%}", i):
            chars[i] = " "
            chars[i + 1] = " "
            errors.append(f"{filename}:{line_of(i)}: unbalanced Jinja tag")
            i += 2
            continue
        if char == "'":
            chars[i] = " "
            in_string = True
            string_line = line_of(i)
            i += 1
            continue
        i += 1
    if in_block_comment:
        errors.append(f"{filename}:{block_comment_line}: unclosed SQL comment or string literal")
    if in_string:
        errors.append(f"{filename}:{string_line}: unclosed SQL comment or string literal")
    if in_expr or in_stmt:
        errors.append(f"{filename}:{expr_line if in_expr else stmt_line}: unbalanced Jinja tag")
    for _, tag_line in jinja_stack:
        errors.append(f"{filename}:{tag_line}: unbalanced Jinja tag")
    masked = "".join(chars)
    open_parens: list[int] = []
    paren_error = False
    for match in re.finditer(r"[()]", masked):
        pos = match.start()
        if match.group() == "(":
            open_parens.append(line_of(pos))
        elif not paren_error and not open_parens:
            errors.append(f"{filename}:{line_of(pos)}: unbalanced parenthesis")
            paren_error = True
        elif open_parens:
            open_parens.pop()
    if open_parens and not paren_error:
        errors.append(f"{filename}:{open_parens[-1]}: unbalanced parenthesis")
    for match in _TRAILING_COMMA_RE.finditer(masked):
        keyword = re.sub(r"\s+", " ", match.group(1).upper())
        errors.append(f"{filename}:{line_of(match.start())}: trailing comma before {keyword}")
    return errors


def _message_text(message: Mapping[str, Any]) -> str:
    """Flatten a chat message's ``content`` (string or content-part list) to text.

    A string content passes through unchanged; a list keeps its string parts and
    the ``text`` field of each dict part joined with spaces. Anything else
    collapses to an empty string.
    """
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return " ".join(parts)
    return ""


def _parse_tool_arguments(raw_arguments: Any) -> dict[str, Any] | None:
    """Parse tool-call arguments into a dict when possible."""
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if isinstance(raw_arguments, str):
        try:
            parsed = json.loads(raw_arguments)
            return parsed if isinstance(parsed, dict) else None
        except ValueError:
            return None
    return None


def _strip_sql_comments_and_strings(text: str) -> str:
    """Replace SQL comments and single-quoted string literals with spaces."""
    chars = list(text)
    n = len(text)
    i = 0
    in_block_comment = False
    in_string = False
    while i < n:
        char = text[i]
        if in_block_comment:
            if char != "\n":
                chars[i] = " "
            if text.startswith("*/", i):
                chars[i + 1] = " "
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue
        if in_string:
            if char != "\n":
                chars[i] = " "
            if char == "'":
                if text.startswith("''", i):
                    chars[i + 1] = " "
                    i += 2
                else:
                    in_string = False
                    i += 1
            else:
                i += 1
            continue
        if text.startswith("--", i):
            end = text.find("\n", i)
            end = n if end == -1 else end
            for idx in range(i, end):
                chars[idx] = " "
            i = end
            continue
        if text.startswith("/*", i):
            chars[i] = " "
            chars[i + 1] = " "
            in_block_comment = True
            i += 2
            continue
        if char == "'":
            chars[i] = " "
            in_string = True
            i += 1
            continue
        i += 1
    return "".join(chars)


def check_egress_payload(
    payload: Mapping[str, Any],
    *,
    allowed_models: Iterable[str] | None = None,
    max_payload_bytes: int = 262_144,
) -> list[str]:
    """Validate an outbound LLM request payload before it leaves the local proxy.

    Checks for forbidden egress keys anywhere in the JSON structure (except
    standard OpenAI/OpenRouter top-level envelope keys), leaked credential
    patterns in message text content, disallowed model names, and oversized
    serialized payloads. Returns human-readable violation strings.
    """
    violations: list[str] = []
    if not isinstance(payload, Mapping):
        return ["egress payload must be a JSON object"]

    def _scan_value(value: Any, path: str, *, depth: int) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                key_str = str(key)
                child_path = f"{path}.{key_str}"
                if not (depth == 0 and key_str in _STANDARD_ENVELOPE_KEYS):
                    if key_str.lower() in FORBIDDEN_EGRESS_KEYS:
                        violations.append(f"forbidden egress key: {child_path}")
                _scan_value(item, child_path, depth=depth + 1)
        elif isinstance(value, (list, tuple)):
            for idx, item in enumerate(value):
                _scan_value(item, f"{path}[{idx}]", depth=depth + 1)

    _scan_value(payload, "body", depth=0)

    messages = payload.get("messages")
    if isinstance(messages, list):
        for idx, msg in enumerate(messages):
            if not isinstance(msg, Mapping):
                continue
            text = _message_text(msg)
            if not text:
                continue
            for label, pattern in _CREDENTIAL_PATTERNS:
                if pattern.search(text):
                    violations.append(
                        f"leaked credential pattern ({label}) in messages[{idx}]"
                    )

    if allowed_models is not None:
        allowed_set = set(allowed_models)
        if allowed_set:
            model = payload.get("model")
            if not isinstance(model, str) or model not in allowed_set:
                violations.append(
                    f"disallowed model {model!r}; allowed: {sorted(allowed_set)}"
                )

    try:
        serialized_size = len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    except (TypeError, ValueError):
        serialized_size = len(repr(payload).encode("utf-8"))
    if serialized_size > max_payload_bytes:
        violations.append(
            f"payload size {serialized_size} bytes exceeds max_payload_bytes {max_payload_bytes}"
        )
    return violations


def check_readonly_sql_tool_calls(
    messages: Iterable[Mapping[str, Any]] | None,
) -> list[str]:
    """Inspect assistant SQL tool calls for mutating keywords, `;` injection, and syntax errors."""
    violations: list[str] = []
    for message in messages or []:
        if not (isinstance(message, Mapping) and message.get("role") == "assistant"):
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            if not isinstance(call, Mapping):
                continue
            function = call.get("function")
            function = function if isinstance(function, Mapping) else {}
            tool_name = str(function.get("name") or call.get("name") or "")
            if tool_name not in _SQL_TOOL_NAMES:
                continue
            args = _parse_tool_arguments(function.get("arguments"))
            if not args:
                continue
            sql = args.get("sql") if isinstance(args.get("sql"), str) else args.get("query")
            if not isinstance(sql, str) or not sql.strip():
                continue
            masked = _strip_sql_comments_and_strings(sql)
            if ";" in masked.strip().rstrip(";").strip():
                violations.append(
                    f"multi-statement SQL injection detected in {tool_name}"
                )
            seen_keywords: set[str] = set()
            for match in _UNSAFE_SQL_KEYWORD_RE.finditer(masked):
                kw = match.group(1).upper()
                if kw not in seen_keywords:
                    seen_keywords.add(kw)
                    violations.append(
                        f"unsafe SQL keyword {kw} in {tool_name}"
                    )
            for syntax_err in check_sql_syntax(sql, filename=tool_name):
                violations.append(f"invalid SQL syntax in {tool_name}: {syntax_err}")
    return violations


def _extract_oscillation_target(tool_name: str, raw_arguments: Any) -> str:
    """Extract a normalized target table, file, or command for semantic oscillation detection."""
    args = _parse_tool_arguments(raw_arguments)
    if not args:
        return ""
    if tool_name == "edit":
        return _stale_tool_file_path(args)
    if tool_name in ("query_warehouse", "get_metric_lineage"):
        for key in _SQL_TARGET_KEYS:
            val = args.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        sql = args.get("sql") if isinstance(args.get("sql"), str) else args.get("query")
        if isinstance(sql, str) and sql.strip():
            tables = sorted({m.group(1).lower() for m in _SQL_TABLE_REF_RE.finditer(sql)})
            if tables:
                return ",".join(tables)
        return ""
    if tool_name == "bash":
        cmd = args.get("command") if isinstance(args.get("command"), str) else args.get("cmd")
        if isinstance(cmd, str) and cmd.strip():
            return cmd.strip().splitlines()[0][:80]
    return ""


def _extract_tool_error_prefix(content: str) -> str:
    """Return a normalized failing error prefix if ``content`` looks like a tool error."""
    stripped = content.strip()
    if not stripped:
        return ""
    if not _ERROR_RESULT_RE.search(stripped[:200]):
        return ""
    first_line = stripped.splitlines()[0].strip()
    return first_line[:80]


def detect_duplicate_tool_calls(messages: Iterable[Mapping[str, Any]] | None) -> list[str]:
    """H-5: detect identical repeated tool calls or semantic SQL/tool oscillation.

    Compares the last assistant turn carrying tool calls against the previous
    one for exact duplicates, and scans the last 6 assistant messages for
    semantic oscillation (>= 3 calls to ``query_warehouse``,
    ``get_metric_lineage``, ``bash``, or ``edit`` with an identical target
    table/file or identical failing result error prefix).
    """
    msg_list = [m for m in (messages or []) if isinstance(m, Mapping)]
    turns: list[list[tuple[str, str]]] = []
    tool_results_by_id: dict[str, str] = {}
    assistant_messages: list[Mapping[str, Any]] = []

    for message in msg_list:
        role = message.get("role")
        if role == "tool":
            call_id = message.get("tool_call_id")
            if isinstance(call_id, str) and call_id:
                tool_results_by_id[call_id] = _message_text(message)
            continue
        if role != "assistant":
            continue
        assistant_messages.append(message)
        tool_calls = message.get("tool_calls")
        if not (isinstance(tool_calls, list) and tool_calls):
            continue
        turn: list[tuple[str, str]] = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function") or {}
            if not isinstance(function, dict):
                function = {}
            function_name = function.get("name") or tool_call.get("name") or ""
            raw_arguments = function.get("arguments", "")
            if isinstance(raw_arguments, (dict, list)):
                canonical_arguments = json.dumps(raw_arguments, sort_keys=True)
            else:
                try:
                    canonical_arguments = json.dumps(json.loads(raw_arguments), sort_keys=True)
                except (TypeError, ValueError):
                    canonical_arguments = str(raw_arguments).strip()
            turn.append((function_name, canonical_arguments))
        if turn:
            turns.append(turn)

    if len(turns) >= 2:
        last_turn, previous_turn = turns[-1], turns[-2]
        if last_turn == previous_turn:
            return [
                f"duplicate tool call {last_turn[0][0]} repeated with identical arguments"
            ]
        for function_name, canonical_arguments in last_turn:
            if (function_name, canonical_arguments) in previous_turn:
                return [
                    f"duplicate tool call {function_name} repeated with identical arguments"
                ]

    target_counts: dict[tuple[str, str], int] = {}
    error_counts: dict[tuple[str, str], int] = {}
    for message in assistant_messages[-6:]:
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, Mapping):
                continue
            function = tool_call.get("function") or {}
            if not isinstance(function, Mapping):
                function = {}
            tool_name = str(function.get("name") or tool_call.get("name") or "")
            if tool_name not in _OSCILLATION_TOOLS:
                continue
            target = _extract_oscillation_target(tool_name, function.get("arguments", ""))
            if target:
                key = (tool_name, target)
                target_counts[key] = target_counts.get(key, 0) + 1
                if target_counts[key] >= 3:
                    return [
                        f"semantic tool oscillation: {tool_name} called {target_counts[key]} times "
                        f"targeting {target!r} in last 6 assistant turns"
                    ]
            call_id = tool_call.get("id")
            if isinstance(call_id, str) and call_id in tool_results_by_id:
                err_prefix = _extract_tool_error_prefix(tool_results_by_id[call_id])
                if err_prefix:
                    err_key = (tool_name, err_prefix)
                    error_counts[err_key] = error_counts.get(err_key, 0) + 1
                    if error_counts[err_key] >= 3:
                        return [
                            f"semantic tool oscillation: {tool_name} failed {error_counts[err_key]} times "
                            f"with {err_prefix!r} in last 6 assistant turns"
                        ]
    return []


_EDIT_MISMATCH_RE = re.compile(
    r"(?:oldstring not found|old_string not found|could not find oldstring|failed to match"
    r"|no exact match|found \d+ matches for oldstring|oldstring and newstring are identical)",
    re.IGNORECASE,
)
_TOOL_JSON_OR_ARGS_ERROR_RE = re.compile(
    r"(?:could not be parsed as json|invalid arguments for tool)",
    re.IGNORECASE,
)
_EDIT_HINT_MARKER = "[HARNESS_EDIT_HINT"
_EDIT_JSON_TRUNCATION_HINT = (
    "\n[HARNESS_EDIT_HINT: Tool call JSON arguments were malformed or truncated. "
    "Keep oldString/newString small (under 40 lines) and prefer targeted edit calls "
    "instead of rewriting whole files.]"
)
_EDIT_FAILURE_HINT = (
    "\n[HARNESS_EDIT_HINT: Exact target text did not match file contents. "
    "Do NOT retry the same edit blindly; call read with offset/limit around "
    "the target lines first, then apply a single accurate edit.]"
)


def inject_edit_failure_hints(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """H-17: append a recovery hint to trailing edit-mismatch tool results.

    Scans the latest trailing ``role == "tool"`` messages (or trailing
    ``tool_result`` content blocks) without mutating ``messages``. If a tool
    result matches an edit-target mismatch error and does not already contain
    ``[HARNESS_EDIT_HINT``, appends ``_EDIT_FAILURE_HINT`` and returns the
    new message list together with the number of hints injected.
    """
    if not isinstance(messages, list) or not messages:
        return (messages, 0)

    new_messages: list[dict[str, Any]] | None = None
    injected = 0

    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, dict):
            break
        role = message.get("role")
        if role == "tool":
            text = _message_text(message)
            hint = None
            if _EDIT_HINT_MARKER not in text:
                if _TOOL_JSON_OR_ARGS_ERROR_RE.search(text):
                    hint = _EDIT_JSON_TRUNCATION_HINT
                elif _EDIT_MISMATCH_RE.search(text):
                    hint = _EDIT_FAILURE_HINT
            if hint is not None:
                content = message.get("content")
                if isinstance(content, list):
                    updated_content: Any = [
                        *content,
                        {"type": "text", "text": hint},
                    ]
                elif isinstance(content, str):
                    updated_content = content + hint
                else:
                    updated_content = hint
                if new_messages is None:
                    new_messages = list(messages)
                new_messages[index] = {**message, "content": updated_content}
                injected += 1
            continue

        content = message.get("content")
        if (
            index == len(messages) - 1
            and isinstance(content, list)
            and any(
                isinstance(block, dict) and block.get("type") == "tool_result"
                for block in content
            )
        ):
            new_content: list[Any] | None = None
            for block_index, block in enumerate(content):
                if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                    continue
                block_text = _message_text({"content": block.get("content")})
                hint = None
                if _EDIT_HINT_MARKER not in block_text:
                    if _TOOL_JSON_OR_ARGS_ERROR_RE.search(block_text):
                        hint = _EDIT_JSON_TRUNCATION_HINT
                    elif _EDIT_MISMATCH_RE.search(block_text):
                        hint = _EDIT_FAILURE_HINT
                if hint is not None:
                    block_content = block.get("content")
                    if isinstance(block_content, list):
                        updated_block_content: Any = [
                            *block_content,
                            {"type": "text", "text": hint},
                        ]
                    elif isinstance(block_content, str):
                        updated_block_content = block_content + hint
                    else:
                        updated_block_content = hint
                    if new_content is None:
                        new_content = list(content)
                    new_content[block_index] = {
                        **block,
                        "content": updated_block_content,
                    }
                    injected += 1
            if new_content is not None:
                if new_messages is None:
                    new_messages = list(messages)
                new_messages[index] = {**message, "content": new_content}
        break

    if new_messages is None:
        return (messages, 0)
    return (new_messages, injected)


_READ_HINT_MARKER = "[HARNESS_READ_HINT"
_READ_PERMISSION_HINT = (
    "\n[HARNESS_READ_HINT: Permission denied — that path is outside this task's "
    "allowed files. Do NOT retry reading it or run find/glob; work strictly with "
    "the allowed task files.]"
)
_READ_BUDGET_MARKER = "[HARNESS_READ_BUDGET_WARNING"
_READ_BUDGET_WARNING = (
    "\n[HARNESS_READ_BUDGET_WARNING: You have performed 3+ consecutive file reads "
    "without applying an edit. Stop re-reading and apply your edit now, or the run "
    "will be terminated by repeated_file_read_without_write.]"
)
_READ_TOOLS = {"read"}
_READ_RESET_TOOLS = {"edit", "write", "apply_patch"}


def _append_tool_text(message: dict[str, Any], text: str) -> dict[str, Any]:
    """Return a copy of a tool message with ``text`` appended to its content."""
    content = message.get("content")
    if isinstance(content, list):
        return {**message, "content": [*content, {"type": "text", "text": text}]}
    if isinstance(content, str):
        return {**message, "content": content + text}
    return {**message, "content": text}


def _append_tool_result_text(block: dict[str, Any], text: str) -> dict[str, Any]:
    """Return a copy of a ``tool_result`` block with ``text`` appended."""
    block_content = block.get("content")
    if isinstance(block_content, list):
        return {**block, "content": [*block_content, {"type": "text", "text": text}]}
    if isinstance(block_content, str):
        return {**block, "content": block_content + text}
    return {**block, "content": text}


def inject_read_guard_hints(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """H-21: append recovery hints for read loops and permission-denied reads.

    Pure function: scans without mutating ``messages``. Appends
    ``_READ_PERMISSION_HINT`` to trailing tool results containing
    ``Permission denied: read``, and ``_READ_BUDGET_WARNING`` to the last tool
    result when >= 3 consecutive ``read`` tool calls appear without an
    ``edit`` / ``write`` / ``apply_patch``. Idempotent via the
    ``[HARNESS_READ_HINT`` / ``[HARNESS_READ_BUDGET_WARNING`` markers.
    """
    if not isinstance(messages, list) or not messages:
        return (messages, 0)

    new_messages: list[dict[str, Any]] | None = None
    injected = 0

    def touch(index: int, message: dict[str, Any]) -> None:
        nonlocal new_messages
        if new_messages is None:
            new_messages = list(messages)
        new_messages[index] = message

    # a) Permission-denied read results (trailing ``tool`` messages, or
    #    trailing ``tool_result`` blocks in the last message).
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, dict):
            break
        if message.get("role") == "tool":
            text = _message_text(message)
            if "Permission denied: read" in text and _READ_HINT_MARKER not in text:
                touch(index, _append_tool_text(message, _READ_PERMISSION_HINT))
                injected += 1
            continue
        content = message.get("content")
        if (
            index == len(messages) - 1
            and isinstance(content, list)
            and any(
                isinstance(block, dict) and block.get("type") == "tool_result"
                for block in content
            )
        ):
            new_content: list[Any] | None = None
            for block_index, block in enumerate(content):
                if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                    continue
                block_text = _message_text({"content": block.get("content")})
                if (
                    "Permission denied: read" in block_text
                    and _READ_HINT_MARKER not in block_text
                ):
                    if new_content is None:
                        new_content = list(content)
                    new_content[block_index] = _append_tool_result_text(
                        block, _READ_PERMISSION_HINT
                    )
                    injected += 1
            if new_content is not None:
                touch(index, {**message, "content": new_content})
        break

    current = new_messages if new_messages is not None else messages

    # b) Count consecutive ``read`` calls across the whole conversation; any
    #    ``edit`` / ``write`` / ``apply_patch`` resets the streak.
    consecutive_reads = 0
    for message in current:
        if not isinstance(message, dict):
            continue
        names: list[str] = []
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function")
                function_name = (
                    function.get("name") if isinstance(function, dict) else None
                )
                names.append(str(function_name or tool_call.get("name") or ""))
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    names.append(str(block.get("name") or ""))
        for name in names:
            if name in _READ_RESET_TOOLS:
                consecutive_reads = 0
            elif name in _READ_TOOLS:
                consecutive_reads += 1

    if consecutive_reads >= 3:
        target_index: int | None = None
        for index in range(len(current) - 1, -1, -1):
            message = current[index]
            if isinstance(message, dict) and message.get("role") == "tool":
                target_index = index
                break
        if target_index is not None:
            message = current[target_index]
            if _READ_BUDGET_MARKER not in _message_text(message):
                touch(target_index, _append_tool_text(message, _READ_BUDGET_WARNING))
                injected += 1
        else:
            last = current[-1] if current else None
            if isinstance(last, dict) and isinstance(last.get("content"), list):
                content = last["content"]
                new_content = None
                for block_index in range(len(content) - 1, -1, -1):
                    block = content[block_index]
                    if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                        continue
                    block_text = _message_text({"content": block.get("content")})
                    if _READ_BUDGET_MARKER not in block_text:
                        if new_content is None:
                            new_content = list(content)
                        new_content[block_index] = _append_tool_result_text(
                            block, _READ_BUDGET_WARNING
                        )
                        injected += 1
                    break
                if new_content is not None:
                    touch(len(current) - 1, {**last, "content": new_content})

    if new_messages is None:
        return (messages, 0)
    return (new_messages, injected)


def _sql_skeleton_lines(text: str) -> list[str]:
    """Summarize a SQL script as ``L<start>-L<end>`` CTE and final-SELECT ranges.

    Tracks parenthesis depth while skipping ``--`` line comments, ``/* */``
    block comments, and single-quoted strings, so each top-level ``<name> AS
    (...)`` body is attributed to the CTE that opens it. Indented ranges are
    returned with two leading spaces, matching :func:`build_file_skeletons`.
    """
    lines: list[str] = []
    depth = 0
    in_block_comment = False
    in_string = False
    starts: list[tuple[int, str]] = []
    final_select: int | None = None
    last_lineno = 0
    for lineno, raw in enumerate(text.splitlines(), 1):
        last_lineno = lineno
        stripped = raw.strip()
        cursor = 0
        while cursor < len(raw):
            char = raw[cursor]
            if in_block_comment:
                if raw[cursor:cursor + 2] == "*/":
                    in_block_comment = False
                    cursor += 2
                    continue
                cursor += 1
                continue
            if in_string:
                if char == "'":
                    in_string = False
                cursor += 1
                continue
            if raw[cursor:cursor + 2] == "--":
                break
            if raw[cursor:cursor + 2] == "/*":
                in_block_comment = True
                cursor += 2
                continue
            if char == "'":
                in_string = True
                cursor += 1
                continue
            if char == "(":
                if depth == 0:
                    match = re.search(
                        r"(?:\bWITH(?:\s+RECURSIVE)?|,)\s*([A-Za-z_][\w$]*)\s+AS\s*$",
                        raw[:cursor],
                        re.IGNORECASE,
                    )
                    if match:
                        starts.append((lineno, match.group(1)))
                depth += 1
            elif char == ")":
                depth -= 1
                if starts and depth == 0:
                    lines.append(f"  L{starts[-1][0]}-L{lineno}: CTE {starts[-1][1]}")
                    starts.pop()
            cursor += 1
        if (
            depth == 0
            and not in_block_comment
            and not in_string
            and final_select is None
            and re.match(r"SELECT\b", stripped, re.IGNORECASE)
        ):
            final_select = lineno
    if final_select is not None:
        lines.append(f"  L{final_select}-L{last_lineno}: final SELECT")
    return lines


def build_file_skeletons(files: Iterable[str], *, line_threshold: int = 250) -> str:
    """Build a compact line-range skeleton map for files longer than ``line_threshold``.

    ``.py`` files are summarized by top-level classes/functions with their line
    spans and up to five argument names, plus one level of nested methods;
    ``.sql`` files by CTE and final-SELECT ranges via :func:`_sql_skeleton_lines`.
    Shorter files, non-matching suffixes, and unreadable files are skipped. The
    return value is a prompt-ready string (empty when nothing qualified) that
    tells the reader to open only the listed line ranges.
    """
    chunks: list[str] = []
    for path in files:
        file_path = Path(path)
        try:
            if not file_path.is_file():
                continue
            text = file_path.read_text()
        except OSError:
            continue
        line_count = len(text.splitlines())
        if line_count <= line_threshold:
            continue
        lines: list[str] = []
        if file_path.suffix == ".py":
            try:
                tree = ast.parse(text)
            except (SyntaxError, OSError):
                continue
            for node in tree.body:
                if isinstance(node, _DEF_NODES):
                    kind = "class" if isinstance(node, ast.ClassDef) else "def"
                    argument_names = (
                        ", ".join(argument.arg for argument in node.args.args[:5])
                        if kind == "def"
                        else ""
                    )
                    lines.append(
                        f"L{node.lineno}-L{node.end_lineno}: {kind} {node.name}"
                        f"({argument_names})"
                    )
                    if isinstance(node, ast.ClassDef):
                        for sub_node in node.body:
                            if isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                sub_args = ", ".join(
                                    argument.arg for argument in sub_node.args.args[:5]
                                )
                                lines.append(
                                    f"  L{sub_node.lineno}-L{sub_node.end_lineno}: "
                                    f"def {sub_node.name}({sub_args})"
                                )
        elif file_path.suffix == ".sql":
            lines = _sql_skeleton_lines(text)
        if lines:
            chunks.append(f"--- {path} ({line_count} lines)\n" + "\n".join(lines))
    if not chunks:
        return ""
    return "[FILE SKELETON MAP -- read only target line ranges]\n" + "\n".join(chunks) + "\n"


def extract_compact_test_failure(stdout: str, stderr: str, max_lines: int = 25) -> str:
    """Compact, bounded digest of a test failure for a micro-repair prompt.

    Scans ``stdout`` + ``stderr`` and keeps ``FAILED`` summary lines, ``E   ``
    assertion lines, import/syntax error lines, and the last traceback file
    location (only the final one is kept). If nothing matched, falls back to the
    tail of the non-empty output capped at ``min(max_lines, 10)`` lines.
    """
    lines = stdout.splitlines() + stderr.splitlines()
    file_pattern = re.compile(r'File "[^"]+", line \d+')
    last_file_index: int | None = None
    picked: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        if file_pattern.search(line):
            last_file_index = index  # Only the last traceback location is kept.
            continue
        if (
            "FAILED" in line
            or line.lstrip().startswith("E   ")
            or "SyntaxError" in line
            or "ImportError" in line
            or "ModuleNotFoundError" in line
        ):
            picked.append((index, line))
    if last_file_index is not None:
        picked.append((last_file_index, lines[last_file_index]))
    picked.sort()
    if picked:
        return "\n".join(line for _, line in picked[:max_lines])
    non_empty = [line for line in lines if line.strip()]
    return "\n".join(non_empty[-min(max_lines, 10) :])


def _stale_tool_file_path(arguments: Any) -> str:
    """Extract the first file path from tool-call arguments (JSON or dict)."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            arguments = None
    if not isinstance(arguments, dict):
        return ""
    for key in _STALE_PATH_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _stale_tool_calls(message: Mapping[str, Any]) -> Iterable[tuple[str, str, str]]:
    """Yield ``(tool_call_id, tool_name, file_path)`` for calls in ``message``.

    Covers OpenAI-style ``tool_calls`` entries and Anthropic-style ``tool_use``
    content blocks.
    """
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            function = function if isinstance(function, dict) else {}
            name = function.get("name") or call.get("name") or ""
            yield str(call.get("id") or ""), str(name), _stale_tool_file_path(
                function.get("arguments")
            )
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                yield str(block.get("id") or ""), str(block.get("name") or ""), _stale_tool_file_path(
                    block.get("input")
                )


def compact_stale_reads(
    messages: list[dict],
    *,
    turn_index: int = 1,
    min_turn: int = 5,
    char_threshold: int = 400,
) -> tuple[list[dict], int]:
    """H-8: drop long read outputs that a later edit made stale.

    Maps ``tool_call_id`` to ``(tool_name, file_path)`` across assistant
    ``tool_calls`` (arguments inspected for ``path``-style keys), records the
    last message index at which each file was touched by an edit/write/patch
    tool, and replaces every earlier read-tool result on that path whose text
    exceeds ``char_threshold`` with a compact placeholder. Returns the new
    (shallow-copied) message list and the number of results compacted; before
    ``min_turn`` or for non-list input the messages pass through untouched.
    """
    if turn_index < min_turn or not isinstance(messages, list):
        return (messages, 0)
    call_map: dict[str, tuple[str, str]] = {}
    last_edit_idx: dict[str, int] = {}
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        for call_id, tool_name, file_path in _stale_tool_calls(message):
            if not call_id:
                continue
            call_map[call_id] = (tool_name.lower(), file_path)
            lowered = tool_name.lower()
            if file_path and any(marker in lowered for marker in _STALE_EDIT_MARKERS):
                last_edit_idx[file_path] = index
    if not last_edit_idx:
        return (messages, 0)
    new_messages = list(messages)
    compacted = 0

    def is_stale_read(call_id: Any, index: int) -> tuple[str, str] | None:
        mapped = call_map.get(call_id) if isinstance(call_id, str) else None
        if mapped is None:
            return None
        tool_name, file_path = mapped
        if file_path not in last_edit_idx or index >= last_edit_idx[file_path]:
            return None
        if not any(marker in tool_name for marker in _STALE_READ_MARKERS):
            return None
        return file_path

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "tool":
            stale = is_stale_read(message.get("tool_call_id"), index)
            if stale is not None and len(_message_text(message)) > char_threshold:
                new_messages[index] = {
                    **message,
                    "content": _STALE_READ_NOTICE.format(path=stale),
                }
                compacted += 1
        content = message.get("content")
        if not isinstance(content, list):
            continue
        new_content: list[Any] | None = None
        for block_index, block in enumerate(content):
            if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                continue
            stale = is_stale_read(block.get("tool_use_id"), index)
            if stale is None or len(_message_text({"content": block.get("content")})) <= char_threshold:
                continue
            if new_content is None:
                new_content = list(content)
            new_content[block_index] = {
                **block,
                "content": _STALE_READ_NOTICE.format(path=stale),
            }
            compacted += 1
        if new_content is not None:
            new_messages[index] = {**message, "content": new_content}
    return (new_messages, compacted)


_CONTEXT_BUDGET_NOTICE = (
    "\n...[HARNESS CONTEXT BUDGET: truncated {omitted_lines} lines / "
    "{omitted_chars} chars to fit {max_context_tokens} token budget]...\n"
)


def estimate_messages_tokens(messages: Iterable[Mapping[str, Any]] | None) -> int:
    """Fast deterministic token estimate across message ``content`` and ``tool_calls``."""
    if not messages:
        return 1
    total_chars = 0
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if isinstance(content, str):
            total_chars += len(content)
        elif content is not None:
            total_chars += len(json.dumps(content, ensure_ascii=False, sort_keys=True, default=str))
        tool_calls = message.get("tool_calls")
        if tool_calls:
            total_chars += len(json.dumps(tool_calls, ensure_ascii=False, sort_keys=True, default=str))
    return max(1, total_chars // 4)


def _truncate_tool_text(
    text: str,
    *,
    head_lines: int,
    tail_lines: int,
    max_context_tokens: int,
) -> tuple[str, bool]:
    """Truncate the middle of a large tool output while keeping head and tail."""
    lines = text.splitlines()
    if len(lines) <= head_lines + tail_lines + 4 and len(text) <= 1200:
        return text, False
    if len(lines) > head_lines + tail_lines:
        head = "\n".join(lines[:head_lines])
        tail = "\n".join(lines[-tail_lines:]) if tail_lines > 0 else ""
        omitted_lines = len(lines) - head_lines - tail_lines
        omitted_chars = max(0, len(text) - len(head) - len(tail))
    else:
        head_chars = 400
        tail_chars = 200
        head = text[:head_chars]
        tail = text[-tail_chars:] if len(text) > head_chars + tail_chars else ""
        omitted_lines = 0
        omitted_chars = max(0, len(text) - len(head) - len(tail))
    marker = _CONTEXT_BUDGET_NOTICE.format(
        omitted_lines=omitted_lines,
        omitted_chars=omitted_chars,
        max_context_tokens=max_context_tokens,
    )
    return f"{head}{marker}{tail}", True


def compact_context_window(
    messages: list[Mapping[str, Any]],
    *,
    max_context_tokens: int = 6000,
    keep_recent_messages: int = 4,
    head_lines: int = 12,
    tail_lines: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Compact older tool outputs so estimated tokens fit within ``max_context_tokens``."""
    tokens_before = estimate_messages_tokens(messages)
    working: list[dict[str, Any]] = [
        dict(m) if isinstance(m, Mapping) else m for m in (messages or [])
    ]
    if len(working) >= 5:
        working, _ = compact_stale_reads(working, turn_index=len(working))

    current_tokens = estimate_messages_tokens(working)
    if current_tokens <= max_context_tokens:
        return working, {
            "tokens_before": tokens_before,
            "tokens_after": current_tokens,
            "truncated_messages": 0,
        }

    start_idx = 1 if (working and isinstance(working[0], Mapping) and working[0].get("role") == "system") else 0
    end_idx = len(working) - max(0, keep_recent_messages)
    truncated_messages = 0

    for idx in range(start_idx, max(start_idx, end_idx)):
        if current_tokens <= max_context_tokens:
            break
        msg = working[idx]
        if not isinstance(msg, Mapping):
            continue
        role = msg.get("role")
        content = msg.get("content")
        msg_modified = False

        if role == "tool":
            if isinstance(content, str):
                new_text, changed = _truncate_tool_text(
                    content,
                    head_lines=head_lines,
                    tail_lines=tail_lines,
                    max_context_tokens=max_context_tokens,
                )
                if changed:
                    working[idx] = {**msg, "content": new_text}
                    msg_modified = True
            elif isinstance(content, list):
                new_blocks = list(content)
                for b_idx, block in enumerate(new_blocks):
                    if isinstance(block, str):
                        new_text, changed = _truncate_tool_text(
                            block,
                            head_lines=head_lines,
                            tail_lines=tail_lines,
                            max_context_tokens=max_context_tokens,
                        )
                        if changed:
                            new_blocks[b_idx] = new_text
                            msg_modified = True
                    elif isinstance(block, Mapping) and isinstance(block.get("text"), str):
                        new_text, changed = _truncate_tool_text(
                            block["text"],
                            head_lines=head_lines,
                            tail_lines=tail_lines,
                            max_context_tokens=max_context_tokens,
                        )
                        if changed:
                            new_blocks[b_idx] = {**block, "text": new_text}
                            msg_modified = True
                if msg_modified:
                    working[idx] = {**msg, "content": new_blocks}
        elif role in ("user", "assistant") and isinstance(content, list):
            new_blocks = list(content)
            for b_idx, block in enumerate(new_blocks):
                if not (isinstance(block, Mapping) and block.get("type") == "tool_result"):
                    continue
                block_content = block.get("content")
                if isinstance(block_content, str):
                    new_text, changed = _truncate_tool_text(
                        block_content,
                        head_lines=head_lines,
                        tail_lines=tail_lines,
                        max_context_tokens=max_context_tokens,
                    )
                    if changed:
                        new_blocks[b_idx] = {**block, "content": new_text}
                        msg_modified = True
                elif isinstance(block_content, list):
                    inner_blocks = list(block_content)
                    inner_changed = False
                    for ib_idx, ib in enumerate(inner_blocks):
                        if isinstance(ib, Mapping) and isinstance(ib.get("text"), str):
                            new_text, changed = _truncate_tool_text(
                                ib["text"],
                                head_lines=head_lines,
                                tail_lines=tail_lines,
                                max_context_tokens=max_context_tokens,
                            )
                            if changed:
                                inner_blocks[ib_idx] = {**ib, "text": new_text}
                                inner_changed = True
                    if inner_changed:
                        new_blocks[b_idx] = {**block, "content": inner_blocks}
                        msg_modified = True
            if msg_modified:
                working[idx] = {**msg, "content": new_blocks}

        if msg_modified:
            truncated_messages += 1
            current_tokens = estimate_messages_tokens(working)

    return working, {
        "tokens_before": tokens_before,
        "tokens_after": current_tokens,
        "truncated_messages": truncated_messages,
    }
