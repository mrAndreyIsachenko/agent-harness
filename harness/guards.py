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
import difflib
import json
import keyword
import re
from bisect import bisect_right
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "FORBIDDEN_EGRESS_KEYS",
    "build_cross_file_symbol_context",
    "build_file_skeletons",
    "check_allowed_files_syntax",
    "check_cot_and_slop_integrity",
    "check_egress_payload",
    "check_readonly_sql_tool_calls",
    "check_sql_syntax",
    "collect_python_top_symbols",
    "compact_context_window",
    "compact_stale_reads",
    "detect_duplicate_tool_calls",
    "detect_edit_oscillation",
    "extract_failing_test_excerpts",
    "estimate_messages_tokens",
    "extract_compact_test_failure",
    "inject_edit_failure_hints",
    "prune_noisy_tool_outputs",
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
    (
        "PRIVATE KEY-----",
        re.compile(
            r"-----BEGIN(?: [A-Z0-9]+){0,3} PRIVATE KEY-----(?:\\n|\r?\n)[A-Za-z0-9+/=]{32,}"
        ),
    ),
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
    file_list = list(files) if files else []
    errors: list[str] = []
    for path in file_list:
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
                    errors.extend(_sibling_import_errors(path, tree, known_files=file_list))
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


def _resolve_importfrom_target(
    path: str, node: ast.ImportFrom, known_files: Iterable[str] | None
) -> Path | None:
    """Resolve the on-disk ``.py`` file an ``ImportFrom`` node points at.

    Relative imports (``node.level >= 1``) climb ``node.level - 1`` parent
    directories from the importing file's directory; absolute imports keep the
    direct sibling first, then match against ``known_files`` path-part suffixes,
    then search up to three ancestor directories. Returns ``None`` when nothing
    exists on disk (unknown module -- not this guard's business).
    """
    base = Path(path).parent
    if node.level >= 1:
        levels_up = node.level - 1
        if levels_up > len(base.parents):
            return None
        target_dir = base.parents[levels_up - 1] if levels_up else base
        if not node.module:
            init = target_dir / "__init__.py"
            return init if init.is_file() else None
        parts = node.module.split(".")
        for candidate in (
            target_dir.joinpath(*parts).with_suffix(".py"),
            target_dir.joinpath(*parts, "__init__.py"),
        ):
            if candidate.is_file():
                return candidate
        return None

    module = node.module or ""
    if not module:
        return None
    parts = module.split(".")
    for candidate in (
        base.joinpath(*parts).with_suffix(".py"),
        base.joinpath(*parts, "__init__.py"),
    ):
        if candidate.is_file():
            return candidate
    rel_py = Path(*parts).with_suffix(".py")
    rel_init = Path(*parts) / "__init__.py"
    for known in known_files or []:
        known_path = Path(known)
        if not known_path.is_file():
            continue
        for rel in (rel_py, rel_init):
            if len(rel.parts) <= len(known_path.parts) and (
                known_path.parts[-len(rel.parts):] == rel.parts
            ):
                return known_path
    for ancestor in list(base.parents)[:3]:
        for candidate in (ancestor / rel_py, ancestor / rel_init):
            if candidate.is_file():
                return candidate
    return None


def _tree_has_dynamic_names(tree: ast.Module) -> bool:
    """True when ``tree`` uses ``globals()`` or a wildcard import."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "globals":
            return True
        if isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
            return True
    return False


def _sibling_import_errors(
    path: str, tree: ast.Module, known_files: Iterable[str] | None = None
) -> list[str]:
    """H-7/H-46: flag ``from <module> import <missing>`` without executing anything.

    For each ``ImportFrom`` whose target module file resolves on disk (simple
    sibling, relative, dotted package, or known-file match) and parses cleanly,
    every non-``*`` imported alias must match a name collected by
    :func:`_sibling_defined_names`. Targets that use ``globals()`` or wildcard
    imports are skipped to avoid false positives, as are unparseable or missing
    targets.
    """
    errors: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        target_file = _resolve_importfrom_target(path, node, known_files)
        if target_file is None:
            continue
        try:
            target_tree = ast.parse(
                target_file.read_text(encoding="utf-8"), filename=str(target_file)
            )
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        if _tree_has_dynamic_names(target_tree):
            continue
        known = _sibling_defined_names(target_tree)
        module_label = node.module or ("." * node.level)
        for alias in node.names:
            if alias.name == "*" or alias.name in known:
                continue
            if target_file.name == "__init__.py":
                pkg_dir = target_file.parent
                if (pkg_dir / f"{alias.name}.py").is_file() or (
                    pkg_dir / alias.name / "__init__.py"
                ).is_file():
                    continue
            errors.append(
                f"{path}:{node.lineno}: cannot import name {alias.name!r} from {module_label!r}"
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


# H-32: edit tools whose arguments carry (file, oldString, newString) tuples.
_EDIT_OSCILLATION_TOOLS = {'edit', 'replace', 'multi_edit', 'patch'}
_EDIT_FILE_KEYS = ('file_path', 'filePath', 'path', 'file')
_EDIT_OLD_KEYS = ('old_string', 'oldString', 'old')
_EDIT_NEW_KEYS = ('new_string', 'newString', 'new')


def _extract_edit_call(tool_name: str, raw_arguments: Any) -> tuple[str, str, str] | None:
    """Return a normalized (filePath, oldString, newString) tuple for an edit call."""
    if tool_name not in _EDIT_OSCILLATION_TOOLS:
        return None
    if isinstance(raw_arguments, (dict, list)):
        arguments = raw_arguments
    else:
        try:
            arguments = json.loads(raw_arguments)
        except (TypeError, ValueError):
            return None
    if not isinstance(arguments, Mapping):
        return None

    def _first(keys: tuple[str, ...]) -> str:
        for key in keys:
            value = arguments.get(key)
            if isinstance(value, str):
                return value
        return ''

    file_path = _first(_EDIT_FILE_KEYS)
    if not file_path:
        return None
    return file_path, _first(_EDIT_OLD_KEYS), _first(_EDIT_NEW_KEYS)


def detect_edit_oscillation(messages: Iterable[Mapping[str, Any]] | None) -> dict[str, str] | None:
    """H-32: detect edit oscillation / untested edit plateaus.

    Scans assistant ``tool_calls`` for edit tools and returns a dict when
    either:
    - ``revert_cycle``: the latest edit on a file exactly reverses an earlier
      edit on the same file (``latest.oldString == prior.newString`` and
      ``latest.newString == prior.oldString``, with non-empty strings), or
    - ``untested_edit_streak``: the same file was edited >= 4 times without any
      intervening ``bash``/``test`` tool call.

    Returns ``None`` when no oscillation is detected.
    """
    edit_calls: list[tuple[str, str, str]] = []
    streaks: dict[str, int] = {}
    for message in (messages or []):
        if not isinstance(message, Mapping):
            continue
        if message.get('role') != 'assistant':
            continue
        tool_calls = message.get('tool_calls')
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, Mapping):
                continue
            function = tool_call.get('function') or {}
            if not isinstance(function, Mapping):
                function = {}
            tool_name = str(function.get('name') or tool_call.get('name') or '')
            if tool_name in ('bash', 'test'):
                streaks.clear()
            call = _extract_edit_call(tool_name, function.get('arguments', ''))
            if call:
                edit_calls.append(call)
                file_path = call[0]
                streaks[file_path] = streaks.get(file_path, 0) + 1
                if streaks[file_path] >= 4:
                    return {'oscillating': True, 'file': file_path,
                            'reason': 'untested_edit_streak'}

    if len(edit_calls) >= 2:
        latest_file, latest_old, latest_new = edit_calls[-1]
        for prior_file, prior_old, prior_new in edit_calls[:-1]:
            if prior_file != latest_file:
                continue
            if (latest_old and latest_new and latest_old == prior_new
                    and latest_new == prior_old):
                return {'oscillating': True, 'file': latest_file, 'reason': 'revert_cycle'}
    return None


_EDIT_MISMATCH_RE = re.compile(
    r"(?:oldstring not found|old_string not found|could not find oldstring|failed to match"
    r"|no exact match|found \d+ matches for oldstring|oldstring and newstring are identical)",
    re.IGNORECASE,
)
_TOOL_JSON_OR_ARGS_ERROR_RE = re.compile(
    r"(?:could not be parsed as json|invalid arguments for tool)",
    re.IGNORECASE,
)
_EDIT_NOOP_RE = re.compile(
    r"oldstring and newstring are identical",
    re.IGNORECASE,
)
_EDIT_AMBIGUOUS_RE = re.compile(
    r"found \d+ matches for oldstring",
    re.IGNORECASE,
)
_EDIT_UNKNOWN_TOOL_RE = re.compile(
    r"no tool named \"(?:execute|run_command|shell|cmd)\" is currently available",
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
_EDIT_NOOP_HINT = (
    "\n[HARNESS_EDIT_HINT: Target change is already present on disk "
    "(oldString == newString). Do NOT repeat this edit; run verification "
    "or finish your response.]"
)
_EDIT_UNKNOWN_TOOL_HINT = (
    "\n[HARNESS_EDIT_HINT: Unknown tool name. Use \"bash\" for shell/pytest "
    "commands, \"read\" to inspect files, and \"edit\" to modify files.]"
)
_EDIT_AMBIGUOUS_HINT = (
    "\n[HARNESS_EDIT_HINT: Multiple matches found. Include 3-5 unique "
    "surrounding context lines in oldString/newString to disambiguate without "
    "re-reading the whole file.]"
)
_EDIT_AMBIGUOUS_LINES_HINT = (
    "\n[HARNESS_EDIT_HINT: Multiple matches found (at lines {lines}). "
    "Include 3-5 unique surrounding context lines in oldString/newString to "
    "disambiguate without re-reading the whole file.]"
)
_EDIT_AMBIGUOUS_MAX_OCCURRENCES = 5


def _find_edit_call_for_result(
    messages: list[dict[str, Any]],
    index: int,
    expected_id: str | None = None,
) -> tuple[str, str, str] | None:
    """Locate the edit tool call that produced the tool result at ``index``.

    Scans assistant messages before ``index`` for OpenAI ``tool_calls`` or
    Anthropic ``tool_use`` blocks. When ``expected_id`` (a ``tool_call_id`` or
    ``tool_use_id``) is given, the matching call wins; otherwise the latest
    edit call found is returned as a fallback.
    """
    fallback: tuple[str, str, str] | None = None
    for message in messages[:index]:
        if not isinstance(message, dict) or message.get('role') != 'assistant':
            continue
        content = message.get('content')
        if isinstance(content, list):
            for block in content:
                if not (isinstance(block, dict) and block.get('type') == 'tool_use'):
                    continue
                call = _extract_edit_call(
                    str(block.get('name') or ''), block.get('input', ''))
                if call:
                    if expected_id and block.get('id') == expected_id:
                        return call
                    fallback = call
        tool_calls = message.get('tool_calls')
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get('function') or {}
            if not isinstance(function, dict):
                function = {}
            name = str(function.get('name') or tool_call.get('name') or '')
            args = function.get('arguments', tool_call.get('input', ''))
            call = _extract_edit_call(name, args)
            if call:
                if expected_id and tool_call.get('id') == expected_id:
                    return call
                fallback = call
    return fallback


def _ambiguous_match_hint(call: tuple[str, str, str]) -> str:
    """Build the ambiguous-match hint, adding on-disk line numbers when known."""
    file_path, old_string = call[0], call[1]
    if file_path and old_string:
        try:
            text = Path(file_path).read_text(errors='replace')
        except OSError:
            text = ''
        if text and old_string in text:
            lines: list[int] = []
            start = 0
            for _ in range(_EDIT_AMBIGUOUS_MAX_OCCURRENCES):
                position = text.find(old_string, start)
                if position < 0:
                    break
                lines.append(text.count('\n', 0, position) + 1)
                start = position + max(len(old_string), 1)
            if lines:
                rendered = ', '.join('L%d' % number for number in lines)
                return _EDIT_AMBIGUOUS_LINES_HINT.format(lines=rendered)
    return _EDIT_AMBIGUOUS_HINT


def _fuzzy_mismatch_hint(call: tuple[str, str, str]) -> str:
    """H-40: fuzzy-locate the closest on-disk text for a failed oldString match."""
    file_path, old_string, new_string = call[0], call[1], call[2]
    if not file_path or not old_string:
        return _EDIT_FAILURE_HINT
    try:
        text = Path(file_path).read_text(errors='replace')
    except OSError:
        return _EDIT_FAILURE_HINT
    if not text:
        return _EDIT_FAILURE_HINT
    if new_string and new_string in text and old_string not in text:
        line = text.count('\n', 0, text.find(new_string)) + 1
        return (
            f"\n[HARNESS_EDIT_HINT: target oldString was not found because "
            f"newString is ALREADY applied in {file_path} at L{line}. "
            f"Do NOT repeat this edit; proceed to the next change or finish.]"
        )
    all_lines = text.splitlines()
    old_lines = old_string.splitlines()
    if not all_lines or not old_lines:
        return _EDIT_FAILURE_HINT
    old_norm_lines = [line.strip() for line in old_lines[:25]]
    old_block_norm = '\n'.join(old_norm_lines)
    win_len = min(len(old_norm_lines), len(all_lines), 25)
    if win_len <= 0:
        return _EDIT_FAILURE_HINT
    def _best_window(indices: Iterable[int], init_idx: int = 0, init_ratio: float = -1.0) -> tuple[int, float]:
        idx, top = init_idx, init_ratio
        for start in indices:
            window_norm = '\n'.join(
                line.strip() for line in all_lines[start:start + win_len])
            ratio = difflib.SequenceMatcher(
                None, old_block_norm, window_norm).ratio()
            if ratio > top:
                top = ratio
                idx = start
        return idx, top

    max_start = len(all_lines) - win_len
    if len(all_lines) > 2500:
        step = max(1, win_len // 3)
        coarse_index, best_ratio = _best_window(range(0, max_start + 1, step))
        refine_start = max(0, coarse_index - 3)
        refine_end = min(max_start, coarse_index + 3)
        best_index, best_ratio = _best_window(
            range(refine_start, refine_end + 1), coarse_index, best_ratio)
    else:
        best_index, best_ratio = _best_window(range(0, max_start + 1))
    if best_ratio < 0.55:
        return _EDIT_FAILURE_HINT
    disp_start = max(0, best_index - 3)
    disp_end = min(len(all_lines), best_index + win_len + 3)
    if disp_end - disp_start > 25:
        disp_end = disp_start + 25
    formatted = [
        f"{ln}: {all_lines[ln - 1]}"
        for ln in range(disp_start + 1, disp_end + 1)
    ]
    return (
        f"{_EDIT_FAILURE_HINT}\n"
        f"Closest on-disk lines in {file_path} "
        f"(L{disp_start + 1}-L{disp_end}):\n"
        + '\n'.join(formatted)
    )


def _select_edit_failure_hint(
    text: str,
    messages: list[dict[str, Any]],
    index: int,
    expected_id: str | None = None,
) -> str | None:
    """H-36: pick the specialized hint for a single edit/tool error result."""
    if _EDIT_HINT_MARKER in text:
        return None
    if _EDIT_NOOP_RE.search(text):
        return _EDIT_NOOP_HINT
    if _EDIT_AMBIGUOUS_RE.search(text):
        call = _find_edit_call_for_result(messages, index, expected_id)
        if call is not None:
            return _ambiguous_match_hint(call)
        return _EDIT_AMBIGUOUS_HINT
    if _EDIT_UNKNOWN_TOOL_RE.search(text):
        return _EDIT_UNKNOWN_TOOL_HINT
    if _TOOL_JSON_OR_ARGS_ERROR_RE.search(text):
        return _EDIT_JSON_TRUNCATION_HINT
    if _EDIT_MISMATCH_RE.search(text):
        call = _find_edit_call_for_result(messages, index, expected_id)
        if call is not None:
            return _fuzzy_mismatch_hint(call)
        return _EDIT_FAILURE_HINT
    return None


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
            hint = _select_edit_failure_hint(
                text, messages, index, message.get("tool_call_id"))
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
                hint = _select_edit_failure_hint(
                    block_text, messages, index, block.get("tool_use_id"))
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
_READ_GUARD_MARKER = "[HARNESS_READ_GUARD"
_READ_PRELOADED_PERMISSION_HINT = (
    "\n[HARNESS_READ_GUARD: This path is outside the tool sandbox, but its "
    "contents/skeleton are ALREADY preloaded in your initial prompt above. Do "
    "NOT call read on it again; proceed directly with your edit on allowed_files.]"
)
_READ_COMPLETE_PRELOADED_HINT = (
    "\n[HARNESS_READ_HINT: This complete file was already preloaded in your "
    "initial prompt. Do NOT call read on it again; proceed directly to edit.]"
)
_READ_TOOLS = {"read"}
_READ_RESET_TOOLS = {"edit", "write", "apply_patch"}


def _requested_read_path(messages: list[dict[str, Any]], result_text: str) -> str:
    """Resolve the path requested by a permission-denied ``read`` tool call.

    Looks up the matching ``tool_call`` (OpenAI) / ``tool_use`` (Anthropic) by
    id and extracts ``filePath`` / ``path`` / ``file_path`` from its
    arguments; falls back to the first path-like token in the error text.
    """
    ids: set[str] = set()
    for match in re.finditer(r'(?:tool_call_id|tool_use_id)["\s:]+([A-Za-z0-9_\-]+)', result_text):
        ids.add(match.group(1))
    for message in messages:
        if not isinstance(message, dict):
            continue
        candidates: list[tuple[str, Any]] = []
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function")
                if isinstance(function, dict):
                    candidates.append((str(tool_call.get("id") or ""), function.get("arguments")))
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    candidates.append((str(block.get("id") or ""), block.get("input")))
        for call_id, arguments in candidates:
            if call_id and call_id in ids:
                path = _path_from_tool_arguments(arguments)
                if path:
                    return path
    match = re.search(r"(/[^\s'\"]+|[\w.\-/]+\.\w+)", result_text)
    return match.group(1) if match else ""


def _path_from_tool_arguments(arguments: Any) -> str:
    raw: Any = arguments
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return ""
    if isinstance(raw, Mapping):
        for key in ("filePath", "path", "file_path"):
            value = raw.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def _first_user_message_text(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            return _message_text(message)
    return ""


_PRELOAD_SECTION_MARKERS = (
    "[PRELOADED_FILE_EXCERPTS",
    "[FILE_SKELETONS",
    "[FILE SKELETON MAP",
)


def _parse_json_str_list(raw: str) -> list[str]:
    """Parse a JSON array of non-empty strings, returning [] on mismatch."""
    try:
        parsed = json.loads(raw.strip())
    except (ValueError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, str) and item]


def _extract_allowed_read_paths_from_prompt(first_user_text: str) -> list[str]:
    """Extract allowed file paths from sandboxed prompt headers."""
    if not first_user_text:
        return []
    paths: list[str] = []
    seen: set[str] = set()

    def _add_all(items: list[str]) -> None:
        for item in items:
            if item not in seen:
                seen.add(item)
                paths.append(item)

    lines = first_user_text.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("Read/edit ONLY the listed files"):
            if ":" in stripped:
                after_colon = stripped.split(":", 1)[1].strip()
                if after_colon.startswith("["):
                    _add_all(_parse_json_str_list(after_colon))
                    continue
            for offset in range(1, 5):
                if idx + offset >= len(lines):
                    break
                candidate = lines[idx + offset].strip()
                if candidate.startswith("["):
                    parsed = _parse_json_str_list(candidate)
                    if parsed:
                        _add_all(parsed)
                        break
        elif stripped.startswith("Read ONLY these source files:"):
            _add_all(_parse_json_str_list(stripped.split(":", 1)[1]))
        elif stripped.startswith("Read-only reference files (do NOT edit):"):
            _add_all(_parse_json_str_list(stripped.split(":", 1)[1]))
    return paths


def _preloaded_hint_for(
    messages: list[dict[str, Any]],
    result_text: str,
    block: dict[str, Any] | None = None,
) -> str:
    """Choose the permission-denied hint based on whether the path was preloaded."""
    path = _requested_read_path(messages, result_text)
    if not path and isinstance(block, dict):
        # Anthropic ``tool_result`` blocks carry the call id on the block, not
        # in the error text; re-resolve via the id.
        call_id = str(block.get("tool_use_id") or block.get("tool_call_id") or "")
        if call_id:
            path = _requested_read_path(messages, f"tool_use_id: {call_id}")
    first_user = _first_user_message_text(messages)
    if (
        path
        and first_user
        and any(marker in first_user for marker in _PRELOAD_SECTION_MARKERS)
        and path in first_user
    ):
        return _READ_PRELOADED_PERMISSION_HINT
    allowed_paths = _extract_allowed_read_paths_from_prompt(first_user)
    if allowed_paths:
        target = path or "<unknown>"
        allowed_str = ", ".join(allowed_paths)
        return (
            f"\n[HARNESS_READ_HINT: Permission denied for '{target}'. "
            f"You ONLY have read permission for: {allowed_str}. "
            "Do NOT call read on any other path; apply your edit directly to the allowed files.]"
        )
    return _READ_PERMISSION_HINT


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


def _is_preloaded_complete_read(
    messages: list[dict[str, Any]],
    call_id: str,
) -> bool:
    """Return True if ``call_id`` is a ``read`` for a COMPLETE FILE PRELOADED path."""
    if not call_id:
        return False
    first_user = _first_user_message_text(messages)
    if "COMPLETE FILE PRELOADED" not in first_user:
        return False
    requested_path = ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict) or str(tool_call.get("id") or "") != call_id:
                    continue
                function = tool_call.get("function")
                name = str((function.get("name") if isinstance(function, dict) else None) or tool_call.get("name") or "")
                if name not in _READ_TOOLS:
                    return False
                args = function.get("arguments") if isinstance(function, dict) else None
                requested_path = _path_from_tool_arguments(args)
                break
        if requested_path:
            break
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if not (isinstance(block, dict) and block.get("type") == "tool_use"):
                    continue
                if str(block.get("id") or "") != call_id:
                    continue
                if str(block.get("name") or "") not in _READ_TOOLS:
                    return False
                requested_path = _path_from_tool_arguments(block.get("input"))
                break
        if requested_path:
            break
    if not requested_path:
        return False
    prefix = f"--- {requested_path} ("
    return any(
        line.startswith(prefix) and "COMPLETE FILE PRELOADED" in line
        for line in first_user.splitlines()
    )


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
            if (
                "Permission denied: read" in text
                and _READ_HINT_MARKER not in text
                and _READ_GUARD_MARKER not in text
            ):
                touch(index, _append_tool_text(
                    message, _preloaded_hint_for(messages, text, block=message)))
                injected += 1
            elif (
                _READ_HINT_MARKER not in text
                and _READ_GUARD_MARKER not in text
                and _is_preloaded_complete_read(messages, str(message.get("tool_call_id") or ""))
            ):
                touch(index, _append_tool_text(message, _READ_COMPLETE_PRELOADED_HINT))
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
                    and _READ_GUARD_MARKER not in block_text
                ):
                    if new_content is None:
                        new_content = list(content)
                    new_content[block_index] = _append_tool_result_text(
                        block, _preloaded_hint_for(messages, block_text, block=block)
                    )
                    injected += 1
                elif (
                    _READ_HINT_MARKER not in block_text
                    and _READ_GUARD_MARKER not in block_text
                    and _is_preloaded_complete_read(
                        messages,
                        str(block.get("tool_use_id") or block.get("tool_call_id") or ""),
                    )
                ):
                    if new_content is None:
                        new_content = list(content)
                    new_content[block_index] = _append_tool_result_text(
                        block, _READ_COMPLETE_PRELOADED_HINT
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


_LINE_RANGE_PATTERNS = (
    re.compile(r"lines?\s+(\d+)\s*[-–—to]+\s*(\d+)", re.IGNORECASE),
    re.compile(r"L(\d+)\s*[-–—]\s*L?(\d+)", re.IGNORECASE),
    re.compile(r"строк[аиуе]?\s+(\d+)\s*[-–—]+\s*(\d+)", re.IGNORECASE),
)

_LINE_SINGLE_PATTERNS = (
    re.compile(r"\bline\s+(\d+)\b", re.IGNORECASE),
    re.compile(r":(\d+):"),
    re.compile(r"(?<![-–—])\bL(\d+)\b(?!\s*[-–—])"),
)


def _parse_prompt_line_ranges(prompt: str) -> list[tuple[int, int]]:
    """Extract normalized ``(lo, hi)`` line ranges (1-based, inclusive) from a prompt.

    Also recognizes single-line references (``line 123``, ``:123:``, ``L123``)
    from Python/pytest tracebacks and converts each into a one-line ``(n, n)``
    range so excerpts capture ``+-5`` context lines around the failure line.
    """
    ranges: list[tuple[int, int]] = []
    for pattern in _LINE_RANGE_PATTERNS:
        for match in pattern.finditer(prompt):
            lo = int(match.group(1))
            hi = int(match.group(2))
            if lo > hi:
                lo, hi = hi, lo
            ranges.append((lo, hi))
    for pattern in _LINE_SINGLE_PATTERNS:
        for match in pattern.finditer(prompt):
            n = int(match.group(1))
            if n >= 1:
                ranges.append((n, n))
    return ranges


_PROMPT_SYMBOL_STOPWORDS: frozenset[str] = frozenset(
    {
        *keyword.kwlist,
        # Built-in types & typing primitives
        "str",
        "int",
        "float",
        "bool",
        "bytes",
        "dict",
        "list",
        "set",
        "tuple",
        "frozenset",
        "type",
        "object",
        "None",
        "True",
        "False",
        "Any",
        "Optional",
        "Union",
        "Iterable",
        "Mapping",
        "Sequence",
        "Callable",
        "Path",
        "Pattern",
        "Match",
        "Exception",
        # Common module / stdlib names
        "re",
        "os",
        "sys",
        "json",
        "ast",
        "io",
        "time",
        "math",
        "copy",
        "enum",
        "uuid",
        "shutil",
        "tempfile",
        "pathlib",
        "typing",
        "unittest",
        "pytest",
        # Generic variable / parameter names
        "path",
        "paths",
        "file",
        "files",
        "line",
        "lines",
        "text",
        "data",
        "value",
        "values",
        "item",
        "items",
        "key",
        "keys",
        "result",
        "results",
        "out",
        "err",
        "msg",
        "args",
        "kwargs",
        "self",
        "cls",
        "prompt",
        "model",
        "payload",
        "name",
        "code",
        "status",
        "error",
        "errors",
        "count",
        "index",
        "idx",
        "start",
        "end",
        "node",
        "tree",
    }
)


def _extract_prompt_symbols(prompt: str) -> list[str]:
    """Return deduplicated identifier names mentioned in backticks or def/class."""
    symbols: list[str] = []
    seen: set[str] = set()
    for symbol in re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", prompt):
        if (
            len(symbol) < 3
            or symbol in _PROMPT_SYMBOL_STOPWORDS
            or symbol.lower() in _PROMPT_SYMBOL_STOPWORDS
        ):
            continue
        if symbol not in seen:
            seen.add(symbol)
            symbols.append(symbol)
    for kind in ("def", "class"):
        for symbol in re.findall(rf"\b{kind}\s+([A-Za-z_][A-Za-z0-9_]*)", prompt):
            if symbol in _PROMPT_SYMBOL_STOPWORDS:
                continue
            if symbol not in seen:
                seen.add(symbol)
                symbols.append(symbol)
    return symbols


def _symbol_line_windows(
    text: str,
    all_lines: list[str],
    line_count: int,
    symbols: list[str],
    *,
    is_python: bool,
    context_lines: int,
    use_ast: bool,
) -> list[tuple[int, int]]:
    """Build line windows around ``symbols`` using AST defs/classes and text matches."""
    windows: list[tuple[int, int]] = []
    ast_matched: set[str] = set()
    if use_ast and is_python:
        symbol_set = set(symbols)
        try:
            tree = ast.parse(text)
        except (SyntaxError, OSError):
            tree = None
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, _DEF_NODES) and node.name in symbol_set:
                    ast_matched.add(node.name)
                    dec_start = min(
                        [d.lineno for d in getattr(node, "decorator_list", ())] + [node.lineno]
                    )
                    end_ln = getattr(node, "end_lineno", None) or node.lineno
                    cap_end = min(line_count, end_ln, node.lineno + 79)
                    windows.append((max(1, dec_start), max(dec_start, cap_end)))
    remaining = [s for s in symbols if s not in ast_matched]
    if remaining:
        seen_linenos: set[int] = set()
        for lineno, line in enumerate(all_lines, 1):
            if lineno in seen_linenos:
                continue
            for symbol in remaining:
                if re.search(rf"\b{re.escape(symbol)}\b", line):
                    seen_linenos.add(lineno)
                    break
        for lineno in sorted(seen_linenos):
            windows.append(
                (max(1, lineno - context_lines), min(line_count, lineno + context_lines))
            )
    return windows


def _cap_merged_windows(merged: list[list[int]], cap: int) -> list[list[int]]:
    """Cap merged ``[start, end]`` windows to at most ``cap`` total lines."""
    capped: list[list[int]] = []
    total = 0
    for start, end in merged:
        span = end - start + 1
        if total >= cap:
            break
        if total + span > cap:
            end = start + (cap - total) - 1
            span = end - start + 1
            if span <= 0:
                break
        capped.append([start, end])
        total += span
    return capped


def build_preloaded_file_excerpts(
    files: Iterable[str],
    prompt: str = "",
    *,
    short_file_max_lines: int = 150,
    max_total_chars: int = 12000,
    read_only_files: Iterable[str] = (),
) -> str:
    """Build pre-loaded exact-line excerpts for ``files`` guided by ``prompt``.

    Files at or under ``short_file_max_lines`` are included whole (and marked
    ``COMPLETE FILE PRELOADED`` when editable); longer files contribute line
    ranges parsed from ``prompt`` (expanded by ``+-5`` context lines) and/or
    ``+-20`` windows (or AST ``def``/``class`` bodies up to 80 lines) around
    prompt symbols, capped at 120 lines per editable file. Read-only reference
    files (``read_only_files``) are marked in their section header, and long
    read-only files without explicit ``L<a>-L<b>`` ranges fall back to
    ``+-15`` windows around prompt symbols, capped at 80 lines. The total
    output is capped at ``max_total_chars``. Returns a prompt-ready string
    (empty when nothing was collected).
    """
    read_only_set = {str(p) for p in read_only_files}
    ranges = _parse_prompt_line_ranges(prompt)
    prompt_symbols = _extract_prompt_symbols(prompt)
    seen: set[str] = set()
    chunks: list[str] = []
    total_chars = 0
    header = (
        "[PRELOADED_FILE_EXCERPTS — exact current lines; apply your edit "
        "directly without re-reading these lines first]:\n"
    )
    for path in files:
        if path in seen:
            continue
        seen.add(path)
        file_path = Path(path)
        try:
            if not file_path.is_file():
                continue
            text = file_path.read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        all_lines = text.splitlines()
        line_count = len(all_lines)
        included: list[str] = []
        if line_count <= short_file_max_lines:
            included = [f"{lineno}: {line}" for lineno, line in enumerate(all_lines, 1)]
        else:
            windows: list[tuple[int, int]] = []
            is_ro = path in read_only_set
            if is_ro and not ranges and prompt_symbols:
                windows = _symbol_line_windows(
                    text,
                    all_lines,
                    line_count,
                    prompt_symbols,
                    is_python=file_path.suffix == ".py",
                    context_lines=15,
                    use_ast=False,
                )
            else:
                for lo, hi in ranges:
                    start = max(1, lo - 5)
                    end = hi + 5
                    if end - start + 1 > 120:
                        end = start + 119
                    if not (1 <= start <= line_count):
                        continue
                    end = min(line_count, end)
                    windows.append((start, end))
                if not is_ro and prompt_symbols:
                    windows.extend(
                        _symbol_line_windows(
                            text,
                            all_lines,
                            line_count,
                            prompt_symbols,
                            is_python=file_path.suffix == ".py",
                            context_lines=20,
                            use_ast=True,
                        )
                    )
            if not windows:
                continue
            windows.sort()
            merged: list[list[int]] = []
            for start, end in windows:
                if merged and start <= merged[-1][1] + 1:
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([start, end])
            if is_ro and not ranges and prompt_symbols:
                merged = _cap_merged_windows(merged, 80)
            elif not is_ro:
                merged = _cap_merged_windows(merged, 120)
            for start, end in merged:
                included.extend(
                    f"{lineno}: {all_lines[lineno - 1]}"
                    for lineno in range(start, end + 1)
                )
        if not included:
            continue
        if path in read_only_set:
            prefix = (
                f"--- {path} ({line_count} lines, READ-ONLY — preloaded below; "
                "do NOT call read or edit on this path)\n"
            )
        elif line_count <= short_file_max_lines:
            prefix = (
                f"--- {path} ({line_count} lines, COMPLETE FILE PRELOADED — "
                "do NOT call read on this file; edit directly)\n"
            )
        else:
            prefix = f"--- {path} ({line_count} lines)\n"
        chunk = prefix + "\n".join(included)
        remaining = max_total_chars - total_chars
        if remaining <= 0:
            break
        if len(chunk) > remaining:
            chunk = chunk[:remaining].rstrip()
            chunks.append(chunk)
            total_chars = max_total_chars
            break
        chunks.append(chunk)
        total_chars += len(chunk)
    if not chunks:
        return ""
    return header + "\n".join(chunks) + "\n\n"


def build_cross_file_symbol_context(
    files: Iterable[str],
    prompt: str = "",
    *,
    directory: str = "",
    read_only_files: Iterable[str] = (),
    max_chars: int = 2400,
) -> str:
    """Build compact imported-symbol signatures from files outside ``files``.

    Parses each editable ``.py`` file, resolves its local imports (including
    relative imports and ``from pkg import submod`` module imports) to ``.py``
    files under ``directory`` (or next to the importing file), and extracts
    compact slices for the needed symbols: function signatures plus first
    docstring line (no bodies), class headers with fields and public method
    signatures, and explicitly imported UPPER_CASE constants. When any file is
    a pytest test file, matching fixtures from nearby ``conftest.py`` files
    are included too. Output is strictly capped at ``max_chars``; returns
    ``""`` when nothing was extracted.
    """
    prompt_symbols = _extract_prompt_symbols(prompt)
    editable_paths = [str(p) for p in files if str(p).endswith(".py")]
    editable_set = set(editable_paths)
    read_only_set = {str(p) for p in read_only_files}
    base = Path(directory).resolve() if directory else None
    base_raw = Path(directory) if directory else None

    def _resolve_module_file(parts: list[str], level: int, importer: Path) -> str | None:
        if not parts:
            return None
        mod_rel = "/".join(parts)
        if level:
            start = importer.parent
            for _ in range(max(0, level - 1)):
                start = start.parent
            bases = [start]
        else:
            bases = [importer.parent]
            if base_raw is not None:
                bases.extend([base_raw, base_raw / "src"])
        seen_bases: set[str] = set()
        for base_dir in bases:
            if str(base_dir) in seen_bases:
                continue
            seen_bases.add(str(base_dir))
            for candidate in (
                base_dir / (mod_rel + ".py"),
                base_dir / mod_rel / "__init__.py",
            ):
                try:
                    resolved = candidate.resolve()
                except OSError:
                    continue
                if base is not None and not resolved.is_relative_to(base):
                    continue
                if resolved.is_file():
                    return str(resolved)
        return None

    # 1-2: collect needed symbols per resolved target module.
    needed: dict[str, set[str]] = {}
    for fpath in editable_paths:
        try:
            text = Path(fpath).read_text(encoding="utf-8")
            tree = ast.parse(text, filename=fpath)
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        importer = Path(fpath)
        module_aliases: set[str] = set()
        imported_names: set[str] = set()
        targets: set[str] = set()
        attr_refs: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    resolved = _resolve_module_file(alias.name.split("."), 0, importer)
                    if resolved:
                        targets.add(resolved)
                    module_aliases.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                mod_parts = (node.module or "").split(".") if node.module else []
                for alias in node.names:
                    resolved = _resolve_module_file(mod_parts, node.level, importer)
                    if resolved:
                        targets.add(resolved)
                        if alias.name != "*":
                            imported_names.add(alias.asname or alias.name)
                    if alias.name != "*":
                        sub = _resolve_module_file(
                            mod_parts + [alias.name], node.level, importer
                        )
                        if sub:
                            targets.add(sub)
                            module_aliases.add(alias.asname or alias.name)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in module_aliases
            ):
                attr_refs.add(node.attr)
        for target in targets:
            bucket = needed.setdefault(target, set())
            bucket.update(prompt_symbols)
            bucket.update(imported_names)
            bucket.update(attr_refs)

    def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
        return f"{prefix} {node.name}({ast.unparse(node.args)}){returns}:"

    def _docline(node: ast.AST) -> str | None:
        doc = ast.get_docstring(node)
        if doc:
            first = doc.strip().splitlines()[0].strip()
            return f'"""{first}"""'
        return None

    def _extract_symbol_lines(
        tree: ast.Module, lines: list[str], names: set[str]
    ) -> list[str]:
        out: list[str] = []
        matched = False

        def emit_function(node: ast.FunctionDef | ast.AsyncFunctionDef, indent: str) -> None:
            out.append(f"{indent}{_signature(node)}")
            doc = _docline(node)
            if doc:
                out.append(f"{indent}    {doc}")
            out.append(f"{indent}    ...")

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in names:
                    matched = True
                    emit_function(node, "")
            elif isinstance(node, ast.ClassDef):
                if node.name not in names:
                    continue
                matched = True
                bases = ", ".join(ast.unparse(b) for b in node.bases)
                out.append(f"class {node.name}({bases}):" if bases else f"class {node.name}:")
                for item in node.body:
                    if isinstance(item, ast.AnnAssign):
                        out.append(f"    {ast.unparse(item)}")
                    elif isinstance(item, ast.Assign):
                        target_names = [
                            t.id for t in item.targets if isinstance(t, ast.Name)
                        ]
                        if target_names and all(
                            n.isupper() and len(n) > 1 for n in target_names
                        ):
                            out.append(f"    {ast.unparse(item)}")
                    elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name.startswith("_") and item.name != "__init__":
                            continue
                        emit_function(item, "    ")
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                target_names: list[str] = []
                if isinstance(node, ast.Assign):
                    target_names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                elif isinstance(node.target, ast.Name):
                    target_names = [node.target.id]
                if any(n in names and n.isupper() and len(n) > 1 for n in target_names):
                    matched = True
                    src_line = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
                    out.append(src_line[:120])
        if not matched:
            # Fallback: no specifically named symbol matched -> top-level public.
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not node.name.startswith("_"):
                        emit_function(node, "")
                elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                    bases = ", ".join(ast.unparse(b) for b in node.bases)
                    out.append(
                        f"class {node.name}({bases}):" if bases else f"class {node.name}:"
                    )
        return out

    sections: list[str] = []
    for target, names in needed.items():
        if target in editable_set:
            continue
        try:
            text = Path(target).read_text(encoding="utf-8")
            tree = ast.parse(text, filename=target)
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        lines = text.splitlines()
        if target in read_only_set:
            if len(lines) <= 150 or names & set(prompt_symbols):
                continue
        symbol_lines = _extract_symbol_lines(tree, lines, names)
        if not symbol_lines:
            continue
        if base_raw is not None:
            try:
                rel = Path(target).relative_to(base_raw).as_posix()
            except ValueError:
                rel = target
        else:
            rel = target
        sections.append(f"# {rel}\n" + "\n".join(symbol_lines))

    # 4: pytest fixture inference from nearby conftest.py files.
    test_files = [
        p
        for p in editable_paths
        if (lambda n: n.startswith("test_") or n.endswith("_test"))(Path(p).stem)
    ]
    if test_files:
        fixture_stop = set(_PROMPT_SYMBOL_STOPWORDS)
        needed_fixtures: set[str] = {
            token
            for token in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", prompt)
            if len(token) >= 3 and token not in fixture_stop and token.lower() not in fixture_stop
        }
        for tpath in test_files:
            try:
                test_tree = ast.parse(Path(tpath).read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, SyntaxError):
                continue
            for node in ast.walk(test_tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                    for arg in (
                        *node.args.posonlyargs,
                        *node.args.args,
                        *node.args.kwonlyargs,
                    ):
                        if arg.arg not in ("self", "cls"):
                            needed_fixtures.add(arg.arg)
        seen_conftests: list[str] = []
        for tpath in test_files:
            current = Path(tpath).parent
            while True:
                conftest = current / "conftest.py"
                if conftest.is_file() and str(conftest) not in editable_set:
                    seen_conftests.append(str(conftest))
                if base_raw is not None and current == base_raw:
                    break
                if current.parent == current:
                    break
                if base_raw is not None and current.parent and not current.parent.is_relative_to(base_raw):
                    break
                current = current.parent
        for conftest_path in dict.fromkeys(seen_conftests):
            try:
                ctext = Path(conftest_path).read_text(encoding="utf-8")
                ctree = ast.parse(ctext, filename=conftest_path)
            except (OSError, UnicodeDecodeError, SyntaxError):
                continue
            clines = ctext.splitlines()
            fixture_lines: list[str] = []
            for node in ctree.body:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                is_fixture = False
                for deco in node.decorator_list:
                    func = deco.func if isinstance(deco, ast.Call) else deco
                    if (isinstance(func, ast.Attribute) and func.attr == "fixture") or (
                        isinstance(func, ast.Name) and func.id == "fixture"
                    ):
                        is_fixture = True
                        break
                if not is_fixture or node.name not in needed_fixtures:
                    continue
                for deco in node.decorator_list:
                    fixture_lines.append("@pytest.fixture" if not isinstance(deco, ast.Call) else "@pytest.fixture(...)")
                returns_suffix = f" -> {ast.unparse(node.returns)}" if node.returns else ""
                fixture_lines.append(f"def {node.name}({ast.unparse(node.args)}){returns_suffix}:")
                doc = _docline(node)
                if doc:
                    fixture_lines.append(f'    {doc}')
                else:
                    for sub in ast.walk(node):
                        if isinstance(sub, (ast.Return, ast.Yield)) and sub.lineno <= len(clines):
                            fixture_lines.append(f"    {clines[sub.lineno - 1].strip()[:120]}")
                            break
            if not fixture_lines:
                continue
            if base_raw is not None:
                try:
                    rel = Path(conftest_path).relative_to(base_raw).as_posix()
                except ValueError:
                    rel = conftest_path
            else:
                rel = conftest_path
            sections.append(f"# {rel} (pytest fixtures)\n" + "\n".join(fixture_lines))

    if not sections:
        return ""
    header = "[Cross-File Symbol Signatures & Fixtures]:\n"
    total = len(header)
    body: list[str] = []
    for section in sections:
        remaining = max_chars - total - 1
        if remaining <= 0:
            break
        if len(section) > remaining:
            section = section[:remaining].rstrip("\n")
            if not section:
                break
            body.append(section)
            break
        body.append(section)
        total += len(section) + 1
    if not body:
        return ""
    return header + "\n".join(body)


def extract_compact_test_failure(
    stdout: str, stderr: str = '', max_lines: int = 25, max_chars: int | None = None
) -> str:
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
        result = "\n".join(line for _, line in picked[:max_lines])
    else:
        non_empty = [line for line in lines if line.strip()]
        result = "\n".join(non_empty[-min(max_lines, 10) :])
    if max_chars is not None and len(result) > max_chars:
        kept: list[str] = []
        total = 0
        for line in result.splitlines():
            if total + len(line) + 1 > max_chars and kept:
                break
            kept.append(line)
            total += len(line) + 1
        result = "\n".join(kept)
    return result


#: Failed pytest nodeids in short summary lines.
_FAILED_NODEID_RE = re.compile(r"(?:FAILED|ERROR)\s+([^\s]+\.py::[^\s]+)")

#: Python traceback source-location lines.
_TRACEBACK_LOC_RE = re.compile(r'File "([^"]+\.py)", line (\d+)(?:, in (\w+))?')

#: Short pytest location lines (``path/file.py:123:``).
_SHORT_LOC_RE = re.compile(r'([^\s"\'()]+\.py):(\d+):?')

#: Maximum number of test files excerpted by :func:`extract_failing_test_excerpts`.
_MAX_TEST_EXCERPT_FILES = 2

#: Maximum excerpt window length per test file.
_MAX_EXCERPT_LINES = 45

_HEADER = "[FAILING_TEST_CONTEXT — READ-ONLY test source; do NOT call read or edit on test files]:"


def _is_test_path(raw_path: str, from_nodeid: bool = False) -> bool:
    """Whether ``raw_path`` looks like a test module."""
    p = Path(raw_path)
    name = p.name
    if name.startswith("test_") or name.endswith("_test.py"):
        return True
    if "tests" in p.parts:
        return True
    return from_nodeid


def extract_failing_test_excerpts(
    test_log: str,
    base_dir: str | Path | None = None,
    *,
    max_chars: int = 3000,
    exclude_files: Iterable[str | Path] | None = None,
) -> str:
    """READ-ONLY source excerpts of the failing pytest tests (H-47).

    Parses ``test_log`` for failed nodeids and traceback/short location lines,
    keeps up to two existing test files on disk, and embeds bounded excerpts of
    the failing test function (or a window around the failing line) so a repair
    attempt never needs to re-read the test file.
    """
    if not test_log:
        return ""
    base = Path(base_dir) if base_dir is not None else None

    def resolve(raw: str) -> Path:
        p = Path(raw)
        return p if p.is_absolute() or base is None else base / p

    excluded: set[Path] = set()
    for item in exclude_files or ():
        excluded.add(resolve(str(item)).resolve())

    # 1. Failed nodeids (order-preserving, deduplicated).
    nodeids: list[str] = []
    seen_nodeids: set[str] = set()
    for match in _FAILED_NODEID_RE.finditer(test_log):
        nodeid = match.group(1)
        if nodeid not in seen_nodeids:
            seen_nodeids.add(nodeid)
            nodeids.append(nodeid)

    # 2. Candidate (path, line) references, in first-seen order per path.
    line_for_path: dict[str, int] = {}
    for pattern in (_TRACEBACK_LOC_RE, _SHORT_LOC_RE):
        for match in pattern.finditer(test_log):
            raw_path, lineno = match.group(1), int(match.group(2))
            if _is_test_path(raw_path):
                line_for_path.setdefault(raw_path, lineno)
    for nodeid in nodeids:
        raw_path = nodeid.split("::", 1)[0]
        if raw_path not in line_for_path:
            line_for_path[raw_path] = 0  # nodeid-origin: no line hint
            if not _is_test_path(raw_path, from_nodeid=True):
                del line_for_path[raw_path]

    # 3. Keep up to 2 unique existing test files on disk.
    excerpts: list[tuple[str, list[str]]] = []
    for raw_path, target_line in line_for_path.items():
        if len(excerpts) >= _MAX_TEST_EXCERPT_FILES:
            break
        resolved = resolve(raw_path)
        if resolved.resolve() in excluded or not resolved.is_file():
            continue
        try:
            lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        window: list[tuple[int, str]] = []
        func_name = None
        class_name = None
        # Target function/class names from the first nodeid of this file, if any.
        for nodeid in nodeids:
            if nodeid.split("::", 1)[0] == raw_path:
                parts = nodeid.split("::")
                if len(parts) >= 3:
                    if parts[1] and parts[1].isidentifier():
                        class_name = parts[1]
                    candidate = parts[-1].split("[", 1)[0]
                    if candidate and candidate.isidentifier():
                        func_name = candidate
                elif len(parts) == 2:
                    candidate = parts[1].split("[", 1)[0]
                    if candidate and candidate.isidentifier():
                        func_name = candidate
                break
        matched_ast = False
        if target_line or func_name:
            try:
                tree = ast.parse("\n".join(lines))
            except SyntaxError:
                tree = None
            if tree is not None:
                matched_node = None
                matched_cls: ast.ClassDef | None = None

                def _visit(body: list[ast.stmt], cls_node: ast.ClassDef | None) -> None:
                    nonlocal matched_node, matched_cls
                    for stmt in body:
                        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            starts = stmt.name.startswith("test_")
                            contains = (
                                target_line
                                and stmt.lineno <= target_line
                                and (stmt.end_lineno or target_line) >= target_line
                            )
                            class_ok = not class_name or (
                                cls_node is not None and cls_node.name == class_name
                            )
                            if (starts and contains) or (
                                func_name and stmt.name == func_name and class_ok
                            ):
                                matched_node, matched_cls = stmt, cls_node
                                return
                        if isinstance(stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                            _visit(
                                stmt.body,
                                stmt if isinstance(stmt, ast.ClassDef) else cls_node,
                            )
                            if matched_node is not None:
                                return

                _visit(tree.body, None)
                node = matched_node
                cls_node = matched_cls
                if node is not None:
                    start = node.lineno
                    end = node.end_lineno or node.lineno
                    if end - start + 1 <= _MAX_EXCERPT_LINES:
                        window = [(n, lines[n - 1]) for n in range(start, end + 1)]
                    else:
                        if target_line:
                            half = _MAX_EXCERPT_LINES // 2
                            lo = max(1, target_line - half)
                            hi = min(len(lines), lo + _MAX_EXCERPT_LINES - 1)
                            lo = max(1, hi - _MAX_EXCERPT_LINES + 1)
                        else:
                            lo, hi = start, min(len(lines), start + _MAX_EXCERPT_LINES - 1)
                        window = [(n, lines[n - 1]) for n in range(lo, hi + 1)]
                    if cls_node is not None:
                        # H-52: prepend class header + setUp/autouse fixture lines.
                        first_method = None
                        for m in cls_node.body:
                            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                first_method = m
                                break
                        if first_method is not None:
                            header_end = min(
                                [first_method.lineno]
                                + [d.lineno for d in first_method.decorator_list]
                            ) - 1
                        else:
                            header_end = cls_node.end_lineno or cls_node.lineno
                        prefix: list[int] = list(
                            range(cls_node.lineno, min(header_end, cls_node.lineno + 19) + 1)
                        )
                        prefix = prefix[:20]
                        setup_lines: list[int] = []
                        for m in cls_node.body:
                            if not isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                continue
                            is_fixture = any(
                                (isinstance(d, ast.Name) and d.id == "fixture")
                                or (isinstance(d, ast.Attribute) and d.attr == "fixture")
                                or (
                                    isinstance(d, ast.Call)
                                    and (
                                        (isinstance(d.func, ast.Name) and d.func.id == "fixture")
                                        or (
                                            isinstance(d.func, ast.Attribute)
                                            and d.func.attr == "fixture"
                                        )
                                    )
                                )
                                for d in m.decorator_list
                            )
                            if m.name in {"setUp", "setUpClass", "__init__"} or is_fixture:
                                lo_m = min([m.lineno] + [d.lineno for d in m.decorator_list])
                                hi_m = m.end_lineno or m.lineno
                                setup_lines.extend(range(lo_m, hi_m + 1))
                        setup_lines = setup_lines[:18]
                        class_prefix = (prefix + setup_lines)[:20]
                        combined = sorted(set(class_prefix + [n for n, _ in window]))
                        window = [(n, lines[n - 1]) for n in combined if 1 <= n <= len(lines)]
                    matched_ast = True
        if not matched_ast:
            if target_line:
                lo = max(1, target_line - 15)
                hi = min(len(lines), target_line + 15)
            else:
                lo, hi = 1, min(30, len(lines))
            window = [(n, lines[n - 1]) for n in range(lo, hi + 1)]
        excerpts.append((raw_path, [f"L{n}: {text}" for n, text in window]))

    if not excerpts:
        return ""
    parts = [_HEADER]
    if nodeids:
        parts.append("Failing nodeids: " + ", ".join(nodeids))
    for raw_path, content_lines in excerpts:
        parts.append(f"--- {raw_path} (READ-ONLY test excerpt)")
        parts.extend(content_lines)
    result = "\n".join(parts)
    if len(result) > max_chars:
        kept: list[str] = []
        total = 0
        for line in result.splitlines():
            if total + len(line) + 1 > max_chars and kept:
                break
            kept.append(line)
            total += len(line) + 1
        result = "\n".join(kept)
    return result


_SWE_PRUNER_MARKER = '[HARNESS SWE-PRUNER:'
_NOISY_TEST_MARKERS = (
    '=== FAILURES ===',
    '=== ERRORS ===',
    'Traceback (most recent call last):',
    'short test summary info',
)


def prune_noisy_tool_outputs(messages: list[dict], max_tool_chars: int = 4000) -> tuple[list[dict], int]:
    """H-31 SWE-Pruner Lite: compact bloated test/traceback tool outputs.

    Scans ``messages`` for ``role == 'tool'`` messages whose string content
    exceeds ``max_tool_chars`` and has not already been pruned. Content that
    looks like pytest/unittest output (failures header, errors header,
    traceback, or short summary) is replaced by a compact digest from
    :func:`extract_compact_test_failure` plus a pruner marker line. Non-test
    outputs (e.g. large file dumps) are left intact.

    Returns ``(new_messages, pruned_count)``. The input list is never mutated;
    when ``pruned_count == 0`` the original message dicts are reused as-is.
    """
    pruned = 0
    new_messages: list[dict] = []
    for message in messages:
        if (
            isinstance(message, dict)
            and message.get('role') == 'tool'
            and isinstance(message.get('content'), str)
            and len(message['content']) > max_tool_chars
            and _SWE_PRUNER_MARKER not in message['content']
            and any(marker in message['content'] for marker in _NOISY_TEST_MARKERS)
        ):
            compacted = extract_compact_test_failure(
                message['content'], max_lines=45, max_chars=max_tool_chars
            )
            new_message = dict(message)
            new_message['content'] = compacted + '\n' + _SWE_PRUNER_MARKER + ' noisy test output compacted]'
            new_messages.append(new_message)
            pruned += 1
        else:
            new_messages.append(message)
    return new_messages, pruned


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
    turn_index: int = 0,
    keep_recent_turns: int = 4,
    min_chars: int = 600,
    min_total_saved_chars: int = 0,
    *,
    min_turn: int = 1,
    char_threshold: int | None = None,
) -> tuple[list[dict], int]:
    """H-8/H-30: drop long read outputs that a later edit made stale.

    Maps ``tool_call_id`` to ``(tool_name, file_path)`` across assistant
    ``tool_calls`` (arguments inspected for ``path``-style keys), records the
    last message index at which each file was touched by an edit/write/patch
    tool, and replaces every earlier read-tool result on that path whose text
    exceeds ``min_chars`` with a compact placeholder. Moonshot-style
    KV-cache-prefix preservation (H-30): messages inside the trailing
    ``keep_recent_turns`` window are never rewritten, and when the total
    characters saved across all candidates is below ``min_total_saved_chars``
    the original list is returned untouched so the upstream KV-cache prefix
    stays valid. Returns the new (shallow-copied) message list and the number
    of results compacted; for non-list input the messages pass through.
    """
    if char_threshold is not None:
        min_chars = char_threshold
    if turn_index < max(1, min_turn) or not isinstance(messages, list):
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
    recent_floor = len(messages) - max(keep_recent_turns, 0)

    def is_stale_read(call_id: Any, index: int) -> tuple[str, str] | None:
        mapped = call_map.get(call_id) if isinstance(call_id, str) else None
        if mapped is None:
            return None
        tool_name, file_path = mapped
        if file_path not in last_edit_idx or index >= last_edit_idx[file_path]:
            return None
        if index >= recent_floor:
            return None
        if not any(marker in tool_name for marker in _STALE_READ_MARKERS):
            return None
        return file_path

    # First pass: collect candidates without mutating anything, so the whole
    # compaction can be vetoed when it would save too little to justify
    # invalidating the upstream KV-cache prefix (H-30).
    candidates: list[tuple[int, int | None, str, int]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "tool":
            stale = is_stale_read(message.get("tool_call_id"), index)
            if stale is not None and len(_message_text(message)) > min_chars:
                candidates.append((index, None, stale, len(_message_text(message))))
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block_index, block in enumerate(content):
            if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                continue
            stale = is_stale_read(block.get("tool_use_id"), index)
            if stale is None:
                continue
            text_len = len(_message_text({"content": block.get("content")}))
            if text_len > min_chars:
                candidates.append((index, block_index, stale, text_len))
    total_saved_chars = sum(
        text_len - len(_STALE_READ_NOTICE.format(path=stale))
        for _, _, stale, text_len in candidates
    )
    if total_saved_chars < min_total_saved_chars:
        return (messages, 0)
    new_messages = list(messages)
    compacted = 0
    for index, block_index, stale, _text_len in candidates:
        notice = _STALE_READ_NOTICE.format(path=stale)
        if block_index is None:
            message = new_messages[index]
            new_messages[index] = {**message, "content": notice}
        else:
            message = new_messages[index]
            content = list(message["content"])
            content[block_index] = {
                **content[block_index],
                "content": notice,
            }
            new_messages[index] = {**message, "content": content}
        compacted += 1
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


def enforce_context_budget(
    payload: dict[str, Any],
    max_tokens: int = 16_000,
    *,
    keep_recent_turns: int = 1,
) -> dict[str, Any]:
    """H-81(b): shrink an oversized outbound payload's context before egress.

    Operates on the request ``payload`` dict in place. When the estimated token
    count of ``payload["messages"]`` exceeds ``max_tokens``, stale read outputs
    are compacted first (preserving the trailing ``keep_recent_turns`` window)
    and, if still over budget, older tool outputs are truncated via
    :func:`compact_context_window`. Returns the payload dict.
    """
    if not isinstance(payload, dict):
        return payload
    messages = payload.get("messages")
    if not isinstance(messages, list) or estimate_messages_tokens(messages) <= max_tokens:
        return payload
    compacted, _ = compact_stale_reads(
        messages,
        turn_index=len(messages),
        keep_recent_turns=keep_recent_turns,
        min_chars=256,
        min_total_saved_chars=256,
    )
    if compacted is not messages:
        payload["messages"] = compacted
        messages = compacted
    if estimate_messages_tokens(messages) > max_tokens:
        compacted, _stats = compact_context_window(messages, max_context_tokens=max_tokens)
        payload["messages"] = compacted
    return payload


def check_cot_and_slop_integrity(messages: list[dict[str, Any]]) -> list[str]:
    """Flag unclosed ``<think>`` blocks, post-``</think>`` collapse, and filler openers."""
    if not isinstance(messages, list):
        return []
    assistant: Mapping[str, Any] | None = None
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "assistant":
            assistant = message
    if assistant is None:
        return []
    text = _message_text(assistant)
    warnings: list[str] = []
    if text.rfind("<think>") > text.rfind("</think>"):
        warnings.append(
            "[COT INTEGRITY WARNING: unclosed <think> block detected; close </think> concisely "
            "before emitting tool call or final answer]"
        )
        response_text = text
    elif "</think>" in text:
        response_text = text[text.rfind("</think>") + len("</think>"):]
        if not response_text.strip() and not assistant.get("tool_calls"):
            warnings.append(
                "[COT INTEGRITY WARNING: empty post-</think> response (length collapse); "
                "provide a concrete answer or tool call after </think>]"
            )
    else:
        response_text = text
    opener = response_text.lstrip().lower()
    if opener.startswith(("certainly! here is", "sure, i'd be happy to", "as an ai")):
        warnings.append(
            "[SLOP GUARD WARNING: omit conversational filler and output direct technical action or artifact]"
        )
    return warnings
