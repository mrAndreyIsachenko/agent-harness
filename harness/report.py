"""Benchmark run summarization and Markdown report rendering.

The helpers in this module are deliberately tolerant of raw runner JSON:
field lookups accept the common key spellings produced by the guard runner
and the guard proxy, and ``proxy_summary`` fills any gaps that the run
record itself does not carry.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from harness.guards import (
    check_readonly_sql_tool_calls,
    compact_stale_reads,
    detect_duplicate_tool_calls,
    estimate_messages_tokens,
)
from harness.routing import apply_provider_routing

__all__ = [
    "summarize_run",
    "aggregate_benchmark_runs",
    "format_benchmark_markdown",
    "scan_state_runs",
    "format_state_stats_markdown",
    "simulate_replay",
    "format_replay_markdown",
]

# Statuses treated as a Pass@1 success (after lowercasing).
PASS_STATUSES = frozenset(
    {
        "pass",
        "passed",
        "success",
        "succeeded",
        "ok",
        "complete",
        "completed",
        "done",
        "0",
        "true",
        "validation_passed_review_required",
        "configuration_verified_no_model_call",
    }
)

_TRUTHY_STRINGS = frozenset({"1", "true", "yes", "y", "on"})


def _first(mapping: dict, keys, default=None):
    """Return the first present, non-None value among ``keys``."""
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return default


def _to_number(value):
    """Coerce ``value`` to int/float, or None when not numeric."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return None
    return None


def _to_int(value) -> int:
    number = _to_number(value)
    if number is None:
        return 0
    return int(number)


def _to_float(value) -> float:
    number = _to_number(value)
    if number is None:
        return 0.0
    return float(number)


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in _TRUTHY_STRINGS
    if isinstance(value, (int, float)):
        return value != 0
    return bool(value)


def _to_str_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, dict):
        return [str(key) for key in value]
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def summarize_run(result: dict, proxy_summary: dict | None = None) -> dict:
    """Normalize one run record into the canonical benchmark shape.

    Values present in ``result`` win; ``proxy_summary`` (guard-proxy
    telemetry) only fills fields the record does not carry. Missing numeric
    fields become 0, missing booleans False, missing lists empty, and an
    unknown status becomes ``"unknown"`` (counted as a failure).
    """
    if not isinstance(result, dict):
        result = {}
    proxy = proxy_summary if isinstance(proxy_summary, dict) else {}

    task_id = _first(result, ("task_id", "taskId", "id", "task"))
    if task_id is None:
        task_id = _first(proxy, ("task_id", "taskId", "id"))

    status = _first(result, ("status", "result", "outcome", "final_status", "exit_status"))
    if status is None:
        status = _first(proxy, ("status", "outcome"))
    status = str(status).strip().lower() if status is not None else "unknown"

    model = _first(result, ("model", "model_id", "modelId", "model_name"))
    if model is None:
        model = _first(proxy, ("model", "model_id"))

    steps = _first(result, ("steps", "step_count", "model_steps", "n_steps", "num_steps", "total_steps"))
    if steps is None:
        steps = _first(proxy, ("steps", "step_count"))

    cost = _first(result, ("cost_usd", "total_cost_usd", "cost", "costUSD", "usd_cost", "spend_usd"))
    if cost is None:
        cost = _first(proxy, ("cost_usd", "cost"))

    micro_repair = _first(result, ("micro_repair_used", "micro_repair", "used_micro_repair", "microRepairUsed"))
    if micro_repair is None:
        # Proxy may report either a flag or a repair-event count.
        count = _first(proxy, ("micro_repair_count", "micro_repairs", "micro_repair"))
        micro_repair = True if (count is not None and _to_int(count) > 0) else _first(proxy, ("micro_repair_used",))

    dispatched = _first(result, ("dispatched_count", "dispatched", "dispatch_count", "dispatches", "num_dispatched"))
    if dispatched is None:
        dispatched = _first(proxy, ("dispatched_count", "dispatched", "dispatches"))

    blocked = _first(result, ("blocked_count", "blocked", "block_count", "num_blocked", "denied_count"))
    if blocked is None:
        blocked = _first(proxy, ("blocked_count", "blocked"))

    sticky = _first(result, ("sticky_provider", "stickyProvider", "sticky", "pinned_provider"))
    if sticky is None:
        sticky = _first(proxy, ("sticky_provider", "sticky", "pinned_provider"))

    degraded = _first(result, ("degraded_providers", "degradedProviders", "degraded", "degraded_list"))
    if degraded is None:
        degraded = _first(proxy, ("degraded_providers", "degraded"))

    return {
        "task_id": task_id,
        "status": status,
        "model": model,
        "steps": _to_number(steps) if steps is not None else 0,
        "cost_usd": _to_float(cost),
        "micro_repair_used": _to_bool(micro_repair) if micro_repair is not None else False,
        "dispatched_count": _to_int(dispatched),
        "blocked_count": _to_int(blocked),
        "sticky_provider": sticky,
        "degraded_providers": _to_str_list(degraded),
    }


def aggregate_benchmark_runs(runs: list[dict]) -> dict:
    """Aggregate normalized run records into benchmark summary metrics.

    Every entry is passed through :func:`summarize_run`; missing numeric
    fields count as 0 and non-passing statuses count against Pass@1.
    """
    normalized = [summarize_run(run) for run in (runs or [])]
    total = len(normalized)

    pass_count = sum(1 for run in normalized if run["status"] in PASS_STATUSES)
    steps = [float(run["steps"]) for run in normalized]
    costs = [run["cost_usd"] for run in normalized]

    return {
        "total_runs": total,
        "pass_count": pass_count,
        "pass_rate": (pass_count / total) if total else 0.0,
        "median_steps": float(statistics.median(steps)) if steps else 0.0,
        "median_cost_usd": float(statistics.median(costs)) if costs else 0.0,
        "total_cost_usd": float(sum(costs)),
        "micro_repair_count": sum(1 for run in normalized if run["micro_repair_used"]),
    }


def format_benchmark_markdown(baseline_runs: list[dict], guarded_runs: list[dict]) -> str:
    """Render a GitHub-flavored Markdown comparison of baseline vs guarded runs."""
    baseline = aggregate_benchmark_runs(baseline_runs or [])
    guarded = aggregate_benchmark_runs(guarded_runs or [])

    rows = [
        (
            "Pass@1 (%)",
            f"{baseline['pass_rate'] * 100:.1f}",
            f"{guarded['pass_rate'] * 100:.1f}",
        ),
        (
            "Median Steps",
            f"{baseline['median_steps']:.1f}",
            f"{guarded['median_steps']:.1f}",
        ),
        (
            "Median Cost (USD)",
            f"{baseline['median_cost_usd']:.4f}",
            f"{guarded['median_cost_usd']:.4f}",
        ),
        (
            "Total Cost (USD)",
            f"{baseline['total_cost_usd']:.4f}",
            f"{guarded['total_cost_usd']:.4f}",
        ),
        (
            "Auto-Repaired Runs",
            str(baseline["micro_repair_count"]),
            str(guarded["micro_repair_count"]),
        ),
    ]

    lines = [
        "# Benchmark Comparison: Baseline vs Guarded Harness",
        "",
        f"Runs compared: baseline {baseline['total_runs']}, guarded {guarded['total_runs']}.",
        "",
        "| Metric | Baseline (Raw Agent) | Guarded (H-1..H-5 Harness) |",
        "| --- | ---: | ---: |",
    ]
    lines.extend(f"| {metric} | {baseline_value} | {guarded_value} |" for metric, baseline_value, guarded_value in rows)
    return "\n".join(lines)


# Request-record keys aggregated when the sibling ``requests.json`` holds a
# list of guard-proxy request records instead of a ready-made summary dict.
_PROXY_COST_KEYS = ("cost_usd", "total_cost_usd", "cost", "costUSD", "spend_usd", "usd_cost")
_PROXY_STEPS_KEYS = ("steps", "step_count", "model_steps")
_PROXY_DISPATCHED_KEYS = ("dispatched_count", "dispatched", "dispatch_count", "dispatches")
_PROXY_BLOCKED_KEYS = ("blocked_count", "blocked", "block_count", "denied_count")
_PROXY_MICRO_REPAIR_KEYS = ("micro_repair_count", "micro_repairs", "micro_repair", "used_micro_repair")


def _summarize_proxy_requests(requests: list) -> dict:
    """Aggregate a list of guard-proxy request records into a summary dict."""
    total_cost = 0.0
    total_steps = 0
    total_dispatched = 0
    total_blocked = 0
    total_micro_repairs = 0
    for request in requests:
        if not isinstance(request, dict):
            continue
        total_cost += _to_float(_first(request, _PROXY_COST_KEYS))
        total_steps += _to_int(_first(request, _PROXY_STEPS_KEYS))
        total_dispatched += _to_int(_first(request, _PROXY_DISPATCHED_KEYS))
        total_blocked += _to_int(_first(request, _PROXY_BLOCKED_KEYS))
        total_micro_repairs += _to_int(_first(request, _PROXY_MICRO_REPAIR_KEYS))
    return {
        "cost_usd": total_cost,
        "steps": total_steps,
        "dispatched_count": total_dispatched,
        "blocked_count": total_blocked,
        "micro_repair_count": total_micro_repairs,
    }


def _load_proxy_summary(path: Path) -> dict | None:
    """Load the optional sibling ``requests.json`` as a proxy summary.

    Accepts either a ready-made summary object or a list of request records
    (aggregated into one summary). Returns ``None`` when the file is absent,
    unreadable, or malformed.
    """
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        return _summarize_proxy_requests(payload)
    return None


def scan_state_runs(state_dir: str | Path) -> list[dict]:
    """Scan a live state directory for guard-runner run records.

    Walks ``<state_dir>/runs/<run>/result.json`` directories oldest-first by
    run-directory mtime, loading each run's optional sibling
    ``requests.json`` guard-proxy telemetry as ``proxy_summary``. A missing
    ``runs/`` directory yields ``[]``; run directories without a readable
    ``result.json`` JSON object are skipped.
    """
    runs_dir = Path(state_dir) / "runs"
    if not runs_dir.is_dir():
        return []

    run_dirs: list[tuple[float, Path]] = []
    for child in runs_dir.iterdir():
        try:
            if not child.is_dir() or not (child / "result.json").is_file():
                continue
            run_dirs.append((child.stat().st_mtime, child))
        except OSError:
            continue
    run_dirs.sort(key=lambda entry: entry[0])

    runs: list[dict] = []
    for _, run_dir in run_dirs:
        try:
            with (run_dir / "result.json").open("r", encoding="utf-8") as handle:
                result = json.load(handle)
        except (OSError, ValueError):
            continue
        if not isinstance(result, dict):
            continue
        runs.append(summarize_run(result, _load_proxy_summary(run_dir / "requests.json")))
    return runs


def format_state_stats_markdown(state_dir: str | Path) -> str:
    """Render a Markdown stats table for runs in a live state directory.

    Runs are grouped by model (an unknown model shows as ``unknown``) with a
    trailing ``ALL`` summary row covering every run.
    """
    runs = scan_state_runs(state_dir)
    if not runs:
        return "No runs found."

    groups: dict[str, list[dict]] = {}
    for run in runs:
        groups.setdefault(str(run.get("model") or "unknown"), []).append(run)

    def _row(label: str, group_runs: list[dict]) -> str:
        agg = aggregate_benchmark_runs(group_runs)
        blocked_total = sum(_to_int(run.get("blocked_count")) for run in group_runs)
        return (
            f"| {label} | {agg['total_runs']} | {agg['pass_rate'] * 100:.1f} "
            f"| {agg['median_steps']:.1f} | {agg['median_cost_usd']:.4f} "
            f"| {agg['total_cost_usd']:.4f} | {blocked_total} "
            f"| {agg['micro_repair_count']} |"
        )

    lines = [
        "# Live State Directory Stats",
        "",
        "| Model | Runs | Pass@1 (%) | Median Steps | Median Cost (USD) "
        "| Total Cost (USD) | Blocked Requests | Auto-Repairs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(_row(model, groups[model]) for model in sorted(groups))
    lines.append(_row("ALL", runs))
    return "\n".join(lines)


def _compact_context_window(messages: list[dict], *, max_context_tokens: int) -> list[dict]:
    """Invoke ``harness.guards.compact_context_window`` when available, with a safe fallback."""
    from harness import guards as _guards

    fn = getattr(_guards, "compact_context_window", None)
    if callable(fn):
        result = fn(messages, max_context_tokens=max_context_tokens)
        if isinstance(result, tuple):
            return list(result[0])
        return list(result)

    compacted, _ = compact_stale_reads(
        list(messages),
        turn_index=max(5, len(messages)),
        min_turn=5,
    )
    if max_context_tokens <= 0:
        return compacted
    while len(compacted) > 2 and estimate_messages_tokens(compacted) > max_context_tokens:
        drop_idx = 1 if (compacted[0].get("role") == "system") else 0
        compacted.pop(drop_idx)
    return compacted


def _load_replay_episodes(records_or_state_dir: list[dict] | str | Path) -> list[dict]:
    """Normalize a list of episode dicts or a state_dir path into episode dicts."""
    if isinstance(records_or_state_dir, (str, Path)):
        path = Path(records_or_state_dir)
        if path.is_file():
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                return []
            try:
                payload = json.loads(text)
            except ValueError:
                episodes: list[dict] = []
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(item, dict):
                        episodes.append(item)
                return episodes
            if isinstance(payload, list):
                return [item for item in payload if isinstance(item, dict)]
            if isinstance(payload, dict):
                for key in ("episodes", "runs", "records"):
                    if isinstance(payload.get(key), list):
                        return [item for item in payload[key] if isinstance(item, dict)]
                return [payload]
            return []

        runs_dir = path / "runs"
        if not runs_dir.is_dir():
            return []
        run_dirs: list[tuple[float, Path]] = []
        for child in runs_dir.iterdir():
            try:
                if child.is_dir():
                    run_dirs.append((child.stat().st_mtime, child))
            except OSError:
                continue
        run_dirs.sort(key=lambda entry: entry[0])

        episodes = []
        for _, run_dir in run_dirs:
            result = None
            result_path = run_dir / "result.json"
            if result_path.is_file():
                try:
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    result = None
                if isinstance(result, dict) and isinstance(result.get("messages"), list):
                    episodes.append(result)
                    continue
            requests_path = run_dir / "requests.json"
            appended = False
            if requests_path.is_file():
                try:
                    req_payload = json.loads(requests_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    req_payload = None
                if isinstance(req_payload, dict) and isinstance(req_payload.get("messages"), list):
                    episodes.append(req_payload)
                    appended = True
                elif isinstance(req_payload, list):
                    with_msgs = [
                        item for item in req_payload
                        if isinstance(item, dict) and isinstance(item.get("messages"), list)
                    ]
                    if with_msgs:
                        episodes.append(with_msgs[-1])
                        appended = True
            if appended:
                continue

            # Fall back to worker.log JSONL (tool_use lines) as synthetic episodes.
            worker_log = run_dir / "worker.log"
            if not worker_log.is_file():
                continue
            try:
                worker_text = worker_log.read_text(encoding="utf-8")
            except OSError:
                continue
            messages: list[dict] = []
            for line in worker_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(obj, dict):
                    continue
                if obj.get("type") != "tool_use":
                    continue
                part = obj.get("part")
                if not isinstance(part, dict):
                    continue
                state = part.get("state") if isinstance(part.get("state"), dict) else {}
                tool_name = str(part.get("tool") or "unknown")
                raw_input = state.get("input", part.get("input"))
                if raw_input is None:
                    raw_input = {}
                raw_output = state.get("output", state.get("error", part.get("output")))
                call_id = f"{run_dir.name}:tool:{len(messages)}"
                messages.append({
                    "role": "assistant",
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(raw_input, default=str),
                        },
                    }],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name,
                    "content": str(raw_output if raw_output is not None else ""),
                })
            if messages:
                episodes.append({
                    "model": (result or {}).get("model") if isinstance(result, dict) else None,
                    "messages": messages,
                })
        return episodes

    return [item for item in (records_or_state_dir or []) if isinstance(item, dict)]


def simulate_replay(
    records_or_state_dir: list[dict] | str | Path,
    *,
    max_context_tokens: int = 6000,
    fallback_teacher_model: str | None = None,
    upstream_mode: str = "openrouter",
) -> dict:
    """Run offline A/B guard and routing simulation over recorded episodes or a state_dir."""
    episodes = _load_replay_episodes(records_or_state_dir)

    total_episodes = len(episodes)
    loop_warnings_triggered = 0
    sql_violations_caught = 0
    escalated_to_teacher_count = 0
    tokens_before = 0
    tokens_after = 0

    for record in episodes:
        raw_messages = record.get("messages")
        messages = [m for m in raw_messages if isinstance(m, dict)] if isinstance(raw_messages, list) else []
        model = _first(record, ("model", "model_id", "modelId", "model_name"))

        loop_warnings_triggered += len(detect_duplicate_tool_calls(messages))
        sql_violations_caught += len(check_readonly_sql_tool_calls(messages))

        before = estimate_messages_tokens(messages) if messages else 0
        compacted = _compact_context_window(messages, max_context_tokens=max_context_tokens)
        after = estimate_messages_tokens(compacted) if compacted else 0
        tokens_before += before
        tokens_after += after

        req_payload: dict = {"messages": compacted}
        if model is not None:
            req_payload["model"] = model
        raw_body = json.dumps(req_payload, ensure_ascii=False).encode("utf-8")
        routed_bytes = apply_provider_routing(
            raw_body,
            "/chat/completions",
            default_model=model,
            turn_index=max(1, len(messages)),
            upstream_mode=upstream_mode,
            fallback_teacher_model=fallback_teacher_model,
        )
        try:
            routed = json.loads(routed_bytes.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            routed = {}
        if (
            fallback_teacher_model
            and isinstance(routed, dict)
            and routed.get("model") == fallback_teacher_model
            and model != fallback_teacher_model
        ):
            escalated_to_teacher_count += 1

    tokens_saved = max(0, tokens_before - tokens_after)
    token_reduction_pct = round((tokens_saved / tokens_before) * 100.0, 2) if tokens_before > 0 else 0.0

    return {
        "total_episodes": total_episodes,
        "loop_warnings_triggered": loop_warnings_triggered,
        "sql_violations_caught": sql_violations_caught,
        "escalated_to_teacher_count": escalated_to_teacher_count,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "tokens_saved": tokens_saved,
        "token_reduction_pct": token_reduction_pct,
    }


def format_replay_markdown(summary: dict) -> str:
    """Render a GitHub-flavored Markdown report table for replay simulation results."""
    data = summary if isinstance(summary, dict) else {}
    rows = [
        ("Total Episodes", str(_to_int(data.get("total_episodes")))),
        ("Loop Warnings Triggered", str(_to_int(data.get("loop_warnings_triggered")))),
        ("SQL Violations Caught", str(_to_int(data.get("sql_violations_caught")))),
        ("Escalated to Teacher", str(_to_int(data.get("escalated_to_teacher_count")))),
        ("Tokens Before", str(_to_int(data.get("tokens_before")))),
        ("Tokens After", str(_to_int(data.get("tokens_after")))),
        ("Tokens Saved", str(_to_int(data.get("tokens_saved")))),
        ("Token Reduction (%)", f"{_to_float(data.get('token_reduction_pct')):.2f}%"),
    ]
    lines = [
        "# Offline Replay Simulation",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {metric} | {value} |" for metric, value in rows)
    return "\n".join(lines)
