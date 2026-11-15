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

__all__ = [
    "summarize_run",
    "aggregate_benchmark_runs",
    "format_benchmark_markdown",
    "scan_state_runs",
    "format_state_stats_markdown",
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
