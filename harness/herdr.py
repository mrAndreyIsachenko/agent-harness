"""Native integration with Herdr (terminal multiplexer for coding agents).

Emits OSC 7501 status sequences to the terminal and reports agent state,
metadata and workspace tokens back to Herdr over its JSON-lines unix socket.
Stdlib only.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

HERDR_SOURCE = "harness"
HERDR_AGENT = "harness"

OSC_7501_STATES = ("idle", "working", "done", "blocked", "error", "clear")
HERDR_PANE_STATES = ("idle", "working", "blocked", "unknown")

_SEQ = 0
_SEQ_LOCK = threading.Lock()

_TASK_ID_SAFE = re.compile(r"[^A-Za-z0-9_.+/]")
_TOKEN_KEY = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


def is_herdr_active(env: dict[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    if env.get("HERDR_ENV") == "1":
        return True
    if env.get("HC_TERMINAL_ID"):
        return True
    return bool(env.get("HERDR_PANE_ID")) and bool(env.get("HERDR_SOCKET_PATH"))


def herdr_bin(env: dict[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    return env.get("HERDR_BIN_PATH") or shutil.which("herdr") or "herdr"


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def format_osc7501(
    state: str,
    *,
    app: str = "harness",
    task_id: str | None = None,
    progress: int | None = None,
    kind: str | None = None,
    title: str | None = None,
    msg: str | None = None,
) -> str:
    if state not in OSC_7501_STATES:
        raise ValueError("invalid OSC 7501 state: %r" % (state,))
    segments = ["state=%s" % state]
    if app:
        segments.append("app=%s" % app[:32])
    if task_id:
        sanitized = _TASK_ID_SAFE.sub("_", task_id).strip("/")
        sanitized = sanitized[:128]
        if sanitized:
            segments.append("id=%s" % sanitized)
    if kind and state == "blocked":
        segments.append("kind=%s" % kind)
    if progress is not None and state in ("working", "blocked"):
        clamped = min(100, max(0, int(progress)))
        segments.append("progress=%d" % clamped)
    if title:
        segments.append("title=%s" % _b64(title))
    if msg:
        segments.append("msg=%s" % _b64(msg))
    return "\x1b]7501;%s\x1b\\" % ":".join(segments)


def emit_osc7501(state: str, *, stream=None, force: bool = False, **kwargs) -> bool:
    stream = stream or sys.stderr
    allowed = (
        force
        or is_herdr_active()
        or os.environ.get("HARNESS_OSC7501") == "1"
        or (hasattr(stream, "isatty") and stream.isatty())
    )
    if not allowed:
        return False
    try:
        title = kwargs.get("title")
        if title and state != "clear":
            stream.write("\x1b]0;%s\x07" % title)
        stream.write(format_osc7501(state, **kwargs))
        stream.flush()
    except OSError:
        return False
    return True


def send_herdr_socket(
    method: str,
    params: dict,
    *,
    env: dict[str, str] | None = None,
    timeout: float = 0.5,
) -> dict | None:
    global _SEQ
    env = os.environ if env is None else env
    socket_path = env.get("HERDR_SOCKET_PATH")
    if not socket_path:
        return None
    with _SEQ_LOCK:
        _SEQ += 1
        seq = _SEQ
    request = {
        "id": "%s:%d:%d" % (HERDR_SOURCE, int(time.time() * 1000), seq),
        "method": method,
        "params": params,
    }
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(socket_path)
            sock.sendall(json.dumps(request).encode("utf-8") + b"\n")
            data = bytearray()
            while b"\n" not in data and len(data) < 65536:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                data.extend(chunk)
        line = bytes(data).split(b"\n", 1)[0]
        return json.loads(line.decode("utf-8"))
    except (OSError, ValueError):
        return None


def report_herdr_task(
    task_id: str,
    state: str,
    *,
    model: str | None = None,
    step: int | None = None,
    max_steps: int | None = None,
    cost: float | None = None,
    budget: float | None = None,
    guard_hits: int = 0,
    reason: str | None = None,
    session_id: str | None = None,
    phase: str | None = None,
    env: dict[str, str] | None = None,
    stream=None,
    force_osc: bool = False,
) -> dict:
    pane_state = {
        "working": "working",
        "blocked": "blocked",
        "done": "idle",
        "error": "blocked",
        "idle": "idle",
        "clear": "unknown",
    }.get(state, "unknown")

    progress = None
    if step is not None and max_steps and max_steps > 0:
        progress = min(100, max(0, int(round((step / max_steps) * 100))))

    parts: list[str] = []
    if phase:
        parts.append(phase)
    if step is not None and max_steps:
        parts.append("step %s/%s" % (step, max_steps))
    if cost is not None and budget:
        parts.append("$%.4f/$%.2f" % (cost, budget))
    elif cost is not None:
        parts.append("$%.4f" % cost)
    if guard_hits > 0:
        parts.append("guards:%d" % guard_hits)
    if reason:
        parts.append("reason:%s" % reason)
    message = " | ".join(parts) or None

    osc_state = state if state in OSC_7501_STATES else "unknown"
    osc_title = "%s (%s)" % (task_id, model.rsplit("/", 1)[-1]) if model else task_id
    # Emit root OSC 7501 record (without id= so herdr-app / herdr-core sets TerminalView.status)
    osc_ok = emit_osc7501(
        osc_state,
        task_id=None,
        progress=progress,
        title=osc_title,
        msg=message,
        stream=stream,
        force=force_osc,
    )
    if task_id:
        # Also emit the sub-record keyed by task_id for multi-record consumers
        emit_osc7501(
            osc_state,
            task_id=task_id,
            progress=progress,
            title=osc_title,
            msg=message,
            stream=stream,
            force=force_osc,
        )

    e = env or os.environ
    sock_ok = False
    pane_id = e.get("HERDR_PANE_ID")
    if pane_id and e.get("HERDR_SOCKET_PATH"):
        resp = send_herdr_socket(
            "pane.report_agent",
            {
                "pane_id": pane_id,
                "source": HERDR_SOURCE,
                "agent": HERDR_AGENT,
                "state": pane_state,
                "message": message,
                "agent_session_id": session_id,
            },
            env=e,
        )
        sock_ok = resp is not None

        tokens = {"task": task_id}
        if model:
            tokens["model"] = model.rsplit("/", 1)[-1]
        if step is not None and max_steps:
            tokens["steps"] = "%s/%s" % (step, max_steps)
        if cost is not None:
            tokens["cost"] = "$%.4f" % cost
        if guard_hits > 0:
            tokens["guards"] = str(guard_hits)
        if phase:
            tokens["phase"] = phase
        if reason:
            tokens["reason"] = reason
        send_herdr_socket(
            "pane.report_metadata",
            {
                "pane_id": pane_id,
                "source": HERDR_SOURCE,
                "title": task_id,
                "display_agent": "harness",
                "tokens": tokens,
            },
            env=e,
        )

        workspace_id = e.get("HERDR_WORKSPACE_ID")
        if workspace_id:
            send_herdr_socket(
                "workspace.report_metadata",
                {
                    "workspace_id": workspace_id,
                    "source": HERDR_SOURCE,
                    "tokens": {"last_task": task_id, "state": state},
                },
                env=e,
            )

    return {
        "task_id": task_id,
        "osc_emitted": osc_ok,
        "socket_reported": sock_ok,
        "progress": progress,
        "message": message,
    }


def release_herdr_agent(
    *, env: dict[str, str] | None = None, stream=None, force_osc: bool = False
) -> bool:
    emit_osc7501("clear", stream=stream, force=force_osc)
    e = env or os.environ
    pane_id = e.get("HERDR_PANE_ID")
    if pane_id and e.get("HERDR_SOCKET_PATH"):
        send_herdr_socket(
            "pane.release_agent",
            {"pane_id": pane_id, "source": HERDR_SOURCE, "agent": HERDR_AGENT},
            env=e,
        )
    return True


import subprocess


def _read_run_directory(run_dir, record) -> str:
    """Resolve a run's directory from the record, falling back to task.json."""
    if isinstance(record, dict):
        directory = str(record.get("directory") or "")
    else:
        directory = str(getattr(record, "directory", None) or "")
    if directory:
        return directory
    if run_dir is None:
        return ""
    task_path = Path(run_dir) / "task.json"
    if not task_path.is_file():
        return ""
    try:
        data = json.loads(task_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if not isinstance(data, dict):
        return ""
    return str(data.get("directory") or "")


def _is_run_alive(result: dict, now_ts: float, mtime: float) -> bool:
    """Best-effort liveness check for a run without ``finished_at``."""
    if result.get("finished_at"):
        return False
    sup_pid = result.get("supervisor_pid")
    if isinstance(sup_pid, int) and sup_pid > 0:
        try:
            os.kill(sup_pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
    # Legacy runs without supervisor_pid: fall back to result.json mtime and,
    # after a grace period, an optional pgrep on the session id.
    if (now_ts - mtime) > 120:
        return False
    if (now_ts - mtime) > 30 and result.get("session_id"):
        try:
            proc = subprocess.run(
                ["/usr/bin/pgrep", "-f", str(result["session_id"])],
                capture_output=True,
                timeout=1,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return proc.returncode == 0
    return True


def _detect_run_project(directory, task_id=None):
    """Best-effort mapping of a run's directory/task_id prefix to a project label."""
    dir_str = str(directory or "")
    tid = str(task_id or "")
    low_dir, low_tid = dir_str.lower(), tid.lower()
    if "small model lab" in low_dir or low_tid.startswith("sml_"):
        return "Small Model Lab"
    if "harness" in low_dir or low_tid.startswith(("harness_", "harness-")):
        return "Harness"
    if ("medvidi" in low_dir
            or any(key in low_dir for key in (
                "analytics-gitops",
                "analytics-airflow-app",
                "worktrees/gitops-",
                "worktrees/airflow-",
            ))
            or low_tid.startswith((
                "medvidi_", "gitops", "airflow", "visual-reconcile",
                "aurora-", "step_journal", "training_",
            ))):
        return "medvidi"
    if directory:
        try:
            name = Path(directory).name
        except (OSError, ValueError):
            return None
        name = str(name or "")
        if name.lower().startswith(("wt-", "gitops-", "airflow-")):
            return "medvidi"
        return name or None
    return None


def _summarize_filtered_runs(filtered_runs, state_path) -> dict:
    """Compute funnel/blocked-reason/tool-error summaries over filtered run records."""
    from harness.report import PASS_STATUSES

    funnel = {
        "runs": len(filtered_runs),
        "passed": 0,
        "blocked": 0,
        "error": 0,
        "running": 0,
        "other": 0,
    }
    reason_counts: dict[str, int] = {}
    tool_error_counts: dict[tuple, int] = {}
    scanned = 0
    for run in filtered_runs:
        record = run if isinstance(run, dict) else getattr(run, "__dict__", {})
        status = str(record.get("status") or "").strip().lower()
        if status in PASS_STATUSES:
            funnel["passed"] += 1
        elif status == "blocked":
            funnel["blocked"] += 1
        elif status in ("error", "failed"):
            funnel["error"] += 1
        elif status in ("pending", "running", "working"):
            funnel["running"] += 1
        else:
            funnel["other"] += 1
        if status == "blocked":
            reason = str(record.get("reason") or "unknown")
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if scanned >= 40:
            continue
        scanned += 1
        run_id = record.get("run_id")
        ev_path = Path(state_path) / "runs" / str(run_id or "") / "events.jsonl"
        if not ev_path.is_file():
            continue
        try:
            ev_lines = ev_path.read_text(encoding="utf-8", errors="replace").splitlines()[-100:]
        except OSError:
            continue
        for raw in ev_lines:
            try:
                ev = json.loads(raw)
            except ValueError:
                continue
            if not isinstance(ev, dict):
                continue
            ev_name = str(ev.get("event") or "")
            if ev_name in ("tool_error", "tool_failure") or (ev.get("tool") and ev.get("error")):
                key = (str(ev.get("tool") or "unknown"), str(ev.get("error") or "error")[:80])
                tool_error_counts[key] = tool_error_counts.get(key, 0) + 1
    top_blocked = sorted(reason_counts.items(), key=lambda kv: -kv[1])[:8]
    top_tool = sorted(tool_error_counts.items(), key=lambda kv: -kv[1])[:8]
    return {"funnel": funnel, "top_blocked_reasons": top_blocked, "top_tool_errors": top_tool}


def _herdr_active_fingerprint_mtime_ns(runs_root, cache_entry, run_id=None):
    """Track mtimes of ``result.json`` for active runs so cached dashboards refresh."""
    active_mtime_ns = 0
    payload = cache_entry.get("payload") if isinstance(cache_entry, dict) else None
    active_runs = payload.get("active_runs", []) if isinstance(payload, dict) else []
    if not isinstance(active_runs, list):
        active_runs = []
    if run_id:
        active_runs = list(active_runs) + [{"run_id": run_id}]
    for active in active_runs:
        if not isinstance(active, dict):
            continue
        try:
            mtime = (runs_root / str(active.get("run_id", "")) / "result.json").stat().st_mtime_ns
        except OSError:
            continue
        active_mtime_ns = max(active_mtime_ns, mtime)
    return active_mtime_ns


def build_herdr_dashboard_data(
    state_dir,
    *,
    run_id=None,
    limit=10,
    days=None,
    project=None,
):
    """Assemble dashboard data (totals, funnel, per-model stats, trace) for Herdr UI."""
    from harness.report import (
        PASS_STATUSES,
        _iter_run_records,
        _to_float,
        _to_int,
        compute_state_insights,
        scan_state_runs,
    )
    from harness.telemetry import format_trace_tree

    state_path = Path(state_dir)

    # Fast-path cache: fingerprint the runs directory so repeat dashboard
    # refreshes within a short window skip the full scan (H-herdr-cache).
    cache_path = state_path / ".herdr-dashboard-cache.json"
    cache_key = "%s:%s:%s" % (days, project or "all", run_id or "")
    run_count = 0
    runs_mtime_ns = 0
    runs_root = state_path / "runs"
    if runs_root.is_dir():
        try:
            runs_mtime_ns = runs_root.stat().st_mtime_ns
            run_count = len(os.listdir(runs_root))
        except OSError:
            run_count = 0
    try:
        dashboard_cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        dashboard_cache = {}
    cache_entry = dashboard_cache.get(cache_key)
    active_mtime_ns = _herdr_active_fingerprint_mtime_ns(runs_root, cache_entry, run_id)
    fingerprint = [run_count, runs_mtime_ns, active_mtime_ns]
    if (
        isinstance(cache_entry, dict)
        and cache_entry.get("fingerprint") == fingerprint
        and (time.time() - _to_float(cache_entry.get("cached_at"))) < 30
    ):
        cached_payload = cache_entry.get("payload")
        if isinstance(cached_payload, dict):
            refreshed = dict(cached_payload)
            alive_now = time.time()
            refreshed["active_runs"] = [
                active for active in cached_payload.get("active_runs", [])
                if isinstance(active, dict) and _is_run_alive(
                    {
                        "finished_at": active.get("finished_at"),
                        "supervisor_pid": active.get("supervisor_pid"),
                        "session_id": active.get("session_id"),
                    },
                    alive_now,
                    alive_now,
                )
            ]
            return refreshed

    runs = scan_state_runs(state_path, days=days)
    insights = compute_state_insights(state_path, days=days)

    project_filter = None
    if project and str(project).lower() != "all":
        project_lower = str(project).strip().lower()
        project_filter = "Small Model Lab" if project_lower == "sml" else str(project)

    def _record_project(record):
        if isinstance(record, dict):
            rec_dir = record.get("directory")
            rec_task = record.get("task_id")
        else:
            rec_dir = getattr(record, "directory", None)
            rec_task = getattr(record, "task_id", None)
        if rec_dir:
            return _detect_run_project(rec_dir, rec_task)
        rid = None
        if isinstance(record, dict):
            rid = (
                record.get("run_id")
                or record.get("id")
                or Path(str(record.get("run_dir") or record.get("artifact") or "")).name
            )
        else:
            rid = (
                getattr(record, "run_id", None)
                or getattr(record, "id", None)
                or Path(str(getattr(record, "run_dir", "") or "")).name
            )
        return _detect_run_project(
            _read_run_directory(
                state_path / "runs" / str(rid or ""),
                record if isinstance(record, dict) else {},
            ),
            rec_task,
        )

    projects = ["all"]
    seen_projects: set[str] = set()
    for run in runs:
        proj = _record_project(run)
        if proj and proj not in seen_projects:
            seen_projects.add(proj)
            projects.append(proj)

    if project_filter:
        filtered_runs = [
            run for run in runs
            if str(_record_project(run) or "").lower() == project_filter.lower()
        ]
        filtered_summary = _summarize_filtered_runs(filtered_runs, state_path)
        funnel = filtered_summary["funnel"]
        blocked_reasons = filtered_summary["top_blocked_reasons"]
        tool_errors = filtered_summary["top_tool_errors"]
        runs_for_stats = filtered_runs
    else:
        funnel = insights.get("task_funnel", {})
        blocked_reasons = insights.get("top_blocked_reasons", [])
        tool_errors = insights.get("top_tool_errors", [])
        runs_for_stats = runs

    total = len(runs_for_stats)
    resolved = 0
    total_cost = 0.0
    blocked_calls = 0
    per_model: dict[str, dict] = {}
    for run in runs_for_stats:
        record = run if isinstance(run, dict) else getattr(run, "__dict__", {})
        status = str(record.get("status") or "").strip().lower()
        if status in PASS_STATUSES:
            resolved += 1
        blocked_calls += _to_int(record.get("blocked_count")) or (1 if status == "blocked" else 0)
        cost = record.get("cost")
        if cost is None:
            cost = record.get("cost_usd") or record.get("total_cost_usd") or 0.0
        total_cost += _to_float(cost)
        model = str(record.get("model") or "unknown")
        bucket = per_model.setdefault(model, {"runs": 0, "passed": 0, "steps": 0, "cost": 0.0})
        bucket["runs"] += 1
        if status in PASS_STATUSES:
            bucket["passed"] += 1
        bucket["steps"] += _to_int(record.get("steps"))
        bucket["cost"] += _to_float(cost)

    def _pct(passed: int, count: int) -> float:
        return round((passed / count * 100.0), 1) if count else 0.0

    by_model = [
        {
            "model": model,
            "runs": bucket["runs"],
            "pass_rate": _pct(bucket["passed"], bucket["runs"]),
            "avg_steps": round(bucket["steps"] / bucket["runs"], 1) if bucket["runs"] else 0.0,
            "total_cost": round(bucket["cost"], 4),
        }
        for model, bucket in per_model.items()
    ]
    by_model.sort(key=lambda item: item["runs"], reverse=True)

    recent_runs: list[dict] = []
    active_runs: list[dict] = []
    now_ts = time.time()
    for run_dir, result in _iter_run_records(state_path, days=days):
        cost_val = result.get("cost")
        if cost_val is None:
            cost_val = result.get("cost_usd") or result.get("observed_cost") or 0.0
        status_raw = str(result.get("status") or "").strip().lower()
        started_at = _to_float(result.get("started_at"))
        finished_at = result.get("finished_at")
        limits = result.get("limits") if isinstance(result.get("limits"), dict) else {}
        max_steps = _to_int(limits.get("max_steps")) or 16
        budget = _to_float(limits.get("available_cost_usd") or limits.get("max_cost_usd"))
        steps_val = _to_int(result.get("steps"))
        dur_val = _to_float(result.get("duration_seconds") or result.get("duration_s") or result.get("elapsed_s"))
        if not dur_val and started_at > 0 and not finished_at:
            dur_val = round(max(0.0, now_ts - started_at), 1)
        entry = {
            "run_id": run_dir.name,
            "task_id": result.get("task_id"),
            "status": result.get("status"),
            "model": result.get("model"),
            "steps": steps_val,
            "max_steps": max_steps,
            "budget": round(budget, 4) if budget else None,
            "cost": float(cost_val),
            "duration_s": dur_val,
            "salvaged_from": result.get("salvaged_from"),
            "reason": result.get("reason"),
            "blocked_count": result.get("blocked_count"),
            "micro_repair_used": result.get("micro_repair_used"),
            "started_at": started_at or None,
            "finished_at": finished_at,
            "project": _detect_run_project(
                _read_run_directory(run_dir, result), result.get("task_id")
            ),
        }
        if entry["project"] and entry["project"] not in seen_projects:
            seen_projects.add(entry["project"])
            projects.append(entry["project"])
        recent_runs.append(entry)
        if status_raw in ("pending", "running", "working") and not finished_at:
            try:
                mtime = (run_dir / "result.json").stat().st_mtime
            except OSError:
                mtime = started_at
            if _is_run_alive(result, now_ts, mtime):
                progress = min(100, max(0, int(round((steps_val / max_steps) * 100)))) if max_steps > 0 else None
                active_runs.append({
                    **entry,
                    "supervisor_pid": result.get("supervisor_pid"),
                    "session_id": result.get("session_id"),
                    "progress": progress,
                    "phase": "micro_repair" if result.get("micro_repair_attempted") else "coding",
                })
    if project_filter:
        recent_runs = [e for e in recent_runs if str(e.get("project") or "").lower() == project_filter.lower()]
        active_runs = [e for e in active_runs if str(e.get("project") or "").lower() == project_filter.lower()]
    recent_runs.reverse()
    active_runs.reverse()
    recent_runs = recent_runs[: max(0, int(limit))]

    selected_run_id = None
    if run_id and (state_path / "runs" / str(run_id)).is_dir():
        selected_run_id = run_id
    elif recent_runs:
        selected_run_id = recent_runs[0]["run_id"]

    trace_tree = ""
    spans: list[dict] = []
    selected_run_detail: dict | None = None
    if selected_run_id:
        from harness.telemetry import synthesize_spans_from_run_dir

        run_dir_path = state_path / "runs" / str(selected_run_id)
        traces_path = run_dir_path / "traces.jsonl"
        parsed: list[dict] = []
        if traces_path.is_file():
            with traces_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        span = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(span, dict):
                        parsed.append(span)
        if not parsed and run_dir_path.is_dir():
            try:
                parsed = list(synthesize_spans_from_run_dir(run_dir_path) or [])
            except Exception:
                parsed = []
        if parsed:
            parsed.sort(key=lambda span: span.get("start_time_ns") or 0)
            trace_tree = format_trace_tree(parsed)
            for span in parsed[:100]:
                attrs = span.get("attributes") if isinstance(span.get("attributes"), dict) else {}
                start_ns = _to_int(span.get("start_time_ns"))
                end_ns = _to_int(span.get("end_time_ns"))
                dur_ms = span.get("duration_ms")
                if dur_ms is None and end_ns >= start_ns > 0:
                    dur_ms = round((end_ns - start_ns) / 1_000_000.0, 2)
                spans.append(
                    {
                        "name": span.get("name"),
                        "kind": span.get("kind") or attrs.get("openinference.span.kind"),
                        "status": span.get("status"),
                        "duration_ms": dur_ms or 0.0,
                        "tool_name": span.get("tool_name") or attrs.get("tool.name"),
                        "guard_reason": (
                            span.get("guard_reason")
                            or attrs.get("guard.reason")
                            or attrs.get("guardrail.name")
                        ),
                        "model": attrs.get("llm.model_name") or attrs.get("gen_ai.request.model"),
                        "provider": attrs.get("llm.provider"),
                        "cost_usd": attrs.get("gen_ai.usage.cost_usd"),
                        "turn_index": attrs.get("harness.turn_index"),
                        "tool_preview": (
                            attrs.get("input.value")
                            or attrs.get("tool.parameters")
                            or attrs.get("output.value")
                            or ""
                        )[:160],
                    }
                )

        if run_dir_path.is_dir():
            def _tail_text(p, max_chars: int = 3500) -> str:
                if not p.is_file():
                    return ""
                try:
                    txt = p.read_text(encoding="utf-8", errors="replace").strip()
                    return txt[-max_chars:] if len(txt) > max_chars else txt
                except OSError:
                    return ""

            res_data: dict = {}
            task_data: dict = {}
            try:
                if (run_dir_path / "result.json").is_file():
                    res_data = json.loads((run_dir_path / "result.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                res_data = {}
            try:
                if (run_dir_path / "task.json").is_file():
                    task_data = json.loads((run_dir_path / "task.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                task_data = {}

            events_tail: list[dict] = []
            ev_path = run_dir_path / "events.jsonl"
            if ev_path.is_file():
                try:
                    ev_lines = ev_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    for raw_ev in ev_lines[-40:]:
                        try:
                            ev = json.loads(raw_ev)
                        except ValueError:
                            continue
                        if not isinstance(ev, dict):
                            continue
                        ev_name = ev.get("event")
                        if ev_name == "request_state":
                            px = ev.get("proxy") or {}
                            reqs = ev.get("requests") or []
                            in_flight = sum(1 for r in reqs if isinstance(r, dict) and r.get("state") == "in_flight")
                            events_tail.append({
                                "time": ev.get("time"),
                                "event": "request_state",
                                "summary": f"dispatched={px.get('dispatched_count', 0)} in_flight={in_flight} cost=${_to_float(ev.get('observed_cost')):.4f} compacted={px.get('compacted_reads', 0)} hints={px.get('edit_hints_injected', 0)}",
                            })
                        else:
                            detail_keys = [f"{k}={v}" for k, v in ev.items() if k not in ("time", "event", "hashes")]
                            events_tail.append({
                                "time": ev.get("time"),
                                "event": ev_name,
                                "summary": " ".join(detail_keys)[:160],
                            })
                except OSError:
                    pass

            # Keep last 12 distinct or newest events
            events_tail = events_tail[-12:]
            limits_obj = res_data.get("limits") if isinstance(res_data.get("limits"), dict) else {}
            cost_now = res_data.get("cost")
            if cost_now is None:
                cost_now = res_data.get("observed_cost") or 0.0
            started_ts = _to_float(res_data.get("started_at"))
            dur_now = _to_float(res_data.get("duration_seconds"))
            if not dur_now and started_ts > 0 and not res_data.get("finished_at"):
                dur_now = round(max(0.0, time.time() - started_ts), 1)

            selected_run_detail = {
                "run_id": str(selected_run_id),
                "task_id": res_data.get("task_id") or task_data.get("task_id"),
                "status": res_data.get("status"),
                "reason": res_data.get("reason") or None,
                "salvaged_from": res_data.get("salvaged_from") or None,
                "model": res_data.get("model") or task_data.get("model"),
                "directory": task_data.get("directory"),
                "files": list(task_data.get("files") or []),
                "prompt": (task_data.get("prompt") or "")[:900],
                "steps": _to_int(res_data.get("steps")),
                "max_steps": _to_int(limits_obj.get("max_steps") or task_data.get("max_steps")) or 16,
                "cost": round(_to_float(cost_now), 4),
                "budget": round(_to_float(limits_obj.get("available_cost_usd") or task_data.get("max_cost_usd")), 4),
                "duration_s": dur_now,
                "test_return_code": res_data.get("test_return_code"),
                "micro_repair_attempted": bool(res_data.get("micro_repair_attempted")),
                "patch": _tail_text(run_dir_path / "changes.patch", 3500),
                "test_log": _tail_text(run_dir_path / "test.log", 2500),
                "events": events_tail,
                "worker_steps": parse_worker_log_steps(run_dir_path, limit=30),
            }

    payload = {
        "totals": {
            "runs": total,
            "resolved": resolved,
            "success_rate": _pct(resolved, total),
            "total_cost": round(total_cost, 4),
            "blocked_calls": blocked_calls,
        },
        "funnel": funnel,
        "by_model": by_model,
        "top_blocked_reasons": [
            {"reason": reason, "count": count}
            for reason, count in blocked_reasons
        ],
        "top_tool_errors": [
            {"tool": tool, "error": error, "count": count}
            for (tool, error), count in tool_errors
        ],
        "projects": projects,
        "filters": {"days": days, "project": project or "all"},
        "pods_sml": collect_pod_and_sml_status(),
        "recent_runs": recent_runs,
        "active_runs": active_runs,
        "selected_run_id": selected_run_id,
        "selected_run_detail": selected_run_detail,
        "trace_tree": trace_tree,
        "spans": spans,
    }

    try:
        active_mtime_ns = _herdr_active_fingerprint_mtime_ns(runs_root, None, run_id)
        for active in active_runs:
            try:
                mtime = (state_path / "runs" / str(active.get("run_id", "")) / "result.json").stat().st_mtime_ns
            except OSError:
                continue
            active_mtime_ns = max(active_mtime_ns, mtime)
        dashboard_cache[cache_key] = {
            "fingerprint": [run_count, runs_mtime_ns, active_mtime_ns],
            "cached_at": time.time(),
            "payload": payload,
        }
        cache_path.write_text(json.dumps(dashboard_cache), encoding="utf-8")
    except OSError:
        pass
    return payload


def parse_worker_log_steps(run_dir, limit: int = 30) -> list[dict]:
    """Parse worker.log / events.jsonl into the last ``limit`` step records."""
    run_dir = Path(run_dir)
    steps: list[dict] = []

    def _preview(text, n: int = 160) -> str:
        return " ".join(str(text or "").split())[:n]

    wlog = run_dir / "worker.log"
    if wlog.is_file():
        try:
            lines = wlog.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            lines = []
        for raw in lines[-400:]:
            try:
                obj = json.loads(raw)
            except ValueError:
                continue
            if not isinstance(obj, dict):
                continue
            part = obj.get("part") if isinstance(obj.get("part"), dict) else {}
            otype = obj.get("type") or part.get("type")
            ts = obj.get("timestamp") or obj.get("time") or obj.get("ts")
            base = {"step": len(steps) + 1, "timestamp": ts, "tool": None, "target": None}
            if otype == "tool_use" or part.get("type") == "tool":
                state = part.get("state") if isinstance(part.get("state"), dict) else {}
                inp = state.get("input") if isinstance(state.get("input"), dict) else {}
                tool = obj.get("tool") or part.get("tool") or part.get("toolName") or part.get("name")
                raw_target = (
                    obj.get("target") or part.get("target") or part.get("path") or part.get("command")
                    or inp.get("filePath") or inp.get("path") or inp.get("file") or inp.get("command")
                )
                target_str = str(raw_target) if raw_target else None
                if target_str and "/" in target_str and not inp.get("command"):
                    parts_p = [p for p in target_str.split("/") if p]
                    target_str = "/".join(parts_p[-2:]) if len(parts_p) > 2 else target_str
                err_val = state.get("error") or part.get("error")
                out_val = state.get("output") or part.get("output")
                step_type = "error" if (err_val or str(state.get("status") or "").lower() in ("error", "failed")) else "tool_call"
                steps.append({
                    **base,
                    "type": step_type,
                    "tool": tool,
                    "target": _preview(target_str) if target_str else None,
                    "preview": _preview(err_val or out_val or part.get("text") or obj.get("text") or raw_target),
                })
            elif otype == "tool_result" or part.get("type") == "tool_result":
                steps.append({
                    **base,
                    "type": "tool_result",
                    "tool": obj.get("tool") or part.get("tool"),
                    "target": None,
                    "preview": _preview(part.get("text") or obj.get("text")),
                })
            elif otype == "text":
                steps.append({
                    **base,
                    "type": "assistant_text",
                    "target": None,
                    "preview": _preview(part.get("text") or obj.get("text")),
                })
            elif otype == "error":
                steps.append({
                    **base,
                    "type": "error",
                    "target": None,
                    "preview": _preview(obj.get("message") or obj.get("error") or part.get("text")),
                })

    ev_path = run_dir / "events.jsonl"
    if ev_path.is_file():
        try:
            ev_lines = ev_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            ev_lines = []
        for raw in ev_lines[-200:]:
            try:
                ev = json.loads(raw)
            except ValueError:
                continue
            if not isinstance(ev, dict):
                continue
            ev_name = str(ev.get("event") or "")
            ts = ev.get("time")
            if "guard" in ev_name or ev.get("guard_reason") or ev.get("hint"):
                steps.append({
                    "step": len(steps) + 1,
                    "timestamp": ts,
                    "type": "guard_hint",
                    "tool": None,
                    "target": None,
                    "preview": _preview(ev.get("reason") or ev.get("hint") or ev_name),
                })
            elif "error" in ev_name:
                steps.append({
                    "step": len(steps) + 1,
                    "timestamp": ts,
                    "type": "error",
                    "tool": ev.get("tool"),
                    "target": None,
                    "preview": _preview(ev.get("error") or ev.get("reason") or ev_name),
                })

    return steps[-max(0, int(limit)):]


def herdr_run_action(action: str, run_id: str, *, state_dir, model: str | None = None) -> dict:
    """Dashboard-driven run actions: retry, escalate, abort."""
    import signal

    state_path = Path(state_dir)
    run_dir = state_path / "runs" / str(run_id)
    action = str(action).lower()

    def _launch(manifest: Path, name: str) -> None:
        inner = [sys.executable, "-m", "harness.cli", "run", str(manifest), "--state-dir", str(state_path)]
        hc = shutil.which("hc")
        if hc:
            argv = [hc, "run", "--bg", "--name", name, "--"] + inner
        else:
            argv = inner
        subprocess.Popen(argv, start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if action in ("retry", "escalate"):
        task_file = run_dir / "task.json"
        if not task_file.is_file():
            return {"status": "error", "action": action, "reason": "missing %s" % task_file}
        try:
            task = json.loads(task_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return {"status": "error", "action": action, "reason": "unreadable task.json: %s" % exc}
        if action == "escalate":
            from harness.routing import select_escalated_model

            fallback = model or "openrouter/deepseek/deepseek-v4.1-flash-1017"
            task["model"] = (
                model
                or select_escalated_model(
                    [{"role": "assistant", "content": "validation_failed"},
                     {"role": "assistant", "content": "blocked"}],
                    current_model=task.get("model"),
                    fallback_teacher_model=fallback,
                )
                or fallback
            )
            manifest_path = run_dir / "escalate.manifest.json"
            name = "escalate-%s" % (task.get("task_id") or run_id)
        else:
            manifest_path = run_dir / "retry.manifest.json"
            name = "retry-%s" % (task.get("task_id") or run_id)
        manifest_path.write_text(json.dumps(task, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _launch(manifest_path, name)
        return {
            "status": "ok",
            "action": action,
            "run_id": str(run_id),
            "manifest": str(manifest_path),
            "model": task.get("model"),
        }

    if action == "abort":
        res_path = run_dir / "result.json"
        res: dict = {}
        if res_path.is_file():
            try:
                res = json.loads(res_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                res = {}
        session_id = res.get("session_id")
        if not session_id:
            ev_path = run_dir / "events.jsonl"
            if ev_path.is_file():
                try:
                    for raw in reversed(ev_path.read_text(encoding="utf-8", errors="replace").splitlines()[-200:]):
                        try:
                            ev = json.loads(raw)
                        except ValueError:
                            continue
                        if isinstance(ev, dict) and ev.get("session_id"):
                            session_id = ev.get("session_id")
                            break
                except OSError:
                    pass
        needles = [s for s in (session_id, str(run_id)) if s]
        killed: list[int] = []
        try:
            ps_out = subprocess.run(
                ["ps", "-eo", "pid,args"], capture_output=True, text=True, timeout=10
            ).stdout
        except (OSError, subprocess.SubprocessError):
            ps_out = ""
        for line in ps_out.splitlines()[1:]:
            parts = line.strip().split(None, 1)
            if len(parts) != 2:
                continue
            pid_s, args_s = parts
            if "opencode-cli" not in args_s:
                continue
            if any(n and n in args_s for n in needles):
                try:
                    pid = int(pid_s)
                    os.kill(pid, signal.SIGTERM)
                    killed.append(pid)
                except (ValueError, OSError):
                    continue
        res["status"] = "blocked"
        res["reason"] = "aborted_from_dashboard"
        res["finished_at"] = time.time()
        try:
            res_path.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except OSError as exc:
            return {"status": "error", "action": action,
                    "reason": "result.json write failed: %s" % exc, "killed_pids": killed}
        session_cleared = False
        if session_id:
            try:
                from harness import finops

                if getattr(finops, "DB", None) and finops.DB.is_file():
                    clear_busy = getattr(finops, "clear_session_busy", None)
                    if callable(clear_busy):
                        clear_busy(finops.DB, session_id, outcome="failed")
                        session_cleared = True
            except Exception:
                session_cleared = False
        return {
            "status": "ok",
            "action": action,
            "run_id": str(run_id),
            "session_id": session_id,
            "killed_pids": killed,
            "session_cleared": session_cleared,
        }

    return {"status": "error", "reason": "unknown action: %r" % (action,)}


DEFAULT_SML_DIR = os.environ.get("HARNESS_SML_DIR", "/Users/mr_a_i/Developer/personal/Small Model Lab")

_SML_METRIC_KEYS = (
    "accuracy", "exact_match", "em", "f1", "pass_rate",
    "bleu", "loss", "win_rate", "score",
)


def collect_pod_and_sml_status(sml_dir=None, workspace_dir=None) -> dict:
    """Collect GPU pod sync status and Small Model Lab external-eval status."""
    sml = Path(sml_dir) if sml_dir else Path(os.environ.get("HARNESS_SML_DIR") or DEFAULT_SML_DIR)
    ws = Path(workspace_dir) if workspace_dir else Path.cwd()

    def _sync_status(base: Path) -> dict:
        info: dict = {"dir": str(base), "pid": None, "pid_alive": False, "log_tail": ""}
        pid_file = base / ".harness" / "pod_sync.pid"
        if pid_file.is_file():
            try:
                info["pid"] = int(pid_file.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                info["pid"] = None
            if info["pid"]:
                try:
                    os.kill(info["pid"], 0)
                    info["pid_alive"] = True
                except OSError:
                    info["pid_alive"] = False
        log_file = base / ".harness" / "pod_sync.log"
        if log_file.is_file():
            try:
                info["log_tail"] = "\n".join(
                    log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-10:])
            except OSError:
                pass
        return info

    pod_status = None
    ps_file = sml / "results" / ".pod_status.json"
    if ps_file.is_file():
        try:
            pod_status = json.loads(ps_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pod_status = None

    evals: list[dict] = []
    evals_dir = sml / "results" / "external_evals"
    if evals_dir.is_dir():
        entries = []
        for p in evals_dir.glob("*.json"):
            try:
                entries.append((p.stat().st_mtime, p))
            except OSError:
                continue
        entries.sort(reverse=True)
        for mtime, p in entries[:20]:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(data, dict):
                continue
            arms = data.get("arms")
            if isinstance(arms, dict):
                arm_names = sorted(arms)
            elif isinstance(arms, list):
                arm_names = [str(a) for a in arms]
            else:
                arm_names = []
            source = data.get("after")
            if not isinstance(source, dict):
                source = data.get("deltas")
            if not isinstance(source, dict):
                source = {}
            metrics: dict = {}
            for key in _SML_METRIC_KEYS:
                if key in source and len(metrics) < 6:
                    value = source[key]
                    metrics[key] = round(float(value), 4) if isinstance(value, (int, float)) else value
            if not metrics:
                for key, value in sorted(source.items()):
                    if key == "elapsed_seconds" or len(metrics) >= 4:
                        continue
                    if isinstance(value, (int, float)):
                        metrics[key] = round(float(value), 4)
            evals.append({
                "file": p.name,
                "model_id": data.get("model_id") or data.get("model"),
                "arms": arm_names,
                "metrics": metrics,
                "updated_at": mtime,
                "size_bytes": p.stat().st_size,
            })
    return {
        "sml_dir": str(sml),
        "workspace_dir": str(ws),
        "pod_sync": {"sml": _sync_status(sml), "workspace": _sync_status(ws)},
        "pod_status": pod_status,
        "external_evals": evals,
    }


def open_herdr_ui(
    *,
    direction: str = "right",
    mod: str = "harness-dashboard",
    runner=subprocess.run,
) -> dict:
    """Open a Herdr app pane for the dashboard; returns a status dict."""
    if direction not in ("right", "down"):
        return {"status": "error", "reason": "invalid direction: %r" % (direction,)}
    app_bin = shutil.which("herdr-app") or "herdr-app"
    try:
        proc = runner(
            [app_bin, "do", "pane.split", json.dumps({"direction": direction, "mod": mod})],
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError) as exc:
        return {"status": "error", "reason": str(exc)}
    if getattr(proc, "returncode", 1) != 0:
        stderr = getattr(proc, "stderr", "") or ""
        stdout = getattr(proc, "stdout", "") or ""
        return {"status": "error", "reason": (stderr or stdout or "pane.split failed").strip()}
    return {"status": "opened", "direction": direction, "mod": mod}


def spawn_herdr_pane(
    command_argv: list[str],
    *,
    title: str | None = None,
    direction: str = "right",
    cwd: str | None = None,
    focus: bool = False,
    env: dict[str, str] | None = None,
    runner=subprocess.run,
) -> dict:
    e = env if env is not None else os.environ
    # Native Herdr.app desktop support (when HC_TERMINAL_ID is set and HERDR_PANE_ID is not)
    if e.get("HC_TERMINAL_ID") and not e.get("HERDR_PANE_ID") and env is None:
        app_bin = shutil.which("herdr-app") or "/Applications/Herdr.app/Contents/MacOS/Herdr"
        hc_bin = shutil.which("hc") or str(os.path.expanduser("~/.local/bin/hc"))
        try:
            proc = runner(
                [app_bin, "do", "pane.split", json.dumps({"direction": direction})],
                capture_output=True, text=True,
            )
            if proc.returncode == 0:
                payload = json.loads(proc.stdout or "{}")
                pane_id = payload.get("pane") or payload.get("terminal")
                term_id = payload.get("terminal")
                if command_argv and term_id:
                    cmd_str = shlex.join(command_argv)
                    if cwd:
                        cmd_str = "cd %s && %s" % (shlex.quote(cwd), cmd_str)
                    runner([hc_bin, "send", term_id, cmd_str], capture_output=True, text=True)
                return {"status": "spawned", "pane_id": pane_id, "terminal_id": term_id, "title": title, "command": command_argv}
        except Exception:  # noqa: BLE001
            pass

    bin_path = herdr_bin(env)
    argv = [
        bin_path, "pane", "split", "--current", "--direction", direction,
        *(["--cwd", cwd] if cwd else []),
        "--focus" if focus else "--no-focus",
    ]
    try:
        proc = runner(argv, capture_output=True, text=True)
    except Exception as exc:  # noqa: BLE001 - reported, never raised
        return {"status": "error", "reason": str(exc)}
    if proc.returncode != 0:
        return {"status": "error", "reason": (proc.stderr or proc.stdout or "pane split failed").strip()}
    pane_id = None
    try:
        payload = json.loads(proc.stdout)
    except ValueError:
        payload = {}
    for candidate in (
        payload.get("result", {}).get("pane", {}).get("pane_id"),
        payload.get("pane", {}).get("pane_id"),
        payload.get("pane_id"),
    ):
        if candidate:
            pane_id = candidate
            break
    if not pane_id:
        return {"status": "error", "reason": "pane_id_missing"}
    if title:
        try:
            rename = runner(
                [bin_path, "pane", "rename", pane_id, title],
                capture_output=True, text=True,
            )
            if rename.returncode != 0:
                return {"status": "error", "reason": (rename.stderr or rename.stdout or "pane rename failed").strip()}
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "reason": str(exc)}
    if command_argv and pane_id:
        try:
            run_proc = runner(
                [bin_path, "pane", "run", pane_id, shlex.join(command_argv)],
                capture_output=True, text=True,
            )
            if getattr(run_proc, "returncode", 0) != 0:
                return {"status": "error", "reason": (run_proc.stderr or run_proc.stdout or "pane run failed").strip()}
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "reason": str(exc)}
    return {"status": "spawned", "pane_id": pane_id, "title": title, "command": command_argv}
