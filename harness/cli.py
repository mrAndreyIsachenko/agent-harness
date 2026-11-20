"""Command-line entry points for the Harness toolkit.

Subcommands:
    preflight <file>...   Validate allowed-file syntax (exit 1 on errors).
    heal --directory <dir> -- <test_argv...>
                          Autonomous pytest failure healer.
    skeleton <file>...    Print line-threshold skeletons for the given files.
    bench-report          Render a Markdown benchmark comparison from two
                          JSON files of run records.
    stats [--state-dir] [--insights]
                          Summarize live state-directory runs as Markdown.
    replay                Run offline A/B replay simulation over episodes
                          or a state directory.
    trace [path]          Print a trace tree from traces.jsonl (or a run
                          directory containing it); defaults to the newest
                          state-dir run and syncs state traces to Phoenix
                          unless --no-sync is given.

The preflight/skeleton implementations live in sibling modules that are
loaded lazily, so this CLI imports cleanly even when those modules are not
present yet (e.g. partial checkouts of the toolkit).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

from harness.proxy import OPENROUTER_UPSTREAM, openrouter_proxy
from harness.report import (
    export_sml_dataset,
    format_benchmark_markdown,
    format_replay_markdown,
    format_state_insights_markdown,
    format_state_stats_markdown,
    simulate_replay,
)
from harness.report import backfill_salvaged_runs
from harness.telemetry import (
    DEFAULT_PHOENIX_ENDPOINT,
    HarnessTracer,
    format_trace_tree,
    synthesize_spans_from_run_dir,
    sync_state_traces_to_phoenix,
)

__all__ = ["main"]

PREFLIGHT_MODULE_CANDIDATES = (
    "harness.preflight",
    "harness.precheck",
    "harness.checks",
    "harness.files",
    "harness.guards",
)
SKELETON_MODULE_CANDIDATES = (
    "harness.skeleton",
    "harness.skeletons",
    "harness.files",
    "harness.skeleton_builder",
    "harness.guards",
)


def _resolve_callable(attribute: str, module_candidates) -> object:
    """Return the named callable from the first candidate module providing it."""
    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        func = getattr(module, attribute, None)
        if callable(func):
            return func
    raise RuntimeError(
        f"could not locate {attribute!r}; tried modules: "
        + ", ".join(module_candidates)
    )


def _load_run_records(path: str, label: str) -> list:
    """Load a JSON list of run records; accept a top-level ``{"runs": [...]}``."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = payload.get("runs")
    if not isinstance(payload, list):
        raise ValueError(f"{label} file {path!r} must contain a JSON list of run records")
    return payload


def _run_preflight(args: argparse.Namespace) -> int:
    try:
        check = _resolve_callable("check_allowed_files_syntax", PREFLIGHT_MODULE_CANDIDATES)
    except RuntimeError as exc:
        print(f"preflight: {exc}", file=sys.stderr)
        return 2

    errors = check(list(args.files))

    # Tolerate a plain boolean "all ok" return.
    if isinstance(errors, bool):
        if errors:
            return 0
        print("preflight: syntax check failed", file=sys.stderr)
        return 1

    errors = list(errors or [])
    for error in errors:
        if isinstance(error, (tuple, list)):
            print(": ".join(str(part) for part in error))
        else:
            print(error)
    return 1 if errors else 0


def _run_skeleton(args: argparse.Namespace) -> int:
    try:
        build = _resolve_callable("build_file_skeletons", SKELETON_MODULE_CANDIDATES)
    except RuntimeError as exc:
        print(f"skeleton: {exc}", file=sys.stderr)
        return 2

    output = build(list(args.files), line_threshold=args.threshold)
    if output is None:
        return 0
    if isinstance(output, str):
        print(output)
    else:
        print(json.dumps(output, indent=2, sort_keys=True, default=str))
    return 0


def _run_bench_report(args: argparse.Namespace) -> int:
    try:
        baseline = _load_run_records(args.baseline, "baseline")
        guarded = _load_run_records(args.guarded, "guarded")
    except (OSError, ValueError) as exc:
        print(f"bench-report: {exc}", file=sys.stderr)
        return 2

    print(format_benchmark_markdown(baseline, guarded))
    return 0


def _load_trace_spans(path: str) -> list:
    """Load spans from a traces.jsonl file or a run directory containing one.

    Tolerates blank lines and unparseable lines in JSONL input, plus a
    top-level ``{"spans": [...]}`` JSON object (or a bare JSON list of spans).
    """
    trace_path = Path(path)
    if trace_path.is_dir():
        candidate = trace_path / "traces.jsonl"
        if not candidate.is_file():
            raise ValueError(f"run directory {path!r} does not contain traces.jsonl")
        trace_path = candidate

    text = trace_path.read_text(encoding="utf-8")
    stripped = text.strip()
    if stripped:
        try:
            payload = json.loads(stripped)
        except ValueError:
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("spans"), list):
            return [span for span in payload["spans"] if isinstance(span, dict)]
        if isinstance(payload, list):
            return [span for span in payload if isinstance(span, dict)]

    spans = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            span = json.loads(line)
        except ValueError:
            continue
        if isinstance(span, dict):
            spans.append(span)
    return spans


def _run_stats(args: argparse.Namespace) -> int:
    if getattr(args, "backfill_salvage", False):
        backfill_salvaged_runs(args.state_dir)
    print(format_state_stats_markdown(args.state_dir))
    if getattr(args, "insights", False):
        print("\n" + format_state_insights_markdown(args.state_dir))
    return 0


def _run_replay(args: argparse.Namespace) -> int:
    target = args.input if args.input is not None else args.state_dir
    try:
        summary = simulate_replay(
            target,
            max_context_tokens=args.max_context_tokens,
            fallback_teacher_model=args.fallback_teacher_model,
            upstream_mode=args.upstream_mode,
        )
    except (OSError, ValueError) as exc:
        print(f"replay: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(format_replay_markdown(summary))
    return 0


def _run_trace(args: argparse.Namespace) -> int:
    if not args.no_sync:
        try:
            sync = sync_state_traces_to_phoenix(args.state_dir, endpoint=args.endpoint)
        except (OSError, ValueError) as exc:
            print(f"trace: phoenix sync failed: {exc}", file=sys.stderr)
            return 2
        print(
            f'[phoenix-sync] endpoint={sync["endpoint"]} reachable={sync["collector_reachable"]} '
            f'newly_synced={sync["newly_synced"]} already_synced={sync["already_synced"]} '
            f'pending={sync["pending_unsynced"]}'
        )
        if args.path is not None and not _flush_external_trace(args.path, args.state_dir, args.endpoint):
            print(f"trace: could not flush external trace {args.path!r} to Phoenix", file=sys.stderr)

    path = args.path
    if path is None:
        newest = _newest_state_run(args.state_dir)
        if newest is None:
            print("trace: no run with traces.jsonl or result.json found under "
                  f"{Path(args.state_dir) / 'runs'}", file=sys.stderr)
            return 2
        path = str(newest)

    trace_path = Path(path)
    try:
        if trace_path.is_dir():
            spans = synthesize_spans_from_run_dir(trace_path)
        else:
            spans = _load_trace_spans(str(trace_path))
    except (OSError, ValueError) as exc:
        print(f"trace: {exc}", file=sys.stderr)
        return 2

    print(format_trace_tree(spans))
    return 0


def _newest_state_run(state_dir: str) -> Path | None:
    """Most recent run directory under ``state_dir``/runs with trace inputs."""
    runs_dir = Path(state_dir) / "runs"
    if not runs_dir.is_dir():
        return None
    candidates = [
        entry for entry in runs_dir.iterdir()
        if entry.is_dir()
        and ((entry / "traces.jsonl").is_file() or (entry / "result.json").is_file())
    ]

    def _mtime(entry: Path) -> float:
        try:
            return entry.stat().st_mtime
        except OSError:
            return 0.0

    return max(candidates, key=_mtime, default=None)


def _flush_external_trace(path: str, state_dir: str, endpoint: str | None) -> bool:
    """Flush a trace outside ``state_dir`` to Phoenix; True when inside or synced."""
    try:
        trace_path = Path(path).resolve()
        state_root = Path(state_dir).resolve()
    except OSError:
        return False
    if trace_path == state_root or state_root in trace_path.parents:
        return True  # already covered by the state-dir sync pass
    if trace_path.is_dir():
        spans = synthesize_spans_from_run_dir(trace_path)
    else:
        spans = _load_trace_spans(str(trace_path))
    if not spans:
        return False
    target = (endpoint
              or os.environ.get("PHOENIX_COLLECTOR_ENDPOINT")
              or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
              or DEFAULT_PHOENIX_ENDPOINT)
    tracer = HarnessTracer(otlp_endpoint=target)
    tracer.spans = list(spans)
    return tracer.flush_otlp(endpoint=target)


def _run_serve(args: argparse.Namespace) -> int:
    proxy_ref: list = [None]

    def _gate() -> str | None:
        if args.max_steps is not None and proxy_ref[0] is not None:
            if getattr(proxy_ref[0], "_turn_count", 0) >= args.max_steps:
                return "step_limit"
        return None

    with openrouter_proxy(
        _gate,
        upstream_url=args.upstream_url,
        default_model=args.default_model,
        allowed_files=args.allowed_file or None,
        host=args.host,
        port=args.port,
        upstream_mode=args.upstream_mode,
        fallback_teacher_model=args.fallback_teacher_model,
        allowed_models=args.allowed_models,
        enforce_egress_guard=args.enforce_egress_guard,
        enforce_readonly_sql=args.enforce_readonly_sql,
        max_context_tokens=args.max_context_tokens,
        sml_results_dir=args.sml_results_dir,
    ) as proxy:
        proxy_ref[0] = proxy
        info = {"base_url": proxy.base_url, "upstream_url": args.upstream_url}
        payload = json.dumps(info)
        print(payload, flush=True)
        if args.print_url_file:
            url_path = Path(args.print_url_file)
            url_path.parent.mkdir(parents=True, exist_ok=True)
            url_path.write_text(payload + "\n", encoding="utf-8")
        if args.duration_seconds is not None:
            proxy.stop_event.wait(timeout=max(0.0, float(args.duration_seconds)))
        else:
            try:
                while not proxy.stop_event.wait(timeout=0.5):
                    pass
            except KeyboardInterrupt:
                pass
    return 0


def _run_run(args: argparse.Namespace) -> int:
    from harness.sandbox import STATE as DEFAULT_SANDBOX_STATE, run_best_of_2, run_task

    state_dir = args.state_dir or os.environ.get("HARNESS_STATE_DIR") or DEFAULT_SANDBOX_STATE
    if getattr(args, "best_of_2", False) and not args.check:
        return int(run_best_of_2(
            args.task_manifest,
            state=state_dir,
            upstream_url=args.upstream_url,
        ))
    result = run_task(
        args.task_manifest,
        state=state_dir,
        check=args.check,
        upstream_url=args.upstream_url,
    )
    if isinstance(result, dict):
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        status = result.get("status")
        return 0 if status in ("passed", "validated", "configuration_verified_no_model_call") else 1
    return int(result)


def _run_heal(args: argparse.Namespace) -> int:
    from harness.sandbox import DEFAULT_MODEL, heal_directory

    test_argv = list(args.test_argv or [])
    if test_argv and test_argv[0] == '--':
        test_argv = test_argv[1:]
    if not test_argv:
        print(json.dumps({'status': 'error', 'reason': 'test_argv_required'}))
        return 2
    result = heal_directory(
        args.directory,
        test_argv,
        dry_run=args.dry_run,
        edit_tests=args.edit_tests,
        model=args.model or DEFAULT_MODEL,
        worktree=args.worktree,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result.get('status') in (
        'already_passing',
        'task_synthesized',
        'validation_passed_review_required',
        'patch_ready_review_required',
        'passed',
        'validated',
    ) else 1


def _run_plan(args: argparse.Namespace) -> int:
    from harness.proxy import GuardError
    from harness.sandbox import execute_plan, validate_plan

    try:
        raw_plan = json.loads(Path(args.plan_manifest).read_text())
        if args.validate_only:
            plan = validate_plan(raw_plan)
            print(json.dumps({
                'plan_id': plan['plan_id'],
                'status': 'validated',
                'waves': [[n['task_id'] for n in w] for w in plan['waves']],
                'max_parallel': plan['max_parallel'],
            }, indent=2, sort_keys=True))
            return 0
        result = execute_plan(raw_plan, state=Path(args.state_dir) if args.state_dir else None)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0 if result.get('status') == 'succeeded' else 1
    except (GuardError, OSError, ValueError) as exc:
        print(json.dumps({
            'status': 'blocked',
            'reason': str(exc) if isinstance(exc, GuardError) else type(exc).__name__,
        }))
        return 1


def _run_export_dataset(args: argparse.Namespace) -> int:
    records = export_sml_dataset(
        args.state_dir,
        output_path=args.output,
        mode=args.mode,
        clean_only=args.clean_only,
        max_steps=args.max_steps,
    )
    if args.output is None:
        for record in records:
            print(json.dumps(record, ensure_ascii=False))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    default_state_dir = os.environ.get(
        "HARNESS_STATE_DIR",
        str(Path.home() / ".local/share/agent-harness/state"),
    )
    parser = argparse.ArgumentParser(
        prog="harness",
        description="Harness toolkit: file preflight, skeletons, benchmark reports.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    preflight = subparsers.add_parser("preflight", help="check allowed-file syntax")
    preflight.add_argument("files", nargs="+", metavar="file", help="files to validate")
    preflight.set_defaults(handler=_run_preflight)

    skeleton = subparsers.add_parser("skeleton", help="print file skeletons")
    skeleton.add_argument("files", nargs="+", metavar="file", help="files to summarize")
    skeleton.add_argument(
        "--threshold",
        type=int,
        default=250,
        help="line threshold above which a file is skeletonized (default: 250)",
    )
    skeleton.set_defaults(handler=_run_skeleton)

    bench_report = subparsers.add_parser("bench-report", help="render a Markdown benchmark comparison")
    bench_report.add_argument("--baseline", required=True, help="JSON file with baseline run records")
    bench_report.add_argument("--guarded", required=True, help="JSON file with guarded run records")
    bench_report.set_defaults(handler=_run_bench_report)

    stats = subparsers.add_parser("stats", help="summarize live state-directory runs as Markdown")
    stats.add_argument(
        "--state-dir",
        default=default_state_dir,
        help="state directory containing runs/ (default: %(default)s)",
    )
    stats.add_argument(
        "--insights",
        action="store_true",
        help="also print task-funnel, blocked-reason and tool-error insights",
    )
    stats.add_argument(
        "--backfill-salvage",
        action="store_true",
        help="retroactively salvage blocked runs whose test passed (H-57) before reporting",
    )
    stats.set_defaults(handler=_run_stats)

    trace = subparsers.add_parser(
        "trace",
        help="print a trace tree from traces.jsonl (default: newest state-dir run)",
    )
    trace.add_argument(
        "path",
        nargs="?",
        default=None,
        metavar="path",
        help="traces.jsonl file or run directory (default: newest run in --state-dir)",
    )
    trace.add_argument(
        "--state-dir",
        default=default_state_dir,
        help="state directory containing runs/ (default: %(default)s)",
    )
    trace.add_argument(
        "--endpoint",
        default=None,
        help="OTLP/Phoenix collector endpoint (default: env vars or built-in Phoenix default)",
    )
    trace.add_argument(
        "--no-sync",
        action="store_true",
        help="skip syncing state traces to Phoenix",
    )
    trace.set_defaults(handler=_run_trace)

    serve = subparsers.add_parser("serve", help="run standalone OpenRouter/vLLM guard proxy")
    serve.add_argument("--host", default="127.0.0.1", help="bind host (default: %(default)s)")
    serve.add_argument("--port", type=int, default=0, help="bind port (default: 0)")
    serve.add_argument(
        "--upstream-url",
        default=OPENROUTER_UPSTREAM,
        help="upstream LLM API base URL (default: %(default)s)",
    )
    serve.add_argument("--default-model", default=None, help="fallback model ID")
    serve.add_argument(
        "--upstream-mode",
        choices=("openrouter", "vllm", "openai"),
        default="openrouter",
        help="upstream routing mode (default: %(default)s)",
    )
    serve.add_argument(
        "--fallback-teacher-model",
        default=None,
        help="fallback teacher model ID for non-OpenRouter upstreams",
    )
    serve.add_argument(
        "--allowed-model",
        dest="allowed_models",
        action="append",
        default=None,
        help="allowed model ID (repeatable)",
    )
    serve.add_argument(
        "--enforce-egress-guard",
        action="store_true",
        help="enforce egress payload guard on dispatched requests",
    )
    serve.add_argument(
        "--enforce-readonly-sql",
        action="store_true",
        help="enforce read-only SQL guard on tool calls",
    )
    serve.add_argument(
        "--allowed-file",
        action="append",
        default=[],
        help="allowed file path for AST preflight (repeatable)",
    )
    serve.add_argument("--max-cost-usd", type=float, default=None, help="optional max cost budget in USD")
    serve.add_argument("--max-steps", type=int, default=None, help="optional max turn count")
    serve.add_argument(
        "--max-context-tokens",
        type=int,
        default=None,
        help="optional max context token budget",
    )
    serve.add_argument(
        "--sml-results-dir",
        default=None,
        help="optional directory to flush per-task SML cost.json manifests",
    )
    serve.add_argument("--print-url-file", default=None, help="optional path to write base_url JSON")
    serve.add_argument(
        "--duration-seconds",
        type=float,
        default=None,
        help="optional bounded run duration in seconds",
    )
    serve.set_defaults(handler=_run_serve)

    run = subparsers.add_parser("run", help="run or validate a sandboxed task manifest")
    run.add_argument("task_manifest", help="path to JSON task manifest")
    run.add_argument("--check", action="store_true", help="validate task manifest without executing")
    run.add_argument("--best-of-2", action="store_true", help="run with best_of_2 auto-escalation strategy")
    run.add_argument("--upstream-url", default=None, help="optional upstream LLM API base URL")
    run.add_argument("--state-dir", default=None, help="optional state directory override")
    run.set_defaults(handler=_run_run)

    heal = subparsers.add_parser("heal", help="autonomous pytest failure healer")
    heal.add_argument("--directory", required=True, help="directory to heal")
    heal.add_argument("--dry-run", action="store_true", help="synthesize without applying edits")
    heal.add_argument("--edit-tests", action="store_true", help="allow editing test files")
    heal.add_argument("--worktree", action="store_true", help="isolate edits in a transactional git worktree")
    heal.add_argument("--model", default=None, help="model ID override (default: sandbox DEFAULT_MODEL)")
    heal.add_argument("test_argv", nargs=argparse.REMAINDER, metavar="-- <test_argv...>")
    heal.set_defaults(handler=_run_heal)

    plan_cmd = subparsers.add_parser('plan', help='execute a parallel worktree DAG plan manifest')
    plan_cmd.add_argument('plan_manifest', help='path to JSON plan manifest')
    plan_cmd.add_argument('--validate-only', action='store_true', help='validate plan DAG without executing')
    plan_cmd.add_argument('--state-dir', default=None, help='optional state directory override')
    plan_cmd.set_defaults(handler=_run_plan)

    replay = subparsers.add_parser(
        "replay",
        help="run offline A/B replay simulation over episodes or a state directory",
    )
    replay.add_argument(
        "--input",
        default=None,
        help="optional path to JSON/JSONL file of episodes or state dir (default: --state-dir)",
    )
    replay.add_argument(
        "--state-dir",
        default=default_state_dir,
        help="state directory containing runs/ when --input is omitted (default: %(default)s)",
    )
    replay.add_argument(
        "--max-context-tokens",
        type=int,
        default=6000,
        help="context token budget for compaction (default: %(default)s)",
    )
    replay.add_argument(
        "--fallback-teacher-model",
        default=None,
        help="optional fallback teacher model ID for escalation simulation",
    )
    replay.add_argument(
        "--upstream-mode",
        default="openrouter",
        help="upstream routing mode (default: %(default)s)",
    )
    replay.add_argument(
        "--json",
        action="store_true",
        help="print raw JSON instead of Markdown",
    )
    replay.set_defaults(handler=_run_replay)

    export_dataset = subparsers.add_parser(
        "export-dataset",
        help="export SFT/DPO fine-tuning records from a state directory",
    )
    export_dataset.add_argument(
        "--state-dir",
        required=True,
        help="state directory containing runs/",
    )
    export_dataset.add_argument(
        "--mode",
        choices=("sft", "dpo"),
        default="sft",
        help="dataset export mode (default: %(default)s)",
    )
    export_dataset.add_argument(
        "--output",
        default=None,
        help="optional JSONL output path (default: print records to stdout)",
    )
    export_dataset.add_argument(
        "--clean-only",
        action="store_true",
        help="exclude runs with micro-repair, salvage, or guard-intervention markers",
    )
    export_dataset.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="keep only runs with a positive step count at or below this cap",
    )
    export_dataset.set_defaults(handler=_run_export_dataset)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``harness`` CLI; returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help(sys.stderr)
        return 2
    return int(handler(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
