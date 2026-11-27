"""Cloud GPU pod orchestration over SSH (stdlib only).

Subcommands wired through ``harness.cli pod``:
- ``preflight``: zero-side-effect GPU/VRAM/.env check on a remote pod.
- ``sync``: rsync results/releases from the pod to local storage (daemonizable).
- ``launch``: start a supervised worker run in tmux on the pod.
- ``wait``: blocking zero-token status monitor for a launched run.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time

from .herdr import report_herdr_task

__all__ = ["SSH_OPTS", "preflight", "sync_pods", "launch", "wait", "logs"]

SSH_OPTS = (
    "-o", "ConnectTimeout=10",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=3",
    "-o", "BatchMode=yes",
)

PID_FILE = ".harness/pod_sync.pid"
LOG_FILE = ".harness/pod_sync.log"
STATUS_FILE = "results/.pod_status.json"

# Markers a live vLLM engine logs once its KV cache has been sized. A staggered
# launch waits for either marker (or enough free VRAM) before starting a worker.
ENGINE_MARKERS = ("VLLM::EngineCore", "Available KV cache memory")
GPU_QUERY = (
    "nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu "
    "--format=csv,noheader,nounits"
)


def _ssh(host: str, command: str, *, check: bool = True,
         timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", *SSH_OPTS, host, command],
        capture_output=True, text=True, check=check, timeout=timeout,
    )


def preflight(host: str, workdir: str, min_free_vram_mib: int = 6144) -> int:
    """Check GPUs, free disk and HF_TOKEN on ``host``; return 0 iff all pass."""
    ok = True
    try:
        gpu = _ssh(host, GPU_QUERY).stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"preflight: nvidia-smi failed: {exc}")
        return 1
    if not gpu:
        print("preflight: no GPUs reported by nvidia-smi")
        ok = False

    for line in gpu.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            ok = False
            continue
        idx, name, total, free = parts[0], parts[1], parts[2], parts[3]
        try:
            free_mib = int(free)
        except ValueError:
            ok = False
            free_mib = -1
        good = free_mib >= min_free_vram_mib
        ok = ok and good
        print(f"gpu[{idx}] {name}: free={free_mib}MiB/{total}MiB "
              f"{'OK' if good else 'BELOW_MIN(' + str(min_free_vram_mib) + ')'}")

    try:
        disk = _ssh(host, "df -BG --output=size,used,avail " + workdir).stdout.strip().splitlines()
        if len(disk) >= 2:
            cols = disk[-1].split()
            print(f"disk[{workdir}]: total={cols[0]} used={cols[1]} avail={cols[2]}")
        else:
            ok = False
            print(f"disk[{workdir}]: df output unavailable")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, IndexError) as exc:
        ok = False
        print(f"disk[{workdir}]: df failed: {exc}")

    env_probe = (
        "test -f {d}/.env && grep -q '^HF_TOKEN=' {d}/.env && echo present || echo missing"
    ).format(d=workdir)
    try:
        hf = _ssh(host, env_probe).stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        hf = "missing"
    hf_ok = hf == "present"
    ok = ok and hf_ok
    print(f"env: HF_TOKEN {'present' if hf_ok else 'MISSING'} in {workdir}/.env")
    print("env note: use UV_NO_SYNC=1 SKIP_UV_SYNC=1 for remote workers")
    print(f"preflight: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, ValueError, TypeError):
        return False
    return True


def _rsync_once(host: str, remote_dir: str, local_dir: str,
                paths: tuple[str, ...]) -> int:
    os.makedirs(local_dir, exist_ok=True)
    rc = 0
    for rel in paths:
        result = subprocess.run(
            ["rsync", "-az", "--partial",
             "--exclude", "external_evals/*/raw/*/samples_*.jsonl",
             f"{host}:{remote_dir.rstrip('/')}/{rel}/", os.path.join(local_dir, rel) + "/"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            err = result.stderr.strip()
            if result.returncode == 23 and "No such file or directory" in err:
                continue
            rc = result.returncode
            print(f"pod sync: rsync {rel} failed: {err}")
        else:
            print(f"pod sync: {rel} synced")
    return rc


def _write_pid_file(path: str, pid: int) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(str(pid))
    os.replace(tmp, path)


def _read_pid(path: str) -> int | None:
    if not os.path.exists(path):
        return None
    try:
        pid = int(open(path, encoding="utf-8").read().strip())
    except (ValueError, OSError):
        return None
    return pid if _pid_alive(pid) else None


def sync_pods(host: str, remote_dir: str, local_dir: str,
              paths: tuple[str, ...] = ("results", "releases"), *,
              daemon: bool = False, stop: bool = False, once: bool = False,
              interval: int = 60, _daemon_loop: bool = False) -> int:
    """Manage rsync pulls from the pod; supports daemon lifecycle."""
    pid_path = os.path.join(local_dir, PID_FILE)
    os.makedirs(os.path.dirname(pid_path), exist_ok=True)

    if _daemon_loop:  # pragma: no cover - exercised indirectly via subprocess
        with open(os.path.join(local_dir, LOG_FILE), "a", encoding="utf-8") as log:
            _write_pid_file(pid_path, os.getpid())
            while True:
                rc = _rsync_once(host, remote_dir, local_dir, paths)
                log.write(f"cycle rc={rc} t={int(time.time())}\n")
                log.flush()
                time.sleep(max(1, interval))

    pid = _read_pid(pid_path)

    if stop:
        if pid is not None:
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"pod sync: sent SIGTERM to daemon pid {pid}")
            except OSError as exc:
                print(f"pod sync: failed to stop pid {pid}: {exc}")
        else:
            print("pod sync: no live daemon to stop")
        if os.path.exists(pid_path):
            os.remove(pid_path)
        rc = _rsync_once(host, remote_dir, local_dir, paths)
        print("pod sync: final pass complete")
        return rc

    if daemon:
        if pid is not None:
            print(f"pod sync: daemon already running (pid {pid})")
            return 0
        os.makedirs(local_dir, exist_ok=True)
        log_handle = open(os.path.join(local_dir, LOG_FILE), "a", encoding="utf-8")
        try:
            child = subprocess.Popen(
                [sys.executable, "-m", "harness.cli", "pod", "sync",
                 "--host", host, "--remote-dir", remote_dir, "--local-dir", local_dir,
                 "--paths", ",".join(paths), "--interval", str(interval),
                 "--_daemon-loop"],
                stdout=log_handle, stderr=log_handle,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                cwd=local_dir,
            )
        finally:
            log_handle.close()
        _write_pid_file(pid_path, child.pid)
        print(f"pod sync: daemon started (pid {child.pid})")
        return 0

    # --once and the bare default both do a single pass.
    return _rsync_once(host, remote_dir, local_dir, paths)


def _free_vram_mib(host: str) -> int | None:
    """Largest free VRAM across GPUs, or ``None`` when nvidia-smi is unavailable."""
    try:
        out = _ssh(
            host,
            "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits",
            check=False,
        ).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None
    values = []
    for line in out.splitlines():
        try:
            values.append(int(line.strip()))
        except ValueError:
            continue
    return max(values) if values else None


def build_supervisor_script(workdir: str, worker_cmd: list[str], *,
                            stagger_vram_mib: int = 6144,
                            on_checkpoint: str | None = None,
                            max_retries: int = 2) -> str:
    """Render the remote bash supervisor used by :func:`launch`."""
    worker = " ".join(shlex.quote(part) for part in worker_cmd)
    marker_re = "|".join(ENGINE_MARKERS)
    checkpoint_hook = on_checkpoint or "true"
    return f"""cd {shlex.quote(workdir)} || exit 1
export PYTHONUNBUFFERED=1 UV_NO_SYNC=1 SKIP_UV_SYNC=1
mkdir -p results results/logs
if [ -s results/worker.log ]; then
  cp -f results/worker.log "results/logs/worker_$(date -u +%Y%m%dT%H%M%SZ).log" 2>/dev/null || true
fi
: > results/worker.log
status() {{ printf '{{"status": "%s", "updated_at": "%s"}}\\n' "$1" "$(date -u +%FT%TZ)" > {STATUS_FILE}; }}
# Staggered start: wait until a peer engine has sized its KV cache or free VRAM
# is high enough for this worker.
i=0
while [ $i -lt 60 ]; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
  if [ -z "$free" ]; then break; fi
  if [ "$free" -ge {stagger_vram_mib} ]; then break; fi
  if grep -qE '{marker_re}' results/vllm.log 2>/dev/null; then break; fi
  sleep 10
  i=$((i + 1))
done
retries=0
while true; do
  status RUNNING
  printf '\\n[supervisor] === attempt %d/%d START %s ===\\n' "$((retries + 1))" "$(( {max_retries} + 1 ))" "$(date -u +%FT%TZ)" >> results/worker.log
  {worker} >> results/worker.log 2>&1
  rc=$?
  printf '\\n[supervisor] === attempt %d exit_code=%d %s ===\\n' "$((retries + 1))" "$rc" "$(date -u +%FT%TZ)" >> results/worker.log
  if [ $rc -eq 0 ]; then status DONE; break; fi
  retries=$((retries + 1))
  if [ $retries -gt {max_retries} ]; then status ERROR; break; fi
  status CHECKPOINT
  {checkpoint_hook}
  sleep 5
done
"""


def launch(host: str, workdir: str, session: str, worker_cmd: list[str], *,
           stagger_vram_mib: int = 6144, on_checkpoint: str | None = None,
           max_retries: int = 2) -> int:
    """Launch a supervised worker in tmux on the pod with VRAM stagger gating."""
    if not worker_cmd:
        print("pod launch: empty worker command")
        return 1

    free = _free_vram_mib(host)
    if free is not None and free < stagger_vram_mib:
        print(f"pod launch: free VRAM {free}MiB < {stagger_vram_mib}MiB; "
              f"supervisor will stagger until ready")

    script = build_supervisor_script(
        workdir, worker_cmd,
        stagger_vram_mib=stagger_vram_mib,
        on_checkpoint=on_checkpoint,
        max_retries=max_retries,
    )
    quoted_session = shlex.quote(session)
    launch_cmd = (
        f"tmux kill-session -t {quoted_session} 2>/dev/null || true; "
        f"tmux new-session -d -s {quoted_session} {shlex.quote(script)}"
    )
    try:
        _ssh(host, launch_cmd, timeout=120)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"pod launch: tmux session failed: {exc}")
        return 1
    print(f"pod launch: session {session!r} started on {host} in {workdir} "
          f"(stagger {stagger_vram_mib}MiB, retries {max_retries})", flush=True)
    if on_checkpoint:
        print(f"pod launch: on-checkpoint hook: {on_checkpoint}", flush=True)
    return 0


def logs(host: str, workdir: str, *, lines: int = 60, follow: bool = False) -> int:
    """Show or follow live worker + lm-eval logs and GPU telemetry on the pod."""
    wdir = shlex.quote(workdir)
    header_cmd = (
        f"cd {wdir} 2>/dev/null || true; "
        f"echo '=== POD STATUS ==='; cat {STATUS_FILE} 2>/dev/null || echo '(no status file)'; "
        f"echo '=== GPU & VRAM ==='; "
        f"nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu "
        f"--format=csv,noheader 2>/dev/null || echo '(nvidia-smi unavailable)'; "
        f"echo '=== DISK ==='; df -h {wdir} 2>/dev/null | tail -n 2 || true; "
        f"echo '=== WORKER LOG (last {int(lines)} lines) ==='; "
        f"tail -n {int(lines)} results/worker.log 2>/dev/null || echo '(no results/worker.log yet)'; "
        f"LIVE_EVAL=$(ls -t results/external_evals/*/raw/*/lm_eval_live.log "
        f"results/external_evals/*/raw/*/*/lm_eval_live.log 2>/dev/null | head -1); "
        f"if [ -n \"$LIVE_EVAL\" ]; then "
        f"echo \"=== ACTIVE LM_EVAL ($LIVE_EVAL) ===\"; tail -n 15 \"$LIVE_EVAL\" 2>/dev/null; fi"
    )
    if not follow:
        proc = subprocess.run(["ssh", *SSH_OPTS, host, header_cmd])
        return proc.returncode
    follow_cmd = (
        header_cmd
        + "; echo '=== FOLLOWING LIVE STREAM (Ctrl+C to detach) ==='; "
        + "tail -n 0 -F results/worker.log "
        + "$(ls -t results/external_evals/*/raw/*/lm_eval_live.log "
        + "results/external_evals/*/raw/*/*/lm_eval_live.log 2>/dev/null | head -2) 2>/dev/null"
    )
    proc = subprocess.run(["ssh", *SSH_OPTS, host, follow_cmd])
    return proc.returncode


def wait(host: str, workdir: str, *, max_wait_sec: int = 180,
         poll_interval_sec: int = 15, local_dir: str | None = None,
         paths: tuple[str, ...] = ("results", "releases"),
         max_unreachable_polls: int = 4) -> int:
    """Blocking status monitor; returns 0 unless the run ends in ERROR.

    Exits early if the remote VM stops/shuts down (``max_unreachable_polls``
    consecutive SSH transport failures with returncode 255 or timeout) so
    callers never hang for hours pinging a stopped instance.
    """
    deadline = time.monotonic() + max_wait_sec
    last_status = None
    last_tail_sig = ""
    poll_idx = 0
    printed = 0
    unreachable_streak = 0
    wdir_q = shlex.quote(workdir)
    probe_cmd = (
        f"cat {wdir_q}/{STATUS_FILE} 2>/dev/null; "
        f"printf '\\n__HARNESS_META__\\n'; "
        f"nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total "
        f"--format=csv,noheader,nounits 2>/dev/null | head -4 | paste -sd ';' -; "
        f"printf '\\n__HARNESS_TAIL__\\n'; "
        f"LIVE_EVAL=$(ls -t {wdir_q}/results/external_evals/*/raw/*/lm_eval_live.log "
        f"{wdir_q}/results/external_evals/*/raw/*/*/lm_eval_live.log 2>/dev/null | head -1); "
        f"if [ -n \"$LIVE_EVAL\" ]; then tr '\\r' '\\n' < \"$LIVE_EVAL\" | grep -v '^$' | tail -n 1; fi; "
        f"tr '\\r' '\\n' < {wdir_q}/results/worker.log 2>/dev/null | grep -v '^$' | tail -n 2"
    )
    while True:
        poll_idx += 1
        ssh_unreachable = False
        gpu_meta = ""
        live_tail = ""
        try:
            result = _ssh(host, probe_cmd, check=False, timeout=15)
            full_out = result.stdout or ""
            if "__HARNESS_META__" in full_out:
                raw_part, rest = full_out.split("__HARNESS_META__", 1)
                raw = raw_part.strip()
                if "__HARNESS_TAIL__" in rest:
                    gpu_part, tail_part = rest.split("__HARNESS_TAIL__", 1)
                    gpu_meta = gpu_part.strip()
                    tail_lines = [ln.strip() for ln in tail_part.splitlines() if ln.strip()]
                    live_tail = tail_lines[-1][:160] if tail_lines else ""
            else:
                raw = full_out.strip()
            if getattr(result, "returncode", 0) == 255 and not raw:
                ssh_unreachable = True
        except (subprocess.TimeoutExpired, OSError):
            raw = ""
            ssh_unreachable = True

        if ssh_unreachable:
            unreachable_streak += 1
            if max_unreachable_polls > 0 and unreachable_streak >= max_unreachable_polls:
                local_status = None
                if local_dir:
                    local_sf = os.path.join(local_dir, STATUS_FILE)
                    try:
                        with open(local_sf, encoding="utf-8") as fh:
                            local_status = json.loads(fh.read()).get("status")
                    except (OSError, ValueError, AttributeError):
                        local_status = None
                final = local_status if local_status in ("DONE", "ERROR", "CHECKPOINT") else "HOST_STOPPED"
                print(f"pod wait: host {host} unreachable ({unreachable_streak} consecutive "
                      f"SSH failures; VM stopped) — exiting with status={final}", flush=True)
                try:
                    report_herdr_task(
                        f'pod:{host}',
                        'error' if final == 'ERROR' else 'done',
                        phase=f'pod {final}',
                        reason='pod_error' if final == 'ERROR' else None,
                    )
                except Exception:
                    pass
                return 1 if final == "ERROR" else 0
        else:
            unreachable_streak = 0

        status = None
        if raw:
            try:
                status = json.loads(raw).get("status")
            except (ValueError, AttributeError):
                status = None

        if status != last_status:
            if printed < 100:
                extra = f" | gpu=[{gpu_meta}]" if gpu_meta else ""
                if live_tail:
                    extra += f" | {live_tail}"
                print(f"pod wait: status={status or 'UNKNOWN'}{extra}", flush=True)
                printed += 1
            last_status = status
            last_tail_sig = live_tail
            pod_state = {'RUNNING': 'working', 'DONE': 'done', 'CHECKPOINT': 'done',
                         'ERROR': 'error'}.get(status or '', 'working')
            try:
                report_herdr_task(f'pod:{host}', pod_state,
                                  phase=f'pod {status or "UNKNOWN"}: {live_tail[:60]}'.rstrip(': '),
                                  reason='pod_error' if status == 'ERROR' else None)
            except Exception:
                pass
        elif (gpu_meta or live_tail) and (live_tail != last_tail_sig or poll_idx % 4 == 0):
            ts = time.strftime("%H:%M:%S")
            gpu_str = f"gpu=[{gpu_meta}]" if gpu_meta else "gpu=n/a"
            tail_str = f" | {live_tail}" if live_tail else ""
            print(f"[{ts}] pod wait: {status or 'RUNNING'} | {gpu_str}{tail_str}", flush=True)
            last_tail_sig = live_tail
            if local_dir:
                try:
                    os.makedirs(os.path.join(local_dir, "results"), exist_ok=True)
                    with open(os.path.join(local_dir, "results", "pod_live.log"), "a", encoding="utf-8") as lf:
                        lf.write(f"[{ts}] {status or 'RUNNING'} | {gpu_str}{tail_str}\n")
                except OSError:
                    pass

        if status in ("CHECKPOINT", "DONE", "ERROR"):
            if printed < 100:
                print(f"pod wait: terminal status {status}", flush=True)
            if local_dir:
                sync_pods(host, workdir, local_dir, paths, once=True)
            return 0 if status != "ERROR" else 1

        if time.monotonic() >= deadline:
            print(f"pod wait: timeout after {max_wait_sec}s "
                  f"(last status={status or 'UNKNOWN'})", flush=True)
            return 0 if status != "ERROR" else 1
        time.sleep(max(1, poll_interval_sec))
