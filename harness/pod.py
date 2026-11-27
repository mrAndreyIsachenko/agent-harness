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

__all__ = ["SSH_OPTS", "preflight", "sync_pods", "launch", "wait"]

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
             f"{host}:{remote_dir.rstrip('/')}/{rel}/", os.path.join(local_dir, rel) + "/"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            rc = result.returncode
            print(f"pod sync: rsync {rel} failed: {result.stderr.strip()}")
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
export UV_NO_SYNC=1 SKIP_UV_SYNC=1
mkdir -p results
status() {{ printf '{{"status": "%s"}}\\n' "$1" > {STATUS_FILE}; }}
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
  {worker} > results/worker.log 2>&1
  rc=$?
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
    try:
        _ssh(host, f"tmux new-session -d -s {shlex.quote(session)} {shlex.quote(script)}",
             timeout=120)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"pod launch: tmux session failed: {exc}")
        return 1
    print(f"pod launch: session {session!r} started on {host} in {workdir} "
          f"(stagger {stagger_vram_mib}MiB, retries {max_retries})")
    if on_checkpoint:
        print(f"pod launch: on-checkpoint hook: {on_checkpoint}")
    return 0


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
    printed = 0
    unreachable_streak = 0
    while True:
        ssh_unreachable = False
        try:
            result = _ssh(host, f"cat {workdir}/{STATUS_FILE} 2>/dev/null",
                          check=False, timeout=15)
            raw = result.stdout.strip()
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
                      f"SSH failures; VM stopped) — exiting with status={final}")
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
            if printed < 25:
                print(f"pod wait: status={status or 'UNKNOWN'}")
                printed += 1
            last_status = status
            pod_state = {'RUNNING': 'working', 'DONE': 'done', 'CHECKPOINT': 'done',
                         'ERROR': 'error'}.get(status or '', 'working')
            try:
                report_herdr_task(f'pod:{host}', pod_state,
                                  phase=f'pod {status or "UNKNOWN"}',
                                  reason='pod_error' if status == 'ERROR' else None)
            except Exception:
                pass

        if status in ("CHECKPOINT", "DONE", "ERROR"):
            if printed < 25:
                print(f"pod wait: terminal status {status}")
            if local_dir:
                sync_pods(host, workdir, local_dir, paths, once=True)
            return 0 if status != "ERROR" else 1

        if time.monotonic() >= deadline:
            print(f"pod wait: timeout after {max_wait_sec}s "
                  f"(last status={status or 'UNKNOWN'})")
            return 0 if status != "ERROR" else 1
        time.sleep(max(1, poll_interval_sec))
