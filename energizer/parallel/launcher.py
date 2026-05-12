from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


RANK_RE = re.compile(r"\brank=(?P<rank>\d+)\b")
SIZE_RE = re.compile(r"\bsize=(?P<size>\d+)\b")
WARN_RE = re.compile(r"\[WARN\]\s+Node with rank (?P<rank>\d+).*(?P<code>\d+)")


@dataclass
class ParallelLaunchConfig:
    hosts: Sequence[str]
    script: str
    script_args: Sequence[str] = field(default_factory=tuple)
    backend: str = "ring"
    cwd: str = "/Users/Shared/projets/energizer"
    python: str = ".venv/bin/python"
    mlx_launch: str = "mlx.launch"
    env: Sequence[str] = field(default_factory=tuple)
    starting_port: int | None = None
    connections_per_ip: int | None = None
    no_verify_script: bool = True


def _as_cwd_path(cwd: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return str(Path(cwd) / path)


def _inner_command(config: ParallelLaunchConfig) -> str:
    python = _as_cwd_path(config.cwd, config.python)
    script = _as_cwd_path(config.cwd, config.script)
    parts = [python, script, *config.script_args]
    return "exec " + " ".join(shlex.quote(part) for part in parts)


def build_mlx_launch_command(config: ParallelLaunchConfig) -> list[str]:
    """
    Build an MLX launch command using the robust bash form.

    MLX rewrites an existing script path to ``python /resolved/local/path``.
    That breaks when local and remote real paths differ, so Energizer launches a
    shell command under ``--no-verify-script`` and runs the shared cwd paths.
    """

    cmd = [
        config.mlx_launch,
        "--backend",
        config.backend,
        "--hosts",
        ",".join(config.hosts),
        "--cwd",
        config.cwd,
    ]
    if config.starting_port is not None:
        cmd.extend(["--starting-port", str(config.starting_port)])
    if config.connections_per_ip is not None:
        cmd.extend(["--connections-per-ip", str(config.connections_per_ip)])
    for item in config.env:
        cmd.extend(["--env", item])
    if config.no_verify_script:
        cmd.append("--no-verify-script")
    cmd.extend(["--", "bash", "-lc", _inner_command(config)])
    return cmd


@dataclass
class RankState:
    rank: int
    host: str
    status: str = "starting"
    last_line: str = ""
    last_seen: float = 0.0
    exit_code: int | None = None


class ParallelMonitor:
    def __init__(self, hosts: Sequence[str]):
        self.started_at = time.time()
        self.lines: list[str] = []
        self.ranks = {
            rank: RankState(rank=rank, host=host, last_seen=self.started_at)
            for rank, host in enumerate(hosts)
        }
        self.world_size = len(hosts)

    def record(self, line: str):
        line = line.rstrip("\n")
        if not line:
            return
        self.lines.append(line)
        self.lines = self.lines[-12:]

        warn = WARN_RE.search(line)
        if warn:
            rank = int(warn.group("rank"))
            state = self.ranks.setdefault(rank, RankState(rank, host="?"))
            state.status = "failed"
            state.exit_code = int(warn.group("code"))
            state.last_line = line
            state.last_seen = time.time()
            return

        rank_match = RANK_RE.search(line)
        if rank_match:
            rank = int(rank_match.group("rank"))
            state = self.ranks.setdefault(rank, RankState(rank, host="?"))
            state.status = "live"
            state.last_line = line
            state.last_seen = time.time()

        size_match = SIZE_RE.search(line)
        if size_match:
            self.world_size = int(size_match.group("size"))

    def mark_finished(self, returncode: int):
        now = time.time()
        for state in self.ranks.values():
            if state.status not in {"failed", "exited"}:
                state.status = "ok" if returncode == 0 else "stopped"
                state.exit_code = returncode if returncode else None
                state.last_seen = now


def _clear() -> str:
    return "\033[2J\033[H"


def _status_color(status: str) -> str:
    if status in {"ok", "live"}:
        return "\033[32m"
    if status in {"failed", "stopped"}:
        return "\033[31m"
    return "\033[33m"


def render_monitor(monitor: ParallelMonitor) -> str:
    elapsed = int(time.time() - monitor.started_at)
    rows = [
        _clear(),
        "\033[1mEnergizer Parallel Monitor\033[0m",
        f"elapsed={elapsed}s world_size={monitor.world_size}",
        "",
        "rank  host                 status      last_seen  last message",
        "----  -------------------  ----------  ---------  ------------",
    ]
    for rank in sorted(monitor.ranks):
        state = monitor.ranks[rank]
        age = max(0.0, time.time() - state.last_seen)
        color = _status_color(state.status)
        status = f"{color}{state.status}\033[0m"
        host = state.host[:19]
        msg = state.last_line[-70:]
        rows.append(f"{rank:<4}  {host:<19}  {status:<19} {age:>6.1f}s  {msg}")

    rows.extend(["", "recent output:"])
    rows.extend(f"  {line}" for line in monitor.lines[-10:])
    return "\n".join(rows)


def launch(config: ParallelLaunchConfig, monitor: bool = False) -> int:
    cmd = build_mlx_launch_command(config)
    if not monitor:
        return subprocess.call(cmd)

    state = ParallelMonitor(config.hosts)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            state.record(line)
            print(render_monitor(state), end="", flush=True)
        returncode = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        returncode = proc.wait()
    state.mark_finished(returncode)
    print(render_monitor(state), flush=True)
    return returncode


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Energizer distributed jobs")
    parser.add_argument("script", help="Python script to run on every host")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    parser.add_argument("--hosts", required=True, help="Comma-separated host IPs")
    parser.add_argument("--backend", default="ring")
    parser.add_argument("--cwd", default="/Users/Shared/projets/energizer")
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--mlx-launch", default="mlx.launch")
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--starting-port", type=int)
    parser.add_argument("--connections-per-ip", type=int)
    parser.add_argument("--monitor", action="store_true")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    config = ParallelLaunchConfig(
        hosts=[host.strip() for host in args.hosts.split(",") if host.strip()],
        script=args.script,
        script_args=tuple(args.script_args),
        backend=args.backend,
        cwd=args.cwd,
        python=args.python,
        mlx_launch=args.mlx_launch,
        env=tuple(args.env),
        starting_port=args.starting_port,
        connections_per_ip=args.connections_per_ip,
    )
    return launch(config, monitor=args.monitor)


if __name__ == "__main__":
    raise SystemExit(main())
