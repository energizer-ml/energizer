from __future__ import annotations

import argparse
import os
import sys
import time

from .telemetry import TelemetryServer


def _write_control(sequence: str) -> None:
    if not sys.stdout.isatty():
        return
    sys.stdout.write(sequence)
    sys.stdout.flush()


def _enter_live_view() -> None:
    _write_control("\033[?1049h\033[?25l\033[H")


def _leave_live_view() -> None:
    _write_control("\033[?25h\033[?1049l")


def _refresh_frame(frame: str) -> None:
    if sys.stdout.isatty():
        sys.stdout.write("\033[H\033[J")
        sys.stdout.write(frame)
        sys.stdout.write("\n")
        sys.stdout.flush()
        return
    print(frame, flush=True)


def _fmt_bytes(value: int) -> str:
    if value >= 1024 * 1024:
        return f"{value / (1024 * 1024):.2f} MB"
    if value >= 1024:
        return f"{value / 1024:.1f} KB"
    return f"{value} B"


def _render(snapshot: dict) -> str:
    width = 100
    if sys.stdout.isatty():
        try:
            width = max(100, os.get_terminal_size().columns)
        except OSError:
            width = 100

    lines = []
    lines.append("ENERGIZER DISTRIBUTED MONITOR".center(width))
    lines.append("-" * width)
    lines.append(
        f"{'rank':<6}{'node':<22}{'events':<10}{'allreduce':<12}{'avg ms':<10}"
        f"{'sent':<14}{'recv':<14}{'last event':<12}"
    )
    lines.append("-" * width)
    for rank_state in snapshot["ranks"]:
        lines.append(
            f"{rank_state['rank']:<6}"
            f"{rank_state['node'][:20]:<22}"
            f"{rank_state['events']:<10}"
            f"{rank_state['allreduce_calls']:<12}"
            f"{rank_state['avg_allreduce_ms']:<10.2f}"
            f"{_fmt_bytes(rank_state['bytes_sent']):<14}"
            f"{_fmt_bytes(rank_state['bytes_received']):<14}"
            f"{rank_state['last_event'][:12]:<12}"
        )
    if not snapshot["ranks"]:
        lines.append("waiting for ranks to connect...")
    lines.append("-" * width)
    return "\n".join(lines)


def serve(addr: str = "0.0.0.0", port: int = 29650, refresh_ms: int = 500) -> None:
    server = TelemetryServer(addr=addr, port=port)
    server.start()
    _enter_live_view()
    try:
        while True:
            _refresh_frame(_render(server.metrics.snapshot()))
            time.sleep(refresh_ms / 1000.0)
    except KeyboardInterrupt:
        pass
    finally:
        _leave_live_view()
        server.stop()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Monitor Energizer distributed runs")
    parser.add_argument("--addr", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=29650)
    parser.add_argument("--refresh-ms", type=int, default=500)
    args = parser.parse_args(argv)
    serve(addr=args.addr, port=args.port, refresh_ms=args.refresh_ms)


if __name__ == "__main__":
    main()
