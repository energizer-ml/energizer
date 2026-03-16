#!/usr/bin/env python3
"""
BACKPACK — Apple Silicon Power Monitor
Tracks CPU · GPU · ANE power in real time via powermetrics.

Usage:
    sudo python3 backpack_monitor.py
    sudo python3 backpack_monitor.py -i 250

Keybinds:
    q   Quit
    p   Pause / Resume
    r   Reset peaks
    h   Cycle history window
    +   Faster sampling
    -   Slower sampling
"""

import os, re, sys, tty, signal, select, argparse
import subprocess, threading, time, termios
from collections import deque
from typing import Optional

# ── ANSI helpers ──────────────────────────────────────────────────────────────

R = "\033[0m"


def fg(r, g, b):
    return f"\033[38;2;{r};{g};{b}m"


def bg(r, g, b):
    return f"\033[48;2;{r};{g};{b}m"


C_CPU = fg(100, 180, 255)
C_GPU = fg(200, 120, 255)
C_ANE = fg(80, 220, 140)
C_WARN = fg(255, 200, 60)
C_HOT = fg(255, 80, 80)
C_DIM = fg(80, 90, 105)
C_WHITE = fg(220, 225, 235)

ANSI_RE = re.compile(r"\033\[[^m]*m|\033\][^\a]*\a")


def vlen(s: str) -> int:
    return len(ANSI_RE.sub("", s))


def rpad(s: str, width: int, fill: str = " ") -> str:
    return s + fill * max(0, width - vlen(s))


def center_str(s: str, width: int) -> str:
    v = vlen(s)
    pad = max(0, width - v)
    return " " * (pad // 2) + s + " " * (pad - pad // 2)


SPARKS = " ▁▂▃▄▅▆▇█"


def sparkline(hist: deque, width: int, col: str) -> str:
    pts = list(hist)[-width:] if hist else []
    mx = max(pts) if pts else 1
    if mx == 0:
        mx = 1
    out = C_DIM + "·" * max(0, width - len(pts)) + R
    for v in pts:
        idx = max(0, min(len(SPARKS) - 1, int((v / mx) * (len(SPARKS) - 1))))
        out += col + SPARKS[idx] + R
    return out


def power_bar(value: int, peak: int, width: int, col: str) -> str:
    if peak <= 0:
        peak = 1
    pct = max(0.0, min(1.0, value / peak))
    filled = int(width * pct)
    color = col if pct < 0.5 else (C_WARN if pct < 0.8 else C_HOT)
    return color + "█" * filled + R + C_DIM + "░" * (width - filled) + R


def fmt_energy(mj: float) -> str:
    if mj >= 1_000_000:
        return f"{mj/1_000_000:.2f}kJ"
    if mj >= 1_000:
        return f"{mj/1_000:.2f}J "
    return f"{mj:.0f}mJ"


def fmt_elapsed(s: float) -> str:
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ── State ─────────────────────────────────────────────────────────────────────

HIST_WINDOWS = [60, 120, 300]


class State:
    def __init__(self, interval_ms: int):
        self.power = {"CPU": 0, "GPU": 0, "ANE": 0}
        self.peak = {"CPU": 0, "GPU": 0, "ANE": 0}
        self.history = {k: deque(maxlen=6000) for k in ("CPU", "GPU", "ANE")}
        self.energy = {"CPU": 0.0, "GPU": 0.0, "ANE": 0.0}
        self.paused = False
        self.interval = interval_ms
        self.hist_idx = 0
        self.start = time.time()
        self.lock = threading.Lock()

    @property
    def hist_window(self):
        return HIST_WINDOWS[self.hist_idx % len(HIST_WINDOWS)]

    def record(self, key: str, val: int):
        if self.paused:
            return
        with self.lock:
            self.power[key] = val
            self.peak[key] = max(self.peak[key], val)
            self.history[key].append(val)
            self.energy[key] += val * (self.interval / 1000.0)

    def rolling_avg(self, key: str) -> float:
        with self.lock:
            h = list(self.history[key])
        pts = int(self.hist_window * 1000 / max(self.interval, 1))
        recent = h[-pts:] if h else []
        return sum(recent) / len(recent) if recent else 0.0

    def reset_peaks(self):
        with self.lock:
            for k in self.peak:
                self.peak[k] = self.power[k]


# ── Render ────────────────────────────────────────────────────────────────────

LOGO = [
    "  ╔╗ ╔═╗╔═╗╦╔═╔═╗╔═╗╔═╗╦╔═",
    "  ╠╩╗╠═╣║  ╠╩╗╠═╝╠═╣║  ╠╩╗",
    "  ╚═╝╩ ╩╚═╝╩ ╩╩  ╩ ╩╚═╝╩ ╩",
]
LOGO_COLS = [fg(60, 160, 255), fg(80, 200, 255), fg(130, 220, 200)]

COMP = {
    "CPU": (C_CPU, "◈"),
    "GPU": (C_GPU, "◉"),
    "ANE": (C_ANE, "◆"),
}


def render(state: State) -> str:
    try:
        W = max(60, os.get_terminal_size().columns)
    except OSError:
        W = 100

    out = []
    add = out.append

    def rule(ch="─"):
        add(C_DIM + ch * W + R)

    # ── Logo ──────────────────────────────────────────────────────────────────
    add("")
    for i, row in enumerate(LOGO):
        add(center_str(LOGO_COLS[i] + "\033[1m" + row + R, W))
    add("")

    # ── Status line ───────────────────────────────────────────────────────────
    elapsed = fmt_elapsed(time.time() - state.start)
    live = C_WARN + "\033[1m ⏸ PAUSED " + R if state.paused else C_ANE + "● LIVE" + R
    status = (
        C_DIM
        + f"  ⏱ {elapsed}   "
        + R
        + C_DIM
        + f"window \033[1m{state.hist_window}s\033[0m   "
        + R
        + C_DIM
        + f"interval \033[1m{state.interval}ms\033[0m   "
        + R
        + live
    )
    add(rpad(status, W))
    rule("─")

    # ── Component sections ────────────────────────────────────────────────────
    BAR_W = max(10, W - 4)  # 2 spaces each side

    with state.lock:
        snap_pw = dict(state.power)
        snap_pk = dict(state.peak)
        snap_en = dict(state.energy)
        snap_hi = {k: deque(v) for k, v in state.history.items()}

    for key in ("CPU", "GPU", "ANE"):
        col, icon = COMP[key]
        val = snap_pw[key]
        peak = snap_pk[key]
        avg = state.rolling_avg(key)
        pct = val / peak if peak > 0 else 0.0
        pcol = col if pct < 0.5 else (C_WARN if pct < 0.8 else C_HOT)

        # Metric row
        label = col + "\033[1m" + f"  {icon} {key}" + R
        v_str = pcol + "\033[1m" + f"  {val:>5} mW" + R
        p_str = pcol + f"  {pct*100:>5.1f}%" + R
        k_str = C_DIM + f"   pk {peak:>5} mW" + R
        a_str = C_DIM + f"  avg {avg:>5.0f} mW" + R
        e_str = C_DIM + f"  Σ {fmt_energy(snap_en[key])}" + R

        add(rpad(label + v_str + p_str + k_str + a_str + e_str, W))

        # Bar
        add("  " + power_bar(val, peak, BAR_W, col))

        # Sparkline
        pts_in_window = int(state.hist_window * 1000 / max(state.interval, 1))
        add("  " + sparkline(snap_hi[key], min(BAR_W, pts_in_window), col))
        add("")

    rule("─")

    # ── Totals ────────────────────────────────────────────────────────────────
    t_now = sum(snap_pw.values())
    t_peak = sum(snap_pk.values())
    t_avg = sum(state.rolling_avg(k) for k in ("CPU", "GPU", "ANE"))
    t_mj = sum(snap_en.values())

    add(
        rpad(
            C_WHITE
            + "\033[1m  Σ TOTAL"
            + R
            + C_WARN
            + "\033[1m"
            + f"  {t_now:>5} mW"
            + R
            + C_DIM
            + f"  pk {t_peak} mW"
            + R
            + C_DIM
            + f"  avg {t_avg:.0f} mW"
            + R
            + C_DIM
            + f"  session Σ {fmt_energy(t_mj)}"
            + R,
            W,
        )
    )
    rule("─")

    # ── Keybinds ──────────────────────────────────────────────────────────────
    def badge(k, d):
        return (
            bg(40, 45, 58) + C_WHITE + "\033[1m" + f" {k} " + R + C_DIM + f" {d}  " + R
        )

    kb = "  "
    for k, d in [
        ("q", "quit"),
        ("p", "pause"),
        ("r", "reset pk"),
        ("h", "history"),
        ("+ -", "interval"),
    ]:
        kb += badge(k, d)
    add(kb)
    add("")

    # Join and prepend cursor-home (no clear = no flicker)
    return "\033[H" + "\r\n".join(out) + "\033[J"


# ── Powermetrics ──────────────────────────────────────────────────────────────

POWER_PATS = {
    "CPU": re.compile(r"CPU Power:.*?(\d+)\s*mW", re.IGNORECASE),
    "GPU": re.compile(r"GPU Power:.*?(\d+)\s*mW", re.IGNORECASE),
    "ANE": re.compile(r"ANE Power:.*?(\d+)\s*mW", re.IGNORECASE),
}


def start_reader(state: State):
    # Prompt for sudo password upfront securely
    subprocess.run(["sudo", "-v"], check=True)

    proc = subprocess.Popen(
        [
            "sudo",
            "-S",
            "powermetrics",
            "--samplers",
            "cpu_power,gpu_power,ane_power",
            "-i",
            str(state.interval),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        universal_newlines=True,
        bufsize=1,  # Line buffered
    )

    def _run():
        # Iterate over readline instead of proc.stdout directly to bypass some buffering issues
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            for key, pat in POWER_PATS.items():
                m = pat.match(line.strip())
                if m:
                    state.record(key, int(m.group(1)))

    threading.Thread(target=_run, daemon=True).start()
    return proc


# ── Keyboard ──────────────────────────────────────────────────────────────────


class KeyReader:
    def __enter__(self):
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, *_):
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def read(self, timeout=0.05) -> Optional[str]:
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        return sys.stdin.read(1) if r else None


# ── Main ──────────────────────────────────────────────────────────────────────


def monitor(interval_ms: int = 500):
    state = State(interval_ms)
    proc = start_reader(state)

    sys.stdout.write("\033[?25l\033[2J\033[H")
    sys.stdout.flush()

    def cleanup(*_):
        sys.stdout.write("\033[?25h" + R + "\n")
        sys.stdout.flush()
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except Exception:
            proc.kill()
        sys.exit(0)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    with KeyReader() as keys:
        while True:
            key = keys.read(timeout=0.08)
            if key:
                k = key.lower()
                if k in ("q", "\x03", "\x04"):
                    cleanup()
                elif k == "p":
                    state.paused = not state.paused
                elif k == "r":
                    state.reset_peaks()
                elif k == "h":
                    state.hist_idx += 1
                elif k == "+":
                    state.interval = max(100, state.interval - 100)
                elif k == "-":
                    state.interval = min(5000, state.interval + 100)

            sys.stdout.write(render(state))
            sys.stdout.flush()
            time.sleep(0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--interval", type=int, default=500)
    args = ap.parse_args()
    monitor(interval_ms=args.interval)


if __name__ == "__main__":
    main()
