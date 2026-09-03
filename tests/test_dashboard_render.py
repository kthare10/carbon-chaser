"""Dashboard rendering tests, at the DOM level.

The recurring bug class here is *invisibility*: state that is correct in the
API but renders as nothing, or renders as a measurement it isn't. A single
stale sample once emitted a lone SVG `moveto` — a path with no line segment,
which draws literally nothing — so intermittent staleness was invisible
while the surrounding solid line still read as continuous measurement. No
amount of state-level assertion catches that; the check has to look at what
was actually drawn.

Skipped automatically when headless Chrome is unavailable.
"""

import http.server
import json
import re
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

STATIC = os.path.join(os.path.dirname(__file__), "..", "carbon_chaser",
                      "static")
CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    shutil.which("google-chrome") or "",
    shutil.which("chromium") or "",
]


def find_chrome():
    for path in CHROME_CANDIDATES:
        if path and os.path.exists(path):
            return path
    return None


def build_state():
    """A history with three regimes: measured, intermittently stale, then a
    site missing entirely."""
    sites = ["CLEM", "TACC", "UTAH", "STAR", "UCSD", "NEWY"]
    zones = {s: f"Z-{s}" for s in sites}
    history = []
    for i in range(30):
        intens, stale = {}, []
        for s in sites:
            if s == "UTAH" and 10 <= i and i % 2:       # flaps every other tick
                intens[s] = 500.0
                stale.append(s)
            elif s == "UCSD" and i >= 20:               # gone entirely
                continue
            else:
                intens[s] = 300.0 + i
        history.append({"sim_t": i * 60, "hour": 6 + i / 10,
                        "intensities": intens, "stale": stale,
                        "active": "CLEM", "step": i * 20, "loss": 0.5,
                        "saved_g": i})
    last = history[-1]
    return {
        "sim_time_s": 1800, "sim_hour": 9.0, "accel": 300.0,
        "status": "running", "health": "ok", "health_note": None,
        "active_site": "CLEM", "managed_sites": sites,
        "sites": {
            s: {"display": s, "lat": 35 + i, "lon": -80 - i * 5,
                "zone": zones[s],
                "intensity": last["intensities"].get(s),
                "data": ("missing" if s not in last["intensities"]
                         else "stale" if s in last["stale"] else "fresh"),
                "managed": True}
            for i, s in enumerate(sites)
        },
        "emissions_g": 100.0, "baseline_g": 150.0, "saved_g": 50.0,
        "migrations": [], "progress": {"step": 600, "loss": 0.5, "acc": 0.8},
        "history": history,
        "carbon_source": {"kind": "trace-replay", "detail": "test",
                          "injected_events": 0},
        "carbon_status": "stale", "carbon_note": "test outage",
        "net_estimates": {}, "net_note": None,
        "measured_rate_gbps": None, "measured_setup_s": None,
        "measured_orchestration_s": None,
    }


def serve(state, directory):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=directory, **k)

        def do_GET(self):
            if self.path.startswith("/api/state"):
                body = json.dumps(state).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/":
                self.path = "/index.html"
            return super().do_GET()

        def log_message(self, *a):
            pass

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    srv = http.server.HTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, port


def dump_dom(chrome, url):
    out = subprocess.run(
        [chrome, "--headless", "--disable-gpu", "--dump-dom",
         "--virtual-time-budget=6000", url],
        capture_output=True, text=True, timeout=180)
    return out.stdout


def main():
    chrome = find_chrome()
    if not chrome:
        print("  SKIP: headless Chrome not found")
        return

    tmp = tempfile.mkdtemp()
    for name in ("index.html", "us-map.js"):
        shutil.copy(os.path.join(STATIC, name), os.path.join(tmp, name))
    # the page loads /static/us-map.js
    os.makedirs(os.path.join(tmp, "static"), exist_ok=True)
    shutil.copy(os.path.join(STATIC, "us-map.js"),
                os.path.join(tmp, "static", "us-map.js"))

    srv, port = serve(build_state(), tmp)
    try:
        dom = dump_dom(chrome, f"http://127.0.0.1:{port}/")
    finally:
        srv.shutdown()

    assert "Carbon Chaser" in dom, "page did not render"

    # 1. intermittent staleness must draw actual line segments, not lone
    #    movetos. Parse per TAG: splitting the DOM on "<path" conflated the
    #    US-map border path (whose `d` is full of L commands) with the site
    #    markers' stroke-dasharray, and passed regardless.
    tags = re.findall(r"<path\b[^>]*>", dom)
    stale_paths = [t for t in tags if "stale-series" in t]
    assert stale_paths, "no stale-series path rendered for stale samples"
    for tag in stale_paths:
        d = re.search(r'd="([^"]*)"', tag)
        assert d, f"stale path has no d: {tag[:80]}"
        assert "L" in d.group(1), (
            "stale path contains only movetos — a lone `M` draws nothing, so "
            f"intermittent staleness is invisible: {d.group(1)[:60]}")
    print(f"  {len(stale_paths)} stale series drawn with real line segments")

    # 2. the data-quality ribbon marks the affected ticks
    marks = re.findall(r'<rect\b[^>]*class="quality-(stale|missing)"', dom)
    assert marks, "no data-quality ribbon marks rendered"
    assert "data quality:" in dom, "ribbon legend missing"
    print(f"  data-quality ribbon: {len(marks)} marks + legend")

    # 3. a missing reading must not be drawn as a value
    assert "no data" in dom, "missing site not labelled 'no data'"
    print("  missing readings labelled 'no data'")

    # 4. a stale reading must be labelled, not shown as a plain measurement
    assert "· stale" in dom, "stale site not labelled on the map"
    print("  stale readings labelled '· stale' on the map")

    # 5. the paused counter must say why
    assert "counter paused" in dom, "paused counter not explained"
    print("  paused CO2 counter explains itself")


if __name__ == "__main__":
    print("test_dashboard_render")
    main()
    print("\nALL PASS")
