"""Live FABRIC dataplane state from the public metrics API.

public-metrics.fabric-testbed.net exposes per-link SNMP gauges
(dataplaneInBits/dataplaneOutBits, labels src_rack/dst_rack, capacity in a
`max` label) through an anonymously queryable Grafana datasource proxy.
From one instant query we get the inter-rack link graph plus current
utilization; BFS gives the shortest rack path between two sites and the
bottleneck headroom along it, which prices a checkpoint transfer.

Fail-open by design: on any fetch error the model reports "no data" and the
engine migrates on carbon alone — the booth demo must not die with the API.
"""

import threading
import time
from collections import deque
from typing import Dict, Optional, Tuple

import requests

MIMIR_PROXY = ("https://public-metrics.fabric-testbed.net/grafana/api/"
               "datasources/proxy/uid/P5A7A89F352A86A10")
DEFAULT_CAPACITY_BPS = 100e9


class FabricNetworkModel:
    def __init__(self, url: str = MIMIR_PROXY, ttl: float = 60.0,
                 efficiency: float = 0.7):
        self.url = url
        self.ttl = ttl
        # fraction of headroom a single transfer realistically gets
        self.efficiency = efficiency
        # (rack_a, rack_b) sorted -> {"util_bps", "cap_bps"}
        self._links: Dict[Tuple[str, str], dict] = {}
        self._fetched_at = 0.0
        self._ok = False
        self._lock = threading.Lock()

    # -- data ---------------------------------------------------------------

    def _query(self, promql: str) -> list:
        resp = requests.get(f"{self.url}/api/v1/query",
                            params={"query": promql}, timeout=10)
        resp.raise_for_status()
        body = resp.json()
        if body.get("status") != "success":
            raise RuntimeError(f"query failed: {body}")
        return body["data"]["result"]

    def _refresh(self):
        with self._lock:
            if time.time() - self._fetched_at < self.ttl:
                return
            self._fetched_at = time.time()  # even on failure, don't hammer
        try:
            links: Dict[Tuple[str, str], dict] = {}
            for metric in ("dataplaneInBits", "dataplaneOutBits"):
                for series in self._query(metric):
                    lb = series["metric"]
                    src, dst = lb.get("src_rack"), lb.get("dst_rack")
                    if not src or not dst or src == dst:
                        continue
                    key = tuple(sorted((src.lower(), dst.lower())))
                    util = float(series["value"][1])
                    cap = float(lb.get("max") or DEFAULT_CAPACITY_BPS)
                    link = links.setdefault(key, {"util_bps": 0.0, "cap_bps": cap})
                    # Series repeat per ruler/ifIndex/direction; keep the
                    # most-loaded reading and largest capacity seen.
                    link["util_bps"] = max(link["util_bps"], util)
                    link["cap_bps"] = max(link["cap_bps"], cap)
            if links:
                with self._lock:
                    self._links = links
                    self._ok = True
        except Exception as e:
            with self._lock:
                self._ok = False
            print(f"[fabric-metrics] refresh failed: {e}", flush=True)

    # -- path math ----------------------------------------------------------

    def _shortest_path(self, src: str, dst: str) -> Optional[list]:
        adj: Dict[str, list] = {}
        for a, b in self._links:
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
        if src not in adj or dst not in adj:
            return None
        prev, seen, q = {}, {src}, deque([src])
        while q:
            cur = q.popleft()
            if cur == dst:
                path = [dst]
                while path[-1] != src:
                    path.append(prev[path[-1]])
                return path[::-1]
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    prev[nxt] = cur
                    q.append(nxt)
        return None

    def estimate(self, src_site: str, dst_site: str, nbytes: Optional[float],
                 rate_hint_bps: Optional[float] = None) -> dict:
        """Estimate a checkpoint transfer src->dst.

        rate_hint_bps, when given, is measured throughput from previous real
        transfers (engine EWMA) and caps the assumed rate — the path headroom
        is an upper bound, not what scp actually achieves.

        Returns {ok, path, hops, headroom_gbps, est_transfer_s}; ok=False
        means no usable data (engine should fail open).
        """
        self._refresh()
        with self._lock:
            if not self._ok:
                return {"ok": False}
            links = dict(self._links)

        path = self._shortest_path(src_site.lower(), dst_site.lower())
        if not path or len(path) < 2:
            return {"ok": False}

        headroom = float("inf")
        for a, b in zip(path, path[1:]):
            link = links[tuple(sorted((a, b)))]
            headroom = min(headroom, max(0.0, link["cap_bps"] - link["util_bps"]))

        usable = headroom * self.efficiency
        calibrated = rate_hint_bps is not None and rate_hint_bps < usable
        if calibrated:
            usable = rate_hint_bps
        est_s = (nbytes * 8 / usable) if (nbytes and usable > 0) else None
        return {
            "ok": True,
            "path": path,
            "hops": len(path) - 1,
            "headroom_gbps": round(headroom / 1e9, 1),
            "rate_gbps_used": round(usable / 1e9, 2),
            "calibrated": calibrated,
            "est_transfer_s": round(est_s, 2) if est_s is not None else None,
        }
