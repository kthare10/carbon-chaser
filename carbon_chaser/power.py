"""Node power providers — the second factor in the CO2 arithmetic.

`CO2 = power x time x carbon_intensity`. With intensity now real (EIA-derived
traces), assumed power is the only synthetic term left, so it gets the same
treatment as carbon: a provider interface, explicit provenance, and a refusal
to let a modelled number pass as a measurement.

Power comes from NVML only. FABRIC GPUs are PCI passthrough, so NVML reads
the board's own sensor inside the VM. There is no assumed-power mode: when a
sample is missing or stale the provider returns None and the engine pauses
emissions accounting for that site rather than substituting a guess.
"""

import json
import os
from typing import Dict, Optional


class PowerProvider:
    def get_kw(self, site: str) -> Optional[float]:
        """Power draw at `site` in kW, or None if unknown."""
        raise NotImplementedError

    def describe(self) -> dict:
        return {"kind": "unknown", "detail": "", "measured": False}


class MeasuredPowerProvider(PowerProvider):
    """Reads NVML samples the workload writes next to its checkpoint.

    Contract (`<workdir>/power.json`, written atomically):
        {"watts": 71.4, "ts": 1787.., "device": "Tesla T4",
         "samples": 120, "window_s": 30}

    A reading older than `max_age_s` is not used: like carbon data, stale
    power must not be integrated as if current. When there is no usable
    sample this returns None and the engine pauses accounting for that site
    — no invented watts.
    """

    def __init__(self, executor, max_age_s: float = 120.0):
        self.executor = executor
        self.max_age_s = max_age_s
        self._last: Dict[str, dict] = {}
        self._stale_or_missing: Dict[str, str] = {}
        self._devices: Dict[str, str] = {}

    def get_kw(self, site: str) -> Optional[float]:
        import time
        sample = None
        reader = getattr(self.executor, "get_power", None)
        if reader is not None:
            try:
                sample = reader(site)
            except Exception:
                sample = None
        if sample and sample.get("watts") is not None:
            age = time.time() - (sample.get("ts") or 0)
            if age <= self.max_age_s:
                self._last[site] = sample
                if sample.get("device"):
                    self._devices[site] = sample["device"]
                self._stale_or_missing.pop(site, None)
                return float(sample["watts"]) / 1000.0
            self._stale_or_missing[site] = f"sample {int(age)}s old"
        else:
            self._stale_or_missing[site] = "no NVML sample"
        return None

    def describe(self) -> dict:
        if self._stale_or_missing:
            worst = ", ".join(f"{k} ({v})" for k, v in
                              sorted(self._stale_or_missing.items())[:3])
            return {"kind": "unavailable",
                    "detail": f"NVML power unavailable for: {worst} — "
                              f"emissions accounting paused for those sites",
                    "measured": False}
        devices = ", ".join(sorted(set(self._devices.values()))) or "GPU"
        return {"kind": "measured",
                "detail": f"NVML board power from {devices} "
                          f"({len(self._last)} site(s) reporting)",
                "measured": True}


def make_power_provider(cfg: Optional[dict], executor) -> PowerProvider:
    """Always measured. There is no assumed-power mode to fall into."""
    cfg = cfg or {}
    return MeasuredPowerProvider(
        executor, max_age_s=float(cfg.get("power_max_age_s", 120)))
