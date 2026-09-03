"""Simulation clock.

All engine/provider logic reads time from a Clock so the demo can run at
accelerated speed (booth mode) or wall-clock speed (live Electricity Maps
feed) without code changes.
"""

import time


class Clock:
    """Wall-clock time; accel is 1."""

    accel = 1.0

    def __init__(self):
        self._epoch = time.time()

    def now(self) -> float:
        """Seconds since the clock was created (sim time)."""
        return time.time() - self._epoch

    def hour_of_day(self) -> float:
        return (time.localtime().tm_hour
                + time.localtime().tm_min / 60.0)


class SimClock(Clock):
    """Accelerated clock: `accel` sim-seconds pass per real second."""

    def __init__(self, accel: float = 120.0, start_hour: float = 6.0):
        super().__init__()
        self.accel = accel
        self._start_hour = start_hour

    def now(self) -> float:
        return (time.time() - self._epoch) * self.accel

    def hour_of_day(self) -> float:
        return (self._start_hour + self.now() / 3600.0) % 24.0
