"""Zone context agent: static zone profile (type, working hours, expected density) from configs/zones.yaml."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from sentinel.schema import Zone


@dataclass(frozen=True)
class ZoneProfile:
    type: str
    hours: str  # "day", "night", or "auto" (from the local clock and day_hours)
    expected_density: str
    day_hours: tuple[int, int] = (7, 22)  # [start, end) hours counted as day when hours = "auto"


class ZoneContext:
    """One zone per source; unknown sources get the default zone."""

    def __init__(
        self, zones: dict[str, ZoneProfile], sources: dict[str, str] | None = None, default: str = "default"
    ) -> None:
        if default not in zones:
            raise ValueError(f"no default zone {default!r} in {sorted(zones)}")
        self.zones = zones
        self.sources = sources or {}
        self.default = default

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ZoneContext":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        zones = {
            name: ZoneProfile(**{**z, "day_hours": tuple(z.get("day_hours", (7, 22)))})
            for name, z in data["zones"].items()
        }
        return cls(zones, data.get("sources") or {})

    def zone_for(self, source: str, now: datetime | None = None) -> Zone:
        profile = self.zones[self.sources.get(source, self.default)]
        hours = profile.hours
        if hours == "auto":
            start, end = profile.day_hours
            hour = (now or datetime.now()).hour
            hours = "day" if start <= hour < end else "night"
        return Zone(type=profile.type, hours=hours, expected_density=profile.expected_density)
