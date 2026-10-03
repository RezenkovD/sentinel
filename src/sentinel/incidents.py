"""Decision agent: incident finite-state machine with hysteresis (SPEC §7).

NORMAL -> ATTENTION when a window has risk >= theta_high; ATTENTION -> ALARM when k of the last n windows are
that high, or at once when the same person holds a weapon in `weapon_confirm_windows` consecutive windows
(from any calm state; decision 30.09.3: a weapon in a single window is often a false detection). Entering
ALARM opens an incident unless it joins the previous one: the same people, or within merge_after_close_s after
it closed.
ALARM -> COOLDOWN when risk drops below theta_low; COOLDOWN -> NORMAL (close) after quiet_s seconds below
theta_low without a break; COOLDOWN -> ALARM when risk is high again. More than max_new_per_minute new
incidents per minute are folded into one summary instead of separate notifications.

Key frames are opaque snapshots handed to
`save_keyframe(incident_id, name, snapshot, highlight=participants)`; the incident machine
only decides which window's snapshot to keep.
"""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from sentinel.config import IncidentsConfig
from sentinel.schema import RiskResult, SceneDescription

EventKind = Literal["open", "update", "escalate", "close", "summary"]
_WEAPON_UK = {"gun": "вогнепальна зброя", "knife": "ніж"}


def weapon_reason(scene: SceneDescription, reason_uk: str) -> str:
    """An alarm raised by a confirmed weapon names it first, whatever the aggregator said (decision 01.10)."""
    kinds = sorted({p.holds for p in scene.persons if p.holds})
    return "Зброя в руці: " + ", ".join(_WEAPON_UK.get(k, k) for k in kinds) + "; " + reason_uk


KeyframeSaver = Callable[..., Path | None]  # (incident_id, name, snapshot, highlight=participants)


class State(Enum):
    NORMAL = "норма"
    ATTENTION = "увага"
    ALARM = "тривога"
    COOLDOWN = "затихання"


@dataclass(frozen=True)
class IncidentEvent:
    kind: EventKind
    incident_id: str
    source: str
    t: float
    level: str
    max_risk: float
    participants: tuple[int, ...]  # every track id involved (for joining incidents)
    reason_uk: str
    keyframe: Path | None
    started_at: float
    duration_s: float
    merged: bool = False  # the incident was re-opened by a new alarm (joined, not a new incident)
    suppressed: int = 0  # summary: incidents folded in during the current minute
    stream_ended: bool = False  # closed because the video / camera stream ended, not by calm
    people: int = 0  # most people involved at once (track ids pile up when the tracker renumbers)
    weapon: bool = False  # the incident was raised or escalated by a confirmed weapon
    decided_by: str = "A"  # where the risk of the window behind this event came from (A or B)


@dataclass
class Incident:
    id: str
    started_at: float
    max_risk: float
    reason_uk: str
    participants: set[int]
    keyframe: Path | None
    weapon: bool
    last_update: float
    people: int = 0
    suppressed: bool = False
    closed_at: float | None = None
    escalations: int = 0
    updates: int = 0
    reopenings: int = 0
    extra: dict = field(default_factory=dict)


class WeaponConfirmation:
    """The same track holds a weapon in `windows` consecutive windows (decision 30.09.3).

    Shared by the incident automaton and the evaluation of the rule on cached episodes.
    """

    def __init__(self, windows: int) -> None:
        self.windows = windows
        self._holders: deque[tuple[float, float, frozenset[int]]] = deque(maxlen=windows)

    def update(self, scene: SceneDescription) -> bool:
        window = scene.window
        if self._holders and abs(self._holders[-1][1] - window.t_start) > 0.011:  # a gap (2-decimal times)
            self._holders.clear()
        self._holders.append(
            (window.t_start, window.t_end, frozenset(p.id for p in scene.persons if p.holds))
        )
        return len(self._holders) == self.windows and bool(
            frozenset.intersection(*(ids for _, _, ids in self._holders))
        )


def participants(scene: SceneDescription) -> set[int]:
    """People involved: those in described pairs or holding a weapon; everyone in view if nobody is."""
    involved = {i for pair in scene.interactions for i in pair.pair}
    involved |= {p.id for p in scene.persons if p.holds}
    return involved or {p.id for p in scene.persons}


class IncidentManager:
    def __init__(self, cfg: IncidentsConfig, source: str, save_keyframe: KeyframeSaver, run_id: str) -> None:
        self.cfg = cfg
        self.source = source
        self.save_keyframe = save_keyframe
        self.run_id = run_id
        self.state = State.NORMAL
        self._high: deque[bool] = deque(maxlen=cfg.n)
        # (risk, snapshot) of the last n windows since ATTENTION began: bounded, so a long ATTENTION spell
        # neither piles up full-resolution frames nor offers a stale key frame
        self._candidates: deque[tuple[float, Any]] = deque(maxlen=cfg.n)
        self._incident: Incident | None = None
        self._last: Incident | None = None
        self._quiet_since: float | None = None
        self._opens: deque[float] = deque()
        self._weapon = WeaponConfirmation(cfg.weapon_confirm_windows)
        self._weapon_confirmed_now = False
        self._summary: tuple[str, float, int] | None = None  # id, minute start, suppressed count
        self._count = 0
        self.last_incident_id: str | None = None
        self._decided_by = "A"

    def on_window(
        self,
        t: float,
        risk: RiskResult,
        scene: SceneDescription,
        snapshot: Any = None,
        decided_by: str = "A",
    ) -> list[IncidentEvent]:
        cfg = self.cfg
        self._decided_by = decided_by
        r = risk.risk
        held = self._weapon_confirmed(scene)
        high = r >= cfg.theta_high
        self._high.append(high)
        events: list[IncidentEvent] = []
        if self.state is State.NORMAL:
            if held or high:
                self._candidates.clear()
                self._candidates.append((r, snapshot))
                if held:
                    events += self._alarm(t, risk, scene, snapshot)
                else:
                    self.state = State.ATTENTION
        elif self.state is State.ATTENTION:
            self._candidates.append((r, snapshot))
            if held or sum(self._high) >= cfg.k:
                events += self._alarm(t, risk, scene, snapshot)
            elif r < cfg.theta_low:
                self.state = State.NORMAL
                self._candidates.clear()
        elif self.state is State.ALARM:
            events += self._continue(t, risk, scene, snapshot, held)
            if r < cfg.theta_low:
                self.state = State.COOLDOWN
                self._quiet_since = t
        else:  # COOLDOWN
            if held or high:
                self.state = State.ALARM
                self._quiet_since = None
                events += self._continue(t, risk, scene, snapshot, held)
            elif r < cfg.theta_low:
                if self._quiet_since is None:
                    self._quiet_since = t
                if t - self._quiet_since >= cfg.quiet_s:
                    events += self._close(t)
            else:
                self._quiet_since = None  # calm must last quiet_s without a break
        return events

    # --- transitions ---

    def _alarm(
        self, t: float, risk: RiskResult, scene: SceneDescription, snapshot: Any
    ) -> list[IncidentEvent]:
        """Enter ALARM: re-open the previous incident if it qualifies, else open a new one."""
        self.state = State.ALARM
        self._quiet_since = None
        people = participants(scene)
        held = self._weapon_confirmed_now
        best_risk, best_snapshot = max(self._candidates or [(risk.risk, snapshot)], key=lambda c: c[0])
        self._candidates.clear()
        last = self._last
        if last is not None and last.closed_at is not None and self._joins(last, people, t):
            self._incident, self._last = last, None
            last.closed_at = None
            last.reopenings += 1
            last.participants |= people
            last.people = max(last.people, len(people))
            was_weapon, last.weapon = last.weapon, last.weapon or held
            self._raise_max(last, risk)
            if held and not was_weapon:
                last.reason_uk = weapon_reason(scene, risk.reason_uk)
            last.last_update = t
            path = (
                None
                if last.suppressed
                else self.save_keyframe(
                    last.id, f"reopen_{last.reopenings}", snapshot, highlight=last.participants
                )
            )
            return [] if last.suppressed else [self._event("update", last, t, keyframe=path, merged=True)]

        self._count += 1
        incident_id = f"{self.run_id}-{self.source}-{self._count:03d}"
        self.last_incident_id = incident_id
        incident = Incident(
            id=incident_id,
            started_at=t,
            max_risk=max(risk.risk, best_risk),
            reason_uk=weapon_reason(scene, risk.reason_uk) if held else risk.reason_uk,
            participants=people,
            keyframe=None,
            weapon=held,
            last_update=t,
            people=len(people),
        )
        self._incident = incident
        while self._opens and t - self._opens[0] > 60.0:
            self._opens.popleft()
        if len(self._opens) >= self.cfg.max_new_per_minute:
            incident.suppressed = True
            return [self._summary_event(t, risk)]
        self._opens.append(t)
        incident.keyframe = self.save_keyframe(incident_id, "keyframe", best_snapshot, highlight=people)
        return [self._event("open", incident, t, keyframe=incident.keyframe)]

    def _continue(
        self, t: float, risk: RiskResult, scene: SceneDescription, snapshot: Any, held: bool
    ) -> list[IncidentEvent]:
        incident = self._incident
        assert incident is not None
        involved = participants(scene)
        incident.participants |= involved
        incident.people = max(incident.people, len(involved))
        self._raise_max(incident, risk)
        if incident.suppressed:
            incident.weapon = incident.weapon or held
            return []
        if held and not incident.weapon:
            incident.weapon = True
            incident.escalations += 1
            incident.reason_uk = weapon_reason(scene, risk.reason_uk)
            incident.last_update = t
            path = self.save_keyframe(
                incident.id, f"escalate_{incident.escalations}", snapshot, highlight=incident.participants
            )
            return [self._event("escalate", incident, t, keyframe=path)]
        if t - incident.last_update >= self.cfg.update_interval_s:
            incident.last_update = t
            incident.updates += 1  # a fresh frame with every update (decision 03.10)
            path = self.save_keyframe(
                incident.id, f"update_{incident.updates}", snapshot, highlight=incident.participants
            )
            return [self._event("update", incident, t, keyframe=path)]
        return []

    def finish(self, t: float) -> list[IncidentEvent]:
        """End of the stream: close an incident that is still open (its message must not stay "ALARM")."""
        self._candidates.clear()
        if self._incident is None:
            self.state = State.NORMAL
            return []
        return self._close(t, stream_ended=True)

    def _close(self, t: float, stream_ended: bool = False) -> list[IncidentEvent]:
        incident = self._incident
        assert incident is not None
        self.state = State.NORMAL
        self._quiet_since = None
        incident.closed_at = t
        self._incident, self._last = None, incident
        if incident.suppressed:
            return []
        return [self._event("close", incident, t, level="normal", stream_ended=stream_ended)]

    # --- helpers ---

    def _weapon_confirmed(self, scene: SceneDescription) -> bool:
        self._weapon_confirmed_now = self._weapon.update(scene)
        return self._weapon_confirmed_now

    def _joins(self, last: Incident, people: set[int], t: float) -> bool:
        since_close = t - (last.closed_at or t)
        if since_close <= self.cfg.merge_after_close_s:
            return True
        limit = self.cfg.merge_participants_max_s
        return bool(last.participants & people) and (limit is None or since_close <= limit)

    @staticmethod
    def _raise_max(incident: Incident, risk: RiskResult) -> None:
        if risk.risk > incident.max_risk:
            incident.max_risk = risk.risk
            # a weapon incident keeps naming the weapon; the holder may be out of view in this window
            weapon = incident.reason_uk.split("; ", 1)[0] if incident.weapon else None
            incident.reason_uk = f"{weapon}; {risk.reason_uk}" if weapon else risk.reason_uk

    def _summary_event(self, t: float, risk: RiskResult) -> IncidentEvent:
        minute = t - t % 60.0
        if self._summary is None or self._summary[1] != minute:
            self._summary = (f"{self.run_id}-{self.source}-summary-{int(minute)}", minute, 0)
        summary_id, _, count = self._summary
        self._summary = (summary_id, minute, count + 1)
        return IncidentEvent(
            kind="summary",
            incident_id=summary_id,
            source=self.source,
            t=t,
            level="alarm",
            max_risk=risk.risk,
            participants=(),
            reason_uk=risk.reason_uk,
            keyframe=None,
            started_at=minute,
            duration_s=0.0,
            suppressed=count + 1,
        )

    def _event(
        self,
        kind: EventKind,
        incident: Incident,
        t: float,
        keyframe: Path | None = None,
        merged: bool = False,
        level: str = "alarm",
        stream_ended: bool = False,
    ) -> IncidentEvent:
        return IncidentEvent(
            kind=kind,
            incident_id=incident.id,
            source=self.source,
            t=t,
            level=level,
            max_risk=round(incident.max_risk, 2),
            participants=tuple(sorted(incident.participants)),
            reason_uk=incident.reason_uk,
            keyframe=keyframe,
            started_at=incident.started_at,
            duration_s=round(t - incident.started_at, 2),
            merged=merged,
            stream_ended=stream_ended,
            people=incident.people,
            weapon=incident.weapon,
            decided_by=self._decided_by,
        )
