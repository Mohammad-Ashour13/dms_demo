from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IncidentPolicyResult:
    qualifies: bool
    onset: bool
    trigger_kinds: tuple[str, ...]


class IncidentTriggerPolicy:
    """Create bounded clips and suppress yawn-only/repeated same-risk clips.

    A material severity escalation may start a second fixed clip only when the
    earlier clip has already ended. This prevents a stale behavior episode from
    suppressing later DROWSY/CRITICAL evidence without returning to the old
    repeated-clip behavior.
    """

    STATE_PRIORITY = {
        "NORMAL": 0,
        "FATIGUE_WARNING": 1,
        "DROWSY": 2,
        "CRITICAL": 3,
    }

    def __init__(
        self,
        trigger_states,
        trigger_violations,
        *,
        suppress_yawn_only=True,
        episode_clear_sec=2.0,
    ):
        self.trigger_states = {str(value) for value in trigger_states}
        self.trigger_violations = {str(value) for value in trigger_violations}
        self.suppress_yawn_only = bool(suppress_yawn_only)
        self.episode_clear_sec = float(episode_clear_sec)
        self.episode_active = False
        self.clear_since: float | None = None
        self.highest_priority = 0

    def evaluate(
        self,
        now: float,
        decision,
        *,
        recording_active: bool | None = None,
    ) -> IncidentPolicyResult:
        state = str(decision.driver_state.value)
        violations = {str(value) for value in decision.violations}
        matched = sorted(violations & self.trigger_violations)
        state_match = state in self.trigger_states
        yawn_only = bool(violations) and violations <= {"YAWNING", "REPEATED_YAWNS"}
        qualifies = bool(matched or state_match)
        if (
            self.suppress_yawn_only
            and yawn_only
            and state not in {"DROWSY", "CRITICAL"}
        ):
            qualifies = False
        kinds = tuple(([state] if state_match else []) + matched)
        priority = max(
            self.STATE_PRIORITY.get(state, 0),
            1 if matched else 0,
        )

        onset = False
        if qualifies:
            self.clear_since = None
            if not self.episode_active:
                self.episode_active = True
                onset = True
                self.highest_priority = priority
            elif priority > self.highest_priority:
                # None preserves the legacy caller contract: without explicit
                # recorder state, assume the current clip still covers escalation.
                onset = recording_active is False
                self.highest_priority = priority
        elif self.episode_active:
            if self.clear_since is None:
                self.clear_since = float(now)
            elif now - self.clear_since >= self.episode_clear_sec:
                self.episode_active = False
                self.clear_since = None
                self.highest_priority = 0
        return IncidentPolicyResult(qualifies, onset, kinds)
