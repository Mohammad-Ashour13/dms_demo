from __future__ import annotations

from collections import deque

from dms_final_system.shared.contracts import (
    AlarmLevel,
    DriverState,
    EvidenceEvent,
    FusionDecision,
    ModelPrediction,
)


STATE_RANK = {
    DriverState.UNKNOWN: -1,
    DriverState.NORMAL: 0,
    DriverState.FATIGUE_WARNING: 1,
    DriverState.DROWSY: 2,
    DriverState.CRITICAL: 3,
}
VIOLATION_KINDS = {
    "YAWNING", "EXCESSIVE_BLINKING", "PROLONGED_EYE_CLOSURE",
    "REPEATED_YAWNS", "PHONE_USE", "SMOKING", "EATING", "DISTRACTION",
    "SEATBELT_MISSING",
}


class FusionStateMachine:
    """Traceable fusion where an experimental model is supporting evidence only."""

    def __init__(
        self,
        ewma_alpha=0.35,
        warning_enter=0.55,
        warning_exit=0.42,
        drowsy_enter=0.75,
        drowsy_exit=0.58,
        warning_persistence_sec=2.0,
        drowsy_persistence_sec=1.5,
        precritical_drowsy_closure_sec=0.50,
        normal_persistence_sec=3.0,
        recovery_persistence_sec=3.0,
        quality_loss_persistence_sec=0.5,
        quality_recovery_persistence_sec=1.0,
        transition_cooldown_sec=1.0,
        evidence_ttl_sec=5.0,
        excessive_blinks_per_min=25,
        excessive_yawns_per_min=2,
        perclos_warning=0.25,
        perclos_exit=0.18,
        perclos_min_valid_sec=20.0,
        perclos_min_coverage=0.67,
        require_evidence_for_drowsy=True,
        require_evidence_for_warning=True,
        **_,
    ):
        self.alpha = float(ewma_alpha)
        self.warning_enter, self.warning_exit = float(warning_enter), float(warning_exit)
        self.drowsy_enter, self.drowsy_exit = float(drowsy_enter), float(drowsy_exit)
        self.warning_persistence = float(warning_persistence_sec)
        self.drowsy_persistence = float(drowsy_persistence_sec)
        self.precritical_drowsy_closure = float(precritical_drowsy_closure_sec)
        self.normal_persistence = float(normal_persistence_sec)
        self.recovery_persistence = float(recovery_persistence_sec)
        self.quality_loss_persistence = float(quality_loss_persistence_sec)
        self.quality_recovery_persistence = float(quality_recovery_persistence_sec)
        self.cooldown = float(transition_cooldown_sec)
        self.default_ttl = float(evidence_ttl_sec)
        self.excessive_blinks = int(excessive_blinks_per_min)
        self.excessive_yawns = int(excessive_yawns_per_min)
        self.perclos_warning = float(perclos_warning)
        self.perclos_exit = float(perclos_exit)
        self.perclos_min_valid = float(perclos_min_valid_sec)
        self.perclos_min_coverage = float(perclos_min_coverage)
        self.require_evidence_for_drowsy = bool(require_evidence_for_drowsy)
        self.require_evidence_for_warning = bool(require_evidence_for_warning)
        self.state = DriverState.UNKNOWN
        self.ewma: float | None = None
        self.last_window_id: str | None = None
        self.candidate: DriverState | None = None
        self.candidate_since: float | None = None
        self.last_transition = float("-inf")
        self.state_entered_at: float | None = None
        self.state_entry_reason = ["INITIAL_STATE"]
        self.critical_recovery_since: float | None = None
        self.quality_lost_since: float | None = None
        self.last_update_at: float | None = None
        self.has_reached_operational_state = False
        self.unknown_due_to_quality = False
        self.post_critical_recovery = False
        self.post_critical_started_at: float | None = None
        self.evidence: deque[EvidenceEvent] = deque()

    def add_evidence(self, events: list[EvidenceEvent]) -> None:
        self.evidence.extend(events)

    def _fresh_evidence(self, now: float) -> list[EvidenceEvent]:
        kept = [item for item in self.evidence if item.is_fresh(now)]
        self.evidence = deque(kept)
        return kept

    def _perclos_flags(self, metrics: dict) -> tuple[bool, bool, bool]:
        if "decision_perclos_30s" in metrics:
            observed = float(metrics.get("decision_valid_observation_sec_30s") or 0.0)
            coverage = float(metrics.get("decision_perclos_coverage_30s") or 0.0)
            perclos = metrics.get("decision_perclos_30s")
        else:
            observed = float(
                metrics.get("valid_observation_sec_30s")
                or metrics.get("valid_observation_sec")
                or 0.0
            )
            coverage = float(metrics.get("perclos_coverage_30s") or 0.0)
            perclos = metrics.get("perclos_30s")
        ready = bool(
            perclos is not None
            and observed >= self.perclos_min_valid
            and coverage >= self.perclos_min_coverage
        )
        enter = ready and float(perclos) >= self.perclos_warning
        remain = ready and float(perclos) > self.perclos_exit
        return ready, enter, remain

    def _target(
        self,
        probability: float,
        events: list[EvidenceEvent],
        metrics: dict,
        model_enabled: bool,
    ) -> tuple[DriverState, list[str], bool]:
        kinds = {event.kind for event in events}
        closure = float(metrics.get("current_closure_sec") or 0.0)
        if (
            metrics.get("eye_signal_valid", False)
            and closure >= 1.5
            and metrics.get("strong_closure_valid", False)
        ):
            return DriverState.CRITICAL, ["CRITICAL_TRUSTED_EYE_CLOSURE"], False

        yawn_active = bool(
            "YAWNING" in kinds
            or
            metrics.get("mouth_state") == "OPEN"
            or float(metrics.get("current_mouth_open_sec") or 0.0) > 0.0
        )
        trustworthy_long_blink = any(
            event.kind == "LONG_BLINK"
            and event.details.get("physiology_trustworthy", True)
            for event in events
        )
        drowsy_event = bool(
            "HEAD_NOD" in kinds or trustworthy_long_blink
        ) and not yawn_active
        partial_closure = bool(metrics.get("sustained_partial_closure", False)) and not yawn_active
        current_strong_bilateral_closure = bool(
            metrics.get("eye_state") == "CLOSED"
            and closure >= self.precritical_drowsy_closure
            and max(
                float(metrics.get("event_left_relative_ear") or 0.0),
                float(metrics.get("event_right_relative_ear") or 0.0),
            ) <= 0.40
        )
        drowsy_physiology = (
            drowsy_event or partial_closure or current_strong_bilateral_closure
        )
        warning_physiology = drowsy_physiology or "YAWNING" in kinds
        perclos_ready, perclos_enter, perclos_remain = self._perclos_flags(metrics)
        model_drowsy = model_enabled and probability >= self.drowsy_enter
        model_warning = model_enabled and probability >= self.warning_enter
        model_risk_pending = bool(model_warning and not warning_physiology)

        if self.post_critical_recovery:
            new_physiology = any(
                event.kind in {"HEAD_NOD", "LONG_BLINK"}
                and self.post_critical_started_at is not None
                and event.monotonic_sec > self.post_critical_started_at
                for event in events
            )
            if new_physiology:
                self.post_critical_recovery = False
                self.post_critical_started_at = None
            elif perclos_ready and not perclos_remain:
                self.post_critical_recovery = False
                self.post_critical_started_at = None
            else:
                return (
                    DriverState.FATIGUE_WARNING,
                    ["POST_CRITICAL_FATIGUE_MEMORY"],
                    model_risk_pending,
                )

        perclos_drowsy = perclos_enter or (
            self.state == DriverState.DROWSY and perclos_remain
        )
        if perclos_drowsy:
            return DriverState.DROWSY, ["PERCLOS_ABOVE_DROWSY_THRESHOLD"], model_risk_pending
        if model_drowsy and (drowsy_physiology or not self.require_evidence_for_drowsy):
            reasons = ["MODEL_ABOVE_DROWSY_ENTER"]
            if drowsy_event:
                reasons.append("STRONG_PHYSIOLOGICAL_EVIDENCE")
            if partial_closure:
                reasons.append("SUSTAINED_PARTIAL_EYE_CLOSURE")
            if current_strong_bilateral_closure:
                reasons.append("CURRENT_STRONG_BILATERAL_CLOSURE")
            return DriverState.DROWSY, reasons, False

        warning_supported = model_warning and (
            warning_physiology or not self.require_evidence_for_warning
        )
        if warning_supported:
            reasons: list[str] = []
            if model_warning:
                reasons.append("MODEL_ABOVE_WARNING_ENTER")
            if model_drowsy and self.require_evidence_for_drowsy and not drowsy_physiology:
                reasons.append("MODEL_DROWSY_BLOCKED_WITHOUT_PHYSIOLOGY")
            if drowsy_event or "YAWNING" in kinds:
                reasons.append("PHYSIOLOGICAL_SUPPORT")
            if partial_closure:
                reasons.append("SUSTAINED_PARTIAL_EYE_CLOSURE")
            return DriverState.FATIGUE_WARNING, reasons, False
        if (
            self.state == DriverState.FATIGUE_WARNING
            and model_enabled
            and probability >= self.warning_exit
            and warning_physiology
        ):
            return DriverState.FATIGUE_WARNING, ["MODEL_ABOVE_WARNING_EXIT_WITH_PHYSIOLOGY"], False
        reasons = ["RISK_BELOW_EXIT_THRESHOLDS"]
        if model_risk_pending:
            reasons.append("MODEL_RISK_PENDING")
        return DriverState.NORMAL, reasons, model_risk_pending

    def _transition(self, target: DriverState, now: float, reasons: list[str]) -> None:
        self.state = target
        self.last_transition = now
        self.state_entered_at = now
        self.state_entry_reason = list(reasons)
        if target != DriverState.UNKNOWN:
            self.has_reached_operational_state = True
            self.unknown_due_to_quality = False

    def _freeze_timers(self, elapsed: float) -> None:
        if elapsed <= 0:
            return
        if self.candidate_since is not None:
            self.candidate_since += elapsed
        if self.critical_recovery_since is not None:
            self.critical_recovery_since += elapsed

    def update(
        self,
        now: float,
        prediction: ModelPrediction | None,
        event_snapshot,
        trustworthy: bool,
        *,
        model_contribution_enabled: bool = True,
        eye_evidence_trustworthy: bool | None = None,
    ) -> FusionDecision:
        previous = self.state
        metrics = event_snapshot.to_dict() if hasattr(event_snapshot, "to_dict") else dict(event_snapshot or {})
        if eye_evidence_trustworthy is None:
            eye_evidence_trustworthy = bool(
                metrics.get("eye_evidence_trustworthy", metrics.get("eye_signal_valid", trustworthy))
            )
        observation_status = str(
            metrics.get(
                "eye_observation_status",
                "VALID" if eye_evidence_trustworthy else "LOST",
            )
        ).upper()
        distance_status = str(metrics.get("driver_distance_status", "UNKNOWN")).upper()
        if not trustworthy or not eye_evidence_trustworthy:
            observation_status = "LOST"

        elapsed = max(0.0, now - self.last_update_at) if self.last_update_at is not None else 0.0
        self.last_update_at = now
        model_enabled = bool(model_contribution_enabled and prediction is not None)
        if model_enabled and prediction is not None and prediction.window_id != self.last_window_id:
            value = prediction.calibrated_probability
            self.ewma = value if self.ewma is None else self.alpha * value + (1 - self.alpha) * self.ewma
            self.last_window_id = prediction.window_id

        events = self._fresh_evidence(now)
        violations = sorted({event.kind for event in events if event.kind in VIOLATION_KINDS})
        if metrics.get("blink_count_60s", 0) >= self.excessive_blinks:
            violations.append("EXCESSIVE_BLINKING")
        if metrics.get("yawn_count_60s", 0) >= self.excessive_yawns:
            violations.append("REPEATED_YAWNS")
        violations = sorted(set(violations))

        if self.state_entered_at is None:
            self.state_entered_at = now

        hold_transition = False
        model_risk_pending = False
        if observation_status in {"GRACE", "LOST"}:
            if self.quality_lost_since is None:
                self.quality_lost_since = now - float(metrics.get("eye_signal_gap_sec") or 0.0)
        else:
            self.quality_lost_since = None

        loss_age = (
            max(0.0, now - self.quality_lost_since)
            if self.quality_lost_since is not None
            else 0.0
        )
        if observation_status == "GRACE":
            target = self.state
            reasons = ["EYE_SIGNAL_GRACE_HOLD"]
            required = 0.0
            hold_transition = True
            self._freeze_timers(elapsed)
        elif observation_status == "LOST" and self.state == DriverState.CRITICAL:
            target = DriverState.CRITICAL
            reasons = ["CRITICAL_EYE_SIGNAL_LOST"]
            required = 0.0
            hold_transition = True
            self.critical_recovery_since = None
        elif observation_status == "LOST" and loss_age < self.quality_loss_persistence:
            target = self.state
            reasons = [
                "DRIVER_TOO_FAR_PENDING"
                if distance_status == "TOO_FAR"
                else "EYE_SIGNAL_LOSS_PENDING"
            ]
            required = 0.0
            hold_transition = True
            self._freeze_timers(elapsed)
        elif observation_status == "LOST":
            target = DriverState.UNKNOWN
            reasons = [
                "DRIVER_TOO_FAR"
                if distance_status == "TOO_FAR"
                else "EYE_SIGNAL_LOST"
            ]
            required = 0.0
        else:
            target, reasons, model_risk_pending = self._target(
                self.ewma or 0.0, events, metrics, model_enabled
            )
            if target in {DriverState.UNKNOWN, DriverState.CRITICAL}:
                required = 0.0
            elif target == DriverState.NORMAL:
                required = (
                    self.quality_recovery_persistence
                    if self.state == DriverState.UNKNOWN
                    and self.has_reached_operational_state
                    and self.unknown_due_to_quality
                    else self.normal_persistence
                )
            elif target == DriverState.DROWSY:
                # The closure duration already supplies persistence. This gives
                # a visible/audio DROWSY stage from 0.5s until CRITICAL at 1.5s.
                required = (
                    0.0
                    if "CURRENT_STRONG_BILATERAL_CLOSURE" in reasons
                    else self.drowsy_persistence
                )
            else:
                required = self.warning_persistence

        if not hold_transition and target != self.candidate:
            self.candidate, self.candidate_since = target, now
        candidate_age = now - (self.candidate_since if self.candidate_since is not None else now)

        if hold_transition:
            pass
        elif target == DriverState.UNKNOWN:
            self.critical_recovery_since = None
            if self.state != target:
                self.unknown_due_to_quality = self.has_reached_operational_state
                self._transition(target, now, reasons)
        elif target == DriverState.CRITICAL:
            self.critical_recovery_since = None
            self.post_critical_recovery = False
            self.post_critical_started_at = None
            if self.state != target:
                self._transition(target, now, reasons)
        elif self.state == DriverState.CRITICAL:
            # Recovery is based only on continuously trustworthy OPEN eyes. It is
            # intentionally independent from a fluctuating lower target state.
            if metrics.get("eye_state") == "OPEN" and eye_evidence_trustworthy:
                if self.critical_recovery_since is None:
                    self.critical_recovery_since = now
                if now - self.critical_recovery_since >= self.recovery_persistence:
                    _, perclos_enter, _ = self._perclos_flags(metrics)
                    if perclos_enter:
                        self.post_critical_recovery = True
                        self.post_critical_started_at = now
                        target = DriverState.FATIGUE_WARNING
                        reasons = ["POST_CRITICAL_FATIGUE_MEMORY"]
                    self._transition(
                        target,
                        now,
                        ["CRITICAL_RECOVERY_COMPLETE", *reasons],
                    )
                    self.candidate, self.candidate_since = target, now
                    self.critical_recovery_since = None
            else:
                self.critical_recovery_since = None
        else:
            self.critical_recovery_since = None
            if STATE_RANK[target] < STATE_RANK[self.state]:
                required = self.recovery_persistence
            cooldown_ok = now - self.last_transition >= self.cooldown
            if target != self.state and candidate_age >= required and cooldown_ok:
                self._transition(target, now, reasons)

        changed = self.state != previous
        recovery_status = "NONE"
        if self.state == DriverState.CRITICAL and self.critical_recovery_since is not None:
            recovery_status = "CONFIRMING_EYE_OPEN"
        elif self.post_critical_recovery:
            recovery_status = "POST_CRITICAL_FATIGUE"
        alarm = {
            DriverState.UNKNOWN: AlarmLevel.INFO,
            DriverState.NORMAL: AlarmLevel.NONE,
            DriverState.FATIGUE_WARNING: AlarmLevel.WARNING,
            DriverState.DROWSY: AlarmLevel.WARNING,
            DriverState.CRITICAL: AlarmLevel.CRITICAL,
        }[self.state]
        if violations and alarm in {AlarmLevel.NONE, AlarmLevel.INFO}:
            alarm = AlarmLevel.WARNING
            reasons = [*reasons, "VIOLATION_ALARM_ACTIVE"]
        return FusionDecision(
            monotonic_sec=now,
            previous_state=previous,
            driver_state=self.state,
            violations=violations,
            alarm_level=alarm,
            reason_codes=reasons,
            smoothed_probability=self.ewma,
            changed=changed,
            target_state=target,
            active_state=self.state,
            state_entry_reason=list(self.state_entry_reason),
            current_reason_codes=list(reasons),
            state_age_sec=max(0.0, now - (self.state_entered_at or now)),
            candidate_age_sec=max(0.0, candidate_age),
            model_contribution_enabled=model_enabled,
            eye_evidence_trustworthy=bool(eye_evidence_trustworthy),
            eye_observation_status=observation_status,
            recovery_status=recovery_status,
            model_risk_pending=model_risk_pending,
        )
