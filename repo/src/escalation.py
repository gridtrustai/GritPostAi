"""Human-AI escalation policy: what the AI may do alone and what needs a human.

The prototype never commands physical equipment. Its automatic actions are
data-side only: quarantine a reading, substitute an estimate for downstream
systems and mark the sensor as suspect. Every action is logged and reversible.
All thresholds live here so they can be reviewed and versioned in one place.
"""
POLICY_VERSION = "escalation-policy-0.1"

AUTO_ACT = "auto_act"              # AI acts alone (data-side only), human notified
HUMAN_REVIEW = "human_review"      # operator must accept / reject / override
HUMAN_APPROVAL = "human_approval"  # high impact: nothing happens without approval
MONITOR = "monitor"                # logged and watched, no action

# Confidence an alert needs before the AI may act alone.
AUTO_ACT_MIN_CONFIDENCE = 0.85
# Below this the alert is only monitored.
REVIEW_MIN_CONFIDENCE = 0.50
# Evidence types where an automatic data-side action is acceptable: they are
# unambiguous data-path faults, and a wrong action is cheap and reversible.
AUTO_ACT_SUBTYPES = {"missing_data", "implausible_value", "frozen_sensor", "physics_mismatch",
                     "meter_offline"}
# Evidence types that always go to a human: suspected falsification can lead to
# action against a customer, so the AI may flag it but never conclude it alone.
ALWAYS_HUMAN_SUBTYPES = {"sustained_deviation"}
# Over-trust guard: if the AI keeps auto-correcting the same meter, the sensor
# is probably failing and silent corrections would hide it from people.
MAX_AUTO_ACTIONS_PER_METER_24H = 6

AUTO_ACTIONS = "quarantine reading; substitute expected value for downstream use; mark sensor suspect"


def route(verdict: str, subtype: str, confidence: float) -> tuple[str, str]:
    """Return (route, automatic_action) for one flagged reading."""
    if verdict == "possible_grid_event":
        return HUMAN_APPROVAL, "hold automated decisions on this feeder until an operator confirms"
    if verdict != "data_quality_issue":
        return MONITOR, ""
    if subtype in ALWAYS_HUMAN_SUBTYPES:
        return HUMAN_REVIEW, ""
    if subtype in AUTO_ACT_SUBTYPES and confidence >= AUTO_ACT_MIN_CONFIDENCE:
        return AUTO_ACT, AUTO_ACTIONS
    if confidence >= REVIEW_MIN_CONFIDENCE:
        return HUMAN_REVIEW, ""
    return MONITOR, ""
