# Escalation policy (version `escalation-policy-0.1`)

This policy defines what GridTrust AI may do on its own and when a human operator must
decide. All thresholds live in `src/escalation.py` and `src/detect_anomalies.py`, are
versioned, and every decision records the policy and model versions that produced it.

**The prototype never controls physical equipment.** Its only automatic actions are
data-side and reversible: quarantine a reading, substitute an expected value for
downstream systems, and mark the sensor as suspect.

## Routes

| Route | Meaning | Who acts |
|---|---|---|
| `auto_act` | AI applies the data-side actions above and logs them | AI; operator notified in a digest and can undo |
| `human_review` | Operator must accept, reject or override, with a reason | Operator |
| `human_approval` | High impact: nothing downstream changes until approved | Operator (priority) |
| `monitor` | Logged and watched, no action | Nobody yet |

## Rules

| Situation | Route | Why |
|---|---|---|
| ≥ 3 of 5 meters on a feeder show abnormal voltage or no supply at the same time | `human_approval` | Likely a genuine grid event; affects many customers; must never be "corrected" as bad data |
| Missing data, implausible value, frozen sensor, physics mismatch, or single meter offline, with confidence ≥ 0.85 | `auto_act` | Unambiguous data-path faults; a wrong action is cheap and reversible |
| Sustained physically consistent deviation (possible falsification) | `human_review`, always | Concluding tampering can lead to action against a customer; the AI may flag it, never conclude it |
| Any other data-quality alert with confidence ≥ 0.50 | `human_review` | Evidence is not strong enough to act alone |
| Confidence < 0.50 | `monitor` | Too weak to interrupt a person |
| More than 6 automatic corrections on one meter within 24 h | `human_review` | Over-trust guard: the sensor is probably failing and silent fixes would hide it |

## Confidence score

`confidence = 1 − Π(1 − wᵢ)` over the evidence that fired. Weights: missing or implausible
0.95; frozen, physics mismatch or meter offline 0.85; sustained deviation or single-meter
voltage out of band 0.60; Isolation Forest 0.40–0.80, scaled by how extreme its score is.
For feeder events: `0.5 + 0.5 × share of meters affected`. It is a **heuristic evidence
score, not a calibrated probability**; calibration against operator-reviewed labels is a
Phase 2 task.

## Over-trust and under-trust of the AI

| Risk | How the design addresses it |
|---|---|
| **Over-trust** (operators rubber-stamp, or the AI silently fixes too much) | AI only acts on data, never on equipment. Grid events and suspected falsification always need a person. Repeated automatic corrections on one meter escalate to a human. Every alert shows its evidence, not just a verdict. Phase 2: track how often operators accept without opening the evidence, and how long they spend per alert. |
| **Under-trust** (operators ignore or always override the AI) | Explanations name the concrete evidence (for example "P = 1.26 kW but V × I = 0.48 kW"), so a person can verify them quickly. False alarms are kept very low so alerts stay meaningful. Phase 2: track override rate per alert type, review the types operators reject most, and retune them. |

## Operator decision record

Each incident in `results/decision_log.jsonl` has `operator_decision` and `operator_reason`
fields (currently empty). In Phase 2 the operator interface fills them with `accept`,
`reject` or `override` plus a free-text reason, a timestamp and the operator ID.
