# Shared learning across the noosphere (design)

## Purpose

Let separate nodes (smart-metered neighbourhoods, factories, microgrids) learn from each
other's anomalies without centralising raw meter readings. A falsification pattern or a
drifting sensor model spotted in one neighbourhood should help every other node catch it
sooner.

## What is shared, what stays local, and why

| Shared | Why it is safe and useful |
|---|---|
| Pattern signatures: event type, which evidence fired, typical magnitude and duration ranges (for example "physics gap +0.3 to +1.0 kW for 6–60 h") | Describes a fault, not a household |
| Rule and threshold proposals with their measured precision and recall | Lets other nodes test an improvement locally |
| Calibration insights per sensor model or firmware (for example "model X drifts +0.02 kW per day after 18 months") | Hardware knowledge, no customer data |
| Aggregate counts: alerts per type, operator override rate per type | Shows which rules people trust |
| Model and policy versions, node category (neighbourhood, factory or microgrid) | Needed to judge relevance |

| Kept local | Why |
|---|---|
| Raw meter time series | Reveal household routines (occupancy, sleep, appliances) |
| Customer identifiers, addresses, account data | Personal data |
| Individual operator decisions and free-text reasons | May identify people or customers |
| Exact event timestamps | Can be linked to known local incidents |

## Exchange protocol

1. A node confirms an incident through operator review (only human-confirmed incidents are shared).
2. It builds a pattern signature with ranges, not raw values, and drops exact timestamps.
3. A local check verifies that no identifier, address or raw series is present; the local sharing policy approves the message.
4. The signature is published to the other nodes (a shared repository or message broker).
5. A receiving node treats it as a **candidate**: it replays the rule on its own local history, measures precision and false positives, and adopts it only if it helps.
6. Adoption or rejection is reported back as an aggregate, so the network learns which patterns generalise.

Example message:

```json
{
  "schema": "gridtrust.pattern/0.1",
  "origin_category": "smart_metered_neighbourhood",
  "pattern_type": "falsified_reading",
  "evidence": ["sustained_deviation"],
  "signature": {"direction": "below_profile", "rolling_z_range": [-4.5, -3.0],
                "physics_consistent": true, "duration_h_range": [24, 72]},
  "local_performance": {"precision": 0.97, "recall": 0.55, "reviewed_incidents": 12},
  "model_version": "gridtrust-phase1-0.1",
  "limitations": "synthetic validation only",
  "created": "2026-10"
}
```

## Phase 2 options

- **Lightweight multi-agent concept:** one agent per node runs detection locally and exchanges only signatures through the protocol above. A 3-node simulation can show a falsification pattern learned in node A being caught earlier in node B.
- **Federated concept:** nodes train local models (for example an autoencoder of normal load) and share only model updates, optionally with secure aggregation or differential-privacy noise, so the shared model never sees raw readings.

Phase 1 implements the single node only; this document is the design.
