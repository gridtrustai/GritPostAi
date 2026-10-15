"""GridTrust AI Phase 1 pipeline: rules + Isolation Forest + feeder cross-check.

For every reading the pipeline produces:
  * a verdict: normal / data_quality_issue / possible_grid_event
  * a heuristic confidence score (0-1, not a calibrated probability)
  * a plain-language explanation listing every piece of evidence
  * a route from the escalation policy (auto_act / human_review /
    human_approval / monitor) and any data-side automatic action

Flagged readings are grouped into incidents and written to an auditable
decision log (JSON Lines). The prototype never controls physical equipment.

Key idea: a single meter behaving oddly is most likely a data problem; several
independent meters on the same feeder changing together is most likely a real
grid event. The two need different responses.
"""
import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

import escalation as esc
from common import (DECISION_LOG, DETECTIONS_CSV, MEASUREMENTS, NOMINAL_VOLTAGE,
                    READINGS_CSV, RESULTS_DIR, UCI_READINGS_CSV, UCI_RESULTS_DIR)

MODEL_VERSION = "gridtrust-phase1-0.1"

# ---- Thresholds (documented in docs/ESCALATION_POLICY.md) -------------------
PHYSICS_ABS_KW = 0.15        # allowed gap between P and V*I/1000 (kW) ...
PHYSICS_REL = 0.10           # ... or 10 % of V*I, whichever is larger
SUSTAINED_WINDOW_H = 12      # window for slow deviations (bias, falsification)
SUSTAINED_Z = 3.0            # mean robust z over the window
VOLTAGE_BAND_V = 15.0        # |V - 230| beyond this is out of band
FEEDER_MIN_METERS = 3        # meters abnormal at once -> feeder-level event
IF_CONTAMINATION = 0.01      # share of readings the model may call unusual
MODEL_FEATURES = ["power_z", "physics_gap_kw", "rolling_z", "peer_voltage_gap_v",
                  "voltage_dev_v"]

# Evidence weights for the heuristic confidence: 1 - prod(1 - w).
EVIDENCE_WEIGHT = {
    "missing_data": 0.95, "implausible_value": 0.95, "meter_offline": 0.85,
    "frozen_sensor": 0.85, "physics_mismatch": 0.85,
    "sustained_deviation": 0.60, "voltage_out_of_band": 0.60,
}
# When several rules fire, the subtype shown to the operator is the first here.
SUBTYPE_PRIORITY = ["missing_data", "implausible_value", "meter_offline", "frozen_sensor",
                    "physics_mismatch", "sustained_deviation", "voltage_out_of_band"]
ROUTE_SEVERITY = {esc.MONITOR: 0, esc.AUTO_ACT: 1, esc.HUMAN_REVIEW: 2, esc.HUMAN_APPROVAL: 3}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-meter features that make each fault type visible."""
    df = df.sort_values(["meter_id", "timestamp"]).reset_index(drop=True)
    hour, weekend = df["timestamp"].dt.hour, df["timestamp"].dt.dayofweek >= 5
    # Expected power: this meter's robust (median) profile for the hour/day type.
    df["expected_kw"] = df.groupby([df["meter_id"], hour, weekend])["active_power_kw"].transform("median")
    resid = df["active_power_kw"] - df["expected_kw"]
    scale = (resid.abs().groupby(df["meter_id"]).transform("median") * 1.4826).clip(lower=0.02)
    df["power_z"] = resid / scale
    # Physics consistency: active power should be close to V * I for these loads.
    df["vi_kw"] = df["voltage_v"] * df["current_a"] / 1000
    df["physics_gap_kw"] = df["active_power_kw"] - df["vi_kw"]
    # Peer comparison: how far this meter's voltage is from the feeder median now.
    df["peer_voltage_gap_v"] = df["voltage_v"] - df.groupby("timestamp")["voltage_v"].transform("median")
    df["voltage_dev_v"] = df["voltage_v"] - NOMINAL_VOLTAGE
    return df


def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    p, v, i = df["active_power_kw"], df["voltage_v"], df["current_a"]
    missing = df[MEASUREMENTS].isna().any(axis=1)
    no_supply = (v < 100) & (p.abs() < 0.05) & (i.abs() < 0.5)
    implausible = ~missing & ~no_supply & ((p < 0) | (p > 15) | (i < 0) | (i > 100)
                                           | (v > 280) | (v < 100))
    previous = df.groupby("meter_id")[MEASUREMENTS].shift(1)
    frozen = ~missing & ~no_supply & (df[MEASUREMENTS] == previous).all(axis=1)
    tolerance = np.maximum(PHYSICS_ABS_KW, PHYSICS_REL * df["vi_kw"].abs())
    physics = ~missing & ~implausible & ~no_supply & (df["physics_gap_kw"].abs() > tolerance)
    v_band = ~missing & ~no_supply & ~implausible & ((v - NOMINAL_VOLTAGE).abs() > VOLTAGE_BAND_V)
    # Slow-deviation signal: rolling mean of power_z over readings that pass every
    # other check, so a stuck, corrupt or glitching value cannot leak forward.
    # It targets deviations that are physically consistent (P matches V x I),
    # which is what careful falsification or a real change in use looks like.
    known_bad = missing | no_supply | implausible | frozen | physics | v_band
    clean_z = df["power_z"].where(~known_bad)
    df["rolling_z"] = clean_z.groupby(df["meter_id"]).transform(
        lambda s: s.rolling(SUSTAINED_WINDOW_H, min_periods=SUSTAINED_WINDOW_H // 2).mean())
    sustained = (~known_bad & (df["rolling_z"].abs() > SUSTAINED_Z) & (df["power_z"].abs() > 1.5)
                 & (np.sign(df["rolling_z"]) == np.sign(df["power_z"])))

    # Feeder cross-check: count meters with abnormal voltage or no supply right now.
    abnormal_v = v_band | no_supply
    df["meters_abnormal"] = abnormal_v.groupby(df["timestamp"]).transform("sum").astype(int)
    df["meters_on_feeder"] = df.groupby("timestamp")["meter_id"].transform("nunique")
    df["feeder_event"] = df["meters_abnormal"] >= FEEDER_MIN_METERS

    df["missing_data"] = missing
    df["implausible_value"] = implausible
    df["meter_offline"] = no_supply & ~df["feeder_event"]
    df["frozen_sensor"] = frozen
    df["physics_mismatch"] = physics
    df["sustained_deviation"] = sustained
    df["voltage_out_of_band"] = v_band & ~df["feeder_event"]
    df["no_supply"] = no_supply
    return df


def apply_model(df: pd.DataFrame) -> pd.DataFrame:
    X = df[MODEL_FEATURES].fillna(0.0)
    model = IsolationForest(n_estimators=200, contamination=IF_CONTAMINATION, random_state=42)
    model.fit(X)
    df["anomaly_score"] = -model.decision_function(X)
    df["model_flag"] = model.predict(X) == -1
    df["score_percentile"] = df["anomaly_score"].rank(pct=True)
    # Transparent driver: the feature that is most extreme versus its usual range.
    med = X.median()
    mad = ((X - med).abs().median() * 1.4826).replace(0, 1e-6)
    robust = ((X - med) / mad).abs()
    df["model_driver"] = robust.idxmax(axis=1)
    return df


def _fmt(x, nd=2):
    return "missing" if pd.isna(x) else f"{x:.{nd}f}"


def explain(row) -> list[str]:
    """One plain-language sentence per piece of evidence."""
    reasons = []
    p, v, i = row["active_power_kw"], row["voltage_v"], row["current_a"]
    if row["feeder_event"]:
        what = "loss of supply" if row["no_supply"] else f"voltage at {_fmt(v, 0)} V (nominal 230 V)"
        reasons.append(
            f"{row['meters_abnormal']}/{row['meters_on_feeder']} meters on {row['feeder_id']} show "
            f"abnormal voltage or supply at the same time ({what} on this meter); independent meters "
            "changing together points to a genuine grid event, not a sensor fault")
    if row["missing_data"]:
        gone = [m for m in MEASUREMENTS if pd.isna(row[m])]
        reasons.append(f"no value received for {', '.join(gone)}")
    if row["implausible_value"]:
        reasons.append(f"values outside physical limits (P={_fmt(p)} kW, V={_fmt(v, 0)} V, "
                       f"I={_fmt(i, 1)} A)")
    if row["meter_offline"]:
        reasons.append("this meter reports no supply while its neighbours on the same feeder are "
                       "energised; meter or communication fault is more likely than an outage")
    if row["frozen_sensor"]:
        reasons.append("power, voltage and current are all identical to the previous reading; "
                       "the sensor or data path appears stuck")
    if row["physics_mismatch"]:
        reasons.append(f"reported power {_fmt(p)} kW disagrees with V x I = {_fmt(row['vi_kw'])} kW "
                       f"(gap {row['physics_gap_kw']:+.2f} kW); the power channel is likely biased, "
                       "drifting or glitching")
    if row["sustained_deviation"]:
        direction = "below" if row["rolling_z"] < 0 else "above"
        reasons.append(f"consumption has stayed {direction} this meter's usual pattern for about "
                       f"{SUSTAINED_WINDOW_H} h (average {row['rolling_z']:+.1f} robust z) while P, V and "
                       "I remain mutually consistent; this could be under-reporting or tampering, or "
                       "a real change in use, so a person must decide")
    if row["voltage_out_of_band"]:
        reasons.append(f"voltage {_fmt(v, 0)} V is outside 230 V +/- {VOLTAGE_BAND_V:.0f} V on this "
                       "meter only; neighbours look normal, so a local sensor or connection fault is "
                       "more likely than a grid event")
    if row["model_flag"]:
        reasons.append(f"Isolation Forest ranks this reading in the most unusual "
                       f"{(1 - row['score_percentile']) * 100:.1f} % (main driver: "
                       f"{row['model_driver']})")
    return reasons


def assess(df: pd.DataFrame) -> pd.DataFrame:
    verdicts, subtypes, confidences, explanations, evidence = [], [], [], [], []
    for _, row in df.iterrows():
        fired = [k for k in SUBTYPE_PRIORITY if row[k]]
        if row["feeder_event"]:
            verdict, subtype = "possible_grid_event", "feeder_wide_change"
            conf = 0.5 + 0.5 * min(row["meters_abnormal"] / row["meters_on_feeder"], 1.0)
        elif fired or row["model_flag"]:
            verdict = "data_quality_issue"
            subtype = fired[0] if fired else "statistical_outlier"
            weights = [EVIDENCE_WEIGHT[k] for k in fired]
            if row["model_flag"]:
                weights.append(float(np.clip(0.4 + (row["score_percentile"] - 0.99) * 40, 0.4, 0.8)))
            conf = 1 - float(np.prod([1 - w for w in weights]))
        else:
            verdict, subtype, conf = "normal", "", 0.0
        reasons = explain(row) if verdict != "normal" else []
        verdicts.append(verdict)
        subtypes.append(subtype)
        confidences.append(round(conf, 3))
        evidence.append(";".join(fired + (["model"] if row["model_flag"] else [])))
        explanations.append(" | ".join(reasons))
    df["verdict"], df["subtype"], df["confidence"] = verdicts, subtypes, confidences
    df["evidence"], df["explanation"] = evidence, explanations

    routes = [esc.route(v, s, c) for v, s, c in zip(df["verdict"], df["subtype"], df["confidence"])]
    df["route"] = [r for r, _ in routes]
    df["automatic_action"] = [a for _, a in routes]
    # Over-trust guard: too many automatic corrections on one meter -> human.
    auto = (df["route"] == esc.AUTO_ACT).astype(int)
    recent = auto.groupby(df["meter_id"]).transform(lambda s: s.rolling(24, min_periods=1).sum())
    guard = (df["route"] == esc.AUTO_ACT) & (recent > esc.MAX_AUTO_ACTIONS_PER_METER_24H)
    df.loc[guard, "route"] = esc.HUMAN_REVIEW
    df.loc[guard, "automatic_action"] = ""
    df.loc[guard, "explanation"] += (" | escalated: more than "
                                     f"{esc.MAX_AUTO_ACTIONS_PER_METER_24H} automatic corrections on "
                                     "this meter in 24 h, so a technician should inspect it")
    df["escalated_by_guard"] = guard
    return df


def build_incidents(df: pd.DataFrame) -> list[dict]:
    """Group consecutive flagged readings into incidents for the decision log."""
    flagged = df[df["verdict"] != "normal"].copy()
    grid = flagged[flagged["verdict"] == "possible_grid_event"].sort_values("timestamp")
    dq = flagged[flagged["verdict"] == "data_quality_issue"].sort_values(["meter_id", "timestamp"])
    groups = []
    if not grid.empty:
        stamps = grid["timestamp"].drop_duplicates()
        block_of = dict(zip(stamps, (stamps.diff() > pd.Timedelta(hours=1)).cumsum()))
        block = grid["timestamp"].map(block_of)
        groups += [g for _, g in grid.groupby(block)]
    if not dq.empty:
        gap = dq.groupby("meter_id")["timestamp"].diff() > pd.Timedelta(hours=2)
        first = dq["meter_id"] != dq["meter_id"].shift()
        groups += [g for _, g in dq.groupby((gap | first).cumsum())]

    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    incidents = []
    for g in sorted(groups, key=lambda g: g["timestamp"].min()):
        top = g.loc[g["confidence"].idxmax()]
        worst = max(g["route"], key=ROUTE_SEVERITY.get)
        evidence_counts = pd.Series(";".join(g["evidence"]).split(";")).value_counts()
        evidence_counts = evidence_counts[evidence_counts.index != ""]
        incidents.append({
            "incident_id": f"INC-{len(incidents) + 1:04d}",
            "created_at": created,
            "verdict": top["verdict"],
            "subtype": g["subtype"].mode().iat[0],
            "feeder_id": top["feeder_id"],
            "meters": sorted(g["meter_id"].unique().tolist()),
            "start": g["timestamp"].min().isoformat(),
            "end": (g["timestamp"].max() + pd.Timedelta(hours=1)).isoformat(),
            "n_readings": int(len(g)),
            "confidence_max": float(g["confidence"].max()),
            "confidence_note": "heuristic evidence score, not a calibrated probability",
            "evidence": {k: int(n) for k, n in evidence_counts.items()},
            "explanation": top["explanation"],
            "route": worst,
            "automatic_action": next((a for a in g["automatic_action"] if a), ""),
            "requires_human": worst in (esc.HUMAN_REVIEW, esc.HUMAN_APPROVAL),
            "escalated_by_overtrust_guard": bool(g["escalated_by_guard"].any()),
            "operator_decision": None,   # filled by the operator UI: accept/reject/override
            "operator_reason": None,
            "model_version": MODEL_VERSION,
            "policy_version": esc.POLICY_VERSION,
        })
    return incidents


def run(readings: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], float]:
    t0 = time.perf_counter()
    df = build_features(readings)
    df = apply_rules(df)
    df = apply_model(df)
    df = assess(df)
    seconds = time.perf_counter() - t0
    return df, build_incidents(df), seconds


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GridTrust AI detection pipeline.")
    parser.add_argument("--uci", action="store_true",
                        help="use the UCI-baseline readings made by generate_dataset.py --uci")
    args = parser.parse_args()
    readings_csv = UCI_READINGS_CSV if args.uci else READINGS_CSV
    out_dir = UCI_RESULTS_DIR if args.uci else RESULTS_DIR
    detections_csv, decision_log = out_dir / DETECTIONS_CSV.name, out_dir / DECISION_LOG.name

    if not readings_csv.exists():
        hint = "--uci <path>" if args.uci else ""
        raise FileNotFoundError(f"{readings_csv} not found. Run `python src/generate_dataset.py {hint}` first.")
    readings = pd.read_csv(readings_csv, parse_dates=["timestamp"])
    df, incidents, seconds = run(readings)

    out_dir.mkdir(parents=True, exist_ok=True)
    keep = ["timestamp", "feeder_id", "meter_id", *MEASUREMENTS, "expected_kw", "power_z",
            "physics_gap_kw", "rolling_z", "anomaly_score", "model_flag", "verdict", "subtype",
            "confidence", "evidence", "route", "automatic_action", "explanation"]
    keep += [c for c in ("label", "category", "event_id") if c in df]
    out = df.sort_values(["timestamp", "meter_id"])[keep]
    num = out.select_dtypes("float").columns
    out[num] = out[num].round(4)
    out.to_csv(detections_csv, index=False)
    with open(decision_log, "w", encoding="utf-8") as fh:
        for inc in incidents:
            fh.write(json.dumps(inc) + "\n")

    print(f"Processed {len(df)} readings in {seconds:.2f} s "
          f"({seconds / len(df) * 1000:.2f} ms per reading)")
    print(df["verdict"].value_counts().to_string())
    print(df.loc[df["verdict"] != "normal", "route"].value_counts().to_string())
    print(f"Saved {detections_csv}")
    print(f"Saved {len(incidents)} incidents to {decision_log}")


if __name__ == "__main__":
    main()
