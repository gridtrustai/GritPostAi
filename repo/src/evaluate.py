"""Evaluate the pipeline against the injected labels and draw the figures.

Metrics follow the challenge brief: detection precision/recall, false-positive
rate, explanation quality and response latency. All results are on SYNTHETIC
data. Thresholds were set while looking at seed 42, so seed 42 is optimistic;
the held-out seeds (same event types, fresh noise) are the more honest check.
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from common import (CATEGORY_COLORS, EVENT_CATEGORY, FIG_DIR, INK, METRICS_JSON, MUTED,
                    READINGS_CSV, RESULTS_DIR, UCI_READINGS_CSV, UCI_RESULTS_DIR, style_axis)
from detect_anomalies import run
from generate_dataset import build

HELD_OUT_SEEDS = [1, 2, 3, 4, 5]
# Which evidence an operator should see first for each injected fault type.
EXPECTED_SUBTYPE = {
    "spike": "physics_mismatch", "drop": "physics_mismatch",
    "sensor_drift": "physics_mismatch", "persistent_bias": "physics_mismatch",
    "frozen_sensor": "frozen_sensor", "missing_data": "missing_data",
    "corrupted_reading": "implausible_value", "falsified_reading": "sustained_deviation",
    "grid_voltage_sag": "feeder_wide_change", "grid_outage": "feeder_wide_change",
}
VERDICT_CATEGORY = {"normal": "normal", "data_quality_issue": "data_quality",
                    "possible_grid_event": "grid_event"}


def evaluate(df: pd.DataFrame, incidents: list[dict], seconds: float) -> dict:
    flagged = df["verdict"] != "normal"
    actual = df["label"] != "normal"
    tp, fp = int((flagged & actual).sum()), int((flagged & ~actual).sum())
    fn, tn = int((~flagged & actual).sum()), int((~flagged & ~actual).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    predicted_cat = df["verdict"].map(VERDICT_CATEGORY)

    per_event = []
    for event_id, g in df[actual].groupby("event_id"):
        kind = g["label"].iat[0]
        hit = g[g["verdict"] != "normal"]
        first = hit["timestamp"].min() if len(hit) else None
        per_event.append({
            "event_id": event_id, "event_type": kind, "category": EVENT_CATEGORY[kind],
            "readings": int(len(g)), "readings_flagged": int(len(hit)),
            "row_recall": round(len(hit) / len(g), 3),
            "detected": bool(len(hit)),
            "detection_delay_h": None if first is None
            else round((first - g["timestamp"].min()) / pd.Timedelta(hours=1), 1),
            "category_correct": round(float((predicted_cat[hit.index] == EVENT_CATEGORY[kind]).mean()), 3)
            if len(hit) else None,
            "main_reason_matches_fault": bool(len(hit)) and
            hit["subtype"].mode().iat[0] == EXPECTED_SUBTYPE[kind],
            "routes": hit["route"].value_counts().to_dict(),
        })

    # Incident level: an incident is a false alarm if it overlaps no injected event.
    false_incidents = 0
    for inc in incidents:
        rows = df[df["meter_id"].isin(inc["meters"])
                  & (df["timestamp"] >= pd.Timestamp(inc["start"]))
                  & (df["timestamp"] < pd.Timestamp(inc["end"]))]
        false_incidents += int((rows["label"] == "normal").all())

    detected_rows = df[flagged & actual]
    return {
        "data": "synthetic",
        "readings": int(len(df)),
        "row_level": {"precision": round(precision, 3), "recall": round(recall, 3),
                      "f1": round(2 * precision * recall / (precision + recall), 3)
                      if precision + recall else 0.0,
                      "false_positive_rate": round(fp / (fp + tn), 4),
                      "tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "event_level": {"events": len(per_event),
                        "detected": sum(e["detected"] for e in per_event),
                        "incidents_logged": len(incidents),
                        "false_alarm_incidents": false_incidents},
        "grid_vs_data_quality": {
            "accuracy_on_detected_readings": round(float(
                (predicted_cat[detected_rows.index] == detected_rows["category"]).mean()), 3),
            "confusion_rows_true_cols_pred": pd.crosstab(df["category"], predicted_cat).to_dict(),
        },
        "explanations": {
            "flagged_readings_with_explanation": round(float(
                (df.loc[flagged, "explanation"].str.len() > 0).mean()), 3),
            "events_where_main_reason_matches_fault": sum(e["main_reason_matches_fault"]
                                                          for e in per_event),
        },
        "latency": {"processing_ms_per_reading": round(seconds / len(df) * 1000, 3),
                    "total_seconds": round(seconds, 3),
                    "note": "batch run on a laptop-class CPU; hourly data, so detection delay "
                            "is measured in hours of data"},
        "routes_for_flagged_readings": df.loc[flagged, "route"].value_counts().to_dict(),
        "per_event": per_event,
    }


def plot_timeline(df: pd.DataFrame, path) -> None:
    """Ground truth vs verdict as swimlanes: one lane per meter, colour = category."""
    codes = {"normal": 0, "data_quality": 1, "grid_event": 2}
    meters = sorted(df["meter_id"].unique())
    stamps = np.sort(df["timestamp"].unique())
    truth = df.pivot(index="meter_id", columns="timestamp", values="category").loc[meters]
    pred = df.assign(c=df["verdict"].map(VERDICT_CATEGORY)).pivot(
        index="meter_id", columns="timestamp", values="c").loc[meters]
    cmap = ListedColormap(["#f4f4f1", CATEGORY_COLORS["data_quality"], CATEGORY_COLORS["grid_event"]])
    fig, axes = plt.subplots(2, 1, figsize=(15, 5.6), sharex=True, constrained_layout=True)
    extent = [matplotlib.dates.date2num(stamps[0]), matplotlib.dates.date2num(stamps[-1]),
              len(meters) - 0.5, -0.5]
    for ax, grid, title in ((axes[0], truth, "Injected events (ground truth)"),
                            (axes[1], pred, "GridTrust AI verdict")):
        ax.imshow(grid.replace(codes).to_numpy(dtype=float), aspect="auto", cmap=cmap,
                  vmin=0, vmax=2, interpolation="nearest", extent=extent)
        ax.set_yticks(range(len(meters)), meters, fontsize=8)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        style_axis(ax)
        ax.grid(False)
    axes[1].xaxis_date()
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=k.replace("_", " "))
               for k, c in CATEGORY_COLORS.items()]
    axes[0].legend(handles=handles, ncol=2, fontsize=8, loc="lower right",
                   bbox_to_anchor=(1.0, 1.0), frameon=False)
    fig.suptitle("Detection timeline on the synthetic feeder", fontweight="bold", color=INK,
                 x=0.01, ha="left")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_evaluation(metrics: dict, path) -> None:
    events = pd.DataFrame(metrics["per_event"]).sort_values("event_id", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True,
                             gridspec_kw={"width_ratios": [3, 2]})
    ax = axes[0]
    colors = [CATEGORY_COLORS[c] for c in events["category"]]
    ax.barh(events["event_type"].str.replace("_", " "), events["row_recall"] * 100,
            color=colors, height=0.6)
    for y, (_, e) in enumerate(events.iterrows()):
        ax.text(e["row_recall"] * 100 + 1.5, y,
                f"{e['row_recall'] * 100:.0f} %  first flag after {e['detection_delay_h']:.0f} h",
                va="center", fontsize=8, color=INK)
    ax.set_xlim(0, 150)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("share of event readings flagged (%)", color=INK, fontsize=9)
    ax.set_title("Detection per injected event", loc="left", fontsize=10, color=INK)
    style_axis(ax)
    ax.grid(axis="x", color="#e4e4e0")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=k.replace("_", " "))
               for k, c in CATEGORY_COLORS.items()]
    ax.legend(handles=handles, fontsize=8, ncol=2, loc="lower right",
              bbox_to_anchor=(1.0, 1.0), frameon=False)

    ax = axes[1]
    order = ["auto_act", "human_review", "human_approval", "monitor"]
    routes = metrics["routes_for_flagged_readings"]
    counts = [routes.get(r, 0) for r in order]
    ax.barh([r.replace("_", " ") for r in order][::-1], counts[::-1], color=MUTED, height=0.6)
    for y, n in enumerate(counts[::-1]):
        ax.text(n + 1, y, str(n), va="center", fontsize=8, color=INK)
    ax.set_xlabel("flagged readings", color=INK, fontsize=9)
    ax.set_title("Where flagged readings were routed", loc="left", fontsize=10, color=INK)
    style_axis(ax)
    ax.grid(axis="x", color="#e4e4e0")
    ax.grid(axis="y", visible=False)
    rl = metrics["row_level"]
    data_name = "synthetic data" if metrics["data"] == "synthetic" else "UCI power baseline"
    fig.suptitle(f"Evaluation on {data_name}: precision {rl['precision']:.2f}, recall "
                 f"{rl['recall']:.2f}, false-positive rate {rl['false_positive_rate'] * 100:.2f} %",
                 fontweight="bold", color=INK, x=0.01, ha="left")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GridTrust AI against injected labels.")
    parser.add_argument("--uci", help="path to LD2011_2014.txt/.zip: evaluate on the real-power baseline")
    parser.add_argument("--start", default="2014-01-06", help="first day of the 30-day UCI window")
    args = parser.parse_args()

    real_power = None
    if args.uci:
        from load_uci import load_clients
        ts, power, _ = load_clients(args.uci, start=args.start)
        real_power = (ts, power)
        readings_csv, out_dir, suffix = UCI_READINGS_CSV, UCI_RESULTS_DIR, "_uci"
    else:
        readings_csv, out_dir, suffix = READINGS_CSV, RESULTS_DIR, ""
    metrics_json = out_dir / METRICS_JSON.name

    readings = pd.read_csv(readings_csv, parse_dates=["timestamp"])
    df, incidents, seconds = run(readings)
    metrics = evaluate(df, incidents, seconds)
    metrics["data"] = ("UCI ElectricityLoadDiagrams20112014 measured power; simulated voltage, "
                       "current and injected events" if args.uci else "synthetic")

    held_out = []
    for seed in HELD_OUT_SEEDS:
        data, _ = build(seed, real_power=real_power)
        d, inc, s = run(data)
        m = evaluate(d, inc, s)
        held_out.append({"seed": seed, **m["row_level"], **m["event_level"],
                         "grid_vs_dq_accuracy": m["grid_vs_data_quality"]["accuracy_on_detected_readings"]})
    ho = pd.DataFrame(held_out)
    metrics["held_out_seeds"] = {
        "seeds": HELD_OUT_SEEDS,
        "mean": ho[["precision", "recall", "false_positive_rate", "grid_vs_dq_accuracy"]]
        .mean().round(3).to_dict(),
        "min_events_detected": int(ho["detected"].min()),
        "max_false_alarm_incidents": int(ho["false_alarm_incidents"].max()),
        "runs": held_out,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_json.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_timeline(df, FIG_DIR / f"detection_timeline{suffix}.png")
    plot_evaluation(metrics, FIG_DIR / f"detection_evaluation{suffix}.png")

    rl, el = metrics["row_level"], metrics["event_level"]
    print(f"Seed 42  precision {rl['precision']}  recall {rl['recall']}  "
          f"FPR {rl['false_positive_rate']}  events {el['detected']}/{el['events']}  "
          f"false-alarm incidents {el['false_alarm_incidents']}")
    print(f"Held-out seeds (mean): {metrics['held_out_seeds']['mean']}")
    print(f"Grid vs data-quality accuracy: "
          f"{metrics['grid_vs_data_quality']['accuracy_on_detected_readings']}")
    print(f"Latency: {metrics['latency']['processing_ms_per_reading']} ms per reading")
    print(f"Saved {metrics_json}")


if __name__ == "__main__":
    main()
