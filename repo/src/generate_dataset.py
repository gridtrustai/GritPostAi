"""Generate a reproducible synthetic feeder: 5 smart meters, hourly, 30 days.

Every injected event is recorded in two places:
  * per row in data/simulated_energy_readings.csv (label, category, event_id)
  * per event in data/injected_events.csv (type, meters, interval, method)

All data is synthetic (is_synthetic = True). It is not STEG data and not a
measured grid dataset. Phase 2 replaces the synthetic load baseline with public
measured consumption data while keeping the same injection code.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (CATEGORY_COLORS, EVENT_CATEGORY, EVENTS_CSV, FIG_DIR, INK,
                    METER_COLORS, NOMINAL_VOLTAGE, READINGS_CSV, UCI_EVENTS_CSV,
                    UCI_READINGS_CSV, UCI_SELECTION_CSV, style_axis)

SEED = 42
N_HOURS = 24 * 30
FEEDER = "feeder_F1"
METERS = [f"meter_{i:02d}" for i in range(1, 6)]
LOAD_SCALE = [0.8, 1.0, 1.15, 1.3, 0.95]       # household size differences
LINE_DROP_V = [-0.4, -0.9, -1.3, -1.8, -2.2]   # meters further down the feeder

# (event_id, type, meters, start_hour, end_hour_exclusive, how it is generated)
EVENTS = [
    ("EV01", "spike", ["meter_01"], 100, 103,
     "power reading x2.6, voltage/current unchanged (measurement glitch)"),
    ("EV02", "drop", ["meter_02"], 150, 154,
     "power reading x0.2, voltage/current unchanged"),
    ("EV03", "sensor_drift", ["meter_03"], 200, 260,
     "power reading + linear offset 0 -> 1.0 kW over 60 h"),
    ("EV04", "persistent_bias", ["meter_04"], 300, 348,
     "power reading x1.25 + 0.3 kW (scale and offset error)"),
    ("EV05", "frozen_sensor", ["meter_05"], 380, 392,
     "all three channels repeat the last good value"),
    ("EV06", "missing_data", ["meter_01"], 430, 437,
     "all three channels missing (NaN)"),
    ("EV07", "corrupted_reading", ["meter_02"], 470, 474,
     "implausible values: power -2.5 kW, voltage 0 V, current 655.35 A"),
    ("EV08", "falsified_reading", ["meter_03"], 520, 580,
     "power and current scaled down smoothly to 60 % (consistent tampering)"),
    ("EV09", "grid_voltage_sag", METERS, 640, 646,
     "all meters: voltage -22 V, power x0.9, current follows physics"),
    ("EV10", "grid_outage", METERS, 690, 693,
     "all meters: power, voltage and current at 0"),
]


def baseline(rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    """Realistic-looking normal behaviour for each meter on a shared feeder."""
    ts = pd.date_range("2026-01-01", periods=N_HOURS, freq="h")
    hour = ts.hour.to_numpy()
    weekday = ts.dayofweek.to_numpy()
    shape = (0.55 + 0.25 * np.exp(-((hour - 8) ** 2) / 6)      # morning
             + 0.75 * np.exp(-((hour - 20) ** 2) / 5)           # evening peak
             + np.where(weekday >= 5, 0.10, 0.0))               # weekend
    # Feeder voltage is shared: it sags slightly when the whole street is busy.
    feeder_v = NOMINAL_VOLTAGE + 1.5 - 2.5 * (shape - shape.mean()) + rng.normal(0, 0.8, N_HOURS)
    out = {}
    for meter, scale, drop in zip(METERS, LOAD_SCALE, LINE_DROP_V):
        power = np.maximum(scale * shape * rng.lognormal(0, 0.10, N_HOURS), 0.05)
        voltage = feeder_v + drop + rng.normal(0, 0.4, N_HOURS)
        current = power * 1000 / voltage + rng.normal(0, 0.05, N_HOURS)
        out[meter] = pd.DataFrame({"timestamp": ts, "active_power_kw": power,
                                   "voltage_v": voltage, "current_a": current})
    return out


def inject(frames: dict[str, pd.DataFrame]) -> None:
    """Apply each event in place and write the labels."""
    for frame in frames.values():
        frame["label"] = "normal"
        frame["category"] = "normal"
        frame["event_id"] = ""
    for event_id, kind, meters, start, end, _ in EVENTS:
        n = end - start
        for meter in meters:
            f = frames[meter]
            rows = f.index[start:end]
            p = f.loc[rows, "active_power_kw"].to_numpy()
            if kind == "spike":
                f.loc[rows, "active_power_kw"] = p * 2.6
            elif kind == "drop":
                f.loc[rows, "active_power_kw"] = p * 0.2
            elif kind == "sensor_drift":
                f.loc[rows, "active_power_kw"] = p + np.linspace(0.0, 1.0, n)
            elif kind == "persistent_bias":
                f.loc[rows, "active_power_kw"] = p * 1.25 + 0.3
            elif kind == "frozen_sensor":
                for col in ("active_power_kw", "voltage_v", "current_a"):
                    f.loc[rows, col] = f.loc[start - 1, col]
            elif kind == "missing_data":
                f.loc[rows, ["active_power_kw", "voltage_v", "current_a"]] = np.nan
            elif kind == "corrupted_reading":
                f.loc[rows, ["active_power_kw", "voltage_v", "current_a"]] = [-2.5, 0.0, 655.35]
            elif kind == "falsified_reading":
                factor = np.concatenate([np.linspace(1.0, 0.6, n // 3), np.full(n - n // 3, 0.6)])
                f.loc[rows, "active_power_kw"] = p * factor
                f.loc[rows, "current_a"] = f.loc[rows, "current_a"].to_numpy() * factor
            elif kind == "grid_voltage_sag":
                v = f.loc[rows, "voltage_v"].to_numpy() - 22.0
                f.loc[rows, "voltage_v"] = v
                f.loc[rows, "active_power_kw"] = p * 0.9
                f.loc[rows, "current_a"] = p * 0.9 * 1000 / v
            elif kind == "grid_outage":
                f.loc[rows, ["active_power_kw", "voltage_v", "current_a"]] = 0.0
            f.loc[rows, "label"] = kind
            f.loc[rows, "category"] = EVENT_CATEGORY[kind]
            f.loc[rows, "event_id"] = event_id


def plot_dataset(data: pd.DataFrame, events: pd.DataFrame, fig_name: str = "dataset_overview.png"):
    """Power per meter (one row each) with injected events shaded by category."""
    fig, axes = plt.subplots(len(METERS), 1, figsize=(15, 10), sharex=True,
                             constrained_layout=True)
    for ax, meter, color in zip(axes, METERS, METER_COLORS):
        d = data[data["meter_id"] == meter]
        ax.plot(d["timestamp"], d["active_power_kw"], color=color, lw=1.0)
        ax.set_ylabel(f"{meter}\nkW", fontsize=8)
        style_axis(ax)
        top = ax.get_ylim()[1]
        for _, ev in events.iterrows():
            if meter in ev["meter_ids"].split(";"):
                t0, t1 = pd.Timestamp(ev["start"]), pd.Timestamp(ev["end"])
                ax.axvspan(t0, t1, color=CATEGORY_COLORS[ev["category"]], alpha=0.18, lw=0)
                if ev["category"] == "data_quality":
                    ax.text(t0, top, f" {ev['event_type'].replace('_', ' ')}",
                            fontsize=7.5, color=INK, va="top")
    sag = events[events["event_type"] == "grid_voltage_sag"].iloc[0]
    axes[0].text(pd.Timestamp(sag["start"]), axes[0].get_ylim()[1],
                 " grid sag + outage (all meters)", fontsize=7.5, color=INK, va="top", ha="right")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.35, label=k.replace("_", " "))
               for k, c in CATEGORY_COLORS.items()]
    axes[0].legend(handles=handles, ncol=2, fontsize=8, loc="lower right",
                   bbox_to_anchor=(1.0, 1.02), frameon=False)
    source = ("measured UCI power, simulated V/I" if (data["power_source"] != "synthetic").any()
              else "synthetic")
    fig.suptitle(f"Feeder ({source}): active power per meter with injected events",
                 fontweight="bold", color=INK, x=0.01, ha="left")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / fig_name
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def baseline_from_power(ts: pd.DatetimeIndex, power: dict[str, np.ndarray],
                        rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    """Real measured power (e.g. UCI) + simulated voltage and current.

    The datasets give power only, so voltage is simulated as a shared feeder
    voltage that sags when the total measured load is high, and current as
    I = P / V. Both are labelled as simulated in the output.
    """
    total = np.sum(list(power.values()), axis=0)
    load = total / total.mean()
    feeder_v = NOMINAL_VOLTAGE + 1.5 - 2.5 * (load - 1.0) + rng.normal(0, 0.8, len(ts))
    out = {}
    for meter, drop in zip(METERS, LINE_DROP_V):
        p = np.maximum(power[meter], 0.0)
        voltage = feeder_v + drop + rng.normal(0, 0.4, len(ts))
        current = p * 1000 / voltage + rng.normal(0, 0.05, len(ts))
        out[meter] = pd.DataFrame({"timestamp": ts, "active_power_kw": p,
                                   "voltage_v": voltage, "current_a": current})
    return out


def build(seed: int = SEED, real_power: tuple | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (readings, events) for one random seed.

    real_power: optional (timestamps, {meter: hourly kW}) from load_uci.py. When
    given, measured power replaces the synthetic household profiles; voltage,
    current and every injected event are still simulated and labelled.
    """
    rng = np.random.default_rng(seed)
    if real_power is None:
        frames, source = baseline(rng), "synthetic"
    else:
        frames, source = baseline_from_power(*real_power, rng), "uci_ld2011_2014"
    inject(frames)
    data = pd.concat([f.assign(feeder_id=FEEDER, meter_id=m) for m, f in frames.items()],
                     ignore_index=True)
    data["power_source"] = source
    data["is_synthetic"] = True if source == "synthetic" else data["label"] != "normal"
    data = data[["timestamp", "feeder_id", "meter_id", "active_power_kw", "voltage_v",
                 "current_a", "label", "category", "event_id", "power_source", "is_synthetic"]]
    data = data.sort_values(["timestamp", "meter_id"]).reset_index(drop=True)
    for col, nd in (("active_power_kw", 4), ("voltage_v", 2), ("current_a", 3)):
        data[col] = data[col].round(nd)

    ts = frames[METERS[0]]["timestamp"]
    events = pd.DataFrame([{
        "event_id": e, "event_type": k, "category": EVENT_CATEGORY[k],
        "meter_ids": ";".join(m), "start": ts[s], "end": ts[t - 1] + pd.Timedelta(hours=1),
        "n_hours": t - s, "generation_method": how, "is_synthetic": True,
    } for e, k, m, s, t, how in EVENTS])
    return data, events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--uci", help="path to LD2011_2014.txt or its .zip (real power baseline)")
    parser.add_argument("--start", default="2014-01-06", help="first day of the 30-day UCI window")
    args = parser.parse_args()

    if args.uci:
        from load_uci import load_clients
        ts, power, selection = load_clients(args.uci, start=args.start, n_hours=N_HOURS,
                                            n_meters=len(METERS))
        data, events = build(SEED, real_power=(ts, power))
        readings_csv, events_csv = UCI_READINGS_CSV, UCI_EVENTS_CSV
        selection.to_csv(UCI_SELECTION_CSV, index=False)
        print("Selected UCI clients:\n" + selection.to_string(index=False))
        print(f"Saved client selection to {UCI_SELECTION_CSV}")
        fig_name = "dataset_overview_uci.png"
    else:
        data, events = build(SEED)
        readings_csv, events_csv, fig_name = READINGS_CSV, EVENTS_CSV, "dataset_overview.png"

    readings_csv.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(readings_csv, index=False)
    events.to_csv(events_csv, index=False)
    print(f"Saved {len(data)} rows ({len(METERS)} meters x {N_HOURS} h) to {readings_csv}")
    print(f"Saved {len(events)} labelled events to {events_csv}")
    print(f"Saved plot to {plot_dataset(data, events, fig_name)}")


if __name__ == "__main__":
    main()
