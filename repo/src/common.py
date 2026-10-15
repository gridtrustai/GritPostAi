"""Shared paths, labels and plotting style for GridTrust AI scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"

READINGS_CSV = DATA_DIR / "simulated_energy_readings.csv"
EVENTS_CSV = DATA_DIR / "injected_events.csv"
DETECTIONS_CSV = RESULTS_DIR / "detection_results.csv"
DECISION_LOG = RESULTS_DIR / "decision_log.jsonl"
METRICS_JSON = RESULTS_DIR / "metrics.json"

# Real-baseline variant (UCI ElectricityLoadDiagrams20112014 power + injected events)
UCI_READINGS_CSV = DATA_DIR / "uci_feeder_readings.csv"
UCI_EVENTS_CSV = DATA_DIR / "uci_injected_events.csv"
UCI_SELECTION_CSV = DATA_DIR / "uci_client_selection.csv"
UCI_RESULTS_DIR = RESULTS_DIR / "uci"

MEASUREMENTS = ["active_power_kw", "voltage_v", "current_a"]
NOMINAL_VOLTAGE = 230.0

# Event type -> category. The category is what the system must tell apart:
# data-quality problems (one sensor or data path is wrong) versus genuine grid
# events (the network itself changed), because they need different responses.
EVENT_CATEGORY = {
    "spike": "data_quality",
    "drop": "data_quality",
    "sensor_drift": "data_quality",
    "persistent_bias": "data_quality",
    "frozen_sensor": "data_quality",
    "missing_data": "data_quality",
    "corrupted_reading": "data_quality",
    "falsified_reading": "data_quality",
    "grid_voltage_sag": "grid_event",
    "grid_outage": "grid_event",
}

# Plot style: one fixed categorical order for meters (never cycled) and two
# category colours for shading. Colours from the validated reference palette.
METER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
CATEGORY_COLORS = {"data_quality": "#6250d6", "grid_event": "#e34948"}
INK = "#2b2b2b"
MUTED = "#6b6b6b"
GRID = "#e4e4e0"


def style_axis(ax):
    """Recessive grid and spines so the data marks lead."""
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.label.set_color(INK)
