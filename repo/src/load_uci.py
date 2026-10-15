"""Load real household-scale power from UCI ElectricityLoadDiagrams20112014.

Dataset: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
File:    LD2011_2014.txt (or the downloaded .zip), semicolon separated, decimal
         comma, first column = timestamp, then one column per client MT_001 ...
         MT_370. Values are kW averaged over 15 minutes. Clients that joined
         after 2011 are 0 before they start.

This module picks a few real clients for a 30-day window, resamples them to
hourly kW and returns them in the same shape as the synthetic generator, so the
same fault injection, detection and evaluation code runs unchanged. Voltage and
current are NOT in the dataset; generate_dataset.py simulates them from the
real power and labels them as simulated.
"""
from pathlib import Path

import numpy as np
import pandas as pd

MAX_P99_KW = 10.0        # keep household-scale clients so physical limits stay meaningful
MIN_MEDIAN_KW = 0.15     # skip near-empty clients
MAX_ZERO_SHARE = 0.01    # client must be active for the whole window
TARGET_MEDIAN_KW = 0.7   # used only if not enough household-scale clients exist


def read_window(path: Path, start: pd.Timestamp, end: pd.Timestamp,
                chunksize: int = 20_000) -> pd.DataFrame:
    """Read only the rows inside [start, end) so the 700 MB file stays manageable."""
    parts = []
    reader = pd.read_csv(path, sep=";", decimal=",", index_col=0, parse_dates=True,
                         chunksize=chunksize, low_memory=False)
    for chunk in reader:
        chunk.index = pd.to_datetime(chunk.index)
        sel = chunk[(chunk.index > start) & (chunk.index <= end)]
        if not sel.empty:
            parts.append(sel)
        if chunk.index.max() > end:
            break
    if not parts:
        raise ValueError(f"No rows between {start} and {end} in {path}")
    df = pd.concat(parts).apply(pd.to_numeric, errors="coerce")
    df.columns = [str(c).strip().strip('"') for c in df.columns]
    return df


def load_clients(path: str | Path, start: str = "2014-01-06", n_hours: int = 720,
                 n_meters: int = 5) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray], pd.DataFrame]:
    """Return (hourly timestamps, {meter_id: hourly kW array}, selection table)."""
    path = Path(path)
    t0 = pd.Timestamp(start)
    t1 = t0 + pd.Timedelta(hours=n_hours)
    raw = read_window(path, t0, t1)
    # Timestamps mark the END of each 15-min interval: shift so each value is
    # assigned to the hour it was consumed in, then average to hourly kW.
    hourly = raw.shift(-1, freq="15min").resample("h").mean()
    hourly = hourly[(hourly.index >= t0) & (hourly.index < t1)]
    if len(hourly) != n_hours:
        raise ValueError(f"Expected {n_hours} hourly rows, got {len(hourly)}; check --start")

    stats = pd.DataFrame({
        "median_kw": hourly.median(),
        "p99_kw": hourly.quantile(0.99),
        "zero_share": (hourly <= 0).mean(),
        "nan_share": hourly.isna().mean(),
    })
    active = stats[(stats["zero_share"] <= MAX_ZERO_SHARE) & (stats["nan_share"] == 0)
                   & (stats["median_kw"] >= MIN_MEDIAN_KW)]
    household = active[active["p99_kw"] <= MAX_P99_KW].sort_values("median_kw")
    if len(household) >= n_meters:
        chosen, scaled = household.head(n_meters), False
    else:
        chosen, scaled = active.sort_values("median_kw").head(n_meters), True
    if len(chosen) < n_meters:
        raise ValueError(f"Only {len(chosen)} usable clients in this window; try another --start")

    power, rows = {}, []
    for i, client in enumerate(chosen.index, start=1):
        meter = f"meter_{i:02d}"
        values = hourly[client].to_numpy(dtype=float)
        factor = TARGET_MEDIAN_KW / np.median(values) if scaled else 1.0
        power[meter] = values * factor
        rows.append({"meter_id": meter, "uci_client": client, "scale_factor": round(factor, 4),
                     "median_kw": round(float(np.median(values * factor)), 3),
                     "p99_kw": round(float(np.quantile(values * factor, 0.99)), 3)})
    return hourly.index, power, pd.DataFrame(rows)
