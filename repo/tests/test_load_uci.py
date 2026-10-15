"""Test the UCI loader on a small file written in the real LD2011_2014.txt format."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from detect_anomalies import run  # noqa: E402
from generate_dataset import build  # noqa: E402
from load_uci import load_clients  # noqa: E402


def write_fake_ld_file(path: Path) -> None:
    rng = np.random.default_rng(0)
    ts = pd.date_range("2013-12-20 00:15", "2014-02-15 00:00", freq="15min")
    hour = (ts.hour + ts.minute / 60).to_numpy()
    shape = 0.6 + 0.8 * np.exp(-((hour - 20) ** 2) / 5)
    cols = {}
    for i in range(1, 11):
        scale = 80.0 if i % 4 == 0 else rng.uniform(0.3, 1.5)   # every 4th client is commercial
        values = scale * shape * rng.lognormal(0, 0.15, len(ts))
        if i == 3:
            values[:3000] = 0.0                                   # client joins late
        cols[f"MT_{i:03d}"] = values
    df = pd.DataFrame(cols, index=ts)
    with open(path, "w") as fh:
        fh.write('"";' + ";".join(f'"{c}"' for c in df.columns) + "\n")
        for t, row in zip(df.index, df.to_numpy()):
            fh.write(f'"{t:%Y-%m-%d %H:%M:%S}";'
                     + ";".join(f"{x:.6f}".replace(".", ",") for x in row) + "\n")


class LoadUciTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        path = Path(cls.tmp.name) / "LD2011_2014.txt"
        write_fake_ld_file(path)
        cls.ts, cls.power, cls.selection = load_clients(path, start="2014-01-06")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_returns_30_days_hourly_for_5_meters(self):
        self.assertEqual(len(self.ts), 720)
        self.assertEqual(len(self.power), 5)
        self.assertTrue(all(len(v) == 720 for v in self.power.values()))

    def test_skips_commercial_and_inactive_clients(self):
        chosen = set(self.selection["uci_client"])
        self.assertFalse(chosen & {"MT_004", "MT_008", "MT_003"})

    def test_pipeline_runs_on_real_power_baseline(self):
        data, _ = build(42, real_power=(self.ts, self.power))
        self.assertTrue((data["power_source"] == "uci_ld2011_2014").all())
        df, incidents, _ = run(data)
        detected = df[(df["label"] != "normal") & (df["verdict"] != "normal")]["event_id"].nunique()
        self.assertEqual(detected, 10)


if __name__ == "__main__":
    unittest.main()
