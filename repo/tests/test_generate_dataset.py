import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import EVENT_CATEGORY  # noqa: E402
from generate_dataset import METERS, N_HOURS, build  # noqa: E402


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.events = build(42)

    def test_expected_columns_exist(self):
        expected = {"timestamp", "feeder_id", "meter_id", "active_power_kw", "voltage_v",
                    "current_a", "label", "category", "event_id", "is_synthetic"}
        self.assertTrue(expected.issubset(self.df.columns))

    def test_shape_is_meters_times_hours(self):
        self.assertEqual(len(self.df), len(METERS) * N_HOURS)

    def test_every_event_type_is_injected(self):
        self.assertEqual(set(self.df["label"]) - {"normal"}, set(EVENT_CATEGORY))

    def test_grid_events_hit_every_meter(self):
        grid = self.df[self.df["category"] == "grid_event"]
        self.assertEqual(set(grid["meter_id"]), set(METERS))

    def test_all_rows_are_marked_synthetic(self):
        self.assertTrue(self.df["is_synthetic"].all())
        self.assertTrue(self.events["is_synthetic"].all())

    def test_generation_is_reproducible(self):
        again, _ = build(42)
        self.assertTrue(self.df.equals(again))


if __name__ == "__main__":
    unittest.main()
