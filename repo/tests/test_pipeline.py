import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import escalation as esc  # noqa: E402
from detect_anomalies import run  # noqa: E402
from generate_dataset import build  # noqa: E402


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data, _ = build(42)
        cls.df, cls.incidents, _ = run(data)

    def test_every_injected_event_is_detected(self):
        events = self.df[self.df["label"] != "normal"]
        detected = events[events["verdict"] != "normal"]["event_id"].unique()
        self.assertEqual(set(detected), set(events["event_id"].unique()))

    def test_grid_events_are_not_called_data_quality(self):
        grid = self.df[(self.df["category"] == "grid_event") & (self.df["verdict"] != "normal")]
        self.assertTrue((grid["verdict"] == "possible_grid_event").all())

    def test_grid_events_always_need_human_approval(self):
        grid = self.df[self.df["verdict"] == "possible_grid_event"]
        self.assertTrue((grid["route"] == esc.HUMAN_APPROVAL).all())

    def test_suspected_falsification_is_never_auto_acted(self):
        sus = self.df[self.df["subtype"] == "sustained_deviation"]
        self.assertFalse((sus["route"] == esc.AUTO_ACT).any())

    def test_every_flagged_reading_has_an_explanation(self):
        flagged = self.df[self.df["verdict"] != "normal"]
        self.assertTrue((flagged["explanation"].str.len() > 0).all())

    def test_confidence_is_between_zero_and_one(self):
        self.assertTrue(self.df["confidence"].between(0, 1).all())

    def test_incidents_have_audit_fields(self):
        required = {"incident_id", "verdict", "confidence_max", "explanation", "route",
                    "model_version", "policy_version", "operator_decision"}
        for inc in self.incidents:
            self.assertTrue(required.issubset(inc))
            pd.Timestamp(inc["start"])


if __name__ == "__main__":
    unittest.main()
