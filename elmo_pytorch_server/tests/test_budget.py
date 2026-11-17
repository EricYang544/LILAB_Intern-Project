import unittest
from elmo_project.budget import resolve_budget


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {"budget": {"hours": 24, "lm_hours": 14, "ner_hours": 5, "feature_hours": 2,
                               "pilot_hours": 1, "reserve_hours": 2, "safety_factor": 1.3},
                    "ner": {"seeds": [13, 21, 42, 87, 100], "max_epochs": 50},
                    "lm": {"max_tokens": 100000000, "eval_every_tokens": 5000000}}
        self.measured = {"lm": {str(i): {"step_seconds": float(i), "words_per_step": 2000, "validation_seconds": 10} for i in (1, 2)},
                         "features": {"depth1": 60, "depth2": 120, "public": 300},
                         "ner": {name: {"train": 15, "dev": 2, "test": 2} for name in ("static", "depth1", "depth2", "public")}}

    def test_fair_common_step_budget(self):
        result = resolve_budget(self.cfg, self.measured)
        self.assertTrue(result["ready"])
        self.assertGreater(result["lm_steps"], 0)
        self.assertLessEqual(result["estimated_seconds"], 24 * 3600)
        self.assertLessEqual(result["ner_max_epochs"], 50)

    def test_slow_public_model_rejected(self):
        self.measured["features"]["public"] = 100000
        self.assertFalse(resolve_budget(self.cfg, self.measured)["ready"])

    def test_seed_count_is_never_silently_reduced(self):
        self.cfg["budget"]["ner_hours"] = 0.001
        result = resolve_budget(self.cfg, self.measured)
        self.assertFalse(result["ready"])
        self.assertEqual(len(self.cfg["ner"]["seeds"]), 5)

    def test_extremely_slow_lm_rejected(self):
        self.measured["lm"]["2"]["validation_seconds"] = 100000
        self.assertFalse(resolve_budget(self.cfg, self.measured)["ready"])


if __name__ == "__main__":
    unittest.main()
