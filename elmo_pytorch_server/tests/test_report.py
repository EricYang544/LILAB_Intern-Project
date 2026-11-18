from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from elmo_project.common import read_json, write_json
from elmo_project.report import report


class ReportTests(unittest.TestCase):
    def test_literature_scores_are_separate_from_measured_results(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            arms = ["static", "depth1", "depth2"]
            write_json(root / "plan.json", {"config": {"ner": {"seeds": [13]}}, "plan_id": "fixture",
                       "arms": arms, "pilot_seconds": 1, "environment": {"gpu": "fixture"}})
            write_json(root / "status.json", {"complete": True, "seconds": 1})
            for i, arm in enumerate(arms):
                folder = root / "ner" / arm / "13"
                metrics = {kind: {"f1": 80.0 + i, "precision": 80.0, "recall": 80.0}
                           for kind in ("ALL", "PER", "ORG", "LOC", "MISC")}
                write_json(folder / "test.json", {"fingerprint": "fixture", "metrics": metrics})
                write_json(folder / "complete.json", {"best_f1": 80, "epoch": 1, "seconds": 1,
                           "peak_vram_bytes": 0, "parameters": 10})
                write_json(folder / "predictions.json", [])
            for depth in (1, 2):
                write_json(root / f"lm_depth{depth}/complete.json", {})
            with patch("elmo_project.report.plot_curves"):
                report(root)
            output = root / "report"
            measured, literature = (output / "report.md").read_text().split("## 문헌 참고값", 1)
            self.assertNotIn("92.22", measured)
            self.assertIn("92.22 ± 0.10", literature)
            self.assertIn("본 실험의 측정값 아님", literature)
            self.assertIn("공개 ELMo 모델은 다운로드하거나 실행하지 않는다", literature)
            self.assertNotIn("public", read_json(output / "summary.json"))
            self.assertNotIn("92.22", (output / "results.csv").read_text())


if __name__ == "__main__":
    unittest.main()
