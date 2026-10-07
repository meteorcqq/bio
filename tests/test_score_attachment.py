"""Exercise the manifest join with synthetic identifiers, without study data."""
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "src" / "attach_qc_scores.py"


class ScoreAttachmentTests(unittest.TestCase):
    def run_join(self, scores):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, score_file, output = (root / name for name in ("manifest.csv", "scores.csv", "output.csv"))
            manifest.write_text("filename,server_source_path\nsynthetic.wav,/synthetic/synthetic.wav\n", encoding="utf-8")
            score_file.write_text("filename,quality_score_q\n" + scores, encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--manifest", str(manifest),
                 "--scores", str(score_file), "--output", str(output)],
                capture_output=True, text=True,
            )
            rows = []
            if output.exists():
                with output.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
            return result, rows

    def test_complete_join(self):
        result, rows = self.run_join("synthetic.wav,0.8\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["quality_score_q"], "0.8")
        self.assertEqual(rows[0]["filename"], "synthetic.wav")

    def test_duplicate_scores_rejected(self):
        result, rows = self.run_join("synthetic.wav,0.8\nsynthetic.wav,0.9\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Duplicate QC score filename", result.stderr)
        self.assertEqual(rows, [])

    def test_missing_recording_score_rejected(self):
        result, rows = self.run_join("different.wav,0.8\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No QC score", result.stderr)
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
