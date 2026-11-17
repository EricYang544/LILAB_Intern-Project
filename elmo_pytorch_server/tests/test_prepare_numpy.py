from pathlib import Path
import tempfile
import unittest
try:
    import numpy as np
except ImportError:
    np = None
from elmo_project.prepare import prepare, verify_prepared


@unittest.skipIf(np is None, "numpy not installed")
class PreparedArrayTests(unittest.TestCase):
    def test_offline_preparation_alignment_and_integrity(self):
        cfg = {"smoke": True, "seed": 42, "sources": {}, "model": {"max_chars": 16},
               "prepare": {"train_tokens": 300, "valid_tokens": 100, "vocab_size": 50, "max_lm_tokens": 12}}
        with tempfile.TemporaryDirectory() as d:
            cfg.update(data_dir=str(Path(d) / "prepared"), raw_dir=str(Path(d) / "raw"))
            manifest = prepare(cfg)
            self.assertEqual(manifest["ner_counts"], {"train": 16, "dev": 4, "test": 4})
            self.assertEqual(manifest["encoded"]["train"]["words"], manifest["selection"]["tokens"])
            root = Path(cfg["data_dir"])
            chars = np.load(root / "lm_chars.npy")
            self.assertEqual(chars.shape[1], 16)
            self.assertTrue(np.all(chars[0] == 0))
            for split in ("train", "valid"):
                lengths = np.load(root / f"lm_{split}.lengths.npy")
                offsets = np.load(root / f"lm_{split}.offsets.npy")
                words = np.memmap(root / f"lm_{split}.input.bin", mode="r", dtype=np.int32)
                self.assertEqual(len(words), int(lengths.sum()))
                self.assertTrue(np.all(words[offsets] == 2))
                self.assertTrue(np.all(words[offsets + lengths - 1] == 3))
            self.assertEqual(prepare(cfg), manifest)
            verify_prepared(root)
            with (root / "lm_train.input.bin").open("ab") as f:
                f.write(b"broken")
            with self.assertRaises(ValueError):
                verify_prepared(root)


if __name__ == "__main__":
    unittest.main()
