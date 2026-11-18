from pathlib import Path
import tempfile
import unittest
import zipfile
try:
    import numpy as np
except ImportError:
    np = None
from elmo_project.prepare import prepare, verify_prepared


@unittest.skipIf(np is None, "numpy not installed")
class PreparedArrayTests(unittest.TestCase):
    def test_production_prepare_without_public_elmo_sources(self):
        # Exercise real downloads/extraction via local URLs, without ELMo URLs.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            archives = {
                "lm": {"training-monolingual.tokenized.shuffled/news.en-1": "Alice works here .\n" * 20,
                       "heldout-monolingual.tokenized.shuffled/news.en-1": "Bob works here .\n" * 20},
                "conll": {name: "Alice NNP B-NP B-PER\n\n" for name in ("train.txt", "valid.txt", "test.txt")},
                "senna": {"words.lst": "Alice\n", "embeddings.txt": " ".join(["0.1"] * 50) + "\n"},
            }
            sources = {}
            for key, files in archives.items():
                archive = root / f"{key}.zip"
                with zipfile.ZipFile(archive, "w") as z:
                    for name, content in files.items():
                        z.writestr(name, content)
                sources[key] = archive.as_uri()
            evaluator = root / "conlleval"
            evaluator.write_text("fixture evaluator\n")
            sources["conlleval"] = evaluator.as_uri()
            cfg = {"smoke": False, "seed": 42, "sources": sources, "model": {"max_chars": 16},
                   "data_dir": str(root / "prepared"), "raw_dir": str(root / "raw"),
                   "prepare": {"train_tokens": 30, "valid_tokens": 20, "vocab_size": 20,
                               "max_lm_tokens": 12, "expected_ner_counts": [1, 1, 1]}}
            manifest = prepare(cfg)
            self.assertEqual(set(manifest["artifacts"]), {"lm", "conll", "senna", "conlleval"})
            self.assertFalse((root / "raw/elmo_options.json").exists())
            self.assertFalse((root / "raw/elmo_weights.hdf5").exists())
            cfg["include_public_elmo"] = False
            self.assertEqual(prepare(cfg), manifest)
            cfg["include_public_elmo"] = True
            with self.assertRaisesRegex(ValueError, "different settings"):
                prepare(cfg)
            cfg["data_dir"] = str(root / "prepared_public")
            for key in ("elmo_options", "elmo_weights"):
                asset = root / key
                asset.write_text("fixture public asset\n")
                sources[key] = asset.as_uri()
            public_manifest = prepare(cfg)
            self.assertTrue({"elmo_options", "elmo_weights"} <= public_manifest["artifacts"].keys())

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
