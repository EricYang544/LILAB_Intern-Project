import copy
from pathlib import Path
import tempfile
import unittest
try:
    import torch
    import numpy as np
    import yaml
    import h5py
except ImportError:
    torch = None
if torch is not None:
    from elmo_project.common import load_config, Deadline, load_checkpoint, read_json
    from elmo_project.prepare import prepare, verify_prepared
    from elmo_project.training import train_lm
    from elmo_project.pilot import pilot
    from elmo_project.run import run
    from elmo_project.features import extract_features, verify_cache
    from elmo_project.training import lm_model

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(torch is None, "PyTorch/numpy/yaml/h5py not installed")
class PipelineTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "runs").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "runs", prefix="test_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cfg = load_config(ROOT / "configs/smoke.yaml")
        for key in ("data_dir", "raw_dir", "run_dir", "pilot_dir"):
            self.cfg[key] = str(self.root / key)
        self.cfg["ner"]["seeds"] = [13]
        self.cfg["ner"]["max_epochs"] = 1
        self.cfg["lm"]["max_tokens"] = 10000
        self.cfg["lm"]["eval_every_tokens"] = 10000
        torch.set_num_threads(1)
        prepare(self.cfg)

    def test_prepared_data_detects_tampering(self):
        path = Path(self.cfg["data_dir"]) / "ner_train.jsonl"
        path.write_text(path.read_text() + "\n")
        with self.assertRaises(ValueError):
            verify_prepared(self.cfg["data_dir"])

    def test_checkpoint_resume_matches_uninterrupted_training(self):
        cfg = self.cfg
        cfg["model"]["dropout"] = 0.2
        full = train_lm(cfg, 1, 4, "test", torch.device("cpu"), Deadline(120))
        expected = load_checkpoint(full)["model"]
        cfg = copy.deepcopy(cfg)
        cfg["run_dir"] = str(self.root / "resumed")
        class Interrupted:
            remaining = 2
            def check(self):
                self.remaining -= 1
                if self.remaining < 0:
                    raise TimeoutError()
        with self.assertRaises(TimeoutError):
            train_lm(cfg, 1, 4, "test", torch.device("cpu"), Interrupted())
        resumed = train_lm(cfg, 1, 4, "test", torch.device("cpu"), Deadline(120))
        actual = load_checkpoint(resumed)["model"]
        for name in expected:
            self.assertTrue(torch.equal(expected[name], actual[name]), name)

    def test_feature_alignment_and_tamper_detection(self):
        model = lm_model(self.cfg, 1)
        cache = self.root / "cache"
        info = extract_features(self.cfg, model, "model-a", cache, torch.device("cpu"), Deadline(120))
        self.assertEqual(info["shape"], [2, 16])
        values = np.load(cache / "test.npy")
        self.assertEqual(values.shape[0], 19)
        with self.assertRaises(ValueError):
            verify_cache(cache, "another-model")
        with (cache / "test.npy").open("ab") as f:
            f.write(b"bad")
        with self.assertRaises(ValueError):
            verify_cache(cache, info["key"])

    def test_public_hdf5_mapping_matches_native_projected_lstm(self):
        from elmo_project.pretrained import PublicELMo
        from elmo_project.common import write_json
        from elmo_project.batches import feature_batch, ner_records
        cfg = self.cfg["model"]
        native = lm_model(self.cfg, 2).eval()
        options = {"char_cnn": {"embedding": {"dim": cfg["char_dim"]}, "filters": cfg["filters"],
                   "max_characters_per_token": cfg["max_chars"], "n_highway": cfg["highways"], "activation": "relu"},
                   "lstm": {"dim": cfg["hidden"], "projection_dim": cfg["projection"], "n_layers": 2,
                   "use_skip_connections": True}}
        weights = self.root / "fixture.hdf5"
        option_path = self.root / "options.json"
        write_json(option_path, options)
        def array(value):
            return value.detach().numpy()
        with h5py.File(weights, "w") as f:
            f["char_embed"] = array(native.char.embedding.weight[1:])
            for i, conv in enumerate(native.char.convs):
                f[f"CNN/W_cnn_{i}"] = array(conv.weight.permute(2, 1, 0).unsqueeze(0))
                f[f"CNN/b_cnn_{i}"] = array(conv.bias)
            for i, highway in enumerate(native.char.highways):
                for name, source in (("gate", "carry"), ("transform", "transform")):
                    layer = getattr(highway, name)
                    f[f"CNN_high_{i}/W_{source}"] = array(layer.weight.T)
                    f[f"CNN_high_{i}/b_{source}"] = array(layer.bias)
            f["CNN_proj/W_proj"] = array(native.char.projection.weight.T)
            f["CNN_proj/b_proj"] = array(native.char.projection.bias)
            for direction, stack in enumerate(native.directions):
                for index, cell in enumerate(stack):
                    prefix = f"RNN_{direction}/RNN/MultiRNNCell/Cell{index}/LSTMCell"
                    kernel = torch.cat([cell.weight_ih_l0, cell.weight_hh_l0], dim=1)
                    pieces = kernel.chunk(4, dim=0)
                    f[prefix + "/W_0"] = array(torch.cat([pieces[i] for i in (0, 2, 1, 3)], dim=0).T)
                    pieces = (cell.bias_ih_l0 + cell.bias_hh_l0).chunk(4)
                    bias = torch.cat([pieces[i] for i in (0, 2, 1, 3)])
                    bias = bias.detach().clone()
                    bias[2 * cfg["hidden"]:3 * cfg["hidden"]] -= 1
                    f[prefix + "/B"] = array(bias)
                    f[prefix + "/W_P_0"] = array(cell.weight_hr_l0.T)
        adapter = PublicELMo(option_path, weights)
        rows = ner_records(self.cfg["data_dir"], "test")
        chars, lengths = feature_batch(rows, list(range(len(rows))), cfg["max_chars"], torch.device("cpu"))
        with torch.no_grad():
            torch.testing.assert_close(native(chars, lengths), adapter(chars, lengths), atol=1e-6, rtol=1e-5)

    def test_end_to_end_and_completed_run_is_idempotent(self):
        self.cfg["lm"]["max_tokens"] = 40
        plan = pilot(self.cfg)
        plan_path = Path(self.cfg["pilot_dir"]) / "resolved_plan.json"
        run(plan_path)
        status_path = Path(self.cfg["run_dir"]) / "status.json"
        before = read_json(status_path)
        self.assertTrue(before["complete"])
        run(plan_path)
        self.assertEqual(before, read_json(status_path))
        for arm in plan["arms"]:
            self.assertTrue((Path(self.cfg["run_dir"]) / "ner" / arm / "13" / "test.json").exists())


if __name__ == "__main__":
    unittest.main()
