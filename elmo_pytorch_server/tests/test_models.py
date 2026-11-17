import itertools
import math
import unittest
try:
    import numpy as np
    import torch
except ImportError:
    torch = None
if torch is not None:
    from elmo_project.models import BiLM, CRF, NER, reverse_valid, mask_from_lengths
    from elmo_project.pretrained import ReferenceCell
    from elmo_project.text import TAGS, char_ids


@unittest.skipIf(torch is None, "PyTorch/numpy not installed")
class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)
        self.config = {"hidden": 16, "projection": 8, "char_dim": 4, "filters": [[1, 4], [2, 8]], "highways": 1, "dropout": 0}
        self.chars = torch.tensor([[char_ids(w, 12) for w in ["<S>", "Alice", "met", "Bob", "</S>"]]])
        self.lengths = torch.tensor([5])

    def test_directions_do_not_leak_across_layers(self):
        model = BiLM(self.config, 10, 2).eval()
        before = model(self.chars, self.lengths)
        future = self.chars.clone()
        future[:, 3] = torch.tensor(char_ids("Paris", 12))
        after = model(future, self.lengths)
        self.assertTrue(torch.allclose(before[:, :3, 1:, :8], after[:, :3, 1:, :8], atol=1e-6))
        past = self.chars.clone()
        past[:, 1] = torch.tensor(char_ids("London", 12))
        after = model(past, self.lengths)
        self.assertTrue(torch.allclose(before[:, 2:, 1:, 8:], after[:, 2:, 1:, 8:], atol=1e-6))
        self.assertFalse(torch.allclose(before[:, 1, 1:, 8:], model(future, self.lengths)[:, 1, 1:, 8:]))

    def test_padding_invariance_and_layer_shapes(self):
        model = BiLM(self.config, 10, 2).eval()
        alone = model(self.chars, self.lengths)
        chars = torch.zeros(2, 8, 12, dtype=torch.long)
        chars[0, :5] = self.chars[0]
        chars[1] = torch.tensor([char_ids(w, 12) for w in ["<S>", "a", "b", "c", "d", "e", "f", "</S>"]])
        output = model(chars, torch.tensor([5, 8]))
        self.assertEqual(tuple(output.shape), (2, 8, 3, 16))
        self.assertTrue(torch.allclose(alone[0], output[0, :5], atol=1e-6))
        self.assertEqual(output[0, 5:].abs().sum().item(), 0)

    def test_reverse_only_valid_positions(self):
        x = torch.arange(12).reshape(2, 3, 2)
        lens = torch.tensor([2, 3])
        y = reverse_valid(x, lens)
        self.assertTrue(torch.equal(reverse_valid(y, lens), x))
        self.assertTrue(torch.equal(y[0, 2], x[0, 2]))

    def test_crf_partition_and_decode_against_enumeration(self):
        tags = ["O", "B-PER", "I-PER"]
        crf = CRF(tags)
        emissions = torch.randn(1, 3, 3)
        mask = torch.ones(1, 3, dtype=torch.bool)
        trans, start = crf.potentials()
        paths, values = [], []
        for path in itertools.product(range(3), repeat=3):
            value = start[path[0]] + crf.end[path[-1]] + sum(emissions[0, i, p] for i, p in enumerate(path))
            value += sum(trans[a, b] for a, b in zip(path, path[1:]))
            paths.append(path)
            values.append(value)
        values = torch.stack(values)
        self.assertTrue(torch.allclose(crf.log_partition(emissions, mask)[0], torch.logsumexp(values, 0)))
        self.assertEqual(crf.decode(emissions, mask)[0], list(paths[values.argmax()]))
        gold = torch.tensor([[1, 2, 0]])
        expected = torch.logsumexp(values, 0) - values[paths.index((1, 2, 0))]
        self.assertTrue(torch.allclose(crf(emissions, gold, mask), expected))

    def test_crf_masking(self):
        crf = CRF(TAGS)
        x = torch.randn(2, 4, len(TAGS))
        mask = mask_from_lengths(torch.tensor([2, 4]), 4)
        self.assertEqual(crf.decode(x, mask)[0], crf.decode(x[:1, :2], mask[:1, :2])[0])
        self.assertTrue(torch.allclose(crf.log_partition(x, mask)[0], crf.log_partition(x[:1, :2], mask[:1, :2])[0]))

    def test_lm_overfits_tiny_sequence(self):
        model = BiLM(self.config, 8, 2)
        targets = torch.tensor([[2, 4, 5, 6, 3]])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
        first = model.loss(self.chars, targets, self.lengths, 2)[0].item()
        for _ in range(35):
            optimizer.zero_grad()
            loss = model.loss(self.chars, targets, self.lengths, 2)[0]
            loss.backward()
            optimizer.step()
        last = model.loss(self.chars, targets, self.lengths, 2)[0].item()
        self.assertLess(last, first * 0.4)

    def test_ner_overfits_tiny_labeled_batch(self):
        config = {"char_dim": 4, "char_filters": 8, "hidden": [8, 4], "dropout": 0,
                  "elmo_dropout": 0, "mix_l2": 0.1}
        model = NER(config, torch.randn(8, 6), TAGS)
        words = torch.tensor([[2, 3, 4], [5, 6, 7]])
        chars = torch.tensor([[char_ids(w, 12) for w in row] for row in
                              [["Alice", "in", "Paris"], ["Bob", "at", "Acme"]]])
        lengths = torch.tensor([3, 3])
        tags = torch.tensor([[1, 0, 5], [1, 0, 3]])
        mask = torch.ones(2, 3, dtype=torch.bool)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
        for _ in range(80):
            optimizer.zero_grad()
            loss = model.loss(model(words, chars, lengths), tags, mask)
            loss.backward()
            optimizer.step()
        model.eval()
        self.assertEqual(model.crf.decode(model(words, chars, lengths), mask), tags.tolist())

    def test_reference_cell_against_numpy_equations(self):
        rng = np.random.default_rng(3)
        kernel = rng.normal(size=(4, 12)).astype(np.float32)
        bias = rng.normal(size=12).astype(np.float32)
        projection = rng.normal(size=(3, 2)).astype(np.float32)
        x = rng.normal(size=(1, 4, 2)).astype(np.float32)
        layer = ReferenceCell(torch.from_numpy(kernel), torch.from_numpy(bias), torch.from_numpy(projection), 0.7, 0.5)
        actual = layer(torch.from_numpy(x), torch.tensor([4])).numpy()
        c, h, expected = np.zeros((1, 3)), np.zeros((1, 2)), []
        sigmoid = lambda v: 1 / (1 + np.exp(-v))
        for t in range(4):
            i, g, f, o = np.split(np.concatenate([x[:, t], h], -1) @ kernel + bias, 4, axis=-1)
            c = np.clip(sigmoid(f + 1) * c + sigmoid(i) * np.tanh(g), -0.7, 0.7)
            h = np.clip((sigmoid(o) * np.tanh(c)) @ projection, -0.5, 0.5)
            expected.append(h.copy())
        np.testing.assert_allclose(actual, np.stack(expected, 1), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
