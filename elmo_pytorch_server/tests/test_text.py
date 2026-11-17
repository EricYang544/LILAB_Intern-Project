import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from elmo_project.text import char_ids, to_iob2, read_conll, spans, score
from elmo_project.prepare import extract, select_lines
from elmo_project.common import read_json, write_json, sha256


class TextTests(unittest.TestCase):
    def test_byte_mapping_and_special_tokens(self):
        self.assertEqual(char_ids("A", 6), [259, 66, 260, 261, 261, 261])
        self.assertEqual(char_ids("<S>", 6)[:3], [259, 257, 260])
        self.assertEqual(char_ids("</S>", 6)[:3], [259, 258, 260])
        self.assertEqual(len(char_ids("한글" * 50)), 50)
        self.assertNotIn(0, char_ids("é"))

    def test_iob1_span_preserving_conversion(self):
        raw = ["I-PER", "I-PER", "B-PER", "O", "I-ORG", "I-LOC"]
        converted = to_iob2(raw)
        self.assertEqual(converted, ["B-PER", "I-PER", "B-PER", "O", "B-ORG", "B-LOC"])
        self.assertEqual(spans(raw), spans(converted))

    def test_entity_not_token_accuracy(self):
        gold = [["B-PER", "I-PER", "O", "B-LOC"]]
        pred = [["B-PER", "O", "O", "B-LOC"]]
        result = score(gold, pred)
        self.assertEqual(result["ALL"]["f1"], 50)
        self.assertEqual(result["PER"]["f1"], 0)
        self.assertEqual(result["LOC"]["f1"], 100)
        self.assertEqual(score([["O"]], [["O"]])["ALL"]["f1"], 0)
        with self.assertRaises(ValueError):
            score(gold, [["O"]])

    def test_conll_boundaries_and_eof(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "data.txt"
            p.write_text("-DOCSTART- -X- -X- O\n\nAlice X X I-PER\nworks X X O\n\n-DOCSTART- -X- -X- O\nParis X X I-LOC")
            rows = read_conll(p)
            self.assertEqual(len(rows), 2)
            self.assertEqual([r["doc_id"] for r in rows], [0, 1])
            self.assertEqual(rows[1]["tags"], ["B-LOC"])

    def test_archive_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            archive = Path(d) / "bad.tar"
            with tarfile.open(archive, "w") as t:
                member = tarfile.TarInfo("../outside")
                member.size = 3
                t.addfile(member, io.BytesIO(b"bad"))
            with self.assertRaises(ValueError):
                extract(archive, Path(d) / "dest")
            self.assertFalse((Path(d) / "outside").exists())

    def test_selection_deterministic_and_excludes_evaluation_text(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a, b = root / "a", root / "b"
            a.write_text("test sentence\ntraining sentence here\n")
            b.write_text("another training example\n")
            c1, m1 = select_lines([a, b], 20, root / "out1", 42, {"test sentence"})
            c2, m2 = select_lines([a, b], 20, root / "out2", 42, {"test sentence"})
            self.assertEqual(c1, c2)
            self.assertEqual(m1, m2)
            self.assertNotIn("test sentence", (root / "out1").read_text())
            self.assertEqual(sha256(root / "out1"), sha256(root / "out2"))

    def test_atomic_json(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.json"
            write_json(p, {"한글": [1, 2]})
            self.assertEqual(read_json(p), {"한글": [1, 2]})
            self.assertFalse(p.with_suffix(".json.part").exists())


if __name__ == "__main__":
    unittest.main()
