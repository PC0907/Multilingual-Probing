import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from study_truth_transport_v2.runners.extract_prompted_states import content_end_positions
from study_truth_transport_v2.runners.prepare_prompts import render
from study_truth_transport_v2.src.data import pure_group_activations


class PipelinePrimitiveTests(unittest.TestCase):
    def test_content_end_ignores_wrapping_special_tokens(self):
        self.assertEqual(content_end_positions([[1, 7, 8, 2], [4, 5]], [[7, 8], [4, 5]]), [2, 1])
        with self.assertRaises(ValueError):
            content_end_positions([[7, 8, 7, 8]], [[7, 8]])

    def test_pure_group_activations_excludes_mixed_group(self):
        rows = [
            {"group_id": "a", "partition": "train", "label": 0},
            {"group_id": "a", "partition": "train", "label": 0},
            {"group_id": "b", "partition": "train", "label": 0},
            {"group_id": "b", "partition": "train", "label": 1},
            {"group_id": "c", "partition": "test", "label": 1},
        ]
        x = np.arange(10, dtype=float).reshape(5, 2)
        vectors, labels, ids, excluded = pure_group_activations(x, rows, "train")
        np.testing.assert_array_equal(vectors, [[1.0, 2.0]])
        np.testing.assert_array_equal(labels, [0])
        self.assertEqual(ids, ["a"])
        self.assertEqual(excluded, ["b"])

    def test_prompt_renderer_ends_activation_at_statement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "dataset" / "data"
            data.mkdir(parents=True)
            (data / "manifest.json").write_text("{}")
            for language in ("en", "de", "ar", "hi", "fr", "es"):
                (data / f"{language}.json").write_text(json.dumps([{
                    "id": "x", "sentence": f"claim-{language}.", "label": 1,
                    "group_id": "g", "partition": "train"
                }]))
            config = {
                "status": "test", "activation_template_id": "t",
                "languages": {language: {"activation_prefix": f"{language}: ",
                    "answer_true": "T", "answer_false": "F",
                    "behavior_templates": ["{statement}", "x {statement}", "y {statement}"]}
                    for language in ("en", "de", "ar", "hi", "fr", "es")}
            }
            config_path = root / "prompts.json"
            config_path.write_text(json.dumps(config))
            output = root / "rendered"
            render(root / "dataset", config_path, output)
            record = json.loads((output / "de.activation.json").read_text())[0]
            self.assertTrue(record["sentence"].endswith(record["raw_statement"]))
            self.assertEqual(record["target_char_end"], len(record["sentence"]))


if __name__ == "__main__":
    unittest.main()
