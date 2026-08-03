import json
import tempfile
import unittest
from pathlib import Path

import torch

from kernelcubed.benchmark import (
    Case,
    ModelSpec,
    SdpaBackend,
    error_metrics,
    make_qkv,
    parse_lengths,
    percentile,
)


class BenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = ModelSpec(
            model_type="qwen3",
            num_hidden_layers=28,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            max_position_embeddings=40960,
        )

    def test_model_spec_reads_qwen_config(self) -> None:
        config = {
            "model_type": "qwen3",
            "hidden_size": 1024,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "max_position_embeddings": 40960,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )
            self.assertEqual(ModelSpec.from_config(path), self.spec)

    def test_qkv_shapes_match_prefill_and_decode(self) -> None:
        cases = ((Case("prefill", 7), 7), (Case("decode", 7), 1))
        for case, query_length in cases:
            q, k, v = make_qkv(
                self.spec,
                case,
                torch.device("cpu"),
                torch.float32,
                seed=10,
            )
            self.assertEqual(q.shape, (query_length, 16, 128))
            self.assertEqual(k.shape, (7, 8, 128))
            self.assertEqual(v.shape, (7, 8, 128))

    def test_sdpa_gqa_output_shape(self) -> None:
        case = Case("prefill", 5)
        q, k, v = make_qkv(
            self.spec,
            case,
            torch.device("cpu"),
            torch.float32,
            seed=0,
        )
        output = SdpaBackend().run(q, k, v, case.phase)
        self.assertEqual(output.shape, q.shape)

    def test_error_metrics_for_identical_tensors(self) -> None:
        value = torch.randn(3, 4)
        maximum, mean, cosine = error_metrics(value, value)
        self.assertEqual(maximum, 0.0)
        self.assertEqual(mean, 0.0)
        self.assertAlmostEqual(cosine, 1.0, places=6)

    def test_parse_lengths_rejects_non_positive_values(self) -> None:
        self.assertEqual(parse_lengths("1, 16,32"), [1, 16, 32])
        with self.assertRaises(Exception):
            parse_lengths("16,0")

    def test_percentile_interpolates(self) -> None:
        self.assertEqual(percentile([1.0], 0.8), 1.0)
        self.assertAlmostEqual(percentile([1.0, 2.0, 3.0], 0.25), 1.5)


if __name__ == "__main__":
    unittest.main()
