import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from kernelcubed.transformers_backend import (
    BACKEND_NAME,
    get_attention_stats,
    kernelcubed_attention_forward,
    register_transformers_backend,
    reset_attention_stats,
)


class TransformersBackendContractTests(unittest.TestCase):
    def test_stats_can_be_reset(self) -> None:
        reset_attention_stats()
        stats = get_attention_stats()
        self.assertEqual(stats.custom_decode_calls, 0)
        self.assertEqual(stats.sdpa_fallback_calls, 0)

    def test_backend_registers_with_transformers(self) -> None:
        name = register_transformers_backend()
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        self.assertEqual(name, BACKEND_NAME)
        self.assertIs(
            ALL_ATTENTION_FUNCTIONS[name],
            kernelcubed_attention_forward,
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TransformersBackendCudaTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_attention_stats()
        self.module = SimpleNamespace(
            num_key_value_groups=2,
            is_causal=True,
            sliding_window=None,
        )

    def test_decode_matches_sdpa_without_copying_cache(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(7)
        query = torch.randn(
            1, 16, 1, 128, device="cuda", dtype=torch.bfloat16,
            generator=generator,
        )
        key = torch.randn(
            1, 8, 769, 128, device="cuda", dtype=torch.bfloat16,
            generator=generator,
        )
        value = torch.randn(
            1, 8, 769, 128, device="cuda", dtype=torch.bfloat16,
            generator=generator,
        )
        expected = F.scaled_dot_product_attention(
            query,
            key,
            value,
            enable_gqa=True,
        ).transpose(1, 2)
        actual, weights = kernelcubed_attention_forward(
            self.module,
            query,
            key,
            value,
            None,
            scaling=128.0**-0.5,
        )
        self.assertIsNone(weights)
        torch.testing.assert_close(
            actual, expected, atol=2e-2, rtol=2e-2
        )
        stats = get_attention_stats()
        self.assertEqual(stats.custom_decode_calls, 1)
        self.assertEqual(stats.sdpa_fallback_calls, 0)

    def test_prefill_falls_back_to_sdpa(self) -> None:
        query = torch.randn(
            1, 16, 3, 128, device="cuda", dtype=torch.bfloat16
        )
        key = torch.randn(
            1, 8, 3, 128, device="cuda", dtype=torch.bfloat16
        )
        value = torch.randn_like(key)
        actual, weights = kernelcubed_attention_forward(
            self.module,
            query,
            key,
            value,
            None,
            scaling=128.0**-0.5,
        )
        self.assertIsNone(weights)
        self.assertEqual(actual.shape, (1, 3, 16, 128))
        stats = get_attention_stats()
        self.assertEqual(stats.custom_decode_calls, 0)
        self.assertEqual(stats.sdpa_fallback_calls, 1)


if __name__ == "__main__":
    unittest.main()
