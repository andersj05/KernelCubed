import unittest

import torch
import torch.nn.functional as F

from kernelcubed.decode_attention import (
    clear_decode_workspaces,
    decode_attention,
    get_decode_workspace,
    validate_decode_inputs,
)


class DecodeAttentionContractTests(unittest.TestCase):
    def test_rejects_wrong_head_dimension_before_build(self) -> None:
        query = torch.empty(1, 16, 64)
        cache = torch.empty(8, 8, 64)
        with self.assertRaisesRegex(ValueError, "128"):
            validate_decode_inputs(query, cache, cache)

    def test_rejects_non_grouped_head_count_before_build(self) -> None:
        query = torch.empty(1, 15, 128, dtype=torch.float16)
        cache = torch.empty(8, 8, 128, dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "divisible"):
            validate_decode_inputs(query, cache, cache)

    def test_rejects_non_qwen_group_size_before_build(self) -> None:
        query = torch.empty(1, 24, 128, dtype=torch.float16)
        cache = torch.empty(8, 8, 128, dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "two query heads"):
            validate_decode_inputs(query, cache, cache)

    def test_workspace_requires_cuda(self) -> None:
        with self.assertRaisesRegex(ValueError, "CUDA"):
            get_decode_workspace(torch.empty(1, 16, 128))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class DecodeAttentionCudaTests(unittest.TestCase):
    def assert_matches_sdpa(
        self,
        sequence_length: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        generator = torch.Generator(device="cuda").manual_seed(sequence_length)
        query = torch.randn(
            1,
            16,
            128,
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        key = torch.randn(
            sequence_length,
            8,
            128,
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        value = torch.randn(
            sequence_length,
            8,
            128,
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        expected = F.scaled_dot_product_attention(
            query.transpose(0, 1).unsqueeze(0),
            key.transpose(0, 1).unsqueeze(0),
            value.transpose(0, 1).unsqueeze(0),
            dropout_p=0.0,
            enable_gqa=True,
        ).squeeze(0).transpose(0, 1)
        actual = decode_attention(query, key, value)
        torch.testing.assert_close(
            actual, expected, atol=2e-2, rtol=2e-2
        )

    def test_direct_bfloat16_matches_sdpa(self) -> None:
        self.assert_matches_sdpa(37)

    def test_split_bfloat16_matches_sdpa(self) -> None:
        self.assert_matches_sdpa(769)

    def test_direct_float16_matches_sdpa(self) -> None:
        self.assert_matches_sdpa(37, torch.float16)

    def test_split_float16_matches_sdpa(self) -> None:
        self.assert_matches_sdpa(769, torch.float16)


    def test_workspace_is_reused_on_same_stream(self) -> None:
        clear_decode_workspaces()
        query = torch.empty(
            1,
            16,
            128,
            device="cuda",
            dtype=torch.bfloat16,
        )
        first = get_decode_workspace(query)
        second = get_decode_workspace(query)
        self.assertIs(first, second)
        self.assertEqual(first.shape, (64, 16, 130))


if __name__ == "__main__":
    unittest.main()
