import unittest

import torch

from kernelcubed.chat import (
    GenerationReport,
    build_parser,
    format_report,
    initial_messages,
    resolve_dtype,
    sampling_options,
)
from kernelcubed.transformers_backend import AttentionStats, BACKEND_NAME


class ChatTests(unittest.TestCase):
    def test_parser_defaults_to_custom_decode(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.attention_backend, BACKEND_NAME)
        self.assertEqual(args.max_context, 4096)
        self.assertFalse(args.thinking)

    def test_greedy_sampling_omits_sampling_parameters(self) -> None:
        self.assertEqual(
            sampling_options(0.0, 0.9),
            {"do_sample": False},
        )

    def test_sampled_generation_includes_temperature_and_top_p(self) -> None:
        self.assertEqual(
            sampling_options(0.7, 0.9),
            {
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
            },
        )

    def test_initial_messages_can_disable_system_prompt(self) -> None:
        self.assertEqual(initial_messages(""), [])
        self.assertEqual(
            initial_messages("hello"),
            [{"role": "system", "content": "hello"}],
        )

    def test_dtype_resolution(self) -> None:
        self.assertIs(resolve_dtype("bfloat16"), torch.bfloat16)
        self.assertIs(resolve_dtype("float16"), torch.float16)

    def test_report_formats_throughput_and_backend_counts(self) -> None:
        report = GenerationReport(
            text="ok",
            output_tokens=10,
            elapsed_seconds=2.0,
            attention=AttentionStats(
                custom_decode_calls=252,
                sdpa_fallback_calls=28,
            ),
        )
        rendered = format_report(report)
        self.assertIn("5.0 tok/s", rendered)
        self.assertIn("custom decode calls=252", rendered)
        self.assertIn("SDPA fallback calls=28", rendered)


if __name__ == "__main__":
    unittest.main()
