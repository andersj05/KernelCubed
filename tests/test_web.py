import argparse
import json
import unittest

from kernelcubed.web import ASSETS, ChatRuntime, build_parser, encode_event


class WebTests(unittest.TestCase):
    def test_web_parser_defaults_to_local_server(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 7860)
        self.assertFalse(args.no_open)

    def test_stream_event_is_compact_ndjson(self) -> None:
        encoded = encode_event({"type": "delta", "text": "hello"})
        self.assertTrue(encoded.endswith(b"\n"))
        self.assertEqual(
            json.loads(encoded),
            {"type": "delta", "text": "hello"},
        )

    def test_all_declared_assets_exist(self) -> None:
        from kernelcubed.web import WEB_ROOT

        for filename, _ in ASSETS.values():
            self.assertTrue((WEB_ROOT / filename).is_file(), filename)

    def test_reset_restores_only_system_prompt(self) -> None:
        args = argparse.Namespace(
            system_prompt="Be concise.",
            model="model",
            attention_backend="sdpa",
            device="cpu",
            dtype="float16",
            max_context=128,
            max_new_tokens=8,
            thinking=False,
            seed=0,
        )
        runtime = ChatRuntime(args)
        runtime.messages.append({"role": "user", "content": "Hi"})
        self.assertTrue(runtime.reset())
        self.assertEqual(
            runtime.messages,
            [{"role": "system", "content": "Be concise."}],
        )


if __name__ == "__main__":
    unittest.main()
