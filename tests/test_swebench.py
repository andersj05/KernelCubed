import json
import unittest

from kernelcubed.swebench import (
    SelectionConfig,
    build_messages,
    extract_patch,
    is_complete_unified_diff,
    is_unified_diff,
    patch_stats,
    report_list_size,
    select_tasks,
    sanitize_oracle_prompt,
)


def make_row(
    instance_id: str,
    repo: str,
    changed_lines: int,
    prompt_chars: int = 400,
) -> dict:
    additions = "\n".join(f"+new_{index}" for index in range(changed_lines))
    patch = (
        f"diff --git a/module.py b/module.py\n"
        f"--- a/module.py\n"
        f"+++ b/module.py\n"
        f"@@ -1 +1,{changed_lines} @@\n"
        f"-old\n{additions}\n"
    )
    return {
        "instance_id": instance_id,
        "repo": repo,
        "base_commit": "abc123",
        "version": "1.0",
        "problem_statement": "Fix the small bug.",
        "text": f"<code>{'x' * prompt_chars}</code>trailing demo",
        "patch": patch,
        "FAIL_TO_PASS": json.dumps(["test_one"]),
    }


class SwebenchTests(unittest.TestCase):
    def test_patch_stats_ignores_headers(self) -> None:
        patch = """diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1,2 +1,3 @@
-old
+new
+extra
"""
        self.assertEqual(patch_stats(patch).files, 1)
        self.assertEqual(patch_stats(patch).added_lines, 2)
        self.assertEqual(patch_stats(patch).removed_lines, 1)
        self.assertEqual(patch_stats(patch).changed_lines, 3)

    def test_selection_is_deterministic_and_repo_bounded(self) -> None:
        rows = [
            make_row("repo_a__pkg-2", "repo_a/pkg", 3),
            make_row("repo_a__pkg-1", "repo_a/pkg", 1),
            make_row("repo_b__pkg-1", "repo_b/pkg", 2),
        ]
        selected = select_tasks(
            rows,
            SelectionConfig(count=2, max_per_repo=1),
        )
        self.assertEqual(
            [task["instance_id"] for task in selected],
            ["repo_a__pkg-1", "repo_b__pkg-1"],
        )
        self.assertNotIn("patch", selected[0])

    def test_extract_patch_removes_thinking_and_fence(self) -> None:
        response = """<think>work</think>
Here is the fix:
```diff
diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1 +1 @@
-old
+new
```
"""
        patch = extract_patch(response)
        self.assertTrue(patch.startswith("diff --git"))
        self.assertTrue(is_unified_diff(patch))
        self.assertNotIn("think", patch)
        self.assertNotIn("```", patch)

    def test_non_diff_is_rejected(self) -> None:
        self.assertFalse(is_unified_diff(extract_patch("I cannot solve this.")))

    def test_complete_diff_checks_hunk_line_counts(self) -> None:
        complete = """--- a/a.py
+++ b/a.py
@@ -1,2 +1,2 @@
-old
+new
 context
"""
        truncated = """--- a/a.py
+++ b/a.py
@@ -1,2 +1,2 @@
-old
+new
"""
        self.assertTrue(is_complete_unified_diff(complete))
        self.assertFalse(is_complete_unified_diff(truncated))

    def test_oracle_prompt_removes_bundled_demo(self) -> None:
        text = "prefix<code>source</code>instructions<patch>demo</patch>"
        prompt = sanitize_oracle_prompt(text, "Fix it")
        self.assertIn("<issue>\nFix it\n</issue>", prompt)
        self.assertIn("<code>source</code>", prompt)
        self.assertNotIn("demo", prompt)

    def test_prompt_contract_requests_only_a_diff(self) -> None:
        task = {"oracle_prompt": "Issue and source"}
        messages = build_messages(task)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Return only a unified git diff", messages[1]["content"])

    def test_report_list_size_supports_lists_and_counts(self) -> None:
        self.assertEqual(report_list_size({"resolved_ids": ["a", "b"]}, "resolved_ids"), 2)
        self.assertEqual(report_list_size({"submitted": 3}, "submitted"), 3)
        self.assertIsNone(report_list_size({}, "missing"))


if __name__ == "__main__":
    unittest.main()
