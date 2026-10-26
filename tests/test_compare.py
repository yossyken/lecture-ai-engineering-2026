"""Local-only comparison tests. All responses and credentials here are fixtures."""

import contextlib
import importlib
import inspect
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


try:
    compare = importlib.import_module("app.compare")
except ModuleNotFoundError:
    compare = None

CANARY = "synthetic-credential-for-local-tests-only"


def response(text="TEST FIXTURE: explanation", finish_reason="STOP"):
    return SimpleNamespace(
        text=text,
        model_version="fixture-model-version",
        candidates=[SimpleNamespace(finish_reason=finish_reason)],
        usage_metadata=SimpleNamespace(
            prompt_token_count=12,
            candidates_token_count=34,
            total_token_count=46,
        ),
    )


class FixtureClient:
    """Replace only the external API boundary; keep persistence and CLI real."""

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.models = self

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        result = next(self.responses)
        if isinstance(result, BaseException):
            raise result
        return result

    def close(self):
        pass


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(compare, "The comparison runner is not implemented yet")
        # Keep filesystem tests in the checkout: Windows sandbox Temp can deny
        # atomic replacement even though ordinary writes are allowed there.
        cache = Path(__file__).resolve().parents[1] / ".pytest_cache"
        cache.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=cache)
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "observations.json"

    def execute(self, responses, *extra):
        client = FixtureClient(responses)
        console = io.StringIO()
        with (
            patch.object(compare, "get_api_key", return_value=CANARY),
            patch.object(compare, "make_client", return_value=client),
            contextlib.redirect_stdout(console),
            contextlib.redirect_stderr(console),
        ):
            code = compare.main(
                ["--execute", "--output", str(self.output), *extra]
            )
        return code, client, console.getvalue()

    def test_default_and_dry_run_do_not_read_credentials_or_create_client(self):
        for args in ([], ["--dry-run"]):
            with (
                self.subTest(args=args),
                patch.object(compare, "get_api_key", side_effect=AssertionError("credential access")),
                patch.object(compare, "make_client", side_effect=AssertionError("client creation")),
                contextlib.redirect_stdout(io.StringIO()) as console,
            ):
                self.assertEqual(compare.main(args), 0)
                plan = json.loads(console.getvalue())
                self.assertEqual(plan["planned_calls"], 6)
                self.assertEqual(plan["runs_per_model"], 3)
                self.assertFalse(plan["executed"])
        self.assertFalse(self.output.exists())

    def test_execute_records_six_independent_alternating_calls(self):
        from google.genai import types

        code, client, console = self.execute([response() for _ in range(6)])
        self.assertEqual(code, 0)
        self.assertEqual(len(client.calls), 6)
        self.assertEqual(
            [call["model"] for call in client.calls],
            ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"] * 3,
        )
        self.assertEqual(len({call["contents"] for call in client.calls}), 1)
        for call in client.calls:
            parsed_config = types.GenerateContentConfig(**call["config"])
            self.assertFalse(call["config"]["automatic_continuation"])
            self.assertFalse(parsed_config.automatic_continuation)
            self.assertNotIn("temperature", call["config"])
            self.assertEqual(call["config"]["max_output_tokens"], 2048)
        saved = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertTrue(saved["complete"])
        self.assertEqual(len(saved["results"]), 6)
        self.assertEqual([r["run"] for r in saved["results"]], [1, 1, 2, 2, 3, 3])
        first = saved["results"][0]
        self.assertEqual(first["response_text"], "TEST FIXTURE: explanation")
        self.assertEqual(first["returned_model"], "fixture-model-version")
        self.assertEqual(first["tokens"], {"input": 12, "output": 34, "total": 46})
        self.assertEqual(first["finish_reason"], "STOP")
        self.assertGreaterEqual(first["elapsed_seconds"], 0)
        self.assertIn("+00:00", first["started_at_utc"])
        self.assertNotIn(CANARY, console + self.output.read_text(encoding="utf-8"))

    def test_failure_stops_without_retry_and_preserves_completed_observations(self):
        code, client, console = self.execute(
            [response(), RuntimeError("upstream included " + CANARY), response()]
        )
        self.assertEqual(code, 1)
        self.assertEqual(len(client.calls), 2)
        saved = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertFalse(saved["complete"])
        self.assertEqual(len(saved["results"]), 1)
        self.assertEqual(saved["stopped_reason"], "request_failed")
        self.assertNotIn("upstream included", console + json.dumps(saved))
        self.assertNotIn(CANARY, console + json.dumps(saved))

    def test_empty_blocked_or_truncated_response_is_recorded_as_incomplete(self):
        fixtures = (
            ("empty", response(text="")),
            ("blocked", response(finish_reason="SAFETY")),
            ("truncated", response(finish_reason="MAX_TOKENS")),
        )
        for label, fixture in fixtures:
            with self.subTest(fixture=fixture):
                self.output = Path(self.temp.name) / (label + ".json")
                code, client, _ = self.execute([fixture, response()])
                self.assertEqual(code, 1)
                self.assertEqual(len(client.calls), 1)
                saved = json.loads(self.output.read_text(encoding="utf-8"))
                self.assertFalse(saved["complete"])
                self.assertEqual(saved["stopped_reason"], "empty_or_blocked_response")

    def test_interrupt_preserves_already_completed_results(self):
        code, client, _ = self.execute([response(), KeyboardInterrupt()])
        self.assertEqual(code, 130)
        self.assertEqual(len(client.calls), 2)
        saved = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertFalse(saved["complete"])
        self.assertEqual(len(saved["results"]), 1)
        self.assertEqual(saved["stopped_reason"], "interrupted")

    def test_response_strings_are_redacted_before_persistence(self):
        code, _, console = self.execute([response(text="echo " + CANARY) for _ in range(6)])
        self.assertEqual(code, 0)
        saved_text = self.output.read_text(encoding="utf-8")
        self.assertNotIn(CANARY, saved_text + console)
        self.assertIn("[REDACTED]", saved_text)

    def test_write_failure_preserves_last_checkpoint_without_raw_error(self):
        replace = os.replace
        replacements = 0

        def fail_third_checkpoint(source, destination):
            nonlocal replacements
            replacements += 1
            if replacements == 3:
                raise OSError("disk fixture included " + CANARY)
            return replace(source, destination)

        with patch.object(os, "replace", side_effect=fail_third_checkpoint):
            code, client, console = self.execute([response() for _ in range(6)])
        self.assertEqual(code, 1)
        self.assertEqual(len(client.calls), 2)
        saved = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertFalse(saved["complete"])
        self.assertEqual(len(saved["results"]), 1)
        self.assertNotIn(CANARY, console)
        self.assertNotIn("Traceback", console)
        self.assertEqual(list(self.output.parent.glob("*.tmp")), [])

    def test_existing_output_is_preserved_without_reading_credentials(self):
        self.output.write_text("original", encoding="utf-8")
        with (
            patch.object(compare, "get_api_key", side_effect=AssertionError("credential access")),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(compare.main(["--execute", "--output", str(self.output)]), 1)
        self.assertEqual(self.output.read_text(encoding="utf-8"), "original")

    def test_empty_question_is_rejected_before_credentials(self):
        with (
            patch.object(compare, "get_api_key", side_effect=AssertionError("credential access")),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(compare.main(["--execute", "--question", "  "]), 1)

    def test_client_uses_gemini_api_and_disables_sdk_retries(self):
        from google import genai

        signature = inspect.signature(genai.Client)
        with patch.object(genai, "Client") as constructor:
            compare.make_client(CANARY)
        signature.bind(**constructor.call_args.kwargs)
        options = constructor.call_args.kwargs["http_options"]
        self.assertEqual(options.retry_options.attempts, 1)
        self.assertIs(constructor.call_args.kwargs.get("vertexai"), False)
        self.assertIs(constructor.call_args.kwargs.get("enterprise"), False)


if __name__ == "__main__":
    unittest.main()
