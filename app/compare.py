"""Day 1 basic comparison. Default: show the plan; --execute: make six calls.

Only the user should run --execute after entering their key in a private local
terminal. Results are observations, not a generated homework reflection.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import time

MODELS = ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite")
RUNS_PER_MODEL = 3
MAX_OUTPUT_TOKENS = 2048
DEFAULT_QUESTION = "LoRAとは何ですか？日本語で3文以内で説明してください。"


def get_api_key():
    # Lazy loading keeps the default plan independent of credentials.
    from app.config import get_api_key as load_key

    return load_key()


def make_client(key):
    from google import genai
    from google.genai import types

    return genai.Client(
        api_key=key,
        vertexai=False,
        enterprise=False,
        http_options=types.HttpOptions(
            timeout=60000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


def redact(value, key):
    """Protect the exact credential even if a server unexpectedly echoes it."""
    if isinstance(value, str):
        return value.replace(key, "[REDACTED]") if key else value
    if isinstance(value, list):
        return [redact(item, key) for item in value]
    if isinstance(value, dict):
        return {name: redact(item, key) for name, item in value.items()}
    return value


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class CheckpointError(Exception):
    """A safe marker; the underlying filesystem error is never printed."""


def save_checkpoint(output, state, key):
    """Replace a complete checkpoint atomically, keeping the prior one on error."""
    temporary = None
    try:
        data = json.dumps(redact(state, key), ensure_ascii=False, indent=2) + "\n"
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, suffix=".tmp", delete=False
        ) as stream:
            temporary = stream.name
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except (OSError, TypeError, ValueError):
        raise CheckpointError() from None
    finally:
        if temporary is not None:
            try:
                Path(temporary).unlink(missing_ok=True)
            except OSError:
                pass


def run_experiment(question, output):
    """Save each completed observation; stop on any failed/blocked call."""
    state = {
        "question": question,
        "models": list(MODELS),
        "runs_per_model": RUNS_PER_MODEL,
        "planned_calls": 6,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "temperature": "model default (not overridden)",
        "started_at_utc": utc_now(),
        "complete": False,
        "results": [],
    }
    key = ""
    client = None
    try:
        # Exclusive creation refuses a race or an existing experiment file.
        output.parent.mkdir(parents=True, exist_ok=True)
        output.touch(exist_ok=False)
    except OSError:
        print("Cannot create a new result file; existing files are preserved.", file=sys.stderr)
        return 1

    exit_code = 1
    try:
        key = get_api_key()
        client = make_client(key)
        save_checkpoint(output, state, key)
        for run in range(1, RUNS_PER_MODEL + 1):
            for model in MODELS:
                started_at = utc_now()
                started = time.perf_counter()
                reply = client.models.generate_content(
                    model=model,
                    contents=question,
                    config={
                        "max_output_tokens": MAX_OUTPUT_TOKENS,
                        "automatic_continuation": False,
                    },
                )
                elapsed = time.perf_counter() - started
                candidates = reply.candidates or []
                finish = str(candidates[0].finish_reason or "") if candidates else ""
                # Enum values and plain strings both occur in SDK responses.
                finish = finish.rsplit(".", 1)[-1]
                text = reply.text or ""
                if not text.strip() or finish != "STOP":
                    state["stopped_reason"] = "empty_or_blocked_response"
                    save_checkpoint(output, state, key)
                    print("Stopped: empty, blocked or unfinished response. Partial results saved.", file=sys.stderr)
                    return 1
                usage = reply.usage_metadata
                state["results"].append({
                    "requested_model": model,
                    "returned_model": reply.model_version,
                    "run": run,
                    "started_at_utc": started_at,
                    "elapsed_seconds": round(elapsed, 3),
                    "tokens": {
                        "input": getattr(usage, "prompt_token_count", None),
                        "output": getattr(usage, "candidates_token_count", None),
                        "total": getattr(usage, "total_token_count", None),
                    },
                    "finish_reason": finish,
                    "response_text": text,
                })
                save_checkpoint(output, state, key)
        state["complete"] = True
        state["finished_at_utc"] = utc_now()
        save_checkpoint(output, state, key)
        print("Completed six calls. Results saved locally; write your own comparison.")
        return 0
    except CheckpointError:
        print("Stopped: result save failed. The last complete checkpoint is preserved.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        state["stopped_reason"] = "interrupted"
        exit_code = 130
    except (Exception, SystemExit):
        # Never print raw exceptions, SDK objects, headers or credentials.
        state["stopped_reason"] = "request_failed"
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
    try:
        save_checkpoint(output, state, key)
    except CheckpointError:
        print("Stopped: result save failed. The last complete checkpoint is preserved.", file=sys.stderr)
        return 1
    print("Stopped without retry. Partial results saved; no raw error details logged.", file=sys.stderr)
    return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="Make six real API calls (user only).")
    mode.add_argument("--dry-run", action="store_true", help="Show the plan without credentials or calls (default).")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.question.strip():
        print("Question must not be empty.", file=sys.stderr)
        return 1
    if not args.execute:
        print(json.dumps({
            "executed": False,
            "question": args.question,
            "models": list(MODELS),
            "runs_per_model": RUNS_PER_MODEL,
            "planned_calls": 6,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "temperature": "model default (not overridden)",
        }, ensure_ascii=False, indent=2))
        return 0
    output = args.output or Path("local_results") / (
        "day1-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json"
    )
    if output.exists():
        print("Output already exists; choose a new file.", file=sys.stderr)
        return 1
    return run_experiment(args.question, output)


if __name__ == "__main__":
    raise SystemExit(main())
