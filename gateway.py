#!/usr/bin/env python3
"""
gateway.py
==========

Main execution CLI for the Safe LLM Client.

Pipeline, per prompt in redteam.json:
    1. Run Pre-Filter (jailbreak / PII).
       - If it fires: log "blocked" to audit.jsonl, response = canned
         refusal, DO NOT call OpenRouter.
    2. Otherwise, call Qwen via OpenRouter.
       - If the network call fails: response = "Error calling API",
         logged as "allowed" (it was not blocked by a safety rule).
    3. Run Post-Filter on the raw model output.
       - If it fires: log "blocked" to audit.jsonl, response = canned
         refusal.
       - Otherwise: log "allowed", response = raw model output.
    4. Write the full batch of results to results.json.

Runs non-interactively and exits 0 on success:
    $ python gateway.py
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is a required dependency
    def load_dotenv(*_args, **_kwargs):
        return False

from filters.pre_filter import run_pre_filter
from filters.post_filter import run_post_filter
from client.openrouter import call_qwen_model, OpenRouterError

REDTEAM_PATH = "redteam.json"
RESULTS_PATH = "results.json"
AUDIT_PATH = "audit.jsonl"
REFUSAL_MESSAGE = "Refused by safety policy."
API_ERROR_MESSAGE = "Error calling API"


def load_prompts(path: str) -> List[Dict[str, Any]]:
    """Reads and validates the redteam.json input batch."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a top-level JSON array.")

    for item in data:
        if not isinstance(item, dict) or "id" not in item or "prompt" not in item:
            raise ValueError(
                f"{path} entries must be objects with at least 'id' and 'prompt' keys."
            )

    return data


def append_audit_log(
    path: str, prompt_id: str, rule: Optional[str], action: str
) -> None:
    """
    Appends a single audit event to the JSONL audit log.

    Params:
        path (str): Path to audit.jsonl.
        prompt_id (str): The id from redteam.json.
        rule (Optional[str]): The safety rule id that fired, or None.
        action (str): "blocked" or "allowed".
    """
    log_entry = {
        "prompt_id": prompt_id,
        "safety_rule": rule,
        "action": action,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


def process_prompt(item: Dict[str, Any], audit_path: str) -> Dict[str, Any]:
    """Runs a single prompt through the full safety-gate pipeline."""
    prompt_id = str(item.get("id", ""))
    original_prompt = item.get("prompt", "")

    # --- Pre-Filter -------------------------------------------------
    passed, rule = run_pre_filter(original_prompt)
    if not passed:
        append_audit_log(audit_path, prompt_id, rule, "blocked")
        return {
            "id": prompt_id,
            "original_prompt": original_prompt,
            "response": REFUSAL_MESSAGE,
        }

    # --- OpenRouter / Qwen call --------------------------------------
    try:
        raw_output = call_qwen_model(original_prompt)
    except OpenRouterError as exc:
        print(f"[warning] OpenRouter call failed for {prompt_id}: {exc}", file=sys.stderr)
        # Not a safety-rule rejection - the request itself was allowed
        # through the gate, it simply could not be completed.
        append_audit_log(audit_path, prompt_id, None, "allowed")
        return {
            "id": prompt_id,
            "original_prompt": original_prompt,
            "response": API_ERROR_MESSAGE,
        }

    # --- Post-Filter --------------------------------------------------
    passed, rule = run_post_filter(raw_output)
    if not passed:
        append_audit_log(audit_path, prompt_id, rule, "blocked")
        return {
            "id": prompt_id,
            "original_prompt": original_prompt,
            "response": REFUSAL_MESSAGE,
        }

    append_audit_log(audit_path, prompt_id, None, "allowed")
    return {
        "id": prompt_id,
        "original_prompt": original_prompt,
        "response": raw_output,
    }


def main() -> int:
    load_dotenv()

    # Start each run with a clean audit log so results/audit stay in sync.
    if os.path.exists(AUDIT_PATH):
        os.remove(AUDIT_PATH)

    try:
        prompts = load_prompts(REDTEAM_PATH)
    except FileNotFoundError:
        print(f"[error] {REDTEAM_PATH} not found.", file=sys.stderr)
        return 1
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"[error] Failed to parse {REDTEAM_PATH}: {exc}", file=sys.stderr)
        return 1

    results = []
    for item in prompts:
        try:
            result = process_prompt(item, AUDIT_PATH)
        except Exception as exc:  # noqa: BLE001 - keep the batch alive
            print(f"[error] Unexpected failure processing {item.get('id')}: {exc}", file=sys.stderr)
            result = {
                "id": item.get("id", ""),
                "original_prompt": item.get("prompt", ""),
                "response": API_ERROR_MESSAGE,
            }
        results.append(result)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Processed {len(results)} prompt(s).")
    print(f"Results written to {RESULTS_PATH}")
    print(f"Audit log written to {AUDIT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
