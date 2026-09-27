"""
Pre-Filter (Input Guardrails)
=============================

Evaluates a raw user prompt *before* it is sent to the LLM. If a rule
fires here, the pipeline must skip the OpenRouter API call entirely
(saving latency/cost and preventing PII from leaving the system).

Rules implemented:
  1. Jailbreak attempt  -> "rule_jailbreak"
     Case-insensitive match on the exact substring:
     "IGNORE ALL PREVIOUS INSTRUCTIONS"
  2. PII (SSN) attempt  -> "rule_pii"
     Matches a US Social Security Number shape: \\d{3}-\\d{2}-\\d{4}
"""

import re
from typing import Optional, Tuple

# Case-insensitive jailbreak phrase detector.
_JAILBREAK_PATTERN = re.compile(
    r"ignore all previous instructions", re.IGNORECASE
)

# US Social Security Number format: XXX-XX-XXXX
_SSN_PATTERN = re.compile(r"\d{3}-\d{2}-\d{4}")


def run_pre_filter(prompt: str) -> Tuple[bool, Optional[str]]:
    """
    Evaluates a prompt against pre-generation security rules.

    Params:
        prompt (str): The raw user input.

    Returns:
        Tuple[bool, Optional[str]]:
            (True, None)          -> the prompt is safe to forward to the LLM.
            (False, "rule_name")  -> a rule fired; do NOT call the model.
    """
    if prompt is None:
        prompt = ""

    if _JAILBREAK_PATTERN.search(prompt):
        return False, "rule_jailbreak"

    if _SSN_PATTERN.search(prompt):
        return False, "rule_pii"

    return True, None
