"""
Post-Filter (Output Guardrails)
===============================

Evaluates the raw text returned by the LLM *after* generation. Even with
pre-filters in place, a model can hallucinate or leak restricted content,
so this acts as the final safety net before anything reaches the user.

Rules implemented:
  1. Confidentiality breach -> "rule_confidential_leak"
     Output contains the literal substring: "[CONFIDENTIAL]"
  2. Dangerous command      -> "rule_dangerous_command"
     Output contains the literal substring: "sudo rm -rf"
"""

import re
from typing import Optional, Tuple

_CONFIDENTIAL_PATTERN = re.compile(re.escape("[CONFIDENTIAL]"))
_DANGEROUS_CMD_PATTERN = re.compile(re.escape("sudo rm -rf"))


def run_post_filter(output_text: str) -> Tuple[bool, Optional[str]]:
    """
    Evaluates model output against post-generation security rules.

    Params:
        output_text (str): The raw text returned by the LLM.

    Returns:
        Tuple[bool, Optional[str]]:
            (True, None)          -> the output is safe to return to the user.
            (False, "rule_name")  -> a rule fired; the output must be replaced
                                      with the canned refusal message.
    """
    if output_text is None:
        output_text = ""

    if _CONFIDENTIAL_PATTERN.search(output_text):
        return False, "rule_confidential_leak"

    if _DANGEROUS_CMD_PATTERN.search(output_text):
        return False, "rule_dangerous_command"

    return True, None
