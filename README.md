# Safe LLM Gateway — Qwen via OpenRouter

A secure gateway client that sits between a batch of prompts and the
Qwen model (served through OpenRouter), enforcing explicit **pre-generation**
and **post-generation** safety gates.

## Architecture

```
redteam.json --> [Pre-Filter] --(pass)--> [OpenRouter/Qwen] --> [Post-Filter] --> results.json
                       |                                              |
                    (blocked)                                    (blocked)
                       |                                              |
                       v                                              v
                                    audit.jsonl
```

- **Pre-Filter** (`filters/pre_filter.py`): blocks jailbreak attempts
  (`IGNORE ALL PREVIOUS INSTRUCTIONS`, case-insensitive) and PII / SSNs
  (`\d{3}-\d{2}-\d{4}`) *before* any API call is made.
- **OpenRouter client** (`client/openrouter.py`): calls
  `POST https://openrouter.ai/api/v1/chat/completions` with the model
  pinned to `qwen/qwen-2.5-7b-instruct`.
- **Post-Filter** (`filters/post_filter.py`): blocks confidentiality
  leaks (`[CONFIDENTIAL]`) and dangerous commands (`sudo rm -rf`) in the
  model's raw output.
- **Orchestrator** (`gateway.py`): reads `redteam.json`, runs every
  prompt through the pipeline above, and writes `results.json` +
  `audit.jsonl`.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY=<your real key>
```

## Run

```bash
python gateway.py
```

This reads `redteam.json`, processes every prompt non-interactively, and
writes:

- `results.json` — final outcome per prompt (`id`, `original_prompt`, `response`)
- `audit.jsonl` — one JSON object per line: `{"prompt_id", "safety_rule", "action"}`

Exits with status code `0` on success. If OpenRouter cannot be reached
for a given prompt, that prompt's `response` is set to
`"Error calling API"` rather than crashing the batch.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests mock the OpenRouter HTTP call so they run fully offline and
deterministically, covering:
- Jailbreak / PII prompts never reach the API and are logged as `blocked`.
- Benign prompts reach the API and are logged as `allowed`.
- A mocked `[CONFIDENTIAL]` / `sudo rm -rf` model response is caught by
  the post-filter and overwritten with the canned refusal.
- A simulated network failure does not crash the pipeline.

## Notes

- No Google Cloud / Vertex AI SDKs are used or required.
- `OPENROUTER_API_KEY` is only ever read from the environment (via
  `.env`), never hard-coded.
- The audit log is rewritten fresh on every run so it stays in sync with
  the corresponding `results.json`.
