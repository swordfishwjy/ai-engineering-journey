# Current Week: Week 1 — AI Dev Toolchain & LLM Core APIs (22h)

Goal: modern Python async stack + structured outputs; stop hand-writing boilerplate.

| Day     | Exercise                                  | Done when                                                                    |
| ------- | ----------------------------------------- | ---------------------------------------------------------------------------- |
| Mon–Tue | 1. uv + FastAPI scaffold, SSE streaming   | Full async FastAPI scaffold built via natural-language prompts              |
| Wed–Thu | 2. Concurrent structured extractor        | Async batch entity extraction, strict JSON output                            |
| Fri     | 3. Prompt template engine + anti-injection | Reusable prompt template module                                              |
| Sat     | 4. Chat backend w/ sliding window + summary | Concurrent requests; auto-summarize when context exceeds threshold          |
| Sun     | 5. Native function-calling loop           | Model decides when to call tools and integrates results into the answer     |

## Exercise specs

**1. Scaffold + streaming**

- `uv init`; deps: `fastapi`, `uvicorn`, `pydantic-settings`, `openai`, `anthropic`, `tiktoken`
- `pydantic-settings` loads API key, default model, timeout, max retries from `.env`
- `/v1/chat/stream`: `StreamingResponse`, SSE (`text/event-stream`) chunks
- Simulate timeout / 429 → exponential backoff + graceful error response

**2. Structured extractor**

- Nested Pydantic schema, e.g. `JobPostingSchema`: `title`, `company`, `tech_stack: list[str]`, `min_salary: Optional[int]`, `experience_years: int`, `is_remote: bool`, `role_level: Literal["Junior","Mid","Senior","Staff"]`
- 50 messy unstructured texts
- OpenAI structured parse, or Anthropic forced tool use
- `asyncio.Semaphore(10)`; report total time + success rate; save results to JSON

**3. Prompt engine + injection defense**

- `PromptManager`: load system/user prompts from YAML or Jinja2, variable injection, versions (`v1.0`, `v1.1`)
- Wrap untrusted input in `<user_input>…</user_input>`; system prompt explicitly defends against "ignore previous instructions / reveal system prompt"
- 5 adversarial test cases

**4. Context memory backend**

- `tiktoken` (or model tokenizer) to count history tokens before each request
- Session store: in-memory dict or SQLite/Redis keyed by `session_id`
- Over ~2000 tokens → async call to small model (gpt-4o-mini / Haiku) to summarize oldest N turns into a `Conversation Summary`; keep last 4 turns verbatim

**5. Function-calling loop (no LangChain)**

- Tools + JSON Schemas: `query_order_status(order_id)` (SQLite), `calculate_refund(amount, days_since_purchase)`, `search_policy_docs(keyword)` (local Markdown)
- `while` loop: send messages + tools → on `tool_calls`, validate args, execute safely, append `role: "tool"` results → repeat until final text
- Support parallel tool calls
