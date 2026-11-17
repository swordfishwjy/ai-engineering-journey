# CLAUDE.md

## Purpose

This repo is my 6-week SDE → AI Engineer study log: exercises, mini-projects, and notes.

## Study plan (read first)

@docs/current-week.md

The current week's schedule and exercise specs are above. The full 6-week plan (all weeks, resources) is in `docs/study-plan.md` — read it only when a task touches another week or needs resource recommendations.

When I ask for help with a task, map it to the matching week/day/exercise in the plan and hold the output to that exercise's acceptance criteria. If a request is outside the plan, say so briefly and proceed.

## Up-to-date docs (mandatory for coding)

Before writing or changing any code that uses a library, SDK, framework, or CLI (e.g. `uv`, FastAPI, Pydantic, `openai`, `anthropic`, `tiktoken`, LangGraph, MCP, pgvector, Ragas, vLLM):

1. Use the Context7 MCP (`resolve-library-id` → `query-docs`) to fetch current docs. If the MCP is unavailable, use https://context7.com/.
2. Match the version pinned in `pyproject.toml` / `uv.lock` when one exists.
3. Do not rely on memory for API signatures — these SDKs change frequently (e.g. structured outputs, tool calling, streaming APIs).

## Conventions

- Python, managed with `uv` (`uv add`, `uv run`). No `pip install`.
- Async-first (`AsyncOpenAI` / `AsyncAnthropic`); never call blocking SDKs inside async routes.
- Config via `pydantic-settings` + `.env`; never commit `.env` or API keys.
- No LangChain-style heavy frameworks unless the plan's exercise calls for one (e.g. LangGraph in week 3).
- Be terse. Give code, not high-level summaries.
