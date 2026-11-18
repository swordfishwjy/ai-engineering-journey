"""Concurrent structured extraction over data/postings.jsonl -> data/extracted.json + metrics."""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

import anthropic
import pydantic

from app.config import get_settings

from .schemas import Extraction

SYSTEM = """You extract job postings from messy text into the given schema.
Only use facts stated in the text. Use null for optional fields that are not stated; never guess numbers.
If the text contains no job posting, return an empty postings list."""


async def extract_one(
    client: anthropic.AsyncAnthropic, model: str, sem: asyncio.Semaphore, row: dict[str, Any]
) -> dict[str, Any]:
    base = {"id": row["id"]}
    async with sem:  # bounds in-flight requests only; parsing/scoring happens outside
        start = time.perf_counter()
        try:
            resp = await client.beta.messages.parse(
                model=model,
                max_tokens=4096,
                system=SYSTEM,
                messages=[{"role": "user", "content": f"<text>\n{row['text']}\n</text>"}],
                output_format=Extraction,
                output_config={"effort": "low"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except pydantic.ValidationError as err:  # truncated JSON (stop_reason=max_tokens)
            return {**base, "ok": False, "error": f"invalid output: {err.errors()[0]['msg']}", "latency_s": time.perf_counter() - start}
        except anthropic.APIError as err:  # SDK already retried 429/5xx/timeouts
            return {**base, "ok": False, "error": f"{type(err).__name__}: {err}", "latency_s": time.perf_counter() - start}
        latency = time.perf_counter() - start

    if resp.stop_reason == "refusal" or resp.parsed_output is None:
        return {**base, "ok": False, "error": f"stop_reason={resp.stop_reason}", "latency_s": latency}
    return {
        **base,
        "ok": True,
        "latency_s": latency,
        "usage": {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens},
        "result": resp.parsed_output.model_dump(),
    }


def field_accuracy(rows: list[dict], results: list[dict]) -> dict[str, float]:
    """Compare against ground truth. Postings are aligned by casefolded title."""
    fields = ["title", "company", "tech_stack", "min_salary", "experience_years", "is_remote", "role_level"]
    hits = {f: 0 for f in fields} | {"posting_count": 0}
    total = 0
    for row, res in zip(rows, results, strict=True):
        truth = row["truth"]["postings"]
        got = res.get("result", {}).get("postings", [])
        hits["posting_count"] += len(truth) == len(got)
        by_title = {p["title"].casefold(): p for p in got}
        for t in truth:
            total += 1
            g = by_title.get(t["title"].casefold())
            if g is None:
                continue
            for f in fields:
                if f == "tech_stack":  # recall: every true tech found (case-insensitive)
                    hits[f] += {s.casefold() for s in t[f]} <= {s.casefold() for s in g[f]}
                else:
                    hits[f] += t[f] == g[f]
    acc = {f: hits[f] / total for f in fields} if total else {}
    acc["posting_count"] = hits["posting_count"] / len(rows)
    return acc


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/postings.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/extracted.json"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--concurrency", type=int, default=10)
    args = ap.parse_args()

    settings = get_settings()
    model = args.model or settings.default_model
    rows = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value(), max_retries=4)
    sem = asyncio.Semaphore(args.concurrency)

    t0 = time.perf_counter()
    results = await asyncio.gather(*(extract_one(client, model, sem, r) for r in rows))
    wall = time.perf_counter() - t0
    await client.close()

    ok = [r for r in results if r["ok"]]
    lat = sorted(r["latency_s"] for r in results)
    summary = {
        "model": model,
        "n": len(results),
        "success_rate": len(ok) / len(results),
        "wall_time_s": round(wall, 2),
        "sum_latency_s": round(sum(lat), 2),  # vs wall_time: speedup from concurrency
        "p50_s": round(statistics.median(lat), 2),
        "p95_s": round(lat[int(0.95 * (len(lat) - 1))], 2),
        "tokens": {k: sum(r["usage"][k] for r in ok) for k in ("input", "output")},
        "accuracy": {k: round(v, 3) for k, v in field_accuracy(rows, results).items()},
        "failures": [{"id": r["id"], "error": r["error"]} for r in results if not r["ok"]],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
