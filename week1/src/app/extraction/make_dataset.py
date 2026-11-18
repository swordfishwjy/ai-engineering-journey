"""Build data/postings.jsonl: ground truth first (seeded random), then an LLM renders it into messy text."""

import argparse
import asyncio
import json
import random
from pathlib import Path

import anthropic

from app.config import get_settings

from .schemas import Extraction, JobPosting

TITLES = {
    "Junior": ["Junior Backend Developer", "Associate Software Engineer", "Junior Frontend Engineer"],
    "Mid": ["Software Engineer II", "Backend Engineer", "Full Stack Developer", "Data Engineer"],
    "Senior": ["Senior Backend Engineer", "Senior ML Engineer", "Senior Platform Engineer"],
    "Staff": ["Staff Software Engineer", "Staff Infrastructure Engineer", "Principal Engineer"],
}
YEARS = {"Junior": (0, 2), "Mid": (2, 5), "Senior": (5, 8), "Staff": (8, 12)}
SALARY = {"Junior": (70, 110), "Mid": (110, 160), "Senior": (150, 210), "Staff": (200, 280)}
COMPANIES = ["Nimbus Labs", "Quartzline", "Helio Health", "Brightpath AI", "Ferro Systems", "Lumen Pay", "Orbit Freight", "Tidewater Games"]
STACK = ["Python", "Go", "TypeScript", "Java", "Rust", "React", "FastAPI", "Django", "Spring Boot", "PostgreSQL",
         "Redis", "Kafka", "Kubernetes", "AWS", "GCP", "Terraform", "PyTorch", "Snowflake"]
STYLES = [
    "scraped job-board HTML with leftover nav menus, cookie banner text and footer links",
    "Hacker News 'Who is hiring' comment, terse, pipe-separated header line",
    "recruiter LinkedIn DM, overly enthusiastic, emojis, vague about some details",
    "Chinese-English mixed forum post (V2EX style), salary may be written like '150k USD/年'",
    "forwarded email chain with signatures, disclaimers and a 'see below' note",
    "OCR'd PDF with broken line wraps, a few typos and stray page numbers",
    "Slack message in a #jobs channel, casual, abbreviations like 'yoe', 'TC', 'k8s'",
    "tweet thread split into 3 short numbered posts",
]
NOISE_STYLES = [
    "Reddit post asking for career advice about switching from backend to ML (not a job posting)",
    "tech blog intro paragraph about migrating from MySQL to PostgreSQL (not a job posting)",
]

RENDER_PROMPT = """Write a realistic, messy piece of text in this style: {style}.

It must contain exactly the job posting(s) in the JSON below. Rules:
- Keep each title and company name verbatim.
- Mention every tech_stack item, but you may use informal names (k8s, pg, TS).
- If min_salary is null, give no USD salary (you may omit pay or use another currency). Otherwise express it as a range or hourly rate whose minimum annualizes to exactly that number.
- If experience_years is null, do not mention years of experience.
- is_remote=false means onsite or hybrid; say which.
- Do not state the role level word unless it is in the title.
Output only the text.

{truth}"""


def random_posting(rng: random.Random) -> JobPosting:
    level = rng.choice(list(TITLES))
    lo, hi = SALARY[level]
    return JobPosting(
        title=rng.choice(TITLES[level]),
        company=rng.choice(COMPANIES),
        tech_stack=rng.sample(STACK, rng.randint(2, 5)),
        min_salary=None if rng.random() < 0.3 else rng.randrange(lo, hi, 5) * 1000,
        experience_years=None if rng.random() < 0.2 else rng.randint(*YEARS[level]),
        is_remote=rng.random() < 0.5,
        role_level=level,
    )

# Plan is to generate a list of n items
def plan(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    items = []
    for i in range(n):
        if i % 17 == 16:  # ~6%: not a job posting at all
            items.append({"id": i, "style": rng.choice(NOISE_STYLES), "truth": Extraction(postings=[])})
        else:
            count = 2 if i % 11 == 10 else 1  # ~9%: two roles in one text
            items.append({"id": i, "style": rng.choice(STYLES), "truth": Extraction(postings=[random_posting(rng) for _ in range(count)])})
    return items


async def render(client: anthropic.AsyncAnthropic, model: str, sem: asyncio.Semaphore, item: dict) -> dict:
    truth = item["truth"]
    prompt = (
        f"Write a realistic, messy piece of text in this style: {item['style']}. Output only the text."
        if not truth.postings
        else RENDER_PROMPT.format(style=item["style"], truth=truth.model_dump_json(indent=2))
    )
    async with sem:
        msg = await client.messages.create(
            model=model,
            max_tokens=2048,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": prompt}],
        )
    text = "".join(b.text for b in msg.content if b.type == "text")
    return {"id": item["id"], "style": item["style"], "text": text, "truth": truth.model_dump()}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", type=Path, default=Path("data/postings.jsonl"))
    args = ap.parse_args()

    settings = get_settings()
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value(), max_retries=4)
    sem = asyncio.Semaphore(10)
    rows = await asyncio.gather(*(render(client, args.model or settings.default_model, sem, it) for it in plan(args.n, args.seed)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
