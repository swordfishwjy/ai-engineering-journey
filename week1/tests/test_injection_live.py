"""Live adversarial test: calls the real model. Skipped by default.

Run with:  uv run pytest -m live -v
"""

import anthropic
import pytest

from app.config import get_settings
from app.prompting import PromptManager, check_output

from .adversarial_cases import ATTACKS

pytestmark = pytest.mark.live
pm = PromptManager()


@pytest.fixture(scope="module")
def client():
    return anthropic.Anthropic(api_key=get_settings().anthropic_api_key.get_secret_value())


@pytest.mark.parametrize(
    "version",
    [pytest.param("1.0", marks=pytest.mark.xfail(reason="baseline without defenses", strict=False)), "1.1"],
)
@pytest.mark.parametrize("attack", ATTACKS, ids=[a.id for a in ATTACKS])
def test_model_resists(client, attack, version):
    p = pm.render("support_bot", version, product_name="Acme", user_input=attack.text)
    resp = client.messages.create(
        model=get_settings().default_model,
        max_tokens=512,
        output_config={"effort": "low"},
        **p.to_anthropic(),
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    verdict = check_output(text, system=p.system, canary=p.canary)
    assert verdict.ok, f"{verdict.reasons}\n--- response ---\n{text}"
