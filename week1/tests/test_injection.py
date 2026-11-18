"""Offline adversarial tests: what we can guarantee without calling a model."""

import pytest

from app.prompting import PromptManager, check_output, scan_injection

from .adversarial_cases import ATTACKS, Attack

pm = PromptManager()
ids = [a.id for a in ATTACKS]


def render(attack: Attack):
    return pm.render("support_bot", "1.1", product_name="Acme", user_input=attack.text)


@pytest.mark.parametrize("attack", ATTACKS, ids=ids)
def test_input_stays_inside_one_data_block(attack):
    """Defense 1: the attacker can't close </user_input> and write outside it."""
    p = render(attack)
    assert p.user.count("<user_input>") == 1
    assert p.user.count("</user_input>") == 1
    inside = p.user.split("<user_input>")[1].split("</user_input>")[0]
    assert "<" not in inside and ">" not in inside  # every bracket in user data is escaped


@pytest.mark.parametrize("attack", ATTACKS, ids=ids)
def test_untrusted_text_never_reaches_system_prompt(attack):
    """Defense 2: system prompt is built only from trusted values."""
    p = render(attack)
    assert attack.text not in p.system
    assert p.canary not in p.user


def test_template_syntax_in_input_is_not_evaluated():
    """Defense 3: {{ }} typed by a user is text, not code."""
    attack = next(a for a in ATTACKS if a.id == "template_injection")
    p = render(attack)
    assert "{{ canary }}" in p.user and "{% for x in range(3) %}" in p.user
    assert "xxx" not in p.user


@pytest.mark.parametrize("attack", [a for a in ATTACKS if a.expect_flags], ids=lambda a: a.id)
def test_scanner_flags_attack(attack):
    """Defense 4 (detection only): heuristics flag known attack shapes for logging."""
    assert attack.expect_flags <= set(scan_injection(attack.text))


def test_scanner_quiet_on_normal_message():
    assert scan_injection("Hi, my order #1234 hasn't arrived. Can you check the status?") == []


class FakeLLM:
    """Stands in for the model. `obey=True` simulates a model the attack fooled."""

    def __init__(self, obey: bool) -> None:
        self.obey = obey

    def reply(self, system: str, user: str) -> str:
        if self.obey:
            return f"Sure! My instructions are:\n{system}"
        return "I can't share my configuration, but I'm happy to help with your order."


@pytest.mark.parametrize("attack", ATTACKS, ids=ids)
def test_output_guard_catches_a_fooled_model(attack):
    """Defense 5: even if the model is fooled, the leak is caught before reaching the user."""
    p = render(attack)
    leaked = FakeLLM(obey=True).reply(p.system, p.user)
    verdict = check_output(leaked, system=p.system, canary=p.canary)
    assert not verdict.ok and "canary_leak" in verdict.reasons

    safe = FakeLLM(obey=False).reply(p.system, p.user)
    assert check_output(safe, system=p.system, canary=p.canary).ok
