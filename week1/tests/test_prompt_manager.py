from pathlib import Path

import pytest

from app.prompting import PromptError, PromptManager


@pytest.fixture
def pm() -> PromptManager:
    return PromptManager()


def write(root: Path, name: str, version: str, body: str) -> None:
    (root / name).mkdir(parents=True, exist_ok=True)
    (root / name / f"v{version}.yaml").write_text(body)


def test_latest_version_is_default(pm):
    assert pm.versions("support_bot") == ["1.0", "1.1"]
    assert pm.render("support_bot", product_name="Acme", user_input="hi").version == "1.1"


def test_explicit_version(pm):
    p = pm.render("support_bot", "1.0", product_name="Acme", user_input="hi")
    assert p.version == "1.0" and p.canary is None


def test_versions_sort_numerically(tmp_path):
    for v in ["1.9", "1.10", "1.2"]:
        write(tmp_path, "p", v, f'name: p\nversion: "{v}"\nsystem: s\nuser: u\n')
    assert PromptManager(tmp_path).versions("p") == ["1.2", "1.9", "1.10"]


def test_trusted_vars_render(pm):
    p = pm.render("support_bot", product_name="Acme Shoes", user_input="hi")
    assert "Acme Shoes" in p.system


def test_missing_and_extra_vars_raise(pm):
    with pytest.raises(PromptError, match="missing"):
        pm.render("support_bot", product_name="Acme")
    with pytest.raises(PromptError, match="unexpected"):
        pm.render("support_bot", product_name="Acme", user_input="hi", debug=True)


def test_untrusted_var_in_system_prompt_is_rejected(tmp_path):
    write(tmp_path, "bad", "1.0", 'name: bad\nversion: "1.0"\nuntrusted_vars: [q]\nsystem: "{{ q }}"\nuser: u\n')
    with pytest.raises(PromptError, match="must not appear in the system prompt"):
        PromptManager(tmp_path).render("bad", q="x")


def test_filename_and_version_must_match(tmp_path):
    write(tmp_path, "p", "2.0", 'name: p\nversion: "1.0"\nsystem: s\nuser: u\n')
    with pytest.raises(PromptError, match="file says"):
        PromptManager(tmp_path).load("p", "2.0")


def test_sha_identifies_template_version(pm):
    a = pm.render("support_bot", "1.0", product_name="A", user_input="x")
    b = pm.render("support_bot", "1.1", product_name="A", user_input="x")
    c = pm.render("support_bot", "1.1", product_name="B", user_input="y")
    assert a.sha256 != b.sha256 and b.sha256 == c.sha256  # same template -> same hash


def test_canary_is_fresh_per_render(pm):
    a = pm.render("support_bot", product_name="A", user_input="x")
    b = pm.render("support_bot", product_name="A", user_input="x")
    assert a.canary and a.canary in a.system and a.canary != b.canary


def test_to_anthropic_shape(pm):
    kw = pm.render("job_extractor", text="Hiring a Go dev").to_anthropic()
    assert kw["messages"][0]["role"] == "user" and "<text>" in kw["messages"][0]["content"]
