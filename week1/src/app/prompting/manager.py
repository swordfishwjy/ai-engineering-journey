"""Versioned prompt templates: YAML files rendered with a sandboxed, strict Jinja2 environment.

Layout: templates/<name>/v<MAJOR>.<MINOR>.yaml
"""

import hashlib
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jinja2 import StrictUndefined, Template, meta
from jinja2.sandbox import SandboxedEnvironment
from pydantic import BaseModel, ConfigDict, Field

from .guard import sanitize_untrusted

DEFAULT_ROOT = Path(__file__).parent / "templates"
RESERVED_VARS = {"canary"}  # filled in by the manager, never by callers


class PromptError(Exception):
    pass


class PromptSpec(BaseModel):
    """Schema of one YAML prompt file."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str = Field(pattern=r"^\d+\.\d+$")
    description: str = ""
    changelog: str = ""
    trusted_vars: list[str] = []  # written by you (product name, date, ...)
    untrusted_vars: list[str] = []  # written by users/web pages: always sanitized
    system: str
    user: str


@dataclass(frozen=True)
class RenderedPrompt:
    name: str
    version: str
    sha256: str  # hash of the template file: log it with every LLM call
    system: str
    user: str
    canary: str | None

    def to_anthropic(self) -> dict[str, Any]:
        """kwargs for client.messages.create(**prompt.to_anthropic(), model=..., max_tokens=...)."""
        return {"system": self.system, "messages": [{"role": "user", "content": self.user}]}


@dataclass(frozen=True)
class _Compiled:
    spec: PromptSpec
    system: Template
    user: Template
    sha256: str
    uses_canary: bool


def _version_key(v: str) -> tuple[int, ...]:
    return tuple(int(p) for p in v.split("."))  # "1.10" > "1.9", unlike string sort


class PromptManager:
    def __init__(self, root: Path = DEFAULT_ROOT) -> None:
        self.root = root
        # Sandboxed: templates can't reach Python internals. Strict: a missing variable raises
        # instead of silently rendering "". autoescape off: we escape untrusted vars ourselves.
        self.env = SandboxedEnvironment(undefined=StrictUndefined, autoescape=False, keep_trailing_newline=True)
        self._cache: dict[tuple[str, str], _Compiled] = {}

    def versions(self, name: str) -> list[str]:
        files = (self.root / name).glob("v*.yaml")
        return sorted((f.stem[1:] for f in files), key=_version_key)

    def _compile(self, name: str, version: str | None) -> _Compiled:
        available = self.versions(name)
        if not available:
            raise PromptError(f"no prompt named {name!r} in {self.root}")
        version = version or available[-1]
        if (name, version) in self._cache:
            return self._cache[(name, version)]
        path = self.root / name / f"v{version}.yaml"
        if not path.exists():
            raise PromptError(f"{name} has no version {version}; available: {available}")

        raw = path.read_text(encoding="utf-8")
        spec = PromptSpec.model_validate(yaml.safe_load(raw))  # safe_load: no arbitrary Python objects
        if (spec.name, spec.version) != (name, version):
            raise PromptError(f"{path}: file says {spec.name} v{spec.version}")

        declared = set(spec.trusted_vars) | set(spec.untrusted_vars)
        used_system = meta.find_undeclared_variables(self.env.parse(spec.system))
        used_user = meta.find_undeclared_variables(self.env.parse(spec.user))
        if leaked := used_system & set(spec.untrusted_vars):
            raise PromptError(f"{path}: untrusted vars {leaked} must not appear in the system prompt")
        if unknown := (used_system | used_user) - declared - RESERVED_VARS:
            raise PromptError(f"{path}: template uses undeclared vars {unknown}")

        compiled = _Compiled(
            spec=spec,
            system=self.env.from_string(spec.system),
            user=self.env.from_string(spec.user),
            sha256=hashlib.sha256(raw.encode()).hexdigest()[:12],
            uses_canary="canary" in used_system,
        )
        self._cache[(name, version)] = compiled
        return compiled

    def load(self, name: str, version: str | None = None) -> PromptSpec:
        return self._compile(name, version).spec

    def render(self, name: str, version: str | None = None, **variables: Any) -> RenderedPrompt:
        c = self._compile(name, version)
        expected = set(c.spec.trusted_vars) | set(c.spec.untrusted_vars)
        if missing := expected - variables.keys():
            raise PromptError(f"{name} v{c.spec.version}: missing vars {missing}")
        if extra := variables.keys() - expected:
            raise PromptError(f"{name} v{c.spec.version}: unexpected vars {extra}")

        # Untrusted values are sanitized here, centrally, so no template can forget to.
        # Values are inserted as plain strings: Jinja never evaluates {{ }} inside a value.
        ctx = {k: sanitize_untrusted(str(v)) if k in c.spec.untrusted_vars else v for k, v in variables.items()}
        canary = f"cnry-{secrets.token_hex(6)}" if c.uses_canary else None
        if canary:
            ctx["canary"] = canary

        return RenderedPrompt(
            name=name,
            version=c.spec.version,
            sha256=c.sha256,
            system=c.system.render(ctx).strip(),
            user=c.user.render(ctx).strip(),
            canary=canary,
        )
