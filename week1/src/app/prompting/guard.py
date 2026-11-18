"""Injection defenses that don't need a model: input sanitizing, input scanning, output checking.

None of these is a complete defense on its own. They are layers around the real one:
the system prompt treating <user_input> as data, plus giving the model as little power as possible.
"""

import re
import unicodedata
from dataclasses import dataclass

MAX_UNTRUSTED_CHARS = 8000

# Zero-width chars, bidi overrides, and Unicode "tag" chars (invisible ASCII smuggling).
_INVISIBLE = re.compile("[​-‏‪-‮⁠-⁤﻿\U000e0000-\U000e007f]")
_ESCAPES = str.maketrans({"&": "&amp;", "<": "&lt;", ">": "&gt;"})


def normalize(text: str) -> str:
    """NFKC folds look-alikes (fullwidth ＜ -> <) so later steps see the real characters."""
    return _INVISIBLE.sub("", unicodedata.normalize("NFKC", text))


def sanitize_untrusted(text: str, max_chars: int = MAX_UNTRUSTED_CHARS) -> str:
    """Make user text safe to place inside <user_input>...</user_input>.

    After escaping, the user cannot produce a real `</user_input>` tag, so they
    cannot "close" the data block and write text that looks like trusted prompt.
    """
    return normalize(text)[:max_chars].translate(_ESCAPES)


_PATTERNS = {
    "override": r"\b(ignore|disregard|forget)\b.{0,40}\b(previous|above|prior|all|earlier)\b.{0,40}\b(instructions?|rules|prompts?)\b",
    "prompt_leak": r"system prompt|your (instructions|rules)|initial prompt|repeat (the )?(text|words) above",
    "role_hijack": r"\b(you are now|act as|pretend to be|developer mode|jailbreak|DAN)\b",
    "fake_delimiter": r"</?\s*(user_input|system|instructions?)\s*>|^\s*(system|assistant)\s*:",
    "zh_override": r"(忽略|无视|忘记).{0,10}(以上|之前|上面|所有).{0,10}(指令|指示|规则|提示)",
    "zh_leak": r"系统提示|你的指令|初始提示",
}
_COMPILED = {name: re.compile(rx, re.IGNORECASE | re.MULTILINE) for name, rx in _PATTERNS.items()}


def scan_injection(text: str) -> list[str]:
    """Heuristic flags for logging/alerting. Easy to bypass: never rely on it to block."""
    clean = normalize(text)
    return [name for name, rx in _COMPILED.items() if rx.search(clean)]


@dataclass(frozen=True)
class OutputVerdict:
    ok: bool
    reasons: tuple[str, ...] = ()


def check_output(response: str, *, system: str, canary: str | None) -> OutputVerdict:
    """Catch the model leaking its system prompt.

    - canary: a random token planted in the system prompt; it should never appear in output.
    - echo: any long system-prompt line copied verbatim. Paraphrased leaks are NOT caught.
    """
    reasons = []
    if canary and canary in response:
        reasons.append("canary_leak")
    if any(len(line) >= 40 and line in response for line in (ln.strip() for ln in system.splitlines())):
        reasons.append("system_prompt_echo")
    return OutputVerdict(ok=not reasons, reasons=tuple(reasons))
