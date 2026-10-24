"""Build chat-formatted prompts for zero- and few-shot impression generation."""

from __future__ import annotations

from pathlib import Path

from .data import Report

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"
DEFAULT_SYSTEM_PROMPT = (
    "You are a board-certified radiologist. Your task is to generate a concise, "
    "clinically accurate impression section based solely on the following findings "
    "from a radiology report. The impression should be written in the style of a "
    "professional radiology report, using formal clinical language, and should not "
    "introduce information not present in the findings."
)


def load_system_prompt(path: str | Path | None = None) -> str:
    p = Path(path) if path else PROMPT_DIR / "system_prompt.txt"
    return p.read_text(encoding="utf-8").strip() if p.exists() else DEFAULT_SYSTEM_PROMPT


def user_turn(findings: str) -> str:
    return f"FINDINGS:\n{findings}\n\nIMPRESSION:"


def build_messages(
    query: Report,
    demos: list[Report] | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    supports_system_role: bool = True,
) -> list[dict]:
    """Demonstrations are encoded as alternating user/assistant turns.

    If a model's chat template has no system role, the instruction is
    prepended to the first user turn instead.
    """
    turns: list[dict] = []
    for d in demos or []:
        turns.append({"role": "user", "content": user_turn(d.findings)})
        turns.append({"role": "assistant", "content": d.impression})
    turns.append({"role": "user", "content": user_turn(query.findings)})

    if supports_system_role:
        return [{"role": "system", "content": system_prompt}] + turns
    turns[0] = {"role": "user", "content": f"{system_prompt}\n\n{turns[0]['content']}"}
    return turns


def postprocess(text: str) -> str:
    """Strip echoed headers and keep only the impression."""
    text = text.strip()
    for prefix in ("IMPRESSION:", "Impression:", "impression:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text.split("\nFINDINGS:")[0].strip()
