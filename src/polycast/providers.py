"""LLM providers.

The pipeline only ever sees `JsonProvider`, so swapping a hosted model for a
local one is a config change, not a refactor. `MockProvider` is deterministic
and offline: tests, CI and the eval harness all run without a key, which keeps
the feedback loop fast when tuning aggregation or calibration.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Protocol

import httpx

POSITIVE = ("growth", "approve", "rise", "pass", "win", "increase", "expand", "record")
NEGATIVE = ("decline", "reject", "fall", "fail", "loss", "ban", "drop", "recession")


def count_words(text: str, words: tuple[str, ...]) -> int:
    """Whole-word count. Substring matching misfires on phrases like
    'cut the policy rate', which are neutral, not negative."""
    return sum(len(re.findall(rf"\b{re.escape(w)}\b", text)) for w in words)


def extract_question(text: str) -> str:
    """Pull the question block out of a prompt, without the retrieved evidence.

    The stub has to reason about the question, not about whatever documents the
    retriever happened to surface.
    """
    match = re.search(r"^question: (.*)$", text, re.M)
    if not match:
        return ""
    lines = [match.group(1)]
    for line in text[match.end() :].splitlines():
        if not line.strip():
            break
        if line.startswith(("resolution_criteria:", "resolve_by:", "reference_class:")):
            break
        lines.append(line)
    return " ".join(lines)


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model reply.

    Models wrap JSON in fences or prepend a sentence even when told not to, so
    we try the strict parse first and fall back to a substring scan.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        return json.loads(fence.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in model reply: {text[:200]}")
    return json.loads(text[start : end + 1])


class JsonProvider(Protocol):
    async def complete_json(self, system: str, user: str) -> dict[str, Any]: ...


def _stable_unit_interval(*parts: str) -> float:
    """Deterministic pseudo-random number in [0, 1), stable across processes."""
    digest = hashlib.sha256("::".join(parts).encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


class MockProvider:
    """Offline stand-in with the same calling convention as a real model.

    Probabilities are derived from lexical sentiment plus a per-persona bias and
    a stable hash of the question, so a panel of analysts genuinely disagrees
    and the aggregator has something to aggregate.
    """

    def __init__(self, seed: str = "polycast") -> None:
        self.seed = seed
        self.calls: list[dict[str, str]] = []

    def _base_rate(self, text: str) -> float:
        subject = extract_question(text) or text
        lowered = subject.lower()
        pos = count_words(lowered, POSITIVE)
        neg = count_words(lowered, NEGATIVE)
        sentiment = (pos - neg) / max(pos + neg, 1)
        jitter = (_stable_unit_interval(self.seed, text) - 0.5) * 0.2
        return min(0.9, max(0.1, 0.5 + 0.3 * sentiment + jitter))

    async def complete_json(self, system: str, user: str) -> dict[str, Any]:
        self.calls.append({"system": system, "user": user})
        if "red_team" in system:
            match = re.search(r"panel_mean=([0-9.]+)", user)
            mean = float(match.group(1)) if match else 0.5
            if mean > 0.75:
                shift, note = -0.05, "panel is anchored on mutually confirming documents"
            elif mean < 0.25:
                shift, note = 0.05, "panel underweights the pull of the reference class"
            else:
                shift, note = 0.0, "no correlated bias detected in the panel spread"
            return {
                "overlooked": [note],
                "probability_shift": shift,
                "notes": f"red team pass over panel_mean={mean:.3f}",
            }
        base = self._base_rate(user)
        match = re.search(r"analyst_persona[=:]\s*(\w+)", user)
        persona = match.group(1) if match else next(
            (p for p in ("base_rate", "bull", "bear") if p in system), "panel"
        )
        shift = {"base_rate": 0.0, "bull": 0.13, "bear": -0.13}.get(persona, 0.0)
        prob = min(0.95, max(0.05, base + shift))
        n_evidence = user.count("doc_id=")
        return {
            "persona": persona,
            "probability": round(prob, 4),
            "reasoning": (
                f"{persona} read of {n_evidence} retrieved documents; "
                f"lexical prior {base:.2f} adjusted by stance {shift:+.2f}."
            ),
            "evidence_ids": re.findall(r"doc_id=(\S+)", user)[:4],
            "confidence": round(min(0.9, 0.35 + 0.1 * n_evidence), 3),
        }


class OpenAICompatibleProvider:
    """Any endpoint speaking /chat/completions: hosted or local (vLLM, Ollama)."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.2,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout
        )

    async def complete_json(self, system: str, user: str) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        resp = await self._client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return extract_json(content)

    async def aclose(self) -> None:
        await self._client.aclose()
