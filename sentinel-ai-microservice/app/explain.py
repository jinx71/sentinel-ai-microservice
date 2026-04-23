"""Plain-language explanation of a flagged reading.

Two tiers, by design:
  1. rule_based_explanation — deterministic, dependency-free, always available.
  2. llm_explanation — richer maintenance guidance via Claude, used ONLY when the
     operator explicitly hits /explain AND a key is configured.

Why two tiers: the rule-based path keeps the core service hermetic and free to test
in CI (no key, no network), while the LLM path adds value in production without ever
becoming a hard dependency or a hidden per-prediction cost.
"""

from __future__ import annotations

import logging

from .config import Settings
from .model import _NOMINAL, FEATURE_ORDER

logger = logging.getLogger(__name__)

_UNITS = {
    "temperature": "C",
    "pressure": "bar",
    "vibration": "mm/s",
    "humidity": "%RH",
    "flow_rate": "L/min",
}


def _deviations(reading: dict) -> list[tuple[str, float]]:
    """Return (feature, z-score) pairs sorted by absolute deviation, worst first."""
    scored: list[tuple[str, float]] = []
    for feature in FEATURE_ORDER:
        mean, std = _NOMINAL[feature]
        z = (reading[feature] - mean) / std if std else 0.0
        scored.append((feature, z))
    scored.sort(key=lambda item: abs(item[1]), reverse=True)
    return scored


def rule_based_explanation(reading: dict, result: dict) -> str:
    """Build a deterministic narrative from the largest sensor deviations."""
    if not result["is_anomaly"]:
        return (
            f"{reading['equipment_id']} is operating within the nominal envelope "
            f"(severity: {result['severity']}). No action required."
        )

    worst = _deviations(reading)[:2]
    parts: list[str] = []
    for feature, z in worst:
        if abs(z) < 1.0:
            continue
        direction = "above" if z > 0 else "below"
        parts.append(
            f"{feature.replace('_', ' ')} is {abs(z):.1f}σ {direction} nominal "
            f"({reading[feature]:.1f} {_UNITS[feature]})"
        )
    drivers = "; ".join(parts) if parts else "a combination of minor deviations"
    return (
        f"{reading['equipment_id']} flagged as {result['severity']}-severity anomaly. "
        f"Primary drivers: {drivers}. Recommend inspecting the affected subsystem and "
        f"reviewing the last maintenance log before continuing the batch."
    )


def llm_explanation(reading: dict, result: dict, settings: Settings) -> tuple[str, str]:
    """Return (text, source). Falls back to rule-based on any failure.

    The anthropic SDK is imported lazily so the core image does not depend on it
    being importable when LLM explanations are disabled.
    """
    if not settings.llm_available:
        return rule_based_explanation(reading, result), "rule_based"

    try:
        import anthropic  # noqa: PLC0415 - intentional lazy import

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        prompt = (
            "You are a GMP pharmaceutical maintenance engineer. In 2-3 sentences, give "
            "a concrete, actionable explanation and next step for this flagged reading. "
            f"Equipment: {reading['equipment_id']}. Severity: {result['severity']}. "
            f"Sensor values: {reading}. Be specific and avoid speculation."
        )
        message = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        ).strip()
        if not text:
            raise ValueError("empty LLM response")
        return text, "llm"
    except Exception:  # noqa: BLE001 - never let enrichment break the endpoint
        logger.exception("LLM explanation failed, using rule-based fallback")
        return rule_based_explanation(reading, result), "rule_based"
