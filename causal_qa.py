"""
Jafar — Causal QA: What-if questions, scenario comparison, natural-language simulation.
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple

from world_model_engine import WorldModelEngine

logger = logging.getLogger(__name__)

# Simple NL patterns for extracting what-if factors
# "raise price" -> price: +1.0
# "decrease demand" -> demand: -1.0
# "increase power by 50%" -> power: +0.5
# "reduce corruption and increase transparency" -> corruption: -1.0, transparency: +1.0
# "raise price by 30%" -> price: +0.3
# "cut funding by half" -> funding: -0.5

INCREASE_SIGNALS = [
    "increase", "increases", "increased", "raising", "raises", "raise",
    "boost", "boosts", "boosted", "grow", "grows", "grew",
    "improve", "improves", "improved", "enhance", "enhances",
    "strengthen", "strengthens", "expand", "expands", "add", "adds",
    "more", "higher", "greater", "up",
]

DECREASE_SIGNALS = [
    "decrease", "decreases", "decreased", "reduce", "reduces", "reduced",
    "lower", "lowers", "lowered", "cut", "cuts", "cutting",
    "weaken", "weakens", "diminish", "diminishes", "remove", "removes",
    "less", "fewer", "lower", "down", "drop", "drops",
    "eliminate", "eliminates", "suppress", "suppresses",
]

PERCENT_PATTERN = re.compile(r"(\d+)%\s*(?:\b|\Z)", re.IGNORECASE)
AMOUNT_PATTERN = re.compile(r"by\s+(half|double|triple|a\s+lot|a\s+little|slightly|significantly)", re.IGNORECASE)

BOOST_MAP = {
    "half": 0.5, "double": 2.0, "triple": 3.0,
    "a lot": 2.0, "a little": 0.25,
    "significantly": 2.0, "slightly": 0.25,
}


def _magnitude_from_text(text: str, default: float = 1.0) -> float:
    """Extract a magnitude multiplier from text like 'by 30%' or 'by half'."""
    m = PERCENT_PATTERN.search(text)
    if m:
        return float(m.group(1)) / 100.0
    m = AMOUNT_PATTERN.search(text)
    if m:
        word = m.group(1).lower().strip()
        return BOOST_MAP.get(word, default)
    return default


def _is_increase_word(word: str) -> bool:
    return word.lower() in INCREASE_SIGNALS


def _is_decrease_word(word: str) -> bool:
    return word.lower() in DECREASE_SIGNALS


def parse_what_if(question: str) -> List[Dict[str, Any]]:
    """Parse a natural-language what-if question into structured factors.

    Examples:
      "what if I raise price" -> [{"concept": "price", "delta": 1.0}]
      "lower taxes and increase spending" -> [{"concept": "taxes", "delta": -1.0}, {"concept": "spending", "delta": 1.0}]
      "increase power by 30%" -> [{"concept": "power", "delta": 0.3}]
    """
    # Normalize
    text = question.lower().strip()
    # Remove common prefixes
    for prefix in ["what if i ", "what if we ", "what if you ", "what happens if i ",
                   "what happens if we ", "what happens when ", "what if "]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    for suffix in [" happen", " happens", " would happen", "?"]:
        if text.endswith(suffix):
            text = text[:-len(suffix)].strip()

    # Split on "and", "&", ","
    parts = re.split(r"\s+(?:and|&|,)\s+", text)
    factors = []

    for part in parts:
        part = part.strip()
        if not part:
            continue
        direction = None
        concept_words = []
        magnitude = _magnitude_from_text(part)
        tokens = part.split()

        # Check first few tokens for direction signals
        remaining = list(tokens)
        found_op = False
        for i, token in enumerate(remaining):
            if _is_increase_word(token):
                direction = 1.0
                found_op = True
            elif _is_decrease_word(token):
                direction = -1.0
                found_op = True
            else:
                continue
            # Everything after the verb is the concept
            concept = " ".join(remaining[i + 1:])
            # Clean trailing noise
            concept = re.sub(r"\s+by\s+(half|double|triple|a\s+lot|a\s+little|slightly|significantly|\d+%)$", "", concept).strip()
            if concept:
                factors.append({
                    "concept": concept,
                    "delta": round(direction * magnitude, 2),
                    "source_phrase": part,
                })
            break

        if not found_op:
            # Maybe it's just a concept with no explicit direction — treat as "increase" by default
            concept = part.strip()
            concept = re.sub(r"\s+by\s+(half|double|triple|a\s+lot|a\s+little|slightly|significantly|\d+%)$", "", concept).strip()
            factors.append({
                "concept": concept,
                "delta": 1.0 * magnitude,
                "source_phrase": part,
            })

    return factors


class CausalQA:
    """Natural-language interface to the causal world model."""

    def __init__(self, wme: WorldModelEngine):
        self.wme = wme

    # ------------------------------------------------------------------
    #  What-if question
    # ------------------------------------------------------------------

    def what_if(self, question: str, max_steps: int = 5) -> Dict[str, Any]:
        """Answer a what-if question in natural language.

        Returns both structured results and a plain-English summary.
        """
        factors = parse_what_if(question)
        if not factors:
            return {
                "question": question,
                "factors": [],
                "error": "Could not understand the question. Try 'what if I increase price?'",
                "summary": "I couldn't parse that question.",
            }

        factors_dict = {f["concept"]: f["delta"] for f in factors}
        sim = self.wme.predict(factors_dict, max_steps=max_steps)
        summary = self._summarize_simulation(question, factors, sim)
        return {
            "question": question,
            "factors": factors,
            "simulation": sim,
            "summary": summary,
        }

    # ------------------------------------------------------------------
    #  Reverse what-if — "what would make X go up/down?"
    # ------------------------------------------------------------------

    def reverse_what_if(self, target_concept: str,
                        desired_direction: str = "increase") -> Dict[str, Any]:
        """Find levers that affect a target concept.

        desired_direction: 'increase' or 'decrease'
        """
        causes = self.wme.get_causes(target_concept)
        direct = []
        for c in causes:
            if desired_direction == "increase" and c["direction"] == "increases":
                direct.append(c)
            elif desired_direction == "decrease" and c["direction"] == "decreases":
                direct.append(c)
            elif desired_direction == "increase" and c["direction"] == "decreases":
                # Reverse: decreasing a decreaser = increase
                direct.append({**c, "_reverse": True})
            elif desired_direction == "decrease" and c["direction"] == "increases":
                direct.append({**c, "_reverse": True})

        summary_parts = []
        if not direct:
            summary_parts.append(f"I don't know of any way to {desired_direction} '{target_concept}' yet.")
        else:
            if desired_direction == "increase":
                summary_parts.append(
                    f"To increase '{target_concept}', you could try:\n"
                )
            else:
                summary_parts.append(
                    f"To decrease '{target_concept}', you could try:\n"
                )
            for c in direct[:5]:
                rev = c.get("_reverse", False)
                if rev and desired_direction == "increase":
                    action = f"Decrease '{c['cause']}'"
                elif rev and desired_direction == "decrease":
                    action = f"Increase '{c['cause']}'"
                else:
                    action = f"Increase '{c['cause']}'" if c["direction"] == "increases" else f"Decrease '{c['cause']}'"
                summary_parts.append(
                    f"  - {action} (strength: {c['strength']}, evidence: {c['evidence_count']})"
                )

        return {
            "target": target_concept,
            "desired_direction": desired_direction,
            "levers": direct,
            "summary": "\n".join(summary_parts),
        }

    # ------------------------------------------------------------------
    #  Scenario comparison
    # ------------------------------------------------------------------

    def compare_scenarios(self, scenarios: Dict[str, Dict[str, float]],
                           max_steps: int = 5) -> Dict[str, Any]:
        """Run and compare multiple what-if scenarios.

        scenarios: {"scenario_name": {"concept": delta, ...}}
        """
        results = {}
        for name, factors in scenarios.items():
            sim = self.wme.predict(factors, max_steps=max_steps)
            results[name] = sim

        summary = self._summarize_comparison(scenarios, results)
        return {
            "scenarios": scenarios,
            "results": results,
            "summary": summary,
        }

    # ------------------------------------------------------------------
    #  Summarizers
    # ------------------------------------------------------------------

    def _summarize_simulation(self, question: str,
                               factors: List[Dict[str, Any]],
                               sim: Dict[str, Any]) -> str:
        """Turn simulation JSON into plain English."""
        factor_lines = []
        for f in factors:
            dir_str = "increase" if f["delta"] > 0 else "decrease"
            factor_lines.append(f"{dir_str} '{f['concept']}' by {abs(f['delta']):.2f}")
        factor_text = " and ".join(factor_lines)

        if "error" in sim:
            return f"I couldn't simulate '{question}': {sim['error']}"

        final = sim.get("final_state", {})
        if not final:
            return f"If we {factor_text}, I don't see any significant downstream effects in my current model."

        lines = [f"If we {factor_text}, here's what I expect:\n"]
        for concept, val in list(final.items())[:8]:
            if val > 0.01:
                lines.append(f"  - '{concept}' would increase (by {abs(val):.4f})")
            elif val < -0.01:
                lines.append(f"  - '{concept}' would decrease (by {abs(val):.4f})")
            else:
                lines.append(f"  - '{concept}' would be relatively unchanged ({val:+.4f})")

        trade_offs = sim.get("trade_offs_detected", [])
        if trade_offs:
            lines.append(f"\n  Trade-off: {trade_offs[0]['increases']} increases but {trade_offs[0]['decreases']} decreases.")

        thresholds = sim.get("thresholds_triggered", [])
        if thresholds:
            lines.append(f"\n  ⚠ Threshold effect: {thresholds[0]['cause']} -> {thresholds[0]['effect']} ({thresholds[0]['reason']})")

        return "\n".join(lines)

    def _summarize_comparison(self, scenarios: Dict[str, Dict[str, float]],
                               results: Dict[str, Dict[str, Any]]) -> str:
        """Compare simulation results across scenarios side by side."""
        if not results:
            return "No scenarios to compare."

        all_concepts = set()
        for name, sim in results.items():
            all_concepts.update(sim.get("final_state", {}).keys())
        all_concepts = sorted(all_concepts)

        lines = ["Scenario comparison:\n"]

        # Header
        header = f"{'Concept':<25}"
        for name in scenarios:
            header += f" | {name:<20}"
        lines.append(header)
        lines.append("-" * len(header))

        for concept in all_concepts:
            row = f"{concept:<25}"
            for name in scenarios:
                val = results[name].get("final_state", {}).get(concept, 0)
                row += f" | {val:>+8.4f}       "
            lines.append(row)

        lines.append("\nScenarios:")
        for name, factors in scenarios.items():
            parts = [f"{'+' if v > 0 else ''}{v} {k}" for k, v in factors.items()]
            lines.append(f"  {name}: {', '.join(parts)}")

        return "\n".join(lines)
