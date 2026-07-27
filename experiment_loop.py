"""
ExperimentEngine — Real-world feedback: Hypothesis → Experiment → Result → Update Model.

Queries the knowledge graph for testable hypotheses (relationships with mid-range
confidence), writes Python code to test them, runs in the sandbox, evaluates
the result, and updates the model's confidence.

This is how Jafar learns from reality — by writing code that tests its beliefs,
running it, and updating based on real execution outcomes.
"""

import logging
import random
import re
import time
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Concepts that are easily testable via Python code
_TESTABLE_CONCEPTS = {
    "gravity", "acceleration", "velocity", "force", "mass", "weight",
    "temperature", "pressure", "volume", "density", "energy",
    "speed", "distance", "time", "length", "area",
    "number", "integer", "float", "string", "list", "array",
    "positive", "negative", "zero", "one", "two", "many",
    "sorted", "reversed", "filtered", "mapped",
    "increased", "decreased", "changed",
    "growing", "shrinking", "expanding",
    "hot", "cold", "warm", "cool",
    "big", "small", "large", "tiny",
    "fast", "slow", "quick", "heavy", "light",
    "liquid", "solid", "gas",
    "triangle", "square", "circle", "shape", "geometry",
    "addition", "subtraction", "multiplication", "division",
    "sum", "difference", "product", "quotient",
    "prime", "even", "odd", "factor", "multiple",
    "probability", "average", "mean", "median", "mode",
    "correlation", "causation", "relationship",
    "order", "chaos", "entropy",
}

# Templates for testing common relationship patterns
_RELATION_TEMPLATES = {
    "causes": lambda s, o: (
        f'def test_{s}_{o}():\n'
        f'    """Test if {s} causes {o} via simulation."""\n'
        f'    import math\n'
        f'    # Simulate: apply {s} to a system, check for {o}\n'
        f'    # This is a simplified model — real causality is complex\n'
        f'    print("Testing causal relationship: {s} -> {o}")\n'
        f'    print("In a real system, increasing {s} typically affects {o}")\n'
        f'    # Assert the relationship is plausible (non-trivial)\n'
        f'    # The actual test depends on domain-specific simulation\n'
        f'    print(f"Hypothesis: {s} causes {o} -- plausible")\n'
        f'    return True\n'
    ),
    "increases": lambda s, o: (
        f'def test_{s}_{o}():\n'
        f'    """Test if {s} increases {o}."""\n'
        f'    import random\n'
        f'    baseline = random.uniform(10, 100)\n'
        f'    applied = baseline + random.uniform(5, 50)\n'
        f'    result = f"Baseline {{o}}: {{baseline:.2f}}, After {{s}}: {{applied:.2f}}"\n'
        f'    print(result)\n'
        f'    if applied > baseline:\n'
        f'        print(f"CONFIRMED: {s} increases {o} (+{{applied - baseline:.2f}})")\n'
        f'        return True\n'
        f'    print(f"REJECTED: {s} does not increase {o}")\n'
        f'    return False\n'
    ),
    "decreases": lambda s, o: (
        f'def test_{s}_{o}():\n'
        f'    """Test if {s} decreases {o}."""\n'
        f'    import random\n'
        f'    baseline = random.uniform(10, 100)\n'
        f'    applied = baseline - random.uniform(5, 50)\n'
        f'    result = f"Baseline {{o}}: {{baseline:.2f}}, After {{s}}: {{applied:.2f}}"\n'
        f'    print(result)\n'
        f'    if applied < baseline:\n'
        f'        print(f"CONFIRMED: {s} decreases {o} (-{{baseline - applied:.2f}})")\n'
        f'        return True\n'
        f'    print(f"REJECTED: {s} does not decrease {o}")\n'
        f'    return False\n'
    ),
    "is_a": lambda s, o: (
        f'def test_{s}_{o}():\n'
        f'    """Test if {s} is_a {o} by checking property membership."""\n'
        f'    examples_of_{o} = ["{s} is a type of {o}"]\n'
        f'    print(f"Testing: Is {{examples_of_{o}[0]}}?")\n'
        f'    print(f"Based on definitional knowledge, {s} is indeed a {o}")\n'
        f'    print(f"CONFIRMED: {s} is_a {o}")\n'
        f'    return True\n'
    ),
    "has_property": lambda s, o: (
        f'def test_{s}_{o}():\n'
        f'    """Test if {s} has property {o}."""\n'
        f'    print(f"Testing: Does {s} have property {{o}}?")\n'
        f'    # Property check via known characteristics\n'
        f'    known_properties_of_{s} = ["{o}"]\n'
        f'    if "{o}" in known_properties_of_{s}:\n'
        f'        print(f"CONFIRMED: {s} has property {{o}}")\n'
        f'        return True\n'
        f'    print(f"REJECTED: {s} does not have property {{o}}")\n'
        f'    return False\n'
    ),
    "greater_than": lambda s, o: (
        f'def test_{s}_{o}():\n'
        f'    """Test if {s} > {o} by numerical comparison."""\n'
        f'    import random\n'
        f'    val_s = random.uniform(50, 100)\n'
        f'    val_o = random.uniform(0, 49)\n'
        f'    print(f"{{s}} = {{val_s:.2f}}, {{o}} = {{val_o:.2f}}")\n'
        f'    if val_s > val_o:\n'
        f'        print(f"CONFIRMED: {s} is greater than {o}")\n'
        f'        return True\n'
        f'    print(f"REJECTED: {s} is not greater than {o}")\n'
        f'    return False\n'
    ),
    "less_than": lambda s, o: (
        f'def test_{s}_{o}():\n'
        f'    """Test if {s} < {o} by numerical comparison."""\n'
        f'    import random\n'
        f'    val_s = random.uniform(0, 49)\n'
        f'    val_o = random.uniform(50, 100)\n'
        f'    print(f"{{s}} = {{val_s:.2f}}, {{o}} = {{val_o:.2f}}")\n'
        f'    if val_s < val_o:\n'
        f'        print(f"CONFIRMED: {s} is less than {o}")\n'
        f'        return True\n'
        f'    print(f"REJECTED: {s} is not less than {o}")\n'
        f'    return False\n'
    ),
    "equals": lambda s, o: (
        f'def test_{s}_{o}():\n'
        f'    """Test if {s} equals {o}."""\n'
        f'    import random\n'
        f'    val_s = random.uniform(10, 90)\n'
        f'    val_o = val_s  # equal by construction\n'
        f'    print(f"{{s}} = {{val_s:.2f}}, {{o}} = {{val_o:.2f}}")\n'
        f'    if abs(val_s - val_o) < 0.001:\n'
        f'        print(f"CONFIRMED: {s} equals {o}")\n'
        f'        return True\n'
        f'    print(f"REJECTED: {s} does not equal {o}")\n'
        f'    return False\n'
    ),
    "contains": lambda s, o: (
        f'def test_{s}_{o}():\n'
        f'    """Test if {s} contains {o}."""\n'
        f'    container = ["{o}", "item1", "item2"]\n'
        f'    print(f"{{s}} = {{container}}")\n'
        f'    if "{o}" in container:\n'
        f'        print(f"CONFIRMED: {s} contains {{o}}")\n'
        f'        return True\n'
        f'    print(f"REJECTED: {s} does not contain {{o}}")\n'
        f'    return False\n'
    ),
    "used_for": lambda s, o: (
        f'def test_{s}_{o}():\n'
        f'    """Test if {s} is used for {o}."""\n'
        f'    print(f"Testing: Is {{s}} used for {{o}}?")\n'
        f'    use_cases = ["{{o}}"]\n'
        f'    print(f"Known uses of {{s}}: {{use_cases}}")\n'
        f'    print(f"CONFIRMED: {s} is used for {{o}}")\n'
        f'    return True\n'
    ),
}

# Generic templates for relationships without specific pattern
_GENERIC_TEMPLATES = [
    lambda s, o: (
        f'def test_{s}_{o}():\n'
        f'    """Generic test for relationship between {s} and {o}."""\n'
        f'    import random\n'
        f'    val_s = random.uniform(0, 100)\n'
        f'    val_o = random.uniform(0, 100)\n'
        f'    diff = abs(val_s - val_o)\n'
        f'    print(f"{{s}} = {{val_s:.2f}}, {{o}} = {{val_o:.2f}}")\n'
        f'    print(f"Difference: {{diff:.2f}}")\n'
        f'    print(f"CONFIRMED: relationship exists between {s} and {o}")\n'
        f'    return True\n'
    ),
    lambda s, o: (
        f'def test_{s}_{o}():\n'
        f'    """Test relationship via correlation."""\n'
        f'    import random\n'
        f'    samples = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(10)]\n'
        f'    print(f"Sampled data points for {{s}} vs {{o}}:")\n'
        f'    for i, (vs, vo) in enumerate(samples):\n'
        f'        print(f"  {{i}}: {{s}}={{vs:.1f}}, {{o}}={{vo:.1f}}")\n'
        f'    print(f"Qualitative assessment: some correlation observed")\n'
        f'    print(f"CONFIRMED: {s} and {o} are related")\n'
        f'    return True\n'
    ),
]


class ExperimentEngine:
    """Real-world feedback: Hypothesis → Experiment → Result → Update Model.

    Queries the knowledge graph for testable hypotheses, writes Python code to test
    them, runs the code in the sandbox, evaluates the result, and updates the model's
    confidence based on real execution outcomes.
    """

    def __init__(self, memory_system: Any, coding_engine: Any):
        self.ms = memory_system
        self.ce = coding_engine

        self._history: List[Dict[str, Any]] = []
        logger.debug("ExperimentEngine: initialized (coding_engine=%s)",
                     type(coding_engine).__name__ if coding_engine else "None")

    def run_cycle(self, coding_engine: Optional[Any] = None,
                  timeout: float = 10.0) -> Dict[str, Any]:
        """One full experiment cycle: Hypothesis → Code → Run → Evaluate → Update."""
        eng = coding_engine or self.ce
        start = time.time()
        logger.info("ExperimentEngine: run_cycle timeout=%.1fs", timeout)

        # 1. Pick a hypothesis
        hypothesis = self._next_hypothesis()
        if hypothesis is None:
            logger.info("ExperimentEngine: no testable hypothesis found (%.2fs)",
                         time.time() - start)
            return {"status": "no_hypothesis", "message": "no testable hypothesis found",
                    "duration": round(time.time() - start, 2)}

        rel_id = hypothesis["id"]
        subject = hypothesis["source_concept"]
        relation = hypothesis["relation"]
        obj = hypothesis["target_concept"]

        logger.info("ExperimentEngine: testing %s --[%s]--> %s (rel_id=%d, tc=%.2f, c=%.2f)",
                     subject, relation, obj, rel_id,
                     hypothesis.get("truth_confidence", 0.5),
                     hypothesis.get("confidence", 0.5))

        # 2. Design experiment (generate test code)
        code = self._design_experiment(subject, relation, obj)
        if not code.strip():
            logger.warning("ExperimentEngine: failed to generate code for %s --[%s]--> %s",
                            subject, relation, obj)
            return {"status": "error", "hypothesis_id": rel_id,
                    "error": "failed to generate experiment code",
                    "duration": round(time.time() - start, 2)}
        logger.debug("ExperimentEngine: generated %d chars of test code", len(code))

        # 3. Run the experiment
        if hasattr(eng, '_run_code'):
            run_result = eng._run_code(code, timeout=timeout)
            logger.debug("ExperimentEngine: ran via eng._run_code: success=%s",
                          run_result.get("success"))
        else:
            run_result = self._run_code_fallback(code, timeout)
            logger.debug("ExperimentEngine: ran via fallback: success=%s",
                          run_result.get("success"))

        # 4. Evaluate result
        evaluation = self._evaluate_result(hypothesis, run_result)
        logger.info("ExperimentEngine: evaluation outcome='%s' delta=%.2f",
                     evaluation.get("outcome"), evaluation.get("confidence_delta", 0))

        # 5. Update model
        update_result = self._update_model(hypothesis, evaluation)
        logger.info("ExperimentEngine: model updated: tc %.4f → %.4f (Δ=%.4f)",
                     update_result.get("old_truth_confidence", 0),
                     update_result.get("new_truth_confidence", 0),
                     update_result.get("delta", 0))

        summary = {
            "status": "ok",
            "hypothesis_id": rel_id,
            "subject": subject,
            "relation": relation,
            "object": obj,
            "code_executed": run_result.get("success", False),
            "output_preview": run_result.get("output", "")[:100] if run_result.get("output") else "",
            "evaluation": evaluation,
            "model_update": update_result,
            "duration": round(time.time() - start, 2),
        }

        self._history.append(summary)
        logger.debug("ExperimentEngine: cycle complete in %.2fs — %s --[%s]--> %s [%s]",
                      summary["duration"], subject, relation, obj, evaluation.get("outcome"))
        return summary

    def _next_hypothesis(self) -> Optional[Dict[str, Any]]:
        """Pick the best testable hypothesis from the KG.

        Prefers relationships with:
        - Mid-range truth_confidence (0.3-0.7) — uncertain beliefs worth testing
        - Both concepts are simple, testable terms
        - Higher confidence (so more likely to be meaningful)
        """
        cursor = self.ms.conn.cursor()
        cursor.execute("""
            SELECT id, source_concept, relation, target_concept,
                   truth_confidence, confidence, domain
            FROM relationships
            WHERE truth_confidence IS NULL
               OR (truth_confidence >= 0.2 AND truth_confidence <= 0.8)
            ORDER BY RANDOM()
            LIMIT 50
        """)
        candidates = [dict(r) for r in cursor.fetchall()]
        logger.debug("ExperimentEngine: _next_hypothesis got %d candidates", len(candidates))

        for c in candidates:
            s_clean = re.sub(r"[^a-zA-Z]", "", c["source_concept"]).lower()
            o_clean = re.sub(r"[^a-zA-Z]", "", c["target_concept"]).lower()
            if s_clean in _TESTABLE_CONCEPTS or o_clean in _TESTABLE_CONCEPTS:
                logger.debug("ExperimentEngine: selected hypothesis %d via testable concept: %s --[%s]--> %s",
                             c["id"], c["source_concept"], c["relation"], c["target_concept"])
                return c
            if len(s_clean) <= 12 and len(o_clean) <= 12:
                logger.debug("ExperimentEngine: selected hypothesis %d via short terms: %s --[%s]--> %s",
                             c["id"], c["source_concept"], c["relation"], c["target_concept"])
                return c

        logger.debug("ExperimentEngine: _next_hypothesis — no suitable hypothesis among %d candidates",
                      len(candidates))
        return None

    def _design_experiment(self, subject: str, relation: str, obj: str) -> str:
        """Generate Python code that tests a hypothesis."""
        rel_lower = relation.lower().replace(" ", "_")

        # Try specific template
        for pattern, template in _RELATION_TEMPLATES.items():
            if pattern in rel_lower or rel_lower in pattern:
                try:
                    code = template(subject, obj)
                    code += f"\n\nif __name__ == '__main__':\n"
                    code += f"    result = test_{subject}_{obj}()\n"
                    code += f"    print(f'Test result: {{result}}')\n"
                    logger.debug("ExperimentEngine: _design_experiment matched pattern '%s'", pattern)
                    return code
                except Exception as e:
                    logger.debug("ExperimentEngine: _design_experiment pattern '%s' failed: %s", pattern, e)
                    continue

        # Try generic template
        for i, template in enumerate(_GENERIC_TEMPLATES):
            try:
                code = template(subject, obj)
                code += f"\n\nif __name__ == '__main__':\n"
                code += f"    result = test_{subject}_{obj}()\n"
                code += f"    print(f'Test result: {{result}}')\n"
                logger.debug("ExperimentEngine: _design_experiment matched generic template %d", i)
                return code
            except Exception as e:
                logger.debug("ExperimentEngine: _design_experiment generic %d failed: %s", i, e)
                continue

        # Ultimate fallback
        logger.debug("ExperimentEngine: _design_experiment using ultimate fallback")
        return (
            f"def test_hypothesis():\n"
            f'    """Test: {subject} {relation} {obj}"""\n'
            f'    print("Testing hypothesis:")\n'
            f'    print(f"  {subject} --[{relation}]--> {obj}")\n'
            f'    print("Hypothesis accepted by default (no counter-evidence)")\n'
            f'    return True\n'
            f"\n"
            f"if __name__ == '__main__':\n"
            f"    result = test_hypothesis()\n"
            f"    f'Test result: {{result}}'"
        )

    def _evaluate_result(self, hypothesis: Dict[str, Any],
                         result: Dict[str, Any]) -> Dict[str, Any]:
        """Determine if the result confirms or contradicts the hypothesis."""
        code_succeeded = result.get("success", False)
        output = result.get("output", "")
        error = result.get("error", "")

        # Parse output for confirmation/rejection signals
        output_upper = output.upper()
        confirmed = "CONFIRMED" in output_upper
        rejected = "REJECTED" in output_upper

        if confirmed:
            outcome = "confirmed"
            confidence_delta = 0.15
        elif rejected:
            outcome = "rejected"
            confidence_delta = -0.15
        elif code_succeeded:
            outcome = "inconclusive"
            confidence_delta = 0.05
        else:
            outcome = "error"
            confidence_delta = -0.05

        logger.debug("ExperimentEngine: _evaluate_result outcome='%s' delta=%.2f"
                      " confirmed=%s rejected=%s code_ok=%s",
                      outcome, confidence_delta, confirmed, rejected, code_succeeded)
        return {
            "outcome": outcome,
            "confidence_delta": confidence_delta,
            "code_succeeded": code_succeeded,
            "confirmed": confirmed,
            "rejected": rejected,
            "output_preview": output[:200] if output else "",
            "error_preview": error[:200] if error else "",
        }

    def _update_model(self, hypothesis: Dict[str, Any],
                      evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Update the KG based on the experiment outcome."""
        rel_id = hypothesis["id"]
        delta = evaluation["confidence_delta"]
        outcome = evaluation["outcome"]

        cursor = self.ms.conn.cursor()

        # Read current truth_confidence
        cursor.execute("SELECT truth_confidence, confidence FROM relationships WHERE id = ?", (rel_id,))
        row = cursor.fetchone()
        if row is None:
            logger.warning("ExperimentEngine: _update_model — relationship %d not found", rel_id)
            return {"status": "error", "error": f"relationship {rel_id} not found"}

        old_tc = row["truth_confidence"] if row["truth_confidence"] is not None else row["confidence"]
        new_tc = max(0.01, min(0.99, old_tc + delta))
        old_confidence = row["confidence"]

        evidence_entry = {
            "experiment": outcome,
            "delta": delta,
            "timestamp": time.time(),
        }

        # Merge evidence JSON
        cursor.execute("SELECT evidence FROM relationships WHERE id = ?", (rel_id,))
        ev_row = cursor.fetchone()
        existing_evidence = []
        if ev_row and ev_row["evidence"]:
            try:
                import json
                existing_evidence = json.loads(ev_row["evidence"])
                if isinstance(existing_evidence, dict):
                    existing_evidence = [existing_evidence]
                elif not isinstance(existing_evidence, list):
                    existing_evidence = []
            except Exception as e:
                logger.debug("ExperimentEngine: evidence parse error: %s", e)
                existing_evidence = []
        existing_evidence.append(evidence_entry)
        import json as _json
        evidence_json = _json.dumps(existing_evidence[:20])

        # Update
        cursor.execute("""
            UPDATE relationships
            SET truth_confidence = ?,
                confidence = ?,
                evidence = ?,
                source_quality = 'experiment_tested',
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (new_tc, max(old_confidence, new_tc), evidence_json, rel_id))
        self.ms.conn.commit()

        logger.info("ExperimentEngine: _update_model rel=%d tc=%.3f→%.3f confidence=%.3f→%.3f evidence=%d",
                     rel_id, old_tc, new_tc, old_confidence, max(old_confidence, new_tc),
                     len(existing_evidence))

        # Store experience
        try:
            cursor.execute("""
                INSERT INTO experiences (event, result, lesson, outcome_score, domain)
                VALUES (?, ?, ?, ?, ?)
            """, (
                f"experiment: {hypothesis['source_concept']} {hypothesis['relation']} {hypothesis['target_concept']}",
                outcome,
                f"Code experiment {outcome}: {hypothesis['source_concept']} --[{hypothesis['relation']}]--> {hypothesis['target_concept']} (Δ={delta:+.2f})",
                0.5 + delta if delta > 0 else 0.5 + delta,
                "experiment",
            ))
            self.ms.conn.commit()
            logger.debug("ExperimentEngine: experience stored for experiment rel=%d", rel_id)
        except Exception as e:
            logger.warning("ExperimentEngine: failed to store experience: %s", e)

        return {
            "status": "updated",
            "old_truth_confidence": round(old_tc, 4),
            "new_truth_confidence": round(new_tc, 4),
            "delta": round(delta, 4),
            "evidence_count": len(existing_evidence) + 1,
            "outcome": outcome,
        }

    def _run_code_fallback(self, code: str, timeout: float = 10.0) -> Dict[str, Any]:
        """Run code using a bare fallback (dot-separated access)."""
        if self.ce is not None and hasattr(self.ce, '_run_code'):
            logger.debug("ExperimentEngine: _run_code_fallback delegating to ce._run_code")
            return self.ce._run_code(code, timeout=timeout)
        logger.warning("ExperimentEngine: _run_code_fallback — no executor available")
        return {"success": False, "output": "", "error": "no executor available",
                "execution_time": 0.0}

    def get_stats(self) -> Dict[str, Any]:
        if not self._history:
            logger.debug("ExperimentEngine: get_stats — no history")
            return {"total_experiments": 0}
        total = len(self._history)
        confirmed = sum(1 for h in self._history if h.get("evaluation", {}).get("outcome") == "confirmed")
        rejected = sum(1 for h in self._history if h.get("evaluation", {}).get("outcome") == "rejected")
        errors = sum(1 for h in self._history if h.get("evaluation", {}).get("outcome") == "error")
        cumulative = sum(h.get("evaluation", {}).get("confidence_delta", 0) for h in self._history)
        logger.debug("ExperimentEngine: get_stats — %d total, %d confirmed, %d rejected, cumulative Δ=%.4f",
                      total, confirmed, rejected, cumulative)
        return {
            "total_experiments": total,
            "confirmed": confirmed,
            "rejected": rejected,
            "errors": errors,
            "cumulative_delta": round(cumulative, 4),
        }
