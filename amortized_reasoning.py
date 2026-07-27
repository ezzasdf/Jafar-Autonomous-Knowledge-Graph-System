"""
Amortized Reasoning Engine — compiles successful multi-step LLM inference
chains into deterministic rule macros.  When a structurally similar context
is encountered later, the macro is executed directly without waking the LLM.

Macro structure:
    - relation_signature : tuple of (relation, direction) for each input triple
    - subject_roles       : generic slot names for subjects (e.g. "X", "Y")
    - object_roles        : generic slot names for objects
    - output_template     : {subject_slot, relation, object_slot}
    - strategy            : chain | analogy | causal | explain
    - confidence          : how often this macro has succeeded
    - hit_count           : number of times used
    - last_used           : timestamp
"""

import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)
debug_logger = logging.getLogger(f"{__name__}.debug")

AMORTIZED_CONFIDENCE_FLOOR = 0.55
MACRO_SIMILARITY_THRESHOLD = 0.80  # cosine similarity threshold for concept matching
MACRO_MAX_CANDIDATES = 10          # max macros to consider per lookup
MACRO_MAX_HITS_BEFORE_PROMOTE = 5  # hits before macro is considered "trusted"


class AmortizedReasoningEngine:
    """Stores, matches, and executes compiled reasoning macros.

    Each macro represents a previously successful multi-step inference
    that can be replayed deterministically when a structurally similar
    input context is detected.
    """

    def __init__(self, memory_system: Any,
                 concept_enhancer: Optional[Any] = None):
        self.memory = memory_system
        self.enhancer = concept_enhancer
        self._init_store()

    # ------------------------------------------------------------------
    #  Macro persistence
    # ------------------------------------------------------------------

    def _init_store(self) -> None:
        cursor = self.memory.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reasoning_macros (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                relation_signature TEXT NOT NULL,
                subject_roles TEXT NOT NULL,
                object_roles TEXT NOT NULL,
                output_template TEXT NOT NULL,
                strategy TEXT NOT NULL DEFAULT 'chain',
                confidence REAL NOT NULL DEFAULT 0.55,
                hit_count INTEGER NOT NULL DEFAULT 0,
                success_count INTEGER NOT NULL DEFAULT 0,
                fail_count INTEGER NOT NULL DEFAULT 0,
                last_used TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_macros_strategy
            ON reasoning_macros(strategy)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_macros_confidence
            ON reasoning_macros(confidence DESC)
        """)
        self.memory.conn.commit()
        debug_logger.debug("reasoning_macros table ready")

    def _rel_signature(self, triples: List[Dict[str, Any]]) -> Tuple[str, ...]:
        """Build a normalised relation signature from input triples.

        Signature = sorted tuple of (relation,) for each triple.
        Directionality is ignored because macros abstract over roles.
        """
        rels = sorted(t.get("relation", "").lower().strip() for t in triples if t.get("relation"))
        return tuple(rels)

    # ------------------------------------------------------------------
    #  Compilation
    # ------------------------------------------------------------------

    def compile_trace(self, input_triples: List[Dict[str, Any]],
                      output_triples: List[Dict[str, Any]],
                      strategy: str = "chain") -> Optional[int]:
        """Compile a successful reasoning trace into a stored macro.

        Args:
            input_triples:  The triples that were passed as context to the LLM
            output_triples: The triples returned by the LLM (the inferences)
            strategy:       Which strategy was used (chain/analogy/causal/explain)

        Returns:
            macro_id if created/updated, None if no output to learn from.
        """
        if not output_triples:
            return None

        sig = self._rel_signature(input_triples)
        sig_json = json.dumps(sig)
        strategy = strategy.lower()

        # Collect all unique concept names from input
        input_concepts = set()
        for t in input_triples:
            s = t.get("source_concept", t.get("subject", "")).lower().strip()
            o = t.get("target_concept", t.get("object", "")).lower().strip()
            if s:
                input_concepts.add(s)
            if o:
                input_concepts.add(o)

        input_list = list(input_concepts)

        # Build generic roles: X0, X1, ... for subjects; Y0, Y1, ... for objects
        subject_roles: Dict[str, str] = {}
        object_roles: Dict[str, str] = {}
        role_counter = 0

        for ot in output_triples:
            s = ot.get("subject", "").lower().strip()
            o = ot.get("object", "").lower().strip()
            rel = ot.get("relation", "").lower().strip()
            if s and s not in subject_roles:
                subject_roles[s] = f"X{role_counter}"
                role_counter += 1
            if o and o not in object_roles:
                object_roles[o] = f"X{role_counter}"
                role_counter += 1

        # Build output templates using role slots
        output_templates = []
        for ot in output_triples:
            s = ot.get("subject", "").lower().strip()
            o = ot.get("object", "").lower().strip()
            rel = ot.get("relation", "").lower().strip()
            s_slot = subject_roles.get(s, s)
            o_slot = object_roles.get(o, s_slot)  # fallback: same slot
            output_templates.append({
                "subject_slot": s_slot,
                "relation": rel,
                "object_slot": o_slot,
            })

        # Check if a macro with the same signature + strategy already exists
        cursor = self.memory.conn.cursor()
        cursor.execute("""
            SELECT id, confidence, hit_count, success_count
            FROM reasoning_macros
            WHERE relation_signature = ? AND strategy = ?
        """, (sig_json, strategy))
        existing = cursor.fetchone()

        if existing:
            # Boost confidence slightly (the pattern was observed again)
            new_conf = min(1.0, existing["confidence"] + 0.05)
            cursor.execute("""
                UPDATE reasoning_macros
                SET subject_roles = ?, object_roles = ?,
                    output_template = ?, confidence = ?,
                    success_count = success_count + 1,
                    last_used = ?
                WHERE id = ?
            """, (
                json.dumps(subject_roles), json.dumps(object_roles),
                json.dumps(output_templates), new_conf,
                datetime.now(timezone.utc).isoformat(), existing["id"],
            ))
            self.memory.conn.commit()
            debug_logger.debug("Macro %d updated (confidence %.2f -> %.2f)",
                               existing["id"], existing["confidence"], new_conf)
            return existing["id"]

        cursor.execute("""
            INSERT INTO reasoning_macros
                (relation_signature, subject_roles, object_roles,
                 output_template, strategy, confidence, hit_count,
                 success_count, last_used)
            VALUES (?, ?, ?, ?, ?, ?, 0, 1, ?)
        """, (
            sig_json,
            json.dumps(subject_roles),
            json.dumps(object_roles),
            json.dumps(output_templates),
            strategy,
            AMORTIZED_CONFIDENCE_FLOOR + 0.15,
            datetime.now(timezone.utc).isoformat(),
        ))
        self.memory.conn.commit()
        macro_id = cursor.lastrowid
        logger.info("Compiled new macro %d (strategy=%s, sig_len=%d, outputs=%d)",
                     macro_id, strategy, len(sig), len(output_templates))
        return macro_id

    # ------------------------------------------------------------------
    #  Lookup
    # ------------------------------------------------------------------

    def find_macro(self, input_triples: List[Dict[str, Any]],
                   strategy: str = "chain",
                   min_confidence: float = AMORTIZED_CONFIDENCE_FLOOR,
                   similarity_threshold: float = MACRO_SIMILARITY_THRESHOLD,
                   ) -> Optional[Dict[str, Any]]:
        """Find the best matching macro for the given input triples.

        Matching strategy:
          1. Filter macros by strategy
          2. Compare relation signatures (exact match after sorting)
          3. If signature matches, verify concept similarity via embedding
          4. Return the highest-confidence matching macro

        Returns macro dict or None.
        """
        sig = self._rel_signature(input_triples)
        sig_json = json.dumps(sig)

        cursor = self.memory.conn.cursor()
        cursor.execute("""
            SELECT * FROM reasoning_macros
            WHERE strategy = ? AND relation_signature = ?
              AND confidence >= ?
            ORDER BY confidence DESC, success_count DESC
            LIMIT ?
        """, (strategy, sig_json, min_confidence, MACRO_MAX_CANDIDATES))
        rows = [dict(r) for r in cursor.fetchall()]

        if not rows:
            debug_logger.debug("No macro matches for sig=%s strategy=%s",
                               sig, strategy)
            return None

        # If we have embedding enhancer, verify concept-level similarity
        if self.enhancer is not None and len(rows) > 0:
            input_concepts = set()
            for t in input_triples:
                s = t.get("source_concept", t.get("subject", "")).lower().strip()
                o = t.get("target_concept", t.get("object", "")).lower().strip()
                if s:
                    input_concepts.add(s)
                if o:
                    input_concepts.add(o)
            input_list = list(input_concepts)

            for row in rows:
                try:
                    subj_roles = json.loads(row["subject_roles"])
                    obj_roles = json.loads(row["object_roles"])
                    # Check that each stored concept maps to a similar input concept
                    all_similar = True
                    for stored_name, slot in subj_roles.items():
                        # Find the best match in input concepts
                        sims = []
                        for ic in input_list:
                            try:
                                s = self.enhancer.compute_similarity(stored_name, ic)
                                sims.append(s)
                            except Exception:
                                sims.append(0.0)
                        best = max(sims) if sims else 0.0
                        if best < similarity_threshold:
                            all_similar = False
                            break
                    if all_similar:
                        for stored_name, slot in obj_roles.items():
                            if stored_name in subj_roles:
                                continue  # already checked
                            sims = []
                            for ic in input_list:
                                try:
                                    s = self.enhancer.compute_similarity(stored_name, ic)
                                    sims.append(s)
                                except Exception:
                                    sims.append(0.0)
                            best = max(sims) if sims else 0.0
                            if best < similarity_threshold:
                                all_similar = False
                                break
                    if all_similar:
                        debug_logger.debug("Macro %d matched via concept similarity", row["id"])
                        return row
                except Exception:
                    continue

            # Fallback: return the highest-confidence macro even without concept match
            debug_logger.debug("No concept-similar macro found; returning best-confidence")
            return rows[0]

        # Without enhancer, return the best-confidence match
        if rows:
            debug_logger.debug("Macro %d matched (no enhancer)", rows[0]["id"])
            return rows[0]
        return None

    # ------------------------------------------------------------------
    #  Execution
    # ------------------------------------------------------------------

    def execute_macro(self, macro: Dict[str, Any],
                      input_triples: List[Dict[str, Any]],
                      ) -> List[Dict[str, Any]]:
        """Instantiate a macro with the concepts from the current context.

        Maps old role slots to the actual concept names present in input_triples,
        then builds output triples using the macro's output template.

        Returns list of inferred triples.
        """
        subj_roles = json.loads(macro["subject_roles"])
        obj_roles = json.loads(macro["object_roles"])
        templates = json.loads(macro["output_template"])

        # Build a reverse mapping: slot_name -> most similar input concept
        input_concepts = set()
        for t in input_triples:
            s = t.get("source_concept", t.get("subject", "")).lower().strip()
            o = t.get("target_concept", t.get("object", "")).lower().strip()
            if s:
                input_concepts.add(s)
            if o:
                input_concepts.add(o)
        input_list = list(input_concepts)

        # Map old concept names to slots, then slots to new concept names
        slot_to_new: Dict[str, str] = {}
        for stored_name, slot in subj_roles.items():
            best_match = self._best_concept_match(stored_name, input_list)
            if best_match:
                slot_to_new[slot] = best_match
            else:
                slot_to_new[slot] = stored_name  # fallback: keep old name

        for stored_name, slot in obj_roles.items():
            if slot in slot_to_new:
                continue
            best_match = self._best_concept_match(stored_name, input_list)
            if best_match:
                slot_to_new[slot] = best_match
            else:
                slot_to_new[slot] = stored_name

        # Build output triples
        results = []
        for tmpl in templates:
            s_slot = tmpl["subject_slot"]
            o_slot = tmpl["object_slot"]
            subject = slot_to_new.get(s_slot, s_slot)
            obj = slot_to_new.get(o_slot, o_slot)
            results.append({
                "subject": subject,
                "relation": tmpl["relation"],
                "object": obj,
                "confidence": macro["confidence"],
                "source": f"amortized_macro:{macro['id']}",
                "source_type": "inferred",
            })

        # Update usage stats
        cursor = self.memory.conn.cursor()
        cursor.execute("""
            UPDATE reasoning_macros
            SET hit_count = hit_count + 1,
                last_used = ?
            WHERE id = ?
        """, (datetime.now(timezone.utc).isoformat(), macro["id"]))
        self.memory.conn.commit()

        logger.info("Executed macro %d: %d inferences (confidence=%.2f)",
                     macro["id"], len(results), macro["confidence"])
        return results

    def _best_concept_match(self, stored_name: str,
                            candidates: List[str]) -> Optional[str]:
        """Find the candidate concept most similar to stored_name.
        
        Returns None when no good match is found, allowing callers to
        fall back to the stored_name for output-only concepts that
        don't appear in the input context.
        """
        if not candidates:
            return None
        if stored_name in candidates:
            return stored_name
        if self.enhancer is None:
            return None  # no enhancer → no way to verify similarity
        best_sim = -1.0
        best_match = None
        for c in candidates:
            try:
                sim = self.enhancer.compute_similarity(stored_name, c)
                if sim > best_sim:
                    best_sim = sim
                    best_match = c
            except Exception:
                continue
        if best_sim >= MACRO_SIMILARITY_THRESHOLD:
            return best_match
        return None

    # ------------------------------------------------------------------
    #  Amortized reasoning cycle (drop-in for TransformerReasoningEngine)
    # ------------------------------------------------------------------

    def try_amortized_inference(self, triples: List[Dict[str, Any]],
                                 strategy: str = "chain",
                                 store: bool = True,
                                 ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
        """Try to answer an inference request without calling the LLM.

        Returns (inferences, macro) if a matching macro was found and executed,
        or (None, None) if no macro matched.
        """
        macro = self.find_macro(triples, strategy=strategy)
        if macro is None:
            return None, None

        inferences = self.execute_macro(macro, triples)
        if not inferences:
            return None, None

        if store:
            with self.memory.batch():
                for inf in inferences:
                    existing = self.memory._reinforce_relationship(
                        inf["subject"], inf["relation"], inf["object"]
                    )
                    if existing is None:
                        self.memory.add_fact_triple(
                            inf["subject"], inf["relation"], inf["object"],
                            source=inf.get("source", "amortized_inference"),
                            confidence=inf["confidence"],
                            source_type=inf.get("source_type", "inferred"),
                        )

        return inferences, macro

    # ------------------------------------------------------------------
    #  Stats & inspection
    # ------------------------------------------------------------------

    def get_macro_count(self) -> int:
        cursor = self.memory.conn.cursor()
        cursor.execute("SELECT COUNT(*) AS cnt FROM reasoning_macros")
        return cursor.fetchone()["cnt"]

    def get_macros(self, strategy: Optional[str] = None,
                   min_confidence: float = 0.0,
                   limit: int = 50) -> List[Dict[str, Any]]:
        cursor = self.memory.conn.cursor()
        if strategy:
            cursor.execute("""
                SELECT * FROM reasoning_macros
                WHERE strategy = ? AND confidence >= ?
                ORDER BY confidence DESC, hit_count DESC
                LIMIT ?
            """, (strategy, min_confidence, limit))
        else:
            cursor.execute("""
                SELECT * FROM reasoning_macros
                WHERE confidence >= ?
                ORDER BY confidence DESC, hit_count DESC
                LIMIT ?
            """, (min_confidence, limit))
        return [dict(r) for r in cursor.fetchall()]

    def record_failure(self, macro_id: int) -> None:
        cursor = self.memory.conn.cursor()
        cursor.execute("""
            UPDATE reasoning_macros
            SET fail_count = fail_count + 1,
                confidence = MAX(0.1, confidence - 0.1),
                last_used = ?
            WHERE id = ?
        """, (datetime.now(timezone.utc).isoformat(), macro_id))
        self.memory.conn.commit()
