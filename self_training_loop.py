"""
Jafar — Self-Training Loop
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import re

logger = logging.getLogger(__name__)

WEAK_CONFIDENCE_INITIAL = 0.3
WEAK_CONFIDENCE_FLOOR = 0.05
VERIFICATION_BOOST = 0.15
CONTRADICTION_PENALTY = 0.2
MAX_VERIFICATION_AGE_DAYS = 7
STRONG_CONFIDENCE_THRESHOLD = 0.7
WEAK_FACT_CLEANUP_AGE_DAYS = 30
INFERENCE_BATCH_SIZE = 50


class SelfTrainingLoop:
    """Self-supervised training loop for the knowledge graph.

    The loop:
      1. Uses the PatternRecognizer to infer new candidate triples
         from existing high-confidence knowledge
      2. Stores them as weak facts in the DB
      3. Verifies them against strong facts (consistency check)
      4. Boosts consistent facts, penalises contradictions
      5. Eventually graduates stable weak facts to full relationships
    """

    def __init__(self, memory_system, pattern_recognizer, experience_memory=None):
        self.memory = memory_system
        self.pr = pattern_recognizer
        self.exp = experience_memory
        self._create_tables()

    def _create_tables(self) -> None:
        self.memory.conn.execute("""
            CREATE TABLE IF NOT EXISTS weak_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                relation TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence REAL DEFAULT 0.3,
                source TEXT DEFAULT 'self_inferred',
                verification_status TEXT DEFAULT 'pending',
                verification_count INTEGER DEFAULT 0,
                inference_context TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_verified_at TIMESTAMP,
                graduated_at TIMESTAMP,
                UNIQUE(subject, relation, object)
            )
        """)
        self.memory.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_weak_facts_status
            ON weak_facts(verification_status)
        """)
        self.memory.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_weak_facts_confidence
            ON weak_facts(confidence)
        """)
        self.memory.conn.commit()

    def infer_candidates(self, max_candidates: int = 20) -> List[Dict[str, Any]]:
        """Phase 1: Generate candidate triples from existing knowledge.

        Uses the PatternRecognizer to predict likely relations between
        known concepts that aren't already linked.
        """
        concepts = self.memory.get_all_concepts()
        if len(concepts) < 3:
            logger.debug("infer_candidates: only %d concepts, need >= 3", len(concepts))
            return []

        existing = {
            (r["source_concept"], r["relation"], r["target_concept"])
            for r in self.memory.conn.execute(
                "SELECT source_concept, relation, target_concept FROM relationships"
            ).fetchall()
        }

        weak_existing = {
            (r["subject"], r["relation"], r["object"])
            for r in self.memory.conn.execute(
                "SELECT subject, relation, object FROM weak_facts"
            ).fetchall()
        }

        logger.info("Relationships: %d, Concepts: %d, Weak facts: %d",
                    len(existing), len(concepts), len(weak_existing))
        logger.info("Starting pair generation (infer candidates)...")

        candidates: List[Dict[str, Any]] = []
        import random

        if not hasattr(self, '_tried_pairs'):
            self._tried_pairs = set()

        # --- Sample promising pairs instead of O(n²) ---
        # Get concepts that appear in relationships (more likely to yield good inferences)
        linked_concepts = set()
        for r in existing:
            linked_concepts.add(r[0])
            linked_concepts.add(r[2])
        linked_list = [c for c in concepts if c in linked_concepts]
        unlinked_list = [c for c in concepts if c not in linked_concepts]
        random.shuffle(linked_list)
        random.shuffle(unlinked_list)

        # Build candidate pairs: prefer linked↔linked, then linked↔unlinked
        candidate_pairs: List[Tuple[str, str]] = []
        max_pairs_to_try = min(max_candidates * 20, 2000)

        for s in linked_list:
            for o in linked_list:
                if s >= o:
                    continue
                if (s, o) not in self._tried_pairs:
                    candidate_pairs.append((s, o))
                if len(candidate_pairs) >= max_pairs_to_try:
                    break
            if len(candidate_pairs) >= max_pairs_to_try:
                break

        if len(candidate_pairs) < max_pairs_to_try:
            for s in linked_list:
                for o in unlinked_list:
                    if (s, o) not in self._tried_pairs:
                        candidate_pairs.append((s, o))
                    if len(candidate_pairs) >= max_pairs_to_try:
                        break
                if len(candidate_pairs) >= max_pairs_to_try:
                    break

        # Also try some unlinked↔unlinked (for discovery)
        if len(candidate_pairs) < max_pairs_to_try and len(unlinked_list) >= 2:
            for i, s in enumerate(unlinked_list[:50]):
                for o in unlinked_list[i + 1:][:10]:
                    if (s, o) not in self._tried_pairs:
                        candidate_pairs.append((s, o))
                    if len(candidate_pairs) >= max_pairs_to_try:
                        break
                if len(candidate_pairs) >= max_pairs_to_try:
                    break

        random.shuffle(candidate_pairs)
        logger.info("Will examine up to %d concept pairs (sampled)", len(candidate_pairs))

        for idx, (s, o) in enumerate(candidate_pairs):
            if s == o:
                continue

            self._tried_pairs.add((s, o))

            if idx % 500 == 0:
                logger.info("Infer progress: %d/%d pairs examined, %d candidates found",
                            idx, len(candidate_pairs), len(candidates))

            results = self.pr.predict_relations(s, o)
            if not results:
                continue

            top = results[0]
            triple_key = (s, top["relation"], o)
            if triple_key in existing or triple_key in weak_existing:
                continue
            if top["confidence"] < 0.15:
                continue

            candidates.append({
                "subject": s,
                "relation": top["relation"],
                "object": o,
                "confidence": top["confidence"],
                "source": "self_inferred",
            })
            if len(candidates) >= max_candidates:
                break

        logger.info("infer_candidates: generated %d candidates from %d pairs examined",
                    len(candidates), len(candidate_pairs))
        return candidates

    def store_weak_facts(self, candidates: List[Dict[str, Any]]) -> int:
        """Phase 2: Store candidates as weak facts with low confidence."""
        stored = 0
        now = datetime.utcnow().isoformat()
        for c in candidates:
            try:
                self.memory.conn.execute("""
                    INSERT OR IGNORE INTO weak_facts
                        (subject, relation, object, confidence, source,
                         inference_context, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    c["subject"].lower().strip(),
                    c["relation"].lower().strip(),
                    c["object"].lower().strip(),
                    WEAK_CONFIDENCE_INITIAL,
                    c.get("source", "self_inferred"),
                    c.get("context"),
                    now,
                ))
                if self.memory.conn.total_changes > 0 or True:
                    stored += 1
            except Exception as e:
                logger.debug("store_weak_facts skipped duplicate: %s", e)

        self.memory.conn.commit()
        logger.debug("store_weak_facts: stored %d weak facts", stored)
        return stored

    def verify_weak_facts(self, batch_size: int = 30) -> Dict[str, Any]:
        """Phase 3: Cross-reference weak facts against confirmed knowledge.

        A weak fact is:
          - verified if it's consistent with a strong relationship
          - contradicted if the opposite relation exists in strong facts
          - left pending if no confirming or contradicting evidence exists
        """
        cursor = self.memory.conn.execute("""
            SELECT wf.* FROM weak_facts wf
            WHERE wf.verification_status IN ('pending', 'verified')
              AND wf.confidence < ?
            ORDER BY wf.confidence ASC
            LIMIT ?
        """, (STRONG_CONFIDENCE_THRESHOLD, batch_size))
        weak_facts = [dict(r) for r in cursor.fetchall()]
        if not weak_facts:
            return {"checked": 0, "verified": 0, "contradicted": 0, "pending": 0}

        verified_count = 0
        contradicted_count = 0
        pending_count = 0
        now = datetime.utcnow().isoformat()

        for wf in weak_facts:
            s, r, o = wf["subject"], wf["relation"], wf["object"]

            strong_match = self.memory.conn.execute("""
                SELECT confidence FROM relationships
                WHERE source_concept=? AND relation=? AND target_concept=?
                  AND confidence >= ?
            """, (s, r, o, STRONG_CONFIDENCE_THRESHOLD)).fetchone()

            contradiction = self._find_contradiction(s, r, o)

            if strong_match:
                new_conf = min(1.0, wf["confidence"] + VERIFICATION_BOOST)
                self.memory.conn.execute("""
                    UPDATE weak_facts
                    SET confidence=?, verification_status='verified',
                        verification_count=verification_count+1,
                        last_verified_at=?
                    WHERE id=?
                """, (new_conf, now, wf["id"]))
                verified_count += 1

                self._record_experience(s, r, o, "verified",
                                        wf["confidence"], new_conf, "self_train")

                if new_conf >= STRONG_CONFIDENCE_THRESHOLD:
                    self._graduate_weak_fact(wf["id"], s, r, o, new_conf)

            elif contradiction:
                new_conf = max(WEAK_CONFIDENCE_FLOOR, wf["confidence"] - CONTRADICTION_PENALTY)
                self.memory.conn.execute("""
                    UPDATE weak_facts
                    SET confidence=?, verification_status='contradicted',
                        verification_count=verification_count+1,
                        last_verified_at=?
                    WHERE id=?
                """, (new_conf, now, wf["id"]))
                contradicted_count += 1

                self._record_experience(s, r, o, "contradicted",
                                        wf["confidence"], new_conf, "self_train")
            else:
                slight_decay = max(WEAK_CONFIDENCE_FLOOR, wf["confidence"] - 0.02)
                self.memory.conn.execute("""
                    UPDATE weak_facts
                    SET confidence=?, last_verified_at=?
                    WHERE id=?
                """, (slight_decay, now, wf["id"]))
                pending_count += 1

        self.memory.conn.commit()

        logger.debug("verify_weak_facts: %d checked — %d verified, %d contradicted, %d pending",
                     len(weak_facts), verified_count, contradicted_count, pending_count)
        return {
            "checked": len(weak_facts),
            "verified": verified_count,
            "contradicted": contradicted_count,
            "pending": pending_count,
        }

    def _find_contradiction(self, subject: str, relation: str,
                            object: str) -> bool:
        """Check if a contradictory strong relationship exists.

        Contradiction = same subject & object with a semantically
        opposite relation, or the same relation with different subject/object
        ordering that implies the opposite.
        """
        cursor = self.memory.conn.execute("""
            SELECT confidence FROM relationships
            WHERE source_concept=? AND target_concept=?
              AND confidence >= ?
        """, (subject, object, STRONG_CONFIDENCE_THRESHOLD))
        existing = cursor.fetchall()

        opposite_relations = {
            "creates": "destroys",
            "benefits_from": "harms",
            "helps": "hurts",
            "strengthens": "weakens",
        }
        opposite = opposite_relations.get(relation)

        if opposite:
            cursor = self.memory.conn.execute("""
                SELECT confidence FROM relationships
                WHERE source_concept=? AND relation=? AND target_concept=?
                  AND confidence >= ?
            """, (subject, opposite, object, STRONG_CONFIDENCE_THRESHOLD))
            if cursor.fetchone():
                return True

        reverse_check = self.memory.conn.execute("""
            SELECT confidence FROM relationships
            WHERE source_concept=? AND relation=? AND target_concept=?
              AND confidence >= ?
        """, (object, relation, subject, STRONG_CONFIDENCE_THRESHOLD))
        return reverse_check.fetchone() is not None

    def _graduate_weak_fact(self, wf_id: int, subject: str, relation: str,
                            object: str, confidence: float) -> None:
        """Promote a weak fact to a full relationship when confident enough."""
        existing = self.memory.conn.execute("""
            SELECT id FROM relationships
            WHERE source_concept=? AND relation=? AND target_concept=?
        """, (subject, relation, object)).fetchone()
        if not existing:
            self.memory.add_fact_triple(subject, relation, object,
                                        source="self_trained",
                                        confidence=confidence)
        self.memory.conn.execute("""
            UPDATE weak_facts
            SET graduated_at=CURRENT_TIMESTAMP,
                verification_status='graduated'
            WHERE id=?
        """, (wf_id,))
        logger.debug("Weak fact graduated: %s %s %s (conf=%.2f)",
                     subject, relation, object, confidence)

    def _record_experience(self, subject: str, relation: str, object: str,
                           outcome: str, conf_before: float, conf_after: float,
                           source: str = "self_train") -> None:
        if self.exp is not None:
            from experience_memory import ExperienceEvent
            self.exp.record_event(ExperienceEvent(
                subject=subject, relation=relation, object=object,
                outcome=outcome, confidence_before=conf_before,
                confidence_after=conf_after, source=source,
            ))

    def run_cycle(self, max_candidates: int = 20,
                  verify_batch: int = 30) -> Dict[str, Any]:
        """Run one full self-training cycle: infer → store → verify.

        Returns a summary of what happened.
        """
        candidates = self.infer_candidates(max_candidates=max_candidates)
        stored = self.store_weak_facts(candidates)
        verify_result = self.verify_weak_facts(batch_size=verify_batch)

        return {
            "candidates_generated": len(candidates),
            "weak_facts_stored": stored,
            "verification": verify_result,
        }

    def get_pending_weak_facts(self, limit: int = 50) -> List[Dict[str, Any]]:
        cursor = self.memory.conn.execute("""
            SELECT * FROM weak_facts
            WHERE verification_status = 'pending'
            ORDER BY created_at ASC
            LIMIT ?
        """, (limit,))
        return [dict(r) for r in cursor.fetchall()]

    def get_graduated_facts(self, limit: int = 50) -> List[Dict[str, Any]]:
        cursor = self.memory.conn.execute("""
            SELECT * FROM weak_facts
            WHERE verification_status = 'graduated'
            ORDER BY graduated_at DESC
            LIMIT ?
        """, (limit,))
        return [dict(r) for r in cursor.fetchall()]

    def cleanup_stale(self, max_age_days: int = WEAK_FACT_CLEANUP_AGE_DAYS) -> int:
        """Remove weak facts that never graduated and are past max age."""
        cutoff = (datetime.utcnow() - timedelta(days=max_age_days)).isoformat()
        cursor = self.memory.conn.execute("""
            DELETE FROM weak_facts
            WHERE verification_status IN ('pending', 'contradicted')
              AND created_at < ?
              AND graduated_at IS NULL
        """, (cutoff,))
        deleted = cursor.rowcount
        self.memory.conn.commit()
        if deleted:
            logger.debug("cleanup_stale: removed %d stale weak facts", deleted)
        return deleted

    def get_stats(self) -> Dict[str, Any]:
        cursor = self.memory.conn.execute("SELECT COUNT(*) as cnt FROM weak_facts")
        total = cursor.fetchone()["cnt"]
        cursor = self.memory.conn.execute("""
            SELECT verification_status, COUNT(*) as cnt
            FROM weak_facts
            GROUP BY verification_status
        """)
        status_breakdown = {r["verification_status"]: r["cnt"]
                           for r in cursor.fetchall()}
        cursor = self.memory.conn.execute("""
            SELECT COUNT(*) as cnt FROM weak_facts
            WHERE verification_status='graduated'
        """)
        graduated = cursor.fetchone()["cnt"]
        cursor = self.memory.conn.execute("""
            SELECT AVG(confidence) as avg_conf FROM weak_facts
        """)
        avg_row = cursor.fetchone()
        avg_confidence = round(avg_row["avg_conf"], 3) if avg_row["avg_conf"] else 0.0
        return {
            "total_weak_facts": total,
            "graduated": graduated,
            "average_confidence": avg_confidence,
            "status_breakdown": status_breakdown,
        }
