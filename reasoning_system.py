"""
Jafar — Reasoning System
Counterfactual and analogical reasoning over relationships
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
import re
import numpy as np

from memory_system import MemorySystem

logger = logging.getLogger(__name__)

INFERENCE_RULES: List[Tuple[str, str, str, float]] = [
    ("seeks", "requires", "benefits_from", 0.8),
    ("is_a", "has", "has", 0.85),
    ("is_a", "is_a", "is_a", 0.9),
    ("is_a", "can", "can", 0.85),
    ("has", "has", "has", 0.6),
    ("can", "requires", "requires", 0.7),
    ("produces", "requires", "depends_on", 0.75),
    ("depends_on", "requires", "depends_on", 0.7),
    ("lives_in", "is_a", "lives_in", 0.8),
]

INFERENCE_CONFIDENCE_FLOOR = 0.4


class ReasoningSystem:
    """Infers new relationships by chaining existing ones and dispatching to tools."""

    def __init__(self, memory: MemorySystem, tool_registry=None):
        self.memory = memory
        self.tools = tool_registry
        logger.debug("ReasoningSystem initialized with %d rules", len(INFERENCE_RULES))

    def tool_use(self, query: str) -> Dict[str, Any]:
        """Route a query to the best tool and return its result."""
        if self.tools is None:
            return {"success": False, "error": "No tool registry available"}
        return self.tools.decide_and_execute(query)

    def infer_all(self, min_confidence: float = 0.5) -> Dict[str, Any]:
        """Run all inference rules and store new inferred relationships."""
        inferred: List[Dict[str, Any]] = []
        seen: set = set()

        for rel1, rel2, inferred_rel, conf_mult in INFERENCE_RULES:
            rule_result = self._apply_rule(rel1, rel2, inferred_rel,
                                           conf_mult, min_confidence, seen)
            inferred.extend(rule_result)
            logger.debug("Rule %s+%s->%s produced %d facts",
                         rel1, rel2, inferred_rel, len(rule_result))

        stored = 0
        for fact in inferred:
            existing = self.memory._reinforce_relationship(
                fact["subject"], fact["relation"], fact["object"]
            )
            if existing is None:
                self.memory.add_fact_triple(
                    fact["subject"], fact["relation"], fact["object"],
                    source=fact.get("source"), confidence=fact["confidence"],
                    truth_confidence=fact.get("truth_confidence"),
                    source_quality=fact.get("source_quality"),
                    source_type=fact.get("source_type"),
                )
                stored += 1
            else:
                logger.debug("Already exists: %s %s %s (conf: %.2f)",
                             fact["subject"], fact["relation"],
                             fact["object"], existing)

        return {
            "inferred_count": len(inferred),
            "newly_stored": stored,
            "rules_applied": len(INFERENCE_RULES),
            "facts": inferred,
        }

    def _apply_rule(
        self, rel1: str, rel2: str, inferred_rel: str,
        conf_mult: float, min_confidence: float,
        seen: set
    ) -> List[Dict[str, Any]]:
        """Apply a single inference rule: A-r1-B + B-r2-C → A-r3-C."""
        cursor = self.memory.conn.cursor()
        cursor.execute("""
            SELECT r1.source_concept AS a,
                   r1.target_concept AS b,
                   r2.target_concept AS c,
                   r1.confidence AS conf1,
                   r2.confidence AS conf2,
                   r1.source AS src1,
                   r2.source AS src2,
                   r1.truth_confidence AS tc1,
                   r2.truth_confidence AS tc2,
                   r1.source_quality AS sq1,
                   r2.source_quality AS sq2,
                   r1.source_type AS st1,
                   r2.source_type AS st2
            FROM relationships r1
            JOIN relationships r2 ON r1.target_concept = r2.source_concept
            WHERE r1.relation = ?
              AND r2.relation = ?
              AND r1.confidence >= ?
              AND r2.confidence >= ?
        """, (rel1, rel2, min_confidence, min_confidence))

        results: List[Dict[str, Any]] = []
        for row in cursor.fetchall():
            key = (row["a"], inferred_rel, row["c"])
            if key in seen:
                logger.debug("Skipping duplicate: %s", key)
                continue
            seen.add(key)

            inferred_conf = min(row["conf1"], row["conf2"]) * conf_mult
            if inferred_conf < INFERENCE_CONFIDENCE_FLOOR:
                logger.debug("Inferred confidence %.2f below floor, skipping %s",
                             inferred_conf, key)
                continue

            source_parts = []
            if row["src1"]:
                source_parts.append(row["src1"])
            if row["src2"]:
                source_parts.append(row["src2"])
            combined_source = "; ".join(
                dict.fromkeys(source_parts)) if source_parts else None

            tc1 = row["tc1"]
            tc2 = row["tc2"]
            match (tc1 is not None, tc2 is not None):
                case (True, True):
                    inferred_tc = min(tc1, tc2)
                case (True, False):
                    inferred_tc = tc1
                case (False, True):
                    inferred_tc = tc2
                case (False, False):
                    inferred_tc = None

            sq1 = row["sq1"]
            sq2 = row["sq2"]
            match (bool(sq1), bool(sq2)):
                case (True, True):
                    inferred_sq = sq1 if sq1 == sq2 else f"{sq1};{sq2}"
                case (True, False):
                    inferred_sq = sq1
                case (False, True):
                    inferred_sq = sq2
                case (False, False):
                    inferred_sq = None

            st1 = row["st1"]
            st2 = row["st2"]
            match (bool(st1), bool(st2)):
                case (True, True):
                    inferred_st = st1 if st1 == st2 else None
                case (True, False):
                    inferred_st = st1
                case (False, True):
                    inferred_st = st2
                case (False, False):
                    inferred_st = None

            if inferred_tc is not None:
                inferred_tc = round(inferred_tc * conf_mult, 3)

            results.append({
                "subject": row["a"],
                "relation": inferred_rel,
                "object": row["c"],
                "confidence": round(inferred_conf, 2),
                "premises": f"{row['a']} {rel1} {row['b']} AND {row['b']} {rel2} {row['c']}",
                "source": combined_source,
                "truth_confidence": inferred_tc,
                "source_quality": inferred_sq,
                "source_type": inferred_st,
            })

        logger.debug("Rule %s+%s->%s: %d after filter",
                     rel1, rel2, inferred_rel, len(results))
        return results

    def cross_validate(self, vector_db, embedding_generator=None,
                       embedding_enhancer=None, top_k: int = 3,
                       min_similarity: float = 0.3) -> Dict[str, Any]:
        """Cross-validate inferred facts against vector DB passages.

        Enhanced with:
          - Reusable embedding_generator (no new instance per call)
          - Concept co-occurrence scoring (subject+object in same passage)
          - Optional graph-enhanced embeddings via embedding_enhancer

        A fact is "supported" if embedding similarity >= min_similarity
        OR subject/object concepts co-occur in the same passage.
        """
        cursor = self.memory.conn.cursor()
        cursor.execute("""
            SELECT * FROM memories WHERE source LIKE 'inferred:%'
            ORDER BY confidence DESC
        """)
        inferred = cursor.fetchall()

        if not inferred:
            return {"validated": 0, "supported": 0, "unsupported": 0, "details": []}

        results = []
        supported_count = 0
        for row in inferred:
            fact_text = row["fact"]
            conf = row["confidence"]

            fact_embedding = self._embed_fact(fact_text, vector_db,
                                              embedding_generator, embedding_enhancer)

            passages = vector_db.search_similar_pages(
                fact_embedding, top_k=top_k
            )

            best_sim = max((p["similarity"] for p in passages), default=0.0)

            parts = fact_text.lower().split()
            fact_concepts = [p for p in parts if len(p) > 2]

            max_co_occurrence = 0.0
            concepts_found = 0
            for p in passages:
                text_lower = p["text"].lower()
                matches = sum(1 for c in fact_concepts if c in text_lower)
                if matches > concepts_found:
                    concepts_found = matches
                if len(fact_concepts) > 1:
                    pair_score = matches / max(len(fact_concepts), 1)
                    if pair_score > max_co_occurrence:
                        max_co_occurrence = pair_score

            combined_score = max(best_sim, max_co_occurrence)
            is_supported = combined_score >= min_similarity
            if is_supported:
                supported_count += 1

            results.append({
                "fact": fact_text,
                "confidence": conf,
                "best_similarity": round(best_sim, 3),
                "concept_co_occurrence": round(max_co_occurrence, 3),
                "combined_score": round(combined_score, 3),
                "supported": is_supported,
                "supporting_passages": passages[:top_k],
            })

        return {
            "validated": len(results),
            "supported": supported_count,
            "unsupported": len(results) - supported_count,
            "details": results,
        }

    def _embed_fact(self, fact_text: str, vector_db,
                    embedding_generator=None,
                    embedding_enhancer=None) -> np.ndarray:
        try:
            if embedding_generator is not None:
                emb = embedding_generator.generate_single_text_embedding(fact_text)
                if embedding_enhancer is not None and hasattr(embedding_enhancer, '_trained') and embedding_enhancer._trained:
                    try:
                        emb = embedding_enhancer.enhance(fact_text)
                    except Exception:
                        pass
                return emb
            from embeddings import EmbeddingGenerator as EG
            from memory_manager import MemoryManager
            eg = EG(MemoryManager())
            return eg.generate_single_text_embedding(fact_text)
        except Exception:
            rng = np.random.default_rng(42)
            return rng.normal(0, 0.1, 384)

    def explain(self, subject: str, relation: str, obj: str) -> Optional[Dict[str, Any]]:
        """Find the inference chain that produced a given fact (if inferred)."""
        cursor = self.memory.conn.cursor()
        cursor.execute("""
            SELECT * FROM memories
            WHERE fact = ? AND source IS NOT NULL
              AND source LIKE 'inferred:%'
        """, (f"{subject} {relation} {obj}",))
        row = cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    def get_stats(self) -> Dict[str, Any]:
        cursor = self.memory.conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) as cnt FROM memories WHERE source LIKE 'inferred:%'
        """)
        inferred_count = cursor.fetchone()["cnt"]
        return {
            "rules": len(INFERENCE_RULES),
            "inferred_memories": inferred_count,
        }
