"""
PlanningAgent — goal-driven multi-step reasoning with evidence scoring.

Goes beyond fixed inference rules:
- Generates multiple reasoning paths for a goal
- Scores each path by evidence strength (direct, indirect, inferred)
- Tracks supporting facts and source provenance
- Ranks paths and selects the best conclusion
- Learns new inference rules from successful paths
"""

import logging
import time
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum

logger = logging.getLogger(__name__)


class EvidenceStrength(Enum):
    DIRECT = 1.0
    STRONG_INFERRED = 0.8
    WEAK_INFERRED = 0.5
    SPECULATIVE = 0.3
    HYPOTHETICAL = 0.15


class ReasoningMode(Enum):
    RULE_ENGINE = "rule_engine"
    GRAPH_EXPLORATION = "graph_exploration"
    CONSENSUS_REASONING = "consensus_reasoning"


@dataclass(slots=True)
class ReasoningContext:
    uncertainty: float = 0.0
    direct_rule_strength: float = 0.0
    conflict_count: int = 0
    concept_count: int = 0


class ReasoningSelector:
    __slots__ = ('uncertainty_threshold', 'strength_threshold', 'conflict_threshold')

    def __init__(self, uncertainty_threshold: float = 0.6,
                 strength_threshold: float = 0.5,
                 conflict_threshold: int = 2):
        self.uncertainty_threshold = uncertainty_threshold
        self.strength_threshold = strength_threshold
        self.conflict_threshold = conflict_threshold

    def select_mode(self, context: ReasoningContext) -> ReasoningMode:
        if context.conflict_count >= self.conflict_threshold:
            return ReasoningMode.CONSENSUS_REASONING
        if context.concept_count == 0:
            return ReasoningMode.GRAPH_EXPLORATION
        if context.uncertainty < self.uncertainty_threshold and context.direct_rule_strength >= self.strength_threshold:
            return ReasoningMode.RULE_ENGINE
        if context.uncertainty >= self.uncertainty_threshold and context.concept_count >= 2:
            return ReasoningMode.GRAPH_EXPLORATION
        return ReasoningMode.RULE_ENGINE

    def select_mode_for_goal(self, goal: str, concepts: List[str],
                              relationships: List[Dict[str, Any]]) -> ReasoningMode:
        if not concepts:
            return ReasoningMode.GRAPH_EXPLORATION

        confidences = [r.get("confidence", 0.5) for r in relationships[:20]]
        if not confidences:
            return ReasoningMode.GRAPH_EXPLORATION

        avg_conf = sum(confidences) / len(confidences)
        uncertainty = 1.0 - avg_conf
        direct_strength = avg_conf

        from collections import Counter
        rels = [r.get("relation", "") for r in relationships]
        rel_counts = Counter(rels)
        conflict_count = sum(1 for v in rel_counts.values() if v > 1)

        ctx = ReasoningContext(
            uncertainty=uncertainty,
            direct_rule_strength=direct_strength,
            conflict_count=conflict_count,
            concept_count=len(concepts),
        )
        return self.select_mode(ctx)


@dataclass(slots=True)
class EvidenceEntry:
    statement: str
    subject: str
    relation: str
    object: str
    confidence: float
    supporting_facts: List[Dict[str, Any]] = field(default_factory=list)
    source_paths: List[List[str]] = field(default_factory=list)
    strength: Dict[str, float] = field(default_factory=lambda: {
        "direct": 0.0, "indirect": 0.0, "inferred": 0.0
    })
    created_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statement": self.statement,
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "confidence": self.confidence,
            "supporting_facts": self.supporting_facts,
            "source_paths": self.source_paths,
            "strength": self.strength,
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class ReasoningStep:
    type: str
    premise: Dict[str, Any]
    conclusion: Dict[str, Any]
    confidence: float
    evidence: Optional[EvidenceEntry] = None
    description: str = ""


@dataclass(slots=True)
class ReasoningPath:
    goal: str
    steps: List[ReasoningStep] = field(default_factory=list)
    score: float = 0.0
    evidence_chain: List[EvidenceEntry] = field(default_factory=list)
    aggregate_confidence: float = 0.0
    path_length: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "steps": [{
                "type": s.type,
                "premise": s.premise,
                "conclusion": s.conclusion,
                "confidence": s.confidence,
                "description": s.description,
            } for s in self.steps],
            "score": self.score,
            "aggregate_confidence": self.aggregate_confidence,
            "path_length": self.path_length,
            "evidence_count": len(self.evidence_chain),
        }


class EvidenceStore:
    """Stores and queries evidence for inferred statements."""

    __slots__ = ('max_entries', '_entries', '_subject_index')

    def __init__(self, max_entries: int = 10000):
        self.max_entries = max_entries
        self._entries: Dict[str, EvidenceEntry] = {}
        self._subject_index: Dict[str, Set[str]] = defaultdict(set)

    def _key(self, subject: str, relation: str, object: str) -> str:
        return f"{subject.lower().strip()}--{relation.lower().strip()}--{object.lower().strip()}"

    def add_evidence(self, subject: str, relation: str, object: str,
                     confidence: float,
                     supporting_facts: Optional[List[Dict[str, Any]]] = None,
                     source_paths: Optional[List[List[str]]] = None,
                     strength: Optional[Dict[str, float]] = None) -> EvidenceEntry:
        key = self._key(subject, relation, object)
        now = time.time()
        stmt = f"{subject} --[{relation}]--> {object}"

        strength = strength or {"direct": 0.0, "indirect": 0.0, "inferred": 0.0}

        entry = EvidenceEntry(
            statement=stmt,
            subject=subject.lower().strip(),
            relation=relation.lower().strip(),
            object=object.lower().strip(),
            confidence=confidence,
            supporting_facts=supporting_facts or [],
            source_paths=source_paths or [],
            strength=strength,
            created_at=now,
        )

        if len(self._entries) >= self.max_entries:
            oldest = min(self._entries.keys(),
                         key=lambda k: self._entries[k].created_at)
            del self._entries[oldest]

        self._entries[key] = entry
        self._subject_index[subject.lower().strip()].add(key)

        return entry

    def get_evidence(self, subject: str, relation: str,
                     object: str) -> Optional[EvidenceEntry]:
        return self._entries.get(self._key(subject, relation, object))

    def get_evidence_by_subject(self, subject: str) -> List[EvidenceEntry]:
        keys = self._subject_index.get(subject.lower().strip(), set())
        return [self._entries[k] for k in keys if k in self._entries]

    def get_strongest_path(self, subject: str, relation: str,
                           object: str) -> Optional[EvidenceEntry]:
        entry = self.get_evidence(subject, relation, object)
        if entry is None:
            return None
        for sf in entry.supporting_facts:
            sk = self._key(sf.get("subject", ""), sf.get("relation", ""),
                           sf.get("object", ""))
            if sk in self._entries:
                child = self._entries[sk]
                if child.confidence > entry.confidence:
                    entry = child
        return entry

    def merge_evidence(self, subject: str, relation: str, object: str,
                       new_entry: EvidenceEntry) -> EvidenceEntry:
        existing = self.get_evidence(subject, relation, object)
        if existing is None:
            return self.add_evidence(
                subject, relation, object,
                new_entry.confidence,
                new_entry.supporting_facts,
                new_entry.source_paths,
                new_entry.strength)

        all_facts = existing.supporting_facts + new_entry.supporting_facts
        seen = set()
        deduped = []
        for f in all_facts:
            fkey = self._key(f.get("subject", ""), f.get("relation", ""),
                             f.get("object", ""))
            if fkey not in seen:
                seen.add(fkey)
                deduped.append(f)

        all_paths = existing.source_paths + new_entry.source_paths
        best_strength = {
            "direct": max(existing.strength.get("direct", 0),
                          new_entry.strength.get("direct", 0)),
            "indirect": max(existing.strength.get("indirect", 0),
                            new_entry.strength.get("indirect", 0)),
            "inferred": max(existing.strength.get("inferred", 0),
                            new_entry.strength.get("inferred", 0)),
        }
        best_conf = max(existing.confidence, new_entry.confidence)
        existing.supporting_facts = deduped
        existing.source_paths = all_paths
        existing.strength = best_strength
        existing.confidence = best_conf
        return existing

    def get_stats(self) -> Dict[str, Any]:
        if not self._entries:
            return {"total_entries": 0}
        confidences = [e.confidence for e in self._entries.values()]
        return {
            "total_entries": len(self._entries),
            "unique_subjects": len(self._subject_index),
            "avg_confidence": sum(confidences) / len(confidences),
            "max_confidence": max(confidences),
            "total_supporting_facts": sum(
                len(e.supporting_facts) for e in self._entries.values()),
        }


@dataclass(slots=True)
class ChainCompatibility:
    r1: str = ""
    r2: str = ""
    inferred_rel: str = ""
    base_score: float = 0.0
    usage_count: int = 0
    success_count: int = 0


class WeightedRelationScorer:
    """Soft scoring engine that replaces hard-coded inference rules.

    Each compatibility entry stores (r1, r2) -> (inferred_rel, base_score).
    score_chain() returns a combined score that factors in confidences and
    path length, enabling multi-path combination and adaptive learning.
    """

    __slots__ = ('_entries',)

    def __init__(self,
                 seed_rules: Optional[List[Tuple[str, str, str, float]]] = None):
        self._entries: Dict[Tuple[str, str], ChainCompatibility] = {}
        if seed_rules:
            for r1, r2, inferred, score in seed_rules:
                self.add_chain(r1, r2, inferred, score)

    def add_chain(self, r1: str, r2: str, inferred_rel: str,
                  base_score: float) -> None:
        key = (r1.strip().lower(), r2.strip().lower())
        self._entries[key] = ChainCompatibility(
            r1=r1.strip().lower(),
            r2=r2.strip().lower(),
            inferred_rel=inferred_rel,
            base_score=min(1.0, max(0.0, base_score)),
        )

    def score_chain(self, r1: str, r2: str, conf1: float = 1.0,
                    conf2: float = 1.0,
                    path_length: int = 1) -> List[Tuple[str, float]]:
        key = (r1.strip().lower(), r2.strip().lower())
        entry = self._entries.get(key)
        if entry is None:
            return []
        path_decay = max(0.7, 1.0 - 0.1 * (path_length - 1))
        raw_score = entry.base_score * min(conf1, conf2) * path_decay
        return [(entry.inferred_rel, round(raw_score, 4))]

    def score_single(self, r: str, conf: float = 1.0
                     ) -> List[Tuple[str, float]]:
        rl = r.strip().lower()
        results = []
        for key, entry in self._entries.items():
            if key[0] == rl:
                score = entry.base_score * conf
                results.append((entry.inferred_rel, round(score, 4)))
        return results

    def learn_chain(self, r1: str, r2: str, inferred_rel: str,
                    compat_boost: float = 0.05) -> None:
        key = (r1.strip().lower(), r2.strip().lower())
        existing = self._entries.get(key)
        if existing:
            existing.base_score = min(1.0, existing.base_score + compat_boost)
            existing.usage_count += 1
            existing.success_count += 1
        else:
            self.add_chain(r1, r2, inferred_rel, 0.4 + compat_boost)

    def infer_relation(self, r1: str, r2: str) -> Optional[str]:
        key = (r1.strip().lower(), r2.strip().lower())
        entry = self._entries.get(key)
        return entry.inferred_rel if entry else None

    def has_chain(self, r1: str, r2: str) -> bool:
        return (r1.strip().lower(), r2.strip().lower()) in self._entries

    def get_chain_count(self) -> int:
        return len(self._entries)

    def get_rules(self) -> List[Tuple[str, str, str, float]]:
        return [(e.r1, e.r2, e.inferred_rel, e.base_score)
                for e in self._entries.values()]

    def get_entries_sorted(self) -> List[ChainCompatibility]:
        return sorted(self._entries.values(),
                      key=lambda e: e.base_score, reverse=True)

    def get_stats(self) -> Dict[str, Any]:
        entries = self._entries.values()
        if not entries:
            return {"chain_count": 0}
        scores = [e.base_score for e in entries]
        usages = [e.usage_count for e in entries]
        return {
            "chain_count": len(entries),
            "avg_base_score": round(sum(scores) / len(scores), 3),
            "max_base_score": round(max(scores), 3),
            "total_usage": sum(usages),
            "avg_usage": round(sum(usages) / len(usages), 1),
        }


@dataclass(slots=True)
class RulePerformance:
    r1: str = ""
    r2: str = ""
    success_count: int = 0
    fail_count: int = 0
    total_uses: int = 0

    @property
    def success_rate(self) -> float:
        return self.success_count / max(self.total_uses, 1)


class RulePerformanceTracker:
    """Tracks success/fail per (r1, r2) chain and self-improves via scorer.

    Strengthens reliable chains (>=80% success, >=3 uses) and weakens
    unreliable ones (<=30% success, >=3 uses).
    """

    __slots__ = ('_tracker', '_scorer')

    def __init__(self, scorer: WeightedRelationScorer):
        self._tracker: Dict[Tuple[str, str], RulePerformance] = {}
        self._scorer = scorer

    def _key(self, r1: str, r2: str) -> Tuple[str, str]:
        return (r1.strip().lower(), r2.strip().lower())

    def record_success(self, r1: str, r2: str) -> None:
        key = self._key(r1, r2)
        if key not in self._tracker:
            self._tracker[key] = RulePerformance(r1=r1, r2=r2)
        self._tracker[key].success_count += 1
        self._tracker[key].total_uses += 1

    def record_fail(self, r1: str, r2: str) -> None:
        key = self._key(r1, r2)
        if key not in self._tracker:
            self._tracker[key] = RulePerformance(r1=r1, r2=r2)
        self._tracker[key].fail_count += 1
        self._tracker[key].total_uses += 1

    def get_performance(self, r1: str, r2: str) -> Optional[RulePerformance]:
        return self._tracker.get(self._key(r1, r2))

    def self_improve(self) -> Dict[str, Any]:
        changes = {"boosted": 0, "decayed": 0, "unchanged": 0}
        for key, perf in self._tracker.items():
            if perf.total_uses < 3:
                changes["unchanged"] += 1
                continue
            rate = perf.success_rate
            if rate >= 0.8:
                existing = self._scorer._entries.get(key)
                if existing:
                    existing.base_score = min(1.0, existing.base_score + 0.03)
                    existing.success_count += perf.success_count
                    changes["boosted"] += 1
            elif rate <= 0.3:
                existing = self._scorer._entries.get(key)
                if existing:
                    existing.base_score *= 0.9
                    existing.success_count += perf.success_count
                    changes["decayed"] += 1
            else:
                changes["unchanged"] += 1
        return changes

    def get_stats(self) -> Dict[str, Any]:
        if not self._tracker:
            return {"tracked_rules": 0}
        rates = [p.success_rate for p in self._tracker.values()]
        usages = [p.total_uses for p in self._tracker.values()]
        return {
            "tracked_rules": len(self._tracker),
            "avg_success_rate": round(sum(rates) / len(rates), 3),
            "total_recorded_uses": sum(usages),
            "min_uses": min(usages),
            "max_uses": max(usages),
        }


class PlanningAgent:
    """Goal-driven multi-step reasoning with evidence scoring."""

    def __init__(self, memory, evidence_store: Optional[EvidenceStore] = None,
                 inference_rules: Optional[List[Tuple[str, str, str, float]]] = None):
        self.memory = memory
        self.evidence_store = evidence_store or EvidenceStore()

        default_rules = [
            ("is_a", "has", "has", 0.85),
            ("is_a", "is_a", "is_a", 0.9),
            ("is_a", "can", "can", 0.85),
            ("has", "has", "has", 0.6),
            ("can", "requires", "requires", 0.7),
            ("produces", "requires", "depends_on", 0.75),
            ("depends_on", "requires", "depends_on", 0.7),
            ("lives_in", "is_a", "lives_in", 0.8),
            ("benefits_from", "requires", "benefits_from", 0.7),
            ("seeks", "requires", "benefits_from", 0.8),
        ]

        self.inference_rules = inference_rules or default_rules

        self._scorer = WeightedRelationScorer(
            seed_rules=self.inference_rules)

        self._learned_rules: List[Tuple[str, str, str, float]] = []
        self._plan_history: List[Dict[str, Any]] = []
        self._rule_success_count: Dict[int, int] = defaultdict(int)
        self._rule_fail_count: Dict[int, int] = defaultdict(int)
        self._selector = ReasoningSelector()
        self._tracker = RulePerformanceTracker(self._scorer)

    def _triple_key(self, s: str, r: str, o: str) -> str:
        return f"{s.lower().strip()}--{r.lower().strip()}--{o.lower().strip()}"

    def _get_relationships(self, concept: str) -> List[Dict[str, Any]]:
        try:
            graph = self.memory.get_concept_graph(concept)
            return graph.get("relationships", [])
        except Exception as e:
            logger.debug("Error getting graph for '%s': %s", concept, e)
            return []

    def _normalize_rel(self, r: Dict[str, Any],
                       concept: str) -> Optional[Dict[str, Any]]:
        target = r.get("target", "").lower().strip()
        relation = r.get("relation", "")
        direction = r.get("direction", "")
        if not target or not relation:
            return None
        if direction == "outgoing":
            r["source_concept"] = concept.lower().strip()
            r["target_concept"] = target
        else:
            r["source_concept"] = target
            r["target_concept"] = concept.lower().strip()
        r["subject"] = r["source_concept"]
        r["object"] = r["target_concept"]
        return r

    def _get_relationship_triples(self) -> List[Dict[str, Any]]:
        try:
            stats = self.memory.get_stats()
            count = stats.get("relationships", 0)
            triples = []
            concepts = self.memory.get_all_concepts()
            seen = set()
            for c in concepts[:200]:
                for r in self._get_relationships(c):
                    key = self._triple_key(
                        r.get("source_concept", r.get("subject", "")),
                        r.get("relation", ""),
                        r.get("target_concept", r.get("object", "")))
                    if key not in seen:
                        seen.add(key)
                        triples.append(r)
            return triples
        except Exception as e:
            logger.debug("Error getting triples: %s", e)
            return []

    def _bfs_paths(self, start: str, goal_concept: str,
                    max_depth: int = 3) -> List[List[Dict[str, Any]]]:
        found_paths = []
        visited_edges = set()
        queue = [[{"concept": start, "relation": None, "direction": None}]]

        while queue and len(found_paths) < 10:
            path = queue.pop(0)
            current = path[-1]["concept"]
            if len(path) > max_depth:
                continue
            if current == goal_concept and len(path) > 1:
                found_paths.append(path)
                continue
            if len(path) > 2:
                edge_key = tuple(
                    (p["concept"], p.get("relation"))
                    for p in path[-2:])
                if edge_key in visited_edges:
                    continue
                visited_edges.add(edge_key)

            for rel in self._get_relationships(current):
                neighbor = rel.get("target", "").lower().strip()
                rel_name = rel.get("relation", "")
                direction = rel.get("direction", "")
                if neighbor and neighbor not in {p["concept"] for p in path}:
                    new_path = path + [{
                        "concept": neighbor,
                        "relation": rel_name,
                        "direction": direction,
                        "rel_data": rel,
                    }]
                    queue.append(new_path)

        return found_paths

    def generate_plans(self, goal: str) -> List[ReasoningPath]:
        goal_lower = goal.lower()
        concepts = set(re.findall(r"'([^']+)'|\"([^\"]+)\"|([A-Z][a-z]+)", goal_lower))
        flat_concepts = []
        for c in concepts:
            for sub in c:
                if sub and len(sub) > 2:
                    flat_concepts.append(sub.strip().lower())

        if not flat_concepts:
            flat_concepts = [w for w in re.split(r"[^a-z]+", goal_lower)
                             if len(w) > 2]

        concepts_in_kg = []
        for c in flat_concepts:
            try:
                matches = self.memory.search_concepts(c, limit=5)
                if matches and isinstance(matches[0], dict):
                    name = matches[0].get("name", "").lower().strip()
                    if name:
                        concepts_in_kg.append(name)
                elif matches:
                    concepts_in_kg.append(str(matches[0]).lower().strip())
            except Exception:
                continue

        concepts_in_kg = list(dict.fromkeys(concepts_in_kg))

        plans = []

        plan_a = self._build_direct_query_plan(goal, concepts_in_kg)
        if plan_a:
            plans.append(plan_a)

        plan_b = self._build_inference_chain_plan(goal, concepts_in_kg)
        if plan_b:
            plans.append(plan_b)

        if len(concepts_in_kg) >= 2:
            for i in range(len(concepts_in_kg)):
                for j in range(i + 1, len(concepts_in_kg)):
                    plan_c = self._build_path_finding_plan(
                        goal, concepts_in_kg[i], concepts_in_kg[j])
                    if plan_c:
                        plans.append(plan_c)

        plan_d = self._build_rule_based_plan(goal, concepts_in_kg)
        if plan_d:
            plans.append(plan_d)

        if not plans:
            plans.append(ReasoningPath(
                goal=goal,
                score=0.0,
                aggregate_confidence=0.0,
            ))

        for p in plans:
            self._score_path(p)

        plans.sort(key=lambda p: p.score, reverse=True)

        return plans

    def _build_direct_query_plan(self, goal: str,
                                  concepts: List[str]) -> Optional[ReasoningPath]:
        if not concepts:
            return None
        steps = []
        evidence_chain = []
        for c in concepts[:3]:
            rels = self._get_relationships(c)
            for r in rels[:5]:
                nr = self._normalize_rel(dict(r), c)
                if nr is None:
                    continue
                source = nr.get("source_concept", "").lower().strip()
                target = nr.get("target_concept", "").lower().strip()
                rel_name = nr.get("relation", "")
                conf = nr.get("confidence", 0.5)
                tc = nr.get("truth_confidence", conf)
                final_conf = max(conf, tc)

                stmt = f"{source} --[{rel_name}]--> {target}"
                strength_type = "direct" if nr.get("source_type") in (
                    "real", None) else "inferred"
                strength = {"direct": 0.0, "indirect": 0.0, "inferred": 0.0}
                if strength_type == "direct":
                    strength["direct"] = final_conf
                else:
                    strength["inferred"] = final_conf

                evidence = self.evidence_store.add_evidence(
                    source, rel_name, target, final_conf,
                    supporting_facts=[{
                        "subject": source,
                        "relation": rel_name,
                        "object": target,
                        "confidence": final_conf,
                        "source": nr.get("source", ""),
                    }],
                    source_paths=[[nr.get("source", "direct")]],
                    strength=strength,
                )
                evidence_chain.append(evidence)

                steps.append(ReasoningStep(
                    type="direct_query",
                    premise={"concept": c, "relationship": rel_name},
                    conclusion={
                        "subject": source,
                        "relation": rel_name,
                        "object": target,
                    },
                    confidence=final_conf,
                    evidence=evidence,
                    description=f"Direct fact: {stmt}",
                ))

        if not steps:
            return None

        path = ReasoningPath(goal=goal, steps=steps,
                             evidence_chain=evidence_chain)
        return path

    def _build_inference_chain_plan(self, goal: str,
                                      concepts: List[str]) -> Optional[ReasoningPath]:
        if len(concepts) < 1:
            return None
        steps = []
        evidence_chain = []
        for c in concepts[:2]:
            rels = self._get_relationships(c)
            for r1 in rels[:3]:
                nr1 = self._normalize_rel(dict(r1), c)
                if nr1 is None:
                    continue
                r1_relation = nr1.get("relation", "").strip().lower()
                source = nr1.get("source_concept", "").lower().strip()
                target = nr1.get("target_concept", "").lower().strip()
                if not target:
                    continue
                c1 = nr1.get("confidence", 0.5)

                target_rels = self._get_relationships(target)
                for r2 in target_rels[:3]:
                    nr2 = self._normalize_rel(dict(r2), target)
                    if nr2 is None:
                        continue
                    r2_relation = nr2.get("relation", "").strip().lower()
                    c2 = nr2.get("confidence", 0.5)

                    chain_results = self._scorer.score_chain(
                        r1_relation, r2_relation, c1, c2)
                    if not chain_results:
                        chain_results = self._scorer.score_chain(
                            r1_relation, r2_relation, 0.5, 0.5)
                    if not chain_results:
                        continue

                    for inferred_rel, inferred_conf in chain_results:
                        if inferred_conf < 0.3:
                            continue

                        inferred_target = nr2.get(
                            "target_concept", "").lower().strip()
                        stmt = (
                            f"{source} --[{inferred_rel}]--> "
                            f"{inferred_target}")

                        strength = {"direct": 0.0, "indirect": 0.6,
                                    "inferred": 0.0}
                        evidence = self.evidence_store.add_evidence(
                            source, inferred_rel, inferred_target,
                            inferred_conf,
                            supporting_facts=[
                                {"subject": source,
                                 "relation": r1_relation,
                                 "object": target,
                                 "confidence": c1},
                                {"subject": target,
                                 "relation": r2_relation,
                                 "object": inferred_target,
                                 "confidence": c2},
                            ],
                            source_paths=[
                                [nr1.get("source", "unknown"),
                                 nr2.get("source", "unknown")]
                            ],
                            strength=strength,
                        )
                        evidence_chain.append(evidence)
                        steps.append(ReasoningStep(
                            type="inference_chain",
                            premise={
                                "rule": {
                                    "r1": r1_relation,
                                    "r2": r2_relation,
                                    "inferred": inferred_rel,
                                    "score": inferred_conf,
                                },
                                "fact1": {"subject": source,
                                          "relation": r1_relation,
                                          "object": target},
                                "fact2": {"subject": target,
                                          "relation": r2_relation,
                                          "object": inferred_target},
                            },
                            conclusion={
                                "subject": source,
                                "relation": inferred_rel,
                                "object": inferred_target,
                            },
                            confidence=inferred_conf,
                            evidence=evidence,
                            description=f"Inferred: {stmt} "
                                        f"(via {r1_relation}→{r2_relation})",
                        ))
        if not steps:
            return None
        return ReasoningPath(goal=goal, steps=steps,
                             evidence_chain=evidence_chain)

    def _build_path_finding_plan(self, goal: str,
                                   concept_a: str,
                                   concept_b: str) -> Optional[ReasoningPath]:
        if concept_a == concept_b:
            return None
        paths = self._bfs_paths(concept_a, concept_b, max_depth=3)
        if not paths:
            return None

        path_confs = []
        contribution_details = []
        for path in paths:
            path_conf = 1.0
            for i in range(1, len(path)):
                rel_data = path[i].get("rel_data", {})
                conf = rel_data.get("confidence", 0.5)
                tc = rel_data.get("truth_confidence", 0.5)
                path_conf = min(path_conf, max(conf, tc))
            path_confs.append(path_conf)
            contribution_details.append({
                "path": [p["concept"] for p in path],
                "confidence": round(path_conf, 4),
            })

        noisy_or = 1.0
        for pc in path_confs:
            noisy_or *= 1.0 - pc
        combined_conf = round(1.0 - noisy_or, 4)
        combined_conf = min(combined_conf, max(path_confs))

        path_count_bonus = min(1.0, len(paths) / 5.0) * 0.05
        combined_conf = min(1.0, combined_conf + path_count_bonus)

        strength = {"direct": 0.0, "indirect": 0.0,
                    "inferred": combined_conf}
        evidence = self.evidence_store.add_evidence(
            concept_a, "related_to", concept_b,
            combined_conf,
            supporting_facts=[
                {"subject": concept_a, "relation": "related_to",
                 "object": concept_b,
                 "confidence": round(pc, 4)}
                for pc in path_confs[:5]],
            source_paths=[d["path"] for d in contribution_details[:5]],
            strength=strength,
        )

        path_summary = "; ".join(
            f"#{i}: {d['path'][0]}→{d['path'][-1]} (c={d['confidence']:.2f})"
            for i, d in enumerate(contribution_details[:5]))

        step = ReasoningStep(
            type="multi_path",
            premise={"start": concept_a, "end": concept_b,
                     "path_count": len(paths)},
            conclusion={
                "subject": concept_a,
                "relation": "related_to",
                "object": concept_b,
            },
            confidence=combined_conf,
            evidence=evidence,
            description=f"Multi-path ({len(paths)} paths): "
                        f"{concept_a} related_to {concept_b} "
                        f"(combined={combined_conf:.2f}) {path_summary}",
        )

        return ReasoningPath(
            goal=goal, steps=[step],
            evidence_chain=[evidence])

    def _build_rule_based_plan(self, goal: str,
                                concepts: List[str]) -> Optional[ReasoningPath]:
        steps = []
        evidence_chain = []

        for c in concepts[:2]:
            rels = self._get_relationships(c)
            for r in rels[:3]:
                nr = self._normalize_rel(dict(r), c)
                if nr is None:
                    continue
                rrel = nr.get("relation", "").strip().lower()
                source = nr.get("source_concept", "").lower().strip()
                target = nr.get("target_concept", "").lower().strip()
                conf = nr.get("confidence", 0.5)

                chain_results = self._scorer.score_single(rrel, conf)
                if not chain_results:
                    continue

                for inferred_rel, inferred_conf in chain_results:
                    if inferred_conf < 0.3:
                        continue

                    strength = {"direct": 0.0, "indirect": 0.0,
                                "inferred": inferred_conf}
                    evidence = self.evidence_store.add_evidence(
                        source, inferred_rel, target,
                        inferred_conf,
                        supporting_facts=[{
                            "subject": source,
                            "relation": rrel,
                            "object": target,
                            "confidence": conf,
                        }],
                        source_paths=[[nr.get("source", "rule-based")]],
                        strength=strength,
                    )
                    evidence_chain.append(evidence)
                    steps.append(ReasoningStep(
                        type="rule_application",
                        premise={"rule": {"r1": rrel, "inferred": inferred_rel,
                                          "score": inferred_conf},
                                 "fact": nr},
                        conclusion={
                            "subject": source,
                            "relation": inferred_rel,
                            "object": target,
                        },
                        confidence=inferred_conf,
                        evidence=evidence,
                        description=f"Rule '{rrel}→{inferred_rel}': "
                                    f"{source} --{inferred_rel}--> {target}",
                    ))

        if not steps:
            return None
        return ReasoningPath(goal=goal, steps=steps,
                             evidence_chain=evidence_chain)

    def _score_path(self, path: ReasoningPath) -> None:
        if not path.steps:
            path.score = 0.0
            path.aggregate_confidence = 0.0
            path.path_length = 0
            return

        confs = [s.confidence for s in path.steps]
        path.aggregate_confidence = sum(confs) / len(confs)

        path_length_bonus = min(1.0, len(path.steps) / 5.0)

        ev = path.evidence_chain
        direct_evidence = 0
        total_evidence = len(ev)
        for e in ev:
            if e.strength.get("direct", 0) > 0:
                direct_evidence += 1
        evidence_ratio = direct_evidence / max(total_evidence, 1)

        diversity = min(1.0, len({s.type for s in path.steps}) / 3.0)

        path.path_length = len(path.steps)
        path.score = (
            path.aggregate_confidence * 0.4 +
            path_length_bonus * 0.15 +
            evidence_ratio * 0.3 +
            diversity * 0.15
        )

    def select_best_path(self, plans: List[ReasoningPath],
                         min_score: float = 0.3) -> Optional[ReasoningPath]:
        valid = [p for p in plans if p.score >= min_score and p.steps]
        if not valid:
            return None
        valid.sort(key=lambda p: p.score, reverse=True)
        return valid[0]

    def execute_plan(self, path: ReasoningPath) -> Dict[str, Any]:
        results = []
        total_stored = 0
        for step in path.steps:
            if step.type == "direct_query":
                pass
            elif step.type in ("inference_chain", "rule_application"):
                subj = step.conclusion.get("subject", "")
                rel = step.conclusion.get("relation", "")
                obj = step.conclusion.get("object", "")
                conf = step.confidence
                if subj and rel and obj and conf >= 0.3:
                    try:
                        self.memory.add_fact_triple(
                            subj, rel, obj,
                            source=f"planning:{step.type}",
                            confidence=conf,
                            source_type="inferred",
                            truth_confidence=conf * 0.9,
                            evidence=json.dumps(step.premise),
                        )
                        total_stored += 1
                    except Exception as e:
                        logger.debug("Failed to store triple: %s", e)

                premise = step.premise
                if "rule" in premise and isinstance(premise["rule"], dict):
                    r1 = premise["rule"].get("r1", "")
                    r2 = premise["rule"].get("r2", "")
                    if r1 and r2:
                        self._scorer.learn_chain(r1, r2, rel,
                                                 compat_boost=0.02)
                        self._tracker.record_success(r1, r2)

            elif step.type == "multi_path":
                pass

            results.append({
                "type": step.type,
                "confidence": step.confidence,
                "conclusion": step.conclusion,
                "description": step.description,
            })

        return {
            "goal": path.goal,
            "plan_score": path.score,
            "aggregate_confidence": path.aggregate_confidence,
            "steps_executed": len(results),
            "triples_stored": total_stored,
            "results": results,
        }

    def run_planning_cycle(self, goals: Optional[List[str]] = None,
                           max_goals: int = 3) -> Dict[str, Any]:
        if goals is None:
            goals = self._generate_goals_from_knowledge(max_goals)

        all_plans = []
        all_results = []
        total_stored = 0

        for goal in goals[:max_goals]:
            plans = self.generate_plans(goal)
            best = self.select_best_path(plans)
            if best:
                result = self.execute_plan(best)
                all_results.append(result)
                total_stored += result.get("triples_stored", 0)
                all_plans.append(best)

            self._plan_history.append({
                "goal": goal,
                "plans_generated": len(plans),
                "best_score": best.score if best else 0,
                "triples_stored": result.get("triples_stored", 0) if best else 0,
                "timestamp": time.time(),
            })

        evidence_stats = self.evidence_store.get_stats()
        rule_stats = self._get_rule_stats()

        return {
            "goals_processed": len(goals[:max_goals]),
            "plans_generated": sum(len(self.generate_plans(g))
                                   for g in goals[:max_goals]),
            "paths_selected": len(all_results),
            "triples_stored": total_stored,
            "evidence_entries": evidence_stats.get("total_entries", 0),
            "rule_effectiveness": rule_stats,
            "history_length": len(self._plan_history),
            "status": "ok",
        }

    def _generate_goals_from_knowledge(self,
                                        max_goals: int = 3) -> List[str]:
        goals = []
        try:
            concepts = self.memory.get_all_concepts()
            weak_concepts = []
            for c in concepts[:50]:
                rels = self._get_relationships(c)
                if len(rels) < 2:
                    weak_concepts.append(c)

            for c in weak_concepts[:max_goals]:
                goals.append(f"Explore '{c}' and its relationships")

            for i in range(len(concepts) - 1):
                if len(goals) >= max_goals:
                    break
                a, b = concepts[i], concepts[i + 1]
                paths = self._bfs_paths(a, b, max_depth=2)
                if not paths:
                    goals.append(
                        f"Find connection between '{a}' and '{b}'")

        except Exception as e:
            logger.debug("Error generating goals: %s", e)

        if not goals:
            goals = ["Explore known concepts and their relationships"]

        return goals[:max_goals]

    def _get_rule_stats(self) -> Dict[str, Any]:
        stats = {}
        for i, rule in enumerate(self.inference_rules):
            success = self._rule_success_count.get(i, 0)
            fail = self._rule_fail_count.get(i, 0)
            total = success + fail
            effectiveness = success / max(total, 1)
            stats[f"rule_{i}_{rule[0]}→{rule[2]}"] = {
                "rule": list(rule),
                "success_count": success,
                "fail_count": fail,
                "effectiveness": round(effectiveness, 3),
            }
        for i, rule in enumerate(self._learned_rules):
            stats[f"learned_rule_{i}_{rule[0]}→{rule[2]}"] = {
                "rule": list(rule),
                "success_count": 0,
                "fail_count": 0,
                "effectiveness": 1.0,
            }
        stats["scorer"] = self._scorer.get_stats()
        return stats

    def learn_new_rule(self, path: ReasoningPath) -> Optional[Tuple[str, str, str, float]]:
        if len(path.steps) < 2:
            return None

        combos = defaultdict(list)
        for i in range(len(path.steps) - 1):
            s1 = path.steps[i]
            s2 = path.steps[i + 1]
            r1 = s1.conclusion.get("relation", "")
            r2 = s2.conclusion.get("relation", "")
            if r1 and r2:
                combos[(r1, r2)].append({
                    "confidence": (s1.confidence + s2.confidence) / 2,
                })

        for (r1, r2), entries in combos.items():
            inferred_r = self._scorer.infer_relation(r1, r2)
            if not inferred_r:
                continue

            avg_conf = max(e["confidence"] for e in entries)
            new_score = round(avg_conf * 0.85, 2)

            if not self._scorer.has_chain(r1, r2):
                self._scorer.learn_chain(r1, r2, inferred_r,
                                         compat_boost=0.0)
            else:
                self._scorer.learn_chain(r1, r2, inferred_r,
                                         compat_boost=0.05)

            new_rule = (r1, r2, inferred_r, new_score)
            if new_rule not in self._learned_rules:
                self._learned_rules.append(new_rule)
                self._tracker.record_success(r1, r2)
                logger.info("Learned new rule: %s -> %s -> %s (conf=%s)",
                            r1, r2, inferred_r, new_rule[3])
                return new_rule

        return None

    def _infer_relation(self, r1: str, r2: str) -> Optional[str]:
        return self._scorer.infer_relation(r1, r2)

    def get_stats(self) -> Dict[str, Any]:
        ev_stats = self.evidence_store.get_stats()
        rule_stats = self._get_rule_stats()
        return {
            "evidence_store": ev_stats,
            "rules": rule_stats,
            "learned_rules_count": len(self._learned_rules),
            "total_rules": len(self.inference_rules) + len(self._learned_rules),
            "plan_history_length": len(self._plan_history),
            "selector": {
                "uncertainty_threshold": self._selector.uncertainty_threshold,
                "strength_threshold": self._selector.strength_threshold,
                "conflict_threshold": self._selector.conflict_threshold,
            },
            "tracker": self._tracker.get_stats(),
        }

    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self._plan_history[-limit:]

    def get_best_path_for_goal(self, goal: str) -> Optional[Dict[str, Any]]:
        plans = self.generate_plans(goal)
        best = self.select_best_path(plans)
        if best is None:
            return None
        return best.to_dict()
