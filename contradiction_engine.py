"""
Contradiction Resolution Engine — resolves knowledge conflicts using
multiple strategies, weights sources by trust, and tracks source reliability.

Components:
  - ContradictionEngine: orchestrates resolution with strategy selection
  - TrustScorer: per-source trust tracking (accuracy over time)
  - SourceWeighter: multi-factor source weighting
  - strategy library: majority_vote, source_trust, recency, source_type_priority
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple, Callable

logger = logging.getLogger(__name__)
debug_logger = logging.getLogger(f"{__name__}.debug")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SourceRecord:
    label: str
    source_type: Optional[str] = None
    quality: str = "0.5"
    total_facts: int = 0
    accurate_facts: int = 0
    contradicted_facts: int = 0
    last_seen: float = 0.0

    @property
    def accuracy(self) -> float:
        if self.total_facts == 0:
            return 0.5
        return self.accurate_facts / self.total_facts


@dataclass
class ResolutionStrategy:
    name: str
    weight: float = 1.0
    enabled: bool = True


@dataclass
class ContradictionGroup:
    subject: str
    relation: str
    facts: List[Dict[str, Any]]
    strategy_used: str = "unresolved"
    winner: Optional[Dict[str, Any]] = None
    losers: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ResolutionReport:
    group_subject: str
    group_relation: str
    strategy: str
    winner: Dict[str, Any]
    losers: List[Dict[str, Any]]
    confidence_delta: float
    timestamp: float = 0.0


class StrategyLibrary:
    """Library of resolution strategies. Each takes a group and returns
    (winner_idx, [loser_idxs], confidence)."""

    @staticmethod
    def majority_vote(group: List[Dict], source_weights: Dict[str, float]) -> Tuple[int, List[int], float]:
        target_votes: Dict[str, List[int]] = defaultdict(list)
        for i, f in enumerate(group):
            key = f.get("target_concept") or f.get("target") or ""
            target_votes[key].append(i)
        best_target = max(target_votes, key=lambda t: len(target_votes[t]))
        winner_idx = target_votes[best_target][0]
        confidence = len(target_votes[best_target]) / len(group)
        loser_indices = set(range(len(group))) - set(target_votes[best_target])
        return winner_idx, sorted(loser_indices), confidence

    @staticmethod
    def source_trust(group: List[Dict], source_weights: Dict[str, float]) -> Tuple[int, List[int], float]:
        best_idx = 0
        best_weight = -1.0
        for i, f in enumerate(group):
            src = f.get("book_source", "") or f.get("source", "") or f.get("target_concept", "") or ""
            w = source_weights.get(src, 0.5)
            if w > best_weight:
                best_weight = w
                best_idx = i
        confidence = best_weight
        losers = [i for i in range(len(group)) if i != best_idx]
        return best_idx, losers, confidence

    @staticmethod
    def recency(group: List[Dict], _weights: Dict[str, float]) -> Tuple[int, List[int], float]:
        best_idx = 0
        best_time = 0.0
        for i, f in enumerate(group):
            t = f.get("created_at", "")
            parsed = _parse_time(t)
            if parsed > best_time:
                best_time = parsed
                best_idx = i
        losers = [i for i in range(len(group)) if i != best_idx]
        return best_idx, losers, 0.6

    @staticmethod
    def source_type_priority(group: List[Dict], _weights: Dict[str, float]) -> Tuple[int, List[int], float]:
        priority = {"real": 5, "inferred": 4, "fiction": 2, "speculative": 1, "hypothetical": 0, "idea": 0}
        best_idx = 0
        best_prio = -1
        for i, f in enumerate(group):
            st = f.get("source_type", "inferred")
            p = priority.get(st, 3)
            if p > best_prio:
                best_prio = p
                best_idx = i
        confidence = (best_prio + 1) / 6.0
        losers = [i for i in range(len(group)) if i != best_idx]
        return best_idx, losers, confidence


def _parse_time(ts: Any) -> float:
    if ts is None:
        return 0.0
    if isinstance(ts, str):
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return dt.timestamp()
        except (ValueError, AttributeError):
            return 0.0
    if isinstance(ts, (int, float)):
        return float(ts)
    return 0.0


# ---------------------------------------------------------------------------
# TrustScorer
# ---------------------------------------------------------------------------

class TrustScorer:
    """Tracks per-source reliability. Learns from contradiction outcomes.

    When a source's facts are contradicted, its trust goes down.
    When they survive (win a conflict), its trust goes up.
    """

    def __init__(self, decay: float = 0.99, min_observations: int = 3):
        self._sources: Dict[str, SourceRecord] = {}
        self.decay = decay
        self.min_observations = min_observations

    def record_source(self, label: str, source_type: Optional[str] = None,
                      quality: str = "0.5") -> None:
        if label not in self._sources:
            self._sources[label] = SourceRecord(
                label=label, source_type=source_type, quality=quality,
                last_seen=time.time(),
            )
        else:
            self._sources[label].last_seen = time.time()

    def record_contradiction_outcome(self, source_label: str, won: bool) -> None:
        if source_label not in self._sources:
            self.record_source(source_label)
        rec = self._sources[source_label]
        rec.total_facts += 1
        if won:
            rec.accurate_facts += 1
        else:
            rec.contradicted_facts += 1

    def get_weight(self, source_label: str) -> float:
        if source_label not in self._sources:
            return 0.5
        rec = self._sources[source_label]
        if rec.total_facts < self.min_observations:
            base = float(rec.quality) if rec.quality else 0.5
            return base * 0.8 + 0.2 * rec.accuracy
        return rec.accuracy

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_sources": len(self._sources),
            "sources": {
                lbl: {
                    "accuracy": rec.accuracy,
                    "total_facts": rec.total_facts,
                    "contradicted": rec.contradicted_facts,
                    "quality": rec.quality,
                    "source_type": rec.source_type,
                }
                for lbl, rec in self._sources.items()
            },
        }

    def get_source_weights(self) -> Dict[str, float]:
        return {lbl: self.get_weight(lbl) for lbl in self._sources}


# ---------------------------------------------------------------------------
# SourceWeighter
# ---------------------------------------------------------------------------

class SourceWeighter:
    """Multi-factor source weighting engine.

    Combines: source_type weight, trust score, quality score, recency.
    """

    TYPE_WEIGHTS = {
        "real": 1.0, "inferred": 0.7, "fiction": 0.3,
        "speculative": 0.2, "hypothetical": 0.1, "idea": 0.1,
    }

    def __init__(self, trust_scorer: Optional[TrustScorer] = None):
        self.trust_scorer = trust_scorer or TrustScorer()

    def weight_fact(self, fact: Dict[str, Any]) -> float:
        source_label = fact.get("book_source", "") or fact.get("source", "") or ""
        source_type = fact.get("source_type", "inferred")

        type_w = self.TYPE_WEIGHTS.get(source_type, 0.5)
        trust_w = self.trust_scorer.get_weight(source_label) if source_label else 0.5
        quality_raw = fact.get("source_quality", 0.5)
        try:
            quality_w = float(quality_raw) if quality_raw else 0.5
        except (ValueError, TypeError):
            quality_w = 0.5

        return 0.4 * type_w + 0.35 * trust_w + 0.25 * quality_w

    def weight_group(self, group: List[Dict]) -> List[float]:
        return [self.weight_fact(f) for f in group]


# ---------------------------------------------------------------------------
# ContradictionEngine
# ---------------------------------------------------------------------------

class ContradictionEngine:
    """Main contradiction resolution engine.

    Finds conflicts, selects the best strategy, resolves, and logs reports.
    """

    def __init__(
        self,
        memory_system,
        trust_scorer: Optional[TrustScorer] = None,
        source_weighter: Optional[SourceWeighter] = None,
        strategies: Optional[List[ResolutionStrategy]] = None,
        auto_resolve: bool = True,
        demotion_factor: float = 0.2,
    ):
        self.memory = memory_system
        self.trust_scorer = trust_scorer or TrustScorer()
        self.source_weighter = source_weighter or SourceWeighter(self.trust_scorer)
        self.auto_resolve = auto_resolve
        self.demotion_factor = demotion_factor
        self.strategies = strategies or [
            ResolutionStrategy("majority_vote", weight=1.0),
            ResolutionStrategy("source_trust", weight=0.8),
            ResolutionStrategy("source_type_priority", weight=0.7),
            ResolutionStrategy("recency", weight=0.5),
        ]
        self._strategy_map = {
            "majority_vote": StrategyLibrary.majority_vote,
            "source_trust": StrategyLibrary.source_trust,
            "source_type_priority": StrategyLibrary.source_type_priority,
            "recency": StrategyLibrary.recency,
        }
        self._resolution_history: List[ResolutionReport] = []

    def find_conflicts(self) -> List[ContradictionGroup]:
        cursor = self.memory.conn.cursor()
        cursor.execute("""
            SELECT id, source_concept, relation, target_concept,
                   source_type, source_quality, truth_confidence,
                   epistemic_status, source AS book_source, created_at
            FROM relationships
            WHERE target_concept IS NOT NULL
            ORDER BY source_concept, relation, target_concept
        """)
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description]
        facts = [dict(zip(cols, row)) for row in rows]

        groups_map: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
        for f in facts:
            groups_map[(f["source_concept"], f["relation"])].append(f)

        groups = []
        for key, fact_list in groups_map.items():
            targets = set(f["target_concept"] for f in fact_list)
            if len(targets) > 1:
                groups.append(ContradictionGroup(
                    subject=key[0], relation=key[1], facts=fact_list,
                ))

        return groups

    def _select_strategy(self, group: ContradictionGroup) -> str:
        active = [s for s in self.strategies if s.enabled]
        if not active:
            return "majority_vote"

        priorities = {"majority_vote": 0, "source_trust": 1,
                      "source_type_priority": 2, "recency": 3}
        n_facts = len(group.facts)
        used_sources = set(f.get("book_source", "") or f.get("source", "") for f in group.facts)
        has_source_info = any(s for s in used_sources)
        has_type_diff = len(set(f.get("source_type", "") for f in group.facts)) > 1

        for s in sorted(active, key=lambda x: priorities.get(x.name, 99)):
            if s.name == "source_trust" and not has_source_info:
                continue
            if s.name == "source_type_priority" and not has_type_diff:
                continue
            if s.name == "recency" and n_facts < 2:
                continue
            return s.name

        return active[0].name

    def resolve_group(self, group: ContradictionGroup, dry_run: bool = False) -> Optional[ResolutionReport]:
        if len(group.facts) < 2:
            return None

        strategy_name = self._select_strategy(group)
        group.strategy_used = strategy_name

        strategy_fn = self._strategy_map.get(strategy_name)
        if strategy_fn is None:
            strategy_fn = StrategyLibrary.majority_vote

        source_weights = self.source_weighter.trust_scorer.get_source_weights()
        winner_idx, loser_idxs, confidence = strategy_fn(group.facts, source_weights)

        winner = group.facts[winner_idx]
        losers = [group.facts[i] for i in loser_idxs]
        group.winner = winner
        group.losers = losers

        old_winner_tc = winner.get("truth_confidence") or 0.5
        new_winner_tc = max(old_winner_tc, confidence)
        winner_delta = new_winner_tc - old_winner_tc

        if not dry_run and self.auto_resolve:
            self._apply_resolution(winner, losers, new_winner_tc)

        winner_source = winner.get("book_source", "") or winner.get("source", "") or ""
        if winner_source:
            self.trust_scorer.record_contradiction_outcome(winner_source, won=True)
        for los in losers:
            los_source = los.get("book_source", "") or los.get("source", "") or ""
            if los_source:
                self.trust_scorer.record_contradiction_outcome(los_source, won=False)

        report = ResolutionReport(
            group_subject=group.subject,
            group_relation=group.relation,
            strategy=strategy_name,
            winner=winner,
            losers=losers,
            confidence_delta=winner_delta,
            timestamp=time.time(),
        )
        self._resolution_history.append(report)
        return report

    def _apply_resolution(self, winner: Dict, losers: List[Dict], new_tc: float) -> None:
        cursor = self.memory.conn.cursor()
        cursor.execute(
            "UPDATE relationships SET truth_confidence = ? WHERE id = ?",
            (round(new_tc, 4), winner["id"]),
        )
        new_status = ("knowledge" if new_tc >= 0.85 else
                      "established" if new_tc >= 0.65 else
                      "plausible" if new_tc >= 0.45 else "speculative")
        cursor.execute(
            "UPDATE relationships SET epistemic_status = ? WHERE id = ?",
            (new_status, winner["id"]),
        )
        for los in losers:
            old_tc = los.get("truth_confidence") or 0.5
            new_loser_tc = old_tc * self.demotion_factor
            cursor.execute(
                "UPDATE relationships SET truth_confidence = ?, epistemic_status = 'contradicted' WHERE id = ?",
                (round(new_loser_tc, 4), los["id"]),
            )
        self.memory.conn.commit()

    def resolve_all(self, dry_run: bool = False) -> List[ResolutionReport]:
        groups = self.find_conflicts()
        reports = []
        for g in groups:
            report = self.resolve_group(g, dry_run=dry_run)
            if report:
                reports.append(report)
        return reports

    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        recent = self._resolution_history[-limit:]
        return [{
            "subject": r.group_subject,
            "relation": r.group_relation,
            "strategy": r.strategy,
            "winner_target": r.winner.get("target_concept", ""),
            "winner_truth": r.winner.get("truth_confidence", 0),
            "losers": [{"target": l.get("target_concept", ""),
                        "truth": l.get("truth_confidence", 0)} for l in r.losers],
            "confidence_delta": r.confidence_delta,
        } for r in recent]

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_resolutions": len(self._resolution_history),
            "strategy_usage": dict(self._strategy_counts()),
            "trust_stats": self.trust_scorer.get_stats(),
        }

    def _strategy_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for r in self._resolution_history:
            counts[r.strategy] += 1
        return dict(counts)
