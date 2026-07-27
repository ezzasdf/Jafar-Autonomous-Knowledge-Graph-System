"""
Active Memory System — activation spreading across the knowledge graph.

Concepts have activation levels that spread to neighbors like neural
excitation, with decay over distance and over time.

Components:
  - ConceptNode: node-level activation state
  - ActivationMemory: manages activation across the graph
  - ActivationDebugger: trace, visualize, snapshot activation
"""

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple

logger = logging.getLogger(__name__)
debug_logger = logging.getLogger(f"{__name__}.debug")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ActivationState:
    concept: str
    activation: float = 0.0
    baseline: float = 0.0
    last_updated: float = 0.0
    source_activation: float = 0.0
    decay_rate: float = 0.1
    spread_decay: float = 0.5

    def tick(self, now: float, global_decay: float = 0.05) -> float:
        elapsed = now - self.last_updated if self.last_updated > 0 else 0
        if elapsed > 0:
            decay_amount = global_decay * elapsed
            self.activation = max(self.baseline, self.activation - decay_amount)
        self.last_updated = now
        return self.activation


@dataclass(slots=True)
class ActivationTrace:
    seed: str
    hops: int
    nodes_activated: int
    path: List[str]
    final_nodes: Dict[str, float]
    elapsed_seconds: float
    spread_paths: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class SpreadingEvent:
    from_concept: str
    to_concept: str
    relation: str
    activation_delta: float
    hop: int


# ---------------------------------------------------------------------------
# ActivationMemory
# ---------------------------------------------------------------------------

class ActivationMemory:
    """Manages activation levels across the concept graph.

    Usage:
        am = ActivationMemory(memory_system)
        am.activate("wizard")
        am.spread(decay=0.5, max_hops=3)
        hot = am.get_active_subgraph(threshold=0.3)
    """

    def __init__(
        self,
        memory_system,
        default_decay: float = 0.05,
        spread_attenuation: float = 0.5,
        max_hops: int = 4,
        min_activation: float = 0.01,
        memory_pathways=None,
    ):
        self.memory = memory_system
        self.default_decay = default_decay
        self.spread_attenuation = spread_attenuation
        self.max_hops = max_hops
        self.min_activation = min_activation

        self._states: Dict[str, ActivationState] = {}
        self._spread_history: List[SpreadingEvent] = []
        self._traces: List[ActivationTrace] = []
        self._last_tick: float = time.time()
        self._pathways = memory_pathways  # Optional MemoryPathways integration

    def activate(self, concept: str, amount: float = 1.0) -> float:
        concept = concept.lower().strip()
        if concept not in self._states:
            self._states[concept] = ActivationState(
                concept=concept, last_updated=time.time(),
            )
        state = self._states[concept]
        state.activation = min(1.0, state.activation + amount)
        state.source_activation = amount
        state.last_updated = time.time()
        debug_logger.debug("Activate: %s += %.2f = %.4f", concept, amount, state.activation)
        return state.activation

    def dampen(self, concept: str, amount: float = 0.5) -> float:
        concept = concept.lower().strip()
        if concept not in self._states:
            return 0.0
        state = self._states[concept]
        state.activation = max(0.0, state.activation - amount)
        return state.activation

    def get_activation(self, concept: str) -> float:
        concept = concept.lower().strip()
        if concept not in self._states:
            return 0.0
        self._tick_concept(concept)
        return self._states[concept].activation

    def _tick_concept(self, concept: str) -> None:
        state = self._states.get(concept)
        if state:
            now = time.time()
            state.tick(now, self.default_decay)

    def _get_neighbors(self, concept: str) -> List[Tuple[str, str, float]]:
        neighbors = []
        cursor = self.memory.conn.cursor()
        cursor.execute(
            "SELECT relation, target_concept, truth_confidence FROM relationships WHERE source_concept = ?",
            (concept,),
        )
        for row in cursor.fetchall():
            rel, target, tc = row
            w = tc if tc is not None else 0.5
            neighbors.append((target, rel, w))

        cursor.execute(
            "SELECT relation, source_concept, truth_confidence FROM relationships WHERE target_concept = ?",
            (concept,),
        )
        for row in cursor.fetchall():
            rel, source, tc = row
            w = tc if tc is not None else 0.5
            neighbors.append((source, rel + "_inv", w))

        return neighbors

    def spread(
        self,
        seed: Optional[str] = None,
        decay: Optional[float] = None,
        max_hops: Optional[int] = None,
    ) -> ActivationTrace:
        t0 = time.time()
        decay = decay if decay is not None else self.spread_attenuation
        max_hops = max_hops if max_hops is not None else self.max_hops

        if seed is not None:
            self.activate(seed, 1.0)

        seeds = [c for c, s in self._states.items() if s.activation > self.min_activation]
        if not seeds:
            return ActivationTrace(seed="", hops=0, nodes_activated=0, path=[],
                                   final_nodes={}, elapsed_seconds=0)

        activated: Dict[str, float] = {}
        visited: Set[str] = set()
        events: List[SpreadingEvent] = []
        paths: List[Dict[str, Any]] = []

        for s in seeds:
            if s in visited:
                continue
            activated[s] = self._states[s].activation
            visited.add(s)

        queue = [(c, 0) for c in seeds]

        while queue:
            current, hop = queue.pop(0)
            if hop >= max_hops:
                continue

            current_act = activated.get(current, 0.0)
            if current_act <= self.min_activation:
                continue

            neighbors = self._get_neighbors(current)
            for neighbor, relation, weight in neighbors:
                delta = current_act * decay * weight
                if neighbor not in activated:
                    activated[neighbor] = 0.0
                activated[neighbor] += delta

                events.append(SpreadingEvent(
                    from_concept=current, to_concept=neighbor,
                    relation=relation, activation_delta=delta, hop=hop + 1,
                ))

                if neighbor not in visited:
                    visited.add(neighbor)
                    paths.append({
                        "from": current, "to": neighbor, "via": relation,
                        "delta": round(delta, 4), "hop": hop + 1,
                    })
                    queue.append((neighbor, hop + 1))

        for conc, act in activated.items():
            if conc in self._states:
                existing = self._states[conc].activation
                self._states[conc].activation = max(existing, act)
                self._states[conc].last_updated = time.time()
            else:
                self._states[conc] = ActivationState(
                    concept=conc, activation=act, last_updated=time.time(),
                )

        self._spread_history.extend(events)

        if self._pathways and paths:
            self._pathways.record_spread_paths(paths)

        trace = ActivationTrace(
            seed=seeds[0] if seeds else "",
            hops=max_hops,
            nodes_activated=len(activated),
            path=list(activated.keys()),
            final_nodes={k: round(v, 4) for k, v in sorted(
                activated.items(), key=lambda x: -x[1],
            )},
            elapsed_seconds=round(time.time() - t0, 4),
            spread_paths=paths,
        )
        self._traces.append(trace)
        return trace

    def get_active_subgraph(self, threshold: float = 0.1) -> List[Dict[str, Any]]:
        now = time.time()
        nodes = []
        for conc, state in self._states.items():
            state.tick(now, self.default_decay)
            if state.activation >= threshold:
                nodes.append({
                    "concept": conc,
                    "activation": round(state.activation, 4),
                    "baseline": round(state.baseline, 4),
                })
        return sorted(nodes, key=lambda x: -x["activation"])

    def get_activated_relationships(self, threshold: float = 0.1) -> List[Dict[str, Any]]:
        active = {n["concept"] for n in self.get_active_subgraph(threshold)}
        if not active:
            return []
        cursor = self.memory.conn.cursor()
        placeholders = ",".join("?" for _ in active)
        cursor.execute(f"""
            SELECT source_concept, relation, target_concept, truth_confidence
            FROM relationships
            WHERE source_concept IN ({placeholders})
               OR target_concept IN ({placeholders})
        """, list(active) + list(active))
        cols = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        results = []
        for row in rows:
            r = dict(zip(cols, row))
            src_act = self._states.get(r["source_concept"], ActivationState("")).activation
            tgt_act = self._states.get(r["target_concept"], ActivationState("")).activation
            r["activation_score"] = round((src_act + tgt_act) / 2, 4)
            results.append(r)
        return sorted(results, key=lambda x: -x["activation_score"])

    def tick_all(self) -> int:
        now = time.time()
        count = 0
        for state in self._states.values():
            before = state.activation
            state.tick(now, self.default_decay)
            if before > self.min_activation and state.activation <= self.min_activation:
                count += 1
        self._last_tick = now
        return count

    def reset(self, concept: Optional[str] = None) -> None:
        if concept is None:
            self._states.clear()
            self._spread_history.clear()
            self._traces.clear()
        else:
            concept = concept.lower().strip()
            if concept in self._states:
                del self._states[concept]
            self._spread_history = [e for e in self._spread_history
                                    if e.from_concept != concept and e.to_concept != concept]

    def set_baseline(self, concept: str, baseline: float) -> None:
        concept = concept.lower().strip()
        if concept not in self._states:
            self._states[concept] = ActivationState(
                concept=concept, last_updated=time.time(),
            )
        self._states[concept].baseline = max(0.0, min(1.0, baseline))

    def get_stats(self) -> Dict[str, Any]:
        active = self.get_active_subgraph(threshold=self.min_activation)
        activations = [n["activation"] for n in active]
        stats = {
            "total_nodes_tracked": len(self._states),
            "active_nodes": len(active),
            "max_activation": max(activations) if activations else 0.0,
            "mean_activation": sum(activations) / len(activations) if activations else 0.0,
            "spread_events": len(self._spread_history),
            "spread_traces": len(self._traces),
            "recent_trace": self._traces[-1].final_nodes if self._traces else {},
        }
        if self._pathways:
            stats["pathways"] = self._pathways.get_stats()
        return stats

    def sigmoid(self, x: float, midpoint: float = 0.5, steepness: float = 5.0) -> float:
        return 1.0 / (1.0 + math.exp(-steepness * (x - midpoint)))


# ---------------------------------------------------------------------------
# ActivationDebugger
# ---------------------------------------------------------------------------

class ActivationDebugger:
    """Debugging and visualization tools for activation memory."""

    def __init__(self, activation_memory: ActivationMemory):
        self.am = activation_memory

    def trace_activation(self, seed: str, hops: int = 3) -> Dict[str, Any]:
        before = self.am.get_active_subgraph(threshold=0.0)
        trace = self.am.spread(seed=seed, max_hops=hops)
        after = self.am.get_active_subgraph(threshold=0.0)
        return {
            "trace": {
                "seed": trace.seed,
                "hops": trace.hops,
                "nodes_activated": trace.nodes_activated,
                "final_nodes": trace.final_nodes,
                "elapsed_seconds": trace.elapsed_seconds,
                "spread_paths": trace.spread_paths,
            },
            "before": {n["concept"]: n["activation"] for n in before},
            "after": {n["concept"]: n["activation"] for n in after},
            "diff": {
                n["concept"]: n["activation"] - before_dict.get(n["concept"], 0)
                for n in after
            } if (before_dict := {n["concept"]: n["activation"] for n in before}) else {},
        }

    def print_activation_map(self, threshold: float = 0.05) -> str:
        active = self.am.get_active_subgraph(threshold=threshold)
        if not active:
            return "(no active nodes)"

        lines = ["Activation Map:", "-" * 40]
        for n in active:
            bar_len = int(n["activation"] * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            baseline_marker = " [B]" if n["baseline"] > 0 else ""
            lines.append(f"  {n['concept']:<20} {bar} {n['activation']:.3f}{baseline_marker}")
        return "\n".join(lines)

    def print_spread_tree(self, seed: str, hops: int = 3) -> str:
        trace = self.am.spread(seed=seed, max_hops=hops)
        lines = [f"Spread Tree from '{trace.seed}':", "-" * 40]
        paths_by_hop: Dict[int, List[Dict]] = defaultdict(list)
        for p in trace.spread_paths:
            paths_by_hop[p["hop"]].append(p)

        for hop in sorted(paths_by_hop.keys()):
            lines.append(f"\n  Hop {hop}:")
            for p in paths_by_hop[hop]:
                lines.append(f"    {p['from']} --[{p['via']}]--> {p['to']}  (+{p['delta']})")
        lines.append(f"\n  Total nodes reached: {trace.nodes_activated}")
        return "\n".join(lines)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "timestamp": time.time(),
            "active_nodes": self.am.get_active_subgraph(threshold=self.am.min_activation),
            "stats": self.am.get_stats(),
            "recent_spread_events": [
                {
                    "from": e.from_concept, "to": e.to_concept,
                    "via": e.relation, "delta": round(e.activation_delta, 4),
                    "hop": e.hop,
                }
                for e in self._recent_events(10)
            ],
        }

    def _recent_events(self, n: int) -> List[SpreadingEvent]:
        return self.am._spread_history[-n:] if self.am._spread_history else []
