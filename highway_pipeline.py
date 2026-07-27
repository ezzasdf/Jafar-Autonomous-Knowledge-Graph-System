"""
Highway Predictive Pipeline — generator-based bi-directional path prediction
with cycle detection, adaptive pruning, and activation boosting.

Pipeline stages (each a generator):
  1. seed_source(seed)         → emit seed concept as root path
  2. bidir_expand(stream)      → BFS forward & backward, detect cycles
  3. enrich(stream)            → decorate with activation levels, boost scores
  4. prune(stream)             → drop paths below dynamic threshold
  5. streaming_topk(stream)    → bounded heap, yield best paths as discovered

Usage:
    hp = HighwayPredictivePipeline(pathways, activation_memory)
    for path in hp.predict("wizard", top_k=5, max_depth=5):
        print(path["path_str"], path["cumulative_score"])
"""

import heapq
import logging
import time
from collections import defaultdict
from typing import Any, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


class HighwayPredictivePipeline:

    def __init__(self, pathways, activation_memory=None):
        self.pathways = pathways
        self.activation = activation_memory

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def predict(
        self,
        seed: str,
        top_k: int = 5,
        max_depth: int = 5,
        min_strength: float = 0.5,
        expansion_factor: int = 3,
        activation_boost: float = 0.15,
        self_reinforce: bool = False,
    ) -> Generator[Dict[str, Any], None, None]:
        """Yield top-K highways from seed, streaming via bounded heap."""
        self._stats = {
            "paths_yielded": 0, "branches_pruned": 0, "total_expanded": 0,
            "cycles_skipped": 0, "heap_evictions": 0,
        }
        t0 = time.time()

        stream = self._seed_source(seed, min_strength)
        stream = self._bidir_expand(stream, max_depth, min_strength, expansion_factor)
        stream = self._enrich(stream, activation_boost)
        stream = self._prune(stream, min_strength)
        yield from self._streaming_topk(stream, top_k)

        elapsed = time.time() - t0
        logger.info(
            "HighwayPrediction | seed=%s depth=%d top_k=%d "
            "yielded=%d pruned=%d expanded=%d cycles=%d evict=%d in %.3fs",
            seed, max_depth, top_k,
            self._stats["paths_yielded"], self._stats["branches_pruned"],
            self._stats["total_expanded"], self._stats["cycles_skipped"],
            self._stats["heap_evictions"], elapsed,
        )

        if self_reinforce and self.pathways:
            self._reinforce_top_paths(seed, top_k)

    # ------------------------------------------------------------------
    # Stage 1 — seed source
    # ------------------------------------------------------------------

    @staticmethod
    def _seed_source(
        seed: str,
        min_strength: float,
    ) -> Generator[Dict[str, Any], None, None]:
        yield {
            "path": [{"concept": seed, "direction": "start", "strength": 1.0, "depth": 0, "frequency": 1}],
            "cumulative_score": 1.0,
            "length": 1,
            "last_concept": seed,
        }

    # ------------------------------------------------------------------
    # Stage 2 — bi-directional expansion with cycle detection
    #
    # Uses a per-level frontier dict. Each level is yielded fully before
    # expanding to the next. Cycle detection: skip edges whose target
    # concept is already in the path.
    # ------------------------------------------------------------------

    def _bidir_expand(
        self,
        stream: Generator,
        max_depth: int,
        min_strength: float,
        expansion_factor: int,
    ) -> Generator[Dict[str, Any], None, None]:
        frontier: Dict[int, List[Dict]] = defaultdict(list)
        for item in stream:
            frontier[0].append(item)

        for current_depth in range(max_depth + 1):
            level = frontier.get(current_depth, [])
            if not level:
                break

            for item in level:
                yield item
                if current_depth >= max_depth:
                    continue

                last = item["path"][-1]
                concept = last["concept"]
                visited = {s["concept"] for s in item["path"]}

                outgoing = self.pathways.get_outgoing(concept, threshold=min_strength, limit=expansion_factor)
                incoming = self.pathways.get_incoming(concept, threshold=min_strength, limit=expansion_factor)

                for edge in outgoing:
                    if edge["to_concept"] in visited:
                        self._stats["cycles_skipped"] += 1
                        continue
                    self._stats["total_expanded"] += 1
                    score = item["cumulative_score"] * (edge["strength"] / 10.0)
                    frontier[current_depth + 1].append({
                        "path": item["path"] + [{
                            "concept": edge["to_concept"],
                            "direction": "forward",
                            "strength": edge["strength"],
                            "depth": current_depth + 1,
                            "frequency": edge.get("frequency", 1),
                        }],
                        "cumulative_score": round(score, 4),
                        "length": current_depth + 2,
                        "last_concept": edge["to_concept"],
                    })

                for edge in incoming:
                    if edge["from_concept"] in visited:
                        self._stats["cycles_skipped"] += 1
                        continue
                    self._stats["total_expanded"] += 1
                    score = item["cumulative_score"] * (edge["strength"] / 10.0)
                    frontier[current_depth + 1].append({
                        "path": item["path"] + [{
                            "concept": edge["from_concept"],
                            "direction": "backward",
                            "strength": edge["strength"],
                            "depth": current_depth + 1,
                            "frequency": edge.get("frequency", 1),
                        }],
                        "cumulative_score": round(score, 4),
                        "length": current_depth + 2,
                        "last_concept": edge["from_concept"],
                    })

    # ------------------------------------------------------------------
    # Stage 3 — enrich with activation data and boost scores
    #
    # Concepts with active activation get a multiplicative boost to
    # their step score, making "hot" paths rise to the top.
    # ------------------------------------------------------------------

    def _enrich(
        self,
        stream: Generator,
        activation_boost: float = 0.15,
    ) -> Generator[Dict[str, Any], None, None]:
        has_activation = self.activation is not None

        for item in stream:
            for step in item["path"]:
                c = step["concept"]
                if has_activation:
                    act = self.activation.get_activation(c)
                    step["activation"] = round(act, 4)
                    if act > 0.01:
                        boost = 1.0 + act * activation_boost
                        step["strength"] = round(step.get("strength", 1.0) * boost, 4)
                else:
                    step["activation"] = 0.0

            item["cumulative_score"] = self._recalc_score(item)
            yield item

    @staticmethod
    def _recalc_score(item: Dict[str, Any]) -> float:
        score = 1.0
        for step in item["path"]:
            s = step.get("strength", 1.0)
            score *= s / 10.0
        return round(score, 4)

    # ------------------------------------------------------------------
    # Stage 4 — adaptive pruning
    #
    # Drops paths whose cumulative score is too low relative to seed
    # strength and path depth. Deeper paths get a higher threshold.
    # ------------------------------------------------------------------

    @staticmethod
    def _prune(
        stream: Generator,
        min_strength: float,
    ) -> Generator[Dict[str, Any], None, None]:
        for item in stream:
            depth_penalty = 1.0 / max(item["length"] - 1, 1)
            threshold = min_strength * 0.1 * depth_penalty
            if item["cumulative_score"] < threshold and item["length"] > 1:
                continue
            yield item

    # ------------------------------------------------------------------
    # Stage 5 — streaming top-K via bounded heap
    #
    # Instead of collecting all paths and sorting at the end, we
    # maintain a min-heap of size top_k. Paths better than the
    # current worst in the heap get yielded immediately.
    # ------------------------------------------------------------------

    def _streaming_topk(
        self,
        stream: Generator,
        top_k: int,
    ) -> Generator[Dict[str, Any], None, None]:
        heap: List[Tuple[float, int, Dict]] = []

        for item in stream:
            neg_score = -item["cumulative_score"]
            seq = len(heap)

            if len(heap) < top_k:
                heapq.heappush(heap, (neg_score, seq, item))
            elif neg_score > heap[0][0]:
                evicted = heapq.heappushpop(heap, (neg_score, seq, item))
                self._stats["heap_evictions"] += 1

        # Drain heap — best first (highest cumulative_score)
        # Sort by negative score: most negative = highest score
        sorted_items = sorted(heap, key=lambda x: x[0])
        for neg_score, _, item in sorted_items:
            path_names = [s["concept"] for s in item["path"]]
            directions = [s["direction"] for s in item["path"]]
            item["path_str"] = " >> ".join(path_names)
            item["directions"] = " >> ".join(directions)
            item["avg_step_strength"] = round(
                sum(s.get("strength", 0) for s in item["path"]) / max(item["length"], 1), 3
            )
            self._stats["paths_yielded"] += 1
            yield item

    # ------------------------------------------------------------------
    # Self-reinforcement: record top paths back as pathway transitions
    # ------------------------------------------------------------------

    def _reinforce_top_paths(self, seed: str, top_k: int):
        """Re-inforce top-scoring predicted paths as real transitions."""
        count = 0
        for path in self.predict(seed, top_k=top_k, max_depth=3, self_reinforce=False):
            steps = path["path"]
            for i in range(len(steps) - 1):
                frm = steps[i]["concept"]
                to = steps[i + 1]["concept"]
                boost = steps[i + 1].get("strength", 0.5) * 0.1
                self.pathways.record_transition(frm, to, activation_level=boost)
                count += 1
        if count:
            logger.info("Self-reinforced %d pathway transitions from top predictions", count)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        return dict(self._stats)

    def get_highway_summary(self, seed: str, top_k: int = 5, max_depth: int = 5) -> str:
        lines = [f"Highway Prediction from '{seed}' (depth={max_depth}, top_k={top_k})",
                 "=" * 60]
        for i, path in enumerate(self.predict(seed, top_k=top_k, max_depth=max_depth), 1):
            score = path["cumulative_score"]
            dirs = path["directions"]
            path_str = path["path_str"]
            lines.append(f"  #{i}  score={score:.4f}  [{dirs}]")
            lines.append(f"       {path_str}")
            for step in path["path"]:
                act = step.get("activation", 0)
                if act > 0.01:
                    lines.append(f"       act={act:.3f} at '{step['concept']}'")
        if len(lines) == 2:
            lines.append("  (no highways found)")
        return "\n".join(lines)
