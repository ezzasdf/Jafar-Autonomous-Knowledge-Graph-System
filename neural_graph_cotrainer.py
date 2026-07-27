"""
Neural + Graph Co-Training for Offline AI Book Reader
Bidirectional self-supervised loop:

  Graph → creates training data for the neural net
  Neural net → improves graph predictions (fills gaps)
  Graph → corrects the neural net via structural constraints

This is how modern hybrid AI evolves — two views teaching each other.
"""

import logging
import random
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict

import numpy as np

from pattern_recognizer import HAS_TORCH

logger = logging.getLogger(__name__)

# transitive relations — if A→B and B→C then A→C should hold
TRANSITIVE_RELATIONS = {"is_a", "located_in", "part_of", "lives_in", "contains"}

# symmetric relations — if A→B then B→A should hold
SYMMETRIC_RELATIONS = {"associated_with", "interacts_with", "related_to"}

# hierarchical relations — subtypes inherit properties
HIERARCHICAL_RELATIONS = {"is_a"}

# opposite pairs — used for contradiction detection
OPPOSITE_RELATIONS = {
    "creates": "destroys",
    "benefits_from": "harms",
    "strengthens": "weakens",
    "eats": "eaten_by",
}

# Relations that should NOT coexist for the same (subject, object)
MUTUALLY_EXCLUSIVE = {
    ("is_a", "hunts"),
    ("is_a", "eats"),
    ("is_a", "eaten_by"),
    ("hunts", "eaten_by"),
}

GRAPH_TRAIN_MIN_RELS = 5
COTRAIN_CYCLE_MIN_IMPROVEMENT = 0.001
MAX_TRAINING_PAIRS = 500


class NeuralGraphCoTrainer:
    """Co-training loop between the neural network and the knowledge graph.

    Three-phase cycle:
      1. Graph → Neural
         Extract labeled (subject, object) embeddings from graph structure.
         Positive pairs = existing strong relationships.
         Hard negatives = graph-perturbed pairs (swap subject/object, flip relations).

      2. Neural → Graph
         Run RelationPredictor over unlinked concept pairs.
         High-confidence predictions get added as weak relationships.
         The neural net fills gaps the graph hasn't captured yet.

      3. Graph → Neural
         Validate neural predictions through graph constraints:
           - Transitivity:  if A→B and B→C, does the NN agree A→C?
           - Hierarchy:     do subtypes inherit supertype relations?
           - Symmetry:      are symmetric relations consistent?
           - Mutual exclusivity: are contradictory relations avoided?
         Inconsistent predictions become hard negatives for the next training round.

    This cycle repeats, each round the graph gets denser and the NN gets more accurate.
    """

    def __init__(self, memory_system, pattern_recognizer, experience_memory=None):
        self.memory = memory_system
        self.pr = pattern_recognizer
        self.exp = experience_memory
        self._cycle_count = 0
        self._history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    #  Phase 1: Graph → Neural — extract training data from graph
    # ------------------------------------------------------------------

    def graph_to_neural_data(self, max_pairs: int = MAX_TRAINING_PAIRS
                             ) -> Dict[str, Any]:
        """Extract labeled training data from the knowledge graph.

        Returns:
            Dict with 'positive' and 'negative' lists of (embedding, label)
            and a summary of what was generated.
        """
        all_rels = self.memory.conn.execute(
            "SELECT source_concept, relation, target_concept, confidence "
            "FROM relationships"
        ).fetchall()
        if len(all_rels) < GRAPH_TRAIN_MIN_RELS:
            return {"status": "skipped", "reason": f"Need >= {GRAPH_TRAIN_MIN_RELS} relationships"}

        concepts = self.memory.get_all_concepts()
        rel_to_idx = {r: i for i, r in enumerate(self.pr.RELATION_LABELS)}

        positives: List[Tuple[np.ndarray, np.ndarray]] = []
        negatives: List[Tuple[np.ndarray, np.ndarray]] = []
        hard_negatives: List[Tuple[np.ndarray, np.ndarray]] = []

        known_pairs: Set[Tuple[str, str, str]] = set()
        strong_by_subject: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
        strong_by_object: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)

        for row in all_rels:
            s, r, o, conf = row["source_concept"], row["relation"], row["target_concept"], row["confidence"]
            known_pairs.add((s, r, o))
            if conf >= 0.7:
                strong_by_subject[s].append((r, o, conf))
                strong_by_object[o].append((s, r, conf))

        POSITIVE_RELATION_THRESHOLD = 0.6

        positive_count = 0
        for row in all_rels:
            if row["confidence"] < POSITIVE_RELATION_THRESHOLD:
                continue
            s, r, o = row["source_concept"], row["relation"], row["target_concept"]
            if r not in rel_to_idx:
                continue
            emb = self.pr._embed(f"{s} [SEP] {o} [SEP]")
            label = np.zeros(len(self.pr.RELATION_LABELS), dtype=np.float32)
            label[rel_to_idx[r]] = 1.0
            positives.append((emb, label))
            positive_count += 1
            if positive_count >= max_pairs:
                break

        negative_count = 0
        # Type A: random (subject, relation, object) not in graph
        neg_target = min(positive_count * 2, max_pairs)
        attempts = 0
        while negative_count < neg_target and attempts < 2000:
            s = random.choice(concepts)
            r = random.choice(self.pr.RELATION_LABELS)
            o = random.choice(concepts)
            if s != o and (s, r, o) not in known_pairs:
                emb = self.pr._embed(f"{s} [SEP] {o} [SEP]")
                label = np.zeros(len(self.pr.RELATION_LABELS), dtype=np.float32)
                negatives.append((emb, label))
                known_pairs.add((s, r, o))
                negative_count += 1
            attempts += 1

        # Type B: hard negatives from graph structure
        hard_count = 0
        for s in strong_by_subject:
            rels = strong_by_subject[s]
            for r, o, _ in rels[:3]:
                wrong_rel = random.choice([rr for rr in self.pr.RELATION_LABELS if rr != r])
                if (s, wrong_rel, o) not in known_pairs:
                    emb = self.pr._embed(f"{s} [SEP] {o} [SEP]")
                    label = np.zeros(len(self.pr.RELATION_LABELS), dtype=np.float32)
                    wrong_idx = rel_to_idx.get(wrong_rel)
                    if wrong_idx is not None:
                        false_positive_idx = rel_to_idx.get(r)
                        if false_positive_idx is not None:
                            label[false_positive_idx] = 0.25
                    hard_negatives.append((emb, label))
                    known_pairs.add((s, wrong_rel, o))
                    hard_count += 1
                    if hard_count >= max_pairs // 2:
                        break
            if hard_count >= max_pairs // 2:
                break

        all_negatives = negatives + hard_negatives
        random.shuffle(all_negatives)
        all_negatives = all_negatives[:max_pairs * 2]

        logger.debug(
            "graph_to_neural: %d positive, %d negative (%d hard)",
            positive_count, len(all_negatives), hard_count,
        )

        return {
            "status": "generated",
            "positive": positive_count,
            "negative": len(all_negatives),
            "hard_negative": hard_count,
            "positive_samples": positives,
            "negative_samples": all_negatives,
        }

    # ------------------------------------------------------------------
    #  Phase 2: Neural → Graph — predict missing links
    # ------------------------------------------------------------------

    def neural_to_graph_predictions(
        self, max_predictions: int = 30, threshold: float = 0.55
    ) -> Dict[str, Any]:
        """Use the neural net to predict missing relationships in the graph.

        For concept pairs not yet linked, runs RelationPredictor.
        High-confidence predictions are added as weak relationships.

        Returns:
            Dict with predictions added, skipped, and per-relation breakdown.
        """
        concepts = self.memory.get_all_concepts()
        if len(concepts) < 3:
            return {"status": "skipped", "reason": "Need >= 3 concepts"}

        existing = {
            (r["source_concept"], r["relation"], r["target_concept"])
            for r in self.memory.conn.execute(
                "SELECT source_concept, relation, target_concept FROM relationships"
            ).fetchall()
        }

        added = 0
        skipped = 0
        relation_counts: Dict[str, int] = defaultdict(int)
        added_details: List[Dict[str, Any]] = []

        logger.info("Concepts: %d, existing relationships: %d", len(concepts), len(existing))
        logger.info("Starting pair generation (neural → graph)...")

        if not hasattr(self, '_neural_skipped_pairs'):
            self._neural_skipped_pairs = set()

        # --- Sample promising concept pairs instead of O(n²) ---
        # Build a set of concepts that have relationships (more likely to yield predictions)
        linked_concepts = set()
        for r in existing:
            linked_concepts.add(r[0])
            linked_concepts.add(r[2])
        linked_list = [c for c in concepts if c in linked_concepts]
        unlinked_list = [c for c in concepts if c not in linked_concepts]
        random.shuffle(linked_list)
        random.shuffle(unlinked_list)

        # Generate candidate pairs: prefer linked↔linked, then linked↔unlinked
        candidate_pairs: List[Tuple[str, str]] = []
        max_candidates = min(max_predictions * 30, 2000)

        for s in linked_list:
            for o in linked_list:
                if s >= o:
                    continue
                if (s, o) not in self._neural_skipped_pairs:
                    candidate_pairs.append((s, o))
                if len(candidate_pairs) >= max_candidates:
                    break
            if len(candidate_pairs) >= max_candidates:
                break

        if len(candidate_pairs) < max_candidates:
            for s in linked_list:
                for o in unlinked_list:
                    if (s, o) not in self._neural_skipped_pairs:
                        candidate_pairs.append((s, o))
                    if len(candidate_pairs) >= max_candidates:
                        break
                if len(candidate_pairs) >= max_candidates:
                    break

        random.shuffle(candidate_pairs)
        examined = 0
        total_candidates = len(candidate_pairs)
        logger.info("Will examine up to %d concept pairs (sampled)", total_candidates)

        for s, o in candidate_pairs:
            if s == o:
                continue

            results = self.pr.predict_relations(s, o)
            examined += 1
            if examined % 500 == 0:
                logger.info("Pair generation progress: %d/%d examined, %d added",
                            examined, total_candidates, added)

            if not results or results[0]["confidence"] < threshold:
                self._neural_skipped_pairs.add((s, o))
                skipped += 1
                continue

            for pred in results:
                if pred["confidence"] < threshold:
                    continue
                key = (s, pred["relation"], o)
                if key in existing:
                    continue
                self.memory.add_fact_triple(
                    s, pred["relation"], o,
                    source="neural_prediction",
                    confidence=round(pred["confidence"] * 0.7, 3),
                )
                existing.add(key)
                relation_counts[pred["relation"]] += 1
                added += 1
                added_details.append({
                    "subject": s, "relation": pred["relation"],
                    "object": o, "confidence": pred["confidence"],
                })
                self._record_experience(
                    s, pred["relation"], o, "inferred",
                    0.0, pred["confidence"], "neural_to_graph"
                )
                if added >= max_predictions:
                    break
            if added >= max_predictions:
                break

        logger.debug(
            "neural_to_graph: added %d, skipped %d, examined %d pairs",
            added, skipped, examined,
        )
        return {
            "status": "completed",
            "added": added,
            "skipped": skipped,
            "examined": examined,
            "relation_breakdown": dict(relation_counts),
            "details": added_details[:10],
        }

    # ------------------------------------------------------------------
    #  Phase 3: Graph → Neural — correct via structural constraints
    # ------------------------------------------------------------------

    def graph_corrects_neural(self, max_corrections: int = 30
                              ) -> Dict[str, Any]:
        """Validate neural predictions against graph structural constraints.

        Constraint types checked:
          - Transitivity:  if A→B and B→C, does the NN predict A→C?
          - Hierarchy:     do subtypes (is_a) inherit supertype relations?
          - Symmetry:      are symmetric relations consistent?
          - Mutual exclusivity: no contradictory pairs for same (s, o)

        Inconsistent predictions become correction labels (hard negatives).

        Returns:
            Dict with corrections applied, per-constraint stats.
        """
        all_rels = self.memory.conn.execute(
            "SELECT source_concept, relation, target_concept, confidence "
            "FROM relationships WHERE confidence >= 0.65"
        ).fetchall()
        if len(all_rels) < 5:
            return {"status": "skipped", "reason": "Need >= 5 strong relationships"}

        strong_by_subject: Dict[str, List[Tuple[str, str, float]]] = defaultdict(list)
        strong_by_relation: Dict[str, List[Tuple[str, str, str, float]]] = defaultdict(list)

        for row in all_rels:
            s, r, o, conf = row["source_concept"], row["relation"], row["target_concept"], row["confidence"]
            strong_by_subject[s].append((r, o, conf))
            strong_by_relation[r].append((s, o, conf))

        corrections: List[Dict[str, Any]] = []
        constraint_stats: Dict[str, int] = defaultdict(int)
        seen_corrections: Set[Tuple[str, str, str]] = set()

        rel_to_idx = {r: i for i, r in enumerate(self.pr.RELATION_LABELS)}

        # ---- Transitivity check ----
        for rel in TRANSITIVE_RELATIONS:
            if rel not in strong_by_relation:
                continue
            chain_pairs = strong_by_relation[rel]
            random.shuffle(chain_pairs)
            for s, mid, s_conf in chain_pairs[:10]:
                if mid not in strong_by_subject:
                    continue
                for r2, o, r2_conf in strong_by_subject[mid][:5]:
                    if r2 != rel or s == o:
                        continue
                    key = (s, rel, o)
                    if key in seen_corrections:
                        continue
                    existing = self.memory.conn.execute(
                        "SELECT confidence FROM relationships "
                        "WHERE source_concept=? AND relation=? AND target_concept=?",
                        (s, rel, o),
                    ).fetchone()
                    if existing and existing["confidence"] >= 0.5:
                        continue

                    nn_results = self.pr.predict_relations(s, o)
                    nn_agrees = any(
                        p["relation"] == rel and p["confidence"] >= 0.3
                        for p in nn_results
                    )
                    if not nn_agrees and nn_results:
                        inferred_conf = min(s_conf, r2_conf) * 0.8
                        corrections.append({
                            "subject": s, "relation": rel, "object": o,
                            "type": "transitivity",
                            "chain": f"{s}→{mid}→{o}",
                            "inferred_confidence": round(inferred_conf, 3),
                            "nn_confidence": nn_results[0]["confidence"] if nn_results else 0,
                        })
                        constraint_stats["transitivity_corrections"] += 1
                        seen_corrections.add(key)
                        if len(corrections) >= max_corrections:
                            break
                if len(corrections) >= max_corrections:
                    break
            if len(corrections) >= max_corrections:
                break

        # ---- Hierarchy check (is_a inheritance) ----
        if "is_a" in strong_by_relation:
            for s, o, conf in strong_by_relation["is_a"][:15]:
                if o not in strong_by_subject:
                    continue
                for r_inherit, t, t_conf in strong_by_subject[o][:5]:
                    if r_inherit == "is_a":
                        continue
                    key = (s, r_inherit, t)
                    if key in seen_corrections:
                        continue
                    existing = self.memory.conn.execute(
                        "SELECT confidence FROM relationships "
                        "WHERE source_concept=? AND relation=? AND target_concept=?",
                        (s, r_inherit, t),
                    ).fetchone()
                    if existing:
                        continue
                    nn_results = self.pr.predict_relations(s, t)
                    nn_agrees = any(
                        p["relation"] == r_inherit and p["confidence"] >= 0.3
                        for p in nn_results
                    )
                    if not nn_agrees:
                        inferred_conf = min(conf, t_conf) * 0.75
                        corrections.append({
                            "subject": s, "relation": r_inherit, "object": t,
                            "type": "hierarchy",
                            "chain": f"{s} is_a {o}  &  {o} {r_inherit} {t}",
                            "inferred_confidence": round(inferred_conf, 3),
                            "nn_confidence": nn_results[0]["confidence"] if nn_results else 0,
                        })
                        constraint_stats["hierarchy_corrections"] += 1
                        seen_corrections.add(key)
                        if len(corrections) >= max_corrections:
                            break
                if len(corrections) >= max_corrections:
                    break

        # ---- Symmetry check ----
        for rel in SYMMETRIC_RELATIONS:
            if rel not in strong_by_relation:
                continue
            for s, o, conf in strong_by_relation[rel][:10]:
                reverse = self.memory.conn.execute(
                    "SELECT confidence FROM relationships "
                    "WHERE source_concept=? AND relation=? AND target_concept=?",
                    (o, rel, s),
                ).fetchone()
                if reverse and reverse["confidence"] >= 0.5:
                    continue
                nn_results = self.pr.predict_relations(o, s)
                nn_agrees = any(
                    p["relation"] == rel and p["confidence"] >= 0.3
                    for p in nn_results
                )
                if not nn_agrees:
                    corrections.append({
                        "subject": o, "relation": rel, "object": s,
                        "type": "symmetry",
                        "chain": f"{s} {rel} {o}  →  {o} {rel} {s}",
                        "inferred_confidence": round(conf * 0.8, 3),
                        "nn_confidence": nn_results[0]["confidence"] if nn_results else 0,
                    })
                    constraint_stats["symmetry_corrections"] += 1
                    seen_corrections.add(key := (o, rel, s))
                    if len(corrections) >= max_corrections:
                        break
            if len(corrections) >= max_corrections:
                break

        # ---- Apply corrections: add missing transitive/hierarchy edges ----
        applied = 0
        for c in corrections[:max_corrections]:
            if c["inferred_confidence"] >= 0.4:
                self.memory.add_fact_triple(
                    c["subject"], c["relation"], c["object"],
                    source=f"graph_constraint_{c['type']}",
                    confidence=c["inferred_confidence"],
                )
                self._record_experience(
                    c["subject"], c["relation"], c["object"], "verified",
                    0.0, c["inferred_confidence"],
                    f"graph_{c['type']}"
                )
                applied += 1

        logger.debug(
            "graph_corrects: %d corrections found, %d applied | stats=%s",
            len(corrections), applied, dict(constraint_stats),
        )
        return {
            "status": "completed",
            "corrections_found": len(corrections),
            "corrections_applied": applied,
            "constraint_stats": dict(constraint_stats),
            "corrections": corrections[:10],
        }

    # ------------------------------------------------------------------
    #  Full co-training cycle
    # ------------------------------------------------------------------

    def run_cotrain_cycle(self, max_pairs: int = 200,
                          max_predictions: int = 25,
                          threshold: float = 0.55,
                          max_corrections: int = 20) -> Dict[str, Any]:
        """Run one complete co-training cycle: Graph→NN → NN→Graph → Graph→NN.

        Returns:
            Dict with all three phase results and cycle metrics.
        """
        self._cycle_count += 1

        phase1 = self.graph_to_neural_data(max_pairs=max_pairs)

        if phase1["status"] == "generated":
            train_result = self._train_on_phase1_data(phase1)
        else:
            train_result = {"status": phase1["status"], "reason": phase1.get("reason")}

        phase2 = self.neural_to_graph_predictions(
            max_predictions=max_predictions, threshold=threshold
        )

        phase3 = self.graph_corrects_neural(max_corrections=max_corrections)

        cycle_result = {
            "cycle": self._cycle_count,
            "phase1_graph_to_neural": {
                "status": phase1["status"],
                "positive": phase1.get("positive", 0),
                "negative": phase1.get("negative", 0),
                "hard_negative": phase1.get("hard_negative", 0),
                "train_result": train_result,
            },
            "phase2_neural_to_graph": {
                "status": phase2["status"],
                "added": phase2.get("added", 0),
                "skipped": phase2.get("skipped", 0),
                "relation_breakdown": phase2.get("relation_breakdown", {}),
            },
            "phase3_graph_corrects_neural": {
                "status": phase3["status"],
                "corrections_found": phase3.get("corrections_found", 0),
                "corrections_applied": phase3.get("corrections_applied", 0),
                "constraint_stats": phase3.get("constraint_stats", {}),
            },
        }

        self._history.append(cycle_result)
        logger.info(
            "Co-train cycle %d: NN←%d  NN→%d  Graph←%d",
            self._cycle_count,
            phase1.get("positive", 0),
            phase2.get("added", 0),
            phase3.get("corrections_applied", 0),
        )
        return cycle_result

    def _train_on_phase1_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Train the RelationPredictor on data extracted from the graph."""
        if not HAS_TORCH or self.pr.relation_predictor is None:
            return {"status": "no_torch"}
        positives = data.get("positive_samples", [])
        negatives = data.get("negative_samples", [])
        all_samples = positives + negatives
        if len(all_samples) < 5:
            return {"status": "skipped", "reason": "Too few training samples"}

        all_embs = np.array([s[0] for s in all_samples])
        all_labels = np.array([s[1] for s in all_samples])

        self.pr.relation_predictor.train()
        optimizer = __import__('torch').optim.Adam(
            self.pr.relation_predictor.parameters(), lr=0.001
        )

        torch = __import__('torch')

        # --- Class-weighted Focal Loss for multi-label imbalance ---
        # Count positive examples per relation across the full dataset
        pos_counts = all_labels.sum(axis=0).astype(np.float32)  # shape: (num_relations,)
        pos_counts = np.maximum(pos_counts, 1.0)  # avoid division by zero
        total_pos = pos_counts.sum()
        num_rels = len(pos_counts)
        # Alpha: inverse frequency, normalized so mean(alpha) ≈ 1
        class_alpha = np.where(
            pos_counts > 0,
            total_pos / (num_rels * pos_counts),
            1.0,
        )
        # Clamp to [0.2, 5.0] to prevent extreme weights
        class_alpha = np.clip(class_alpha, 0.2, 5.0)
        alpha_tensor = torch.from_numpy(class_alpha).float()
        gamma = 2.0

        def focal_loss(pred, target):
            """α-balanced Focal Loss for multi-label classification.
               FL(p_t) = -α_t * (1-p_t)^γ * log(p_t)
            """
            bce = -(target * torch.log(pred + 1e-8) +
                    (1 - target) * torch.log(1 - pred + 1e-8))
            pt = torch.where(target > 0.5, pred, 1 - pred)
            modulating = (1 - pt) ** gamma
            # per-sample alpha: broadcast class_alpha per active label
            alpha = torch.where(target > 0.5, alpha_tensor, 1 - alpha_tensor)
            return (alpha * modulating * bce).mean()

        criterion = focal_loss

        dataset_size = len(all_embs)
        indices = np.random.permutation(dataset_size)
        split = int(dataset_size * 0.8)
        train_idx = indices[:split]
        val_idx = indices[split:]

        epochs = 15
        final_loss = 0.0
        final_acc = 0.0

        for epoch in range(epochs):
            self.pr.relation_predictor.train()
            batch_loss = 0.0
            batch_count = 0
            for i in range(0, max(len(train_idx), 1), 16):
                batch_i = train_idx[i:i + 16]
                x = torch.from_numpy(all_embs[batch_i]).float()
                y = torch.from_numpy(all_labels[batch_i]).float()
                pred = self.pr.relation_predictor(x)
                loss = criterion(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                batch_count += 1
            if batch_count:
                final_loss = batch_loss / batch_count

            self.pr.relation_predictor.eval()
            with torch.no_grad():
                x_val = torch.from_numpy(all_embs[val_idx]).float()
                y_val = torch.from_numpy(all_labels[val_idx]).float()
                pred_val = self.pr.relation_predictor(x_val)
                final_acc = ((pred_val >= 0.5) == (y_val >= 0.5)).float().mean().item()

        self.pr.relation_predictor.eval()
        self.pr._rp_trained = True
        logger.debug(
            "Co-train NN update: %d samples, val_acc=%.3f, loss=%.4f",
            len(all_samples), final_acc, final_loss,
        )
        # Log class weight distribution
        alpha_rounded = {r: round(float(a), 2)
                        for r, a in zip(self.pr.RELATION_LABELS, class_alpha)}
        max_weight = float(class_alpha.max())
        min_weight = float(class_alpha.min())
        logger.debug(
            "FocalLoss α range: %.2f–%.2f (γ=%.1f) | highest: %s | lowest: %s",
            min_weight, max_weight, gamma,
            self.pr.RELATION_LABELS[int(np.argmax(class_alpha))] if np.any(class_alpha > 0) else "–",
            self.pr.RELATION_LABELS[int(np.argmin(class_alpha))] if np.any(class_alpha > 0) else "–",
        )
        return {
            "status": "trained",
            "samples": len(all_samples),
            "positive": len(positives),
            "negative": len(negatives),
            "val_accuracy": round(final_acc, 3),
            "final_loss": round(final_loss, 4),
            "focal_gamma": gamma,
            "class_alpha_range": [round(min_weight, 2), round(max_weight, 2)],
            "class_alpha_weights": alpha_rounded,
        }

    def _record_experience(self, subject: str, relation: str, obj: str,
                           outcome: str, conf_before: float, conf_after: float,
                           source: str = "cotrain") -> None:
        if self.exp is not None:
            from experience_memory import ExperienceEvent
            self.exp.record_event(ExperienceEvent(
                subject=subject, relation=relation, object=obj,
                outcome=outcome, confidence_before=conf_before,
                confidence_after=conf_after, source=source,
            ))

    # ------------------------------------------------------------------
    #  Debug & metrics
    # ------------------------------------------------------------------

    def get_debug_state(self) -> Dict[str, Any]:
        """Return the full debug state of the co-trainer."""
        graph_stats = self.memory.get_stats() if hasattr(self.memory, 'get_stats') else {}
        return {
            "cycle_count": self._cycle_count,
            "graph": {
                "relationships": graph_stats.get("relationships", 0),
                "concepts": graph_stats.get("concepts", 0),
                "high_confidence": graph_stats.get("high_confidence_relationships", 0),
            },
            "neural": {
                "rp_trained": getattr(self.pr, '_rp_trained', False),
                "scorer_trained": getattr(self.pr, '_trained', False),
                "has_torch": HAS_TORCH,
            },
            "cycle_history": self._history[-5:] if self._history else [],
            "total_cycles": self._cycle_count,
        }

    def get_consistency_score(self) -> Dict[str, Any]:
        """Measure graph-neural consistency.

        Samples random concept pairs and checks if the graph's
        relationships agree with the neural net's top predictions.
        """
        concepts = self.memory.get_all_concepts()
        if len(concepts) < 3:
            return {"score": 0.0, "checked": 0, "consistent": 0}

        import random as rnd
        rnd.shuffle(concepts)

        consistent = 0
        checked = 0
        inconsistencies: List[Dict[str, Any]] = []

        for s in concepts[:20]:
            graph_rels = self.memory.conn.execute(
                "SELECT relation, target_concept, confidence FROM relationships "
                "WHERE source_concept=? AND confidence >= 0.5",
                (s,),
            ).fetchall()
            if not graph_rels:
                continue
            for row in graph_rels[:3]:
                o = row["target_concept"]
                nn_results = self.pr.predict_relations(s, o)
                if not nn_results:
                    continue
                top_nn_rel = nn_results[0]["relation"]
                top_nn_conf = nn_results[0]["confidence"]
                graph_rel = row["relation"]
                graph_conf = row["confidence"]

                if top_nn_rel == graph_rel and top_nn_conf >= 0.3:
                    consistent += 1
                elif top_nn_conf >= 0.5 and top_nn_rel != graph_rel:
                    inconsistencies.append({
                        "subject": s, "graph_relation": graph_rel,
                        "graph_confidence": graph_conf,
                        "nn_relation": top_nn_rel,
                        "nn_confidence": top_nn_conf,
                    })
                checked += 1

        score = round(consistent / max(checked, 1), 3)
        logger.debug("Consistency: %d/%d = %.3f", consistent, checked, score)
        return {
            "score": score,
            "checked": checked,
            "consistent": consistent,
            "inconsistencies": inconsistencies[:5],
        }

    def get_cotrain_history(self) -> List[Dict[str, Any]]:
        """Return the full cycle history for analysis."""
        return list(self._history)
