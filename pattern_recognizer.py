"""
Pattern Recognizer for Offline AI Book Reader
Neural network-based pattern recognition for improved relationship extraction and scoring.

Text / triples / questions -> Neural Network -> Embeddings / classification / scoring

What it does:
  - predicts relationships regex might miss (higher recall)
  - scores confidence better than rule-based heuristics
  - detects semantic similarity between concepts
  - improves inference quality via neural feature extraction
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import re
import os
import json


LIGATURE_MAP = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi",
    "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st",
    "\u00df": "ss", "\u00c6": "AE", "\u00e6": "ae",
    "\u0152": "OE", "\u0153": "oe",
}


def normalize_ligatures(text: str) -> str:
    """Replace Unicode ligatures with their ASCII equivalents for regex matching."""
    for lig, replacement in LIGATURE_MAP.items():
        text = text.replace(lig, replacement)
    return text

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("torch not available — pattern recognizer will use embedding similarity only")


class RelationScorer(nn.Module):
    """Small MLP: embedding -> confidence score for candidate triples."""

    def __init__(self, input_dim: int = 384, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = torch.sigmoid(self.fc3(x))
        return x


class RelationPredictor(nn.Module):
    """Multi-label relation classifier.
    Takes (subject, object, context) embedding -> scores for ALL known relation types.
    Much more powerful than regex — learns which relations are likely between
    any two concepts based on semantic context.

    Input: embedding of "[CLS] subject [SEP] object [SEP] context [SEP]"
    Output: sigmoid scores per relation type (0..1)
    """

    def __init__(self, input_dim: int = 384, hidden_dim: int = 256,
                 num_relations: int = 18):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_relations)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = torch.sigmoid(self.fc3(x))
        return x


class PatternRecognizer:
    """Neural pattern recognizer for relationship extraction and scoring.

    Uses a small PyTorch MLP on top of sentence embeddings to:
      1. Score candidate triples (valid vs invalid)
      2. Infer relationship type from text context
      3. Find semantically similar concepts
      4. Train on existing relationships for self-improvement
    """

    RELATION_LABELS = [
        "is_a", "has", "can", "uses", "seeks", "creates", "rejects",
        "depends_on", "eats", "hunts", "lives_in", "contains", "produces",
        "requires", "controls", "benefits_from", "causes", "prevents",
    ]

    # Verb patterns for candidate generation (beyond regex)
    CANDIDATE_PATTERNS = [
        (r"(\w+(?:\s+\w+)?)\s+(?:are|is)\s+(?:a|an|the)?\s*(\w+(?:\s+\w+)?)", "is_a"),
        (r"(\w+(?:\s+\w+)?)\s+(?:has|have|possesses?)\s+(\w+(?:\s+\w+)?)", "has"),
        (r"(\w+(?:\s+\w+)?)\s+(?:uses?|employs?|applies?)\s+(\w+(?:\s+\w+)?)", "uses"),
        (r"(\w+(?:\s+\w+)?)\s+(?:controls?|rules?|governs?|manages?)\s+(\w+(?:\s+\w+)?)", "controls"),
        (r"(\w+(?:\s+\w+)?)\s+(?:seeks?|desires?|wants?|pursues?)\s+(\w+(?:\s+\w+)?)", "seeks"),
        (r"(\w+(?:\s+\w+)?)\s+(?:creates?|builds?|makes?|produces?)\s+(\w+(?:\s+\w+)?)", "creates"),
        (r"(\w+(?:\s+\w+)?)\s+(?:rejects?|avoids?|fears?|hates?)\s+(\w+(?:\s+\w+)?)", "rejects"),
        (r"(\w+(?:\s+\w+)?)\s+(?:depends?\s+on|relies?\s+on)\s+(\w+(?:\s+\w+)?)", "depends_on"),
        (r"(\w+(?:\s+\w+)?)\s+(?:can|could|will)\s+(\w+(?:\s+\w+)?)", "can"),
        (r"(\w+(?:\s+\w+)?)\s+(?:hunts?|chases?|catches?|captures?)\s+(\w+(?:\s+\w+)?)", "hunts"),
        (r"(\w+(?:\s+\w+)?)\s+(?:lives?\s+in|lives?\s+on|grows?\s+in|found\s+in)\s+(\w+(?:\s+\w+)?)", "lives_in"),
        (r"(\w+(?:\s+\w+)?)\s+(?:contains?|includes?|consists?\s+of)\s+(\w+(?:\s+\w+)?)", "contains"),
        (r"(\w+(?:\s+\w+)?)\s+(?:produces?|generates?|causes?|leads?\s+to)\s+(\w+(?:\s+\w+)?)", "produces"),
        (r"(\w+(?:\s+\w+)?)\s+(?:requires?|needs?|demands?)\s+(\w+(?:\s+\w+)?)", "requires"),
        (r"(\w+(?:\s+\w+)?)\s+(?:benefits?\s+from|profits?\s+from|gains?\s+from)\s+(\w+(?:\s+\w+)?)", "benefits_from"),
        (r"(\w+(?:\s+\w+)?)\s+(?:eat|consume|feed\s+on)\s+(\w+(?:\s+\w+)?)", "eats"),
    ]

    def __init__(self, embedding_generator, memory_system=None):
        self.embedder = embedding_generator
        self.memory = memory_system
        self.model: Optional[RelationScorer] = None
        self._trained = False
        self._device = "cpu"
        self.relation_predictor: Optional[RelationPredictor] = None
        self._rp_trained = False
        self._feedback_negatives: List[Tuple[str, str, str, float]] = []
        self._noise_filter_stats: Dict[str, Any] = {
            "total_checked": 0, "total_removed": 0, "last_threshold": 0.3
        }

        if HAS_TORCH:
            self.model = RelationScorer()
            self.model.eval()
            num_rels = len(self.RELATION_LABELS)
            self.relation_predictor = RelationPredictor(
                num_relations=num_rels
            )
            self.relation_predictor.eval()
            logger.debug("RelationPredictor initialized — %d relation types", num_rels)
        else:
            logger.debug("PyTorch unavailable — RelationPredictor disabled")

    def _embed(self, text: str) -> np.ndarray:
        logger.debug("Embedding text (%d chars): %.60s...", len(text), text.strip())
        return self.embedder.generate_single_text_embedding(text)

    def _score_with_nn(self, emb: np.ndarray) -> float:
        if self.model is None or not HAS_TORCH:
            logger.debug("NN scorer unavailable — returning default 0.5")
            return 0.5
        with torch.no_grad():
            t = torch.from_numpy(emb).float().unsqueeze(0)
            out = self.model(t).item()
        logger.debug("NN score: %.4f", out)
        return float(out)

    def _generate_candidates(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract candidate triples from text using regex patterns."""
        text = normalize_ligatures(text)
        candidates: List[Tuple[str, str, str]] = []
        seen = set()
        for pat, rel in self.CANDIDATE_PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                key = (m.group(1).lower().strip(), rel, m.group(2).lower().strip())
                if key not in seen:
                    seen.add(key)
                    candidates.append(key)
        logger.debug("Generated %d candidate triples from text", len(candidates))
        return candidates

    def predict_relationships(self, text: str,
                              threshold: float = 0.25) -> List[Dict[str, Any]]:
        """Predict relationships from text using NN-scored candidates.

        Returns triples whose neural confidence exceeds threshold.
        Lower threshold = higher recall (catches what regex alone might miss
        by scoring candidates on semantic similarity to known patterns).
        """
        logger.debug("predict_relationships called — text length=%d, threshold=%.2f",
                      len(text), threshold)
        sentences = re.split(r'[.!?]+', text)
        results = []
        total_candidates = 0

        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10:
                continue
            sent_emb = self._embed(sent)
            candidates = self._generate_candidates(sent)
            total_candidates += len(candidates)

            for subj, rel, obj in candidates:
                nn_score = self._score_with_nn(sent_emb)
                combined = 0.6 * nn_score + 0.4 * self._rule_confidence(sent, subj, rel, obj)
                logger.debug("Triple: %s -> [%s] -> %s | nn=%.3f rule=%.3f combined=%.3f",
                              subj, rel, obj, nn_score,
                              self._rule_confidence(sent, subj, rel, obj), combined)

                if combined >= threshold:
                    results.append({
                        "subject": subj,
                        "relation": rel,
                        "object": obj,
                        "confidence": round(combined, 3),
                        "nn_score": round(nn_score, 3),
                        "source_snippet": sent[:120],
                    })

        # Deduplicate by (subj, rel, obj)
        seen = set()
        deduped = []
        for r in results:
            key = (r["subject"], r["relation"], r["object"])
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        logger.debug("predict_relationships: %d candidates -> %d after threshold+dedup",
                      total_candidates, len(deduped))
        return deduped

    def filter_noisy_triples(self, triples: List[Dict[str, Any]],
                              threshold: float = 0.3,
                              return_debug: bool = False) -> List[Dict[str, Any]]:
        """Filter noisy/implausible triples using the neural scorer.

        Each triple is re-scored by the neural model in context.
        Triples scoring below threshold are likely regex false positives
        and are removed. Updates internal noise filter stats.

        Args:
            triples: List of triple dicts with subject, relation, object keys
            threshold: Minimum combined score to keep a triple (default 0.3)
            return_debug: If True, also return (filtered, debug_info)

        Returns:
            Filtered list of triples, or (filtered, debug_info) if return_debug
        """
        logger.debug("filter_noisy_triples: %d triples, threshold=%.2f",
                      len(triples), threshold)
        kept: List[Dict[str, Any]] = []
        removed: List[Dict[str, Any]] = []
        for t in triples:
            ctx = t.get("source_snippet",
                        f"{t['subject']} {t['relation']} {t['object']}")
            emb = self._embed(ctx)
            nn_score = self._score_with_nn(emb)
            rule_score = self._rule_confidence(
                ctx, t["subject"], t["relation"], t["object"]
            )
            combined = 0.6 * nn_score + 0.4 * rule_score
            logger.debug("Noise check: %s -> [%s] -> %s | nn=%.3f rule=%.3f combined=%.3f",
                          t["subject"], t["relation"], t["object"],
                          nn_score, rule_score, combined)
            if combined >= threshold:
                kept.append(t)
            else:
                removed.append({
                    "triple": t, "combined_score": round(combined, 3),
                    "nn_score": round(nn_score, 3), "reason": "below_threshold"
                })

        self._noise_filter_stats["total_checked"] += len(triples)
        self._noise_filter_stats["total_removed"] += len(removed)
        self._noise_filter_stats["last_threshold"] = threshold
        logger.debug("filter_noisy_triples: %d kept, %d removed (%.1f%%)",
                      len(kept), len(removed),
                      len(removed) / max(len(triples), 1) * 100)
        if return_debug:
            return kept, {
                "total": len(triples),
                "kept": len(kept),
                "removed": len(removed),
                "removal_rate": round(len(removed) / max(len(triples), 1), 3),
                "removed_details": removed[:10],
                "cumulative_stats": dict(self._noise_filter_stats),
            }
        return kept

    def get_noise_debug(self) -> Dict[str, Any]:
        """Return cumulative noise filter statistics for debugging."""
        return dict(self._noise_filter_stats)

    def score_confidence(self, text: str, subject: str,
                         relation: str, obj: str) -> float:
        """Neural confidence score for an existing triple in context."""
        emb = self._embed(text)
        nn_score = self._score_with_nn(emb)
        rule_score = self._rule_confidence(text, subject, relation, obj)
        final = round(0.6 * nn_score + 0.4 * rule_score, 3)
        logger.debug("score_confidence: %s -> [%s] -> %s = %.3f", subject, relation, obj, final)
        return final

    def find_similar_concepts(self, concept: str,
                              top_k: int = 10) -> List[Dict[str, Any]]:
        """Find semantically similar concepts using embedding cosine similarity."""
        logger.debug("find_similar_concepts: '%s' top_k=%d", concept, top_k)
        concept_emb = self._embed(concept)
        if self.memory is None:
            logger.debug("No memory system — returning empty similarity list")
            return []
        all_concepts = self.memory.get_all_concepts()
        logger.debug("Comparing against %d stored concepts", len(all_concepts))
        scored: List[Dict[str, Any]] = []
        for c in all_concepts:
            if c.lower().strip() == concept.lower().strip():
                continue
            c_emb = self._embed(c)
            sim = float(np.dot(concept_emb, c_emb) / (
                np.linalg.norm(concept_emb) * np.linalg.norm(c_emb) + 1e-8
            ))
            scored.append({"concept": c, "similarity": sim})
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        logger.debug("Top result: '%s' (sim=%.4f)",
                      scored[0]["concept"] if scored else "NONE",
                      scored[0]["similarity"] if scored else 0)
        return scored[:top_k]

    def find_similar_relationships(self, relation: str,
                                   top_k: int = 10) -> List[Dict[str, Any]]:
        """Find relationships whose context embeddings are similar."""
        rel_emb = self._embed(relation)
        if self.memory is None:
            return []
        rels = self.memory.get_relationships(relation) if hasattr(
            self.memory, "get_relationships") else []
        scored = []
        for r in rels:
            ctx = f"{r['source_concept']} {r['relation']} {r['target_concept']}"
            r_emb = self._embed(ctx)
            sim = float(np.dot(rel_emb, r_emb) / (
                np.linalg.norm(rel_emb) * np.linalg.norm(r_emb) + 1e-8
            ))
            scored.append({**r, "similarity": sim})
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]

    def train_from_memory(self, epochs: int = 30,
                          lr: float = 0.001) -> Dict[str, Any]:
        """Train the neural scorer on existing relationships vs random negatives.

        Positive examples: existing relationships from the DB
        Negative examples: random subject-relation-object pairs not in DB
        """
        if self.model is None or not HAS_TORCH:
            logger.debug("train_from_memory skipped — PyTorch unavailable")
            return {"status": "no_torch", "error": "PyTorch not available"}
        if self.memory is None:
            logger.debug("train_from_memory skipped — no memory system")
            return {"status": "no_memory", "error": "No memory system"}

        all_rels = self.memory.conn.execute(
            "SELECT source_concept, relation, target_concept FROM relationships"
        ).fetchall()
        logger.debug("train_from_memory: %d existing relationships found", len(all_rels))

        if len(all_rels) < 5:
            logger.debug("train_from_memory skipped — only %d relationships", len(all_rels))
            return {"status": "skipped", "error": f"Only {len(all_rels)} relationships, need >= 5"}

        all_concepts = self.memory.get_all_concepts()
        logger.debug("train_from_memory: %d concepts available", len(all_concepts))

        features, labels = [], []

        # Positive: embed context of existing relationships
        for row in all_rels:
            ctx = f"{row['source_concept']} {row['relation']} {row['target_concept']}"
            emb = self._embed(ctx)
            features.append(emb)
            labels.append(1.0)
        logger.debug("Generated %d positive samples", len(all_rels))

        # Negative: random false triples
        import random
        known = {(r["source_concept"], r["relation"], r["target_concept"]) for r in all_rels}
        neg_target = min(len(all_rels), 200)
        attempts = 0
        while len([l for l in labels if l == 0.0]) < neg_target and attempts < 5000:
            s = random.choice(all_concepts)
            r = random.choice(self.RELATION_LABELS)
            o = random.choice(all_concepts)
            if (s, r, o) not in known and s != o:
                ctx = f"{s} {r} {o}"
                emb = self._embed(ctx)
                features.append(emb)
                labels.append(0.0)
                known.add((s, r, o))
            attempts += 1
        logger.debug("Generated %d negative samples in %d attempts",
                      len([l for l in labels if l == 0.0]), attempts)

        # Include feedback negatives as hard negative examples
        fb_neg = 0
        for fb_subj, fb_rel, fb_obj, fb_label in self._feedback_negatives:
            if fb_label < 0.5:  # only negative feedback
                key = (fb_subj, fb_rel, fb_obj)
                if key not in known:
                    ctx = f"{fb_subj} {fb_rel} {fb_obj}"
                    emb = self._embed(ctx)
                    features.append(emb)
                    labels.append(0.0)
                    known.add(key)
                    fb_neg += 1
        if fb_neg:
            logger.debug("Added %d hard negatives from feedback", fb_neg)

        features = np.array(features)
        labels_arr = np.array(labels)

        # Train
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        dataset_size = len(features)
        indices = np.random.permutation(dataset_size)
        split = int(dataset_size * 0.8)
        train_idx, val_idx = indices[:split], indices[split:]
        logger.debug("Training set: %d, Validation set: %d", len(train_idx), len(val_idx))

        train_losses, val_accs = [], []

        for epoch in range(epochs):
            # Training
            self.model.train()
            batch_loss = 0.0
            batch_count = 0
            for i in range(0, len(train_idx), 16):
                batch_i = train_idx[i:i + 16]
                x = torch.from_numpy(features[batch_i]).float()
                y = torch.from_numpy(labels_arr[batch_i]).float().unsqueeze(1)
                pred = self.model(x)
                loss = criterion(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                batch_count += 1
            train_losses.append(batch_loss / max(batch_count, 1))

            # Validation
            self.model.eval()
            with torch.no_grad():
                x_val = torch.from_numpy(features[val_idx]).float()
                y_val = torch.from_numpy(labels_arr[val_idx]).float().unsqueeze(1)
                pred_val = self.model(x_val)
                acc = ((pred_val >= 0.5) == (y_val >= 0.5)).float().mean().item()
                val_accs.append(acc)

            if (epoch + 1) % 10 == 0:
                logger.debug("Epoch %d/%d — train_loss=%.4f val_acc=%.3f",
                              epoch + 1, epochs, train_losses[-1], val_accs[-1])

        self.model.eval()
        self._trained = True
        logger.info("Training complete — final val acc: %.3f, loss: %.4f",
                     val_accs[-1] if val_accs else 0, train_losses[-1] if train_losses else 0)
        return {
            "status": "trained",
            "epochs": epochs,
            "samples": dataset_size,
            "positive": len(all_rels),
            "negative": len([l for l in labels if l == 0.0]),
            "final_val_accuracy": round(val_accs[-1], 3) if val_accs else 0,
            "final_train_loss": round(train_losses[-1], 3) if train_losses else 0,
        }

    def predict_relations(self, subject: str, obj: str,
                          context: str = "") -> List[Dict[str, Any]]:
        """Predict which relations exist between subject and object given context.

        This is the GOLD method — it learns what relations are likely between
        ANY two concepts, not just ones found via regex patterns.

        Args:
            subject: The subject concept (A)
            obj: The object concept (B)
            context: Optional surrounding text for disambiguation

        Returns:
            List of {relation, confidence} sorted descending by confidence
        """
        logger.debug("predict_relations: '%s' <-> '%s' (context=%d chars)",
                      subject, obj, len(context))

        if not HAS_TORCH or self.relation_predictor is None:
            logger.debug("RelationPredictor unavailable — falling back to embedding similarity")
            return self._predict_relations_fallback(subject, obj, context)

        # Encode subject + object + context as a single embedding
        combined = f"{subject} [SEP] {obj} [SEP] {context}" if context else f"{subject} [SEP] {obj}"
        emb = self._embed(combined)

        # Run through multi-label classifier
        self.relation_predictor.eval()
        with torch.no_grad():
            t = torch.from_numpy(emb).float().unsqueeze(0)
            scores = self.relation_predictor(t).squeeze(0).numpy()

        # Build result list sorted by confidence
        results = []
        for i, rel in enumerate(self.RELATION_LABELS):
            conf = float(scores[i])
            if conf >= 0.1:  # low floor — let user filter
                results.append({"relation": rel, "confidence": round(conf, 3)})

        results.sort(key=lambda x: x["confidence"], reverse=True)

        logger.debug("predict_relations: top result — %s (%.3f)",
                      results[0]["relation"] if results else "NONE",
                      results[0]["confidence"] if results else 0)
        return results

    def _predict_relations_fallback(self, subject: str, obj: str,
                                     context: str = "") -> List[Dict[str, Any]]:
        """Fallback when RelationPredictor is unavailable.
        Scores relations by embedding similarity between (subject + relation + object)
        and context."""
        logger.debug("Using fallback relation predictor")
        results = []
        base_text = context if context else f"{subject} {obj}"
        base_emb = self._embed(base_text)

        for rel in self.RELATION_LABELS:
            triple_text = f"{subject} {rel} {obj}"
            triple_emb = self._embed(triple_text)
            sim = float(np.dot(base_emb, triple_emb) / (
                np.linalg.norm(base_emb) * np.linalg.norm(triple_emb) + 1e-8
            ))
            conf = max(0.0, min(1.0, sim))
            results.append({"relation": rel, "confidence": round(conf, 3)})

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results

    def train_relation_predictor(self, epochs: int = 50,
                                  lr: float = 0.001) -> Dict[str, Any]:
        """Train the multi-label relation classifier on DB relationships.

        Positive: existing (subject, object) pairs with their known relations
        Negative: random (subject, object) pairs NOT in DB (all relations = 0)
        """
        logger.debug("train_relation_predictor called — epochs=%d, lr=%f", epochs, lr)
        if not HAS_TORCH or self.relation_predictor is None:
            return {"status": "no_torch", "error": "PyTorch or RelationPredictor unavailable"}
        if self.memory is None:
            return {"status": "no_memory", "error": "No memory system"}

        all_rels = self.memory.conn.execute(
            "SELECT source_concept, relation, target_concept FROM relationships"
        ).fetchall()
        logger.debug("Found %d existing relationships", len(all_rels))

        if len(all_rels) < 10:
            return {"status": "skipped",
                    "error": f"Only {len(all_rels)} relationships, need >= 10 for multi-label training"}

        all_concepts = self.memory.get_all_concepts()
        num_rels = len(self.RELATION_LABELS)
        rel_to_idx = {r: i for i, r in enumerate(self.RELATION_LABELS)}

        # Group relationships by (subject, object) pair
        pair_relations: Dict[Tuple[str, str], List[str]] = {}
        for row in all_rels:
            key = (row["source_concept"], row["target_concept"])
            pair_relations.setdefault(key, []).append(row["relation"])

        logger.debug("Grouped into %d unique (subject, object) pairs", len(pair_relations))

        features, targets = [], []

        # Positive examples: each (s, o) pair with its known relations
        for (s, o), rels in pair_relations.items():
            combined = f"{s} [SEP] {o} [SEP]"
            emb = self._embed(combined)
            features.append(emb)
            label = np.zeros(num_rels, dtype=np.float32)
            for r in rels:
                if r in rel_to_idx:
                    label[rel_to_idx[r]] = 1.0
            targets.append(label)

        # Negative examples: random (s, o) pairs with no known relation
        import random
        known_pairs = set(pair_relations.keys())
        neg_target = len(pair_relations)
        neg_count = 0
        attempts = 0
        while neg_count < neg_target and attempts < 5000:
            s = random.choice(all_concepts)
            o = random.choice(all_concepts)
            if s != o and (s, o) not in known_pairs:
                combined = f"{s} [SEP] {o} [SEP]"
                emb = self._embed(combined)
                features.append(emb)
                targets.append(np.zeros(num_rels, dtype=np.float32))
                known_pairs.add((s, o))
                neg_count += 1
            attempts += 1
        logger.debug("Generated %d negative samples", neg_count)

        features = np.array(features)
        targets_arr = np.array(targets)
        logger.debug("Dataset shape: features=%s targets=%s", features.shape, targets_arr.shape)

        # Train
        self.relation_predictor.train()
        optimizer = torch.optim.Adam(self.relation_predictor.parameters(), lr=lr)
        criterion = nn.BCELoss()

        dataset_size = len(features)
        indices = np.random.permutation(dataset_size)
        split = int(dataset_size * 0.8)
        train_idx, val_idx = indices[:split], indices[split:]

        train_losses, val_accs = [], []

        for epoch in range(epochs):
            self.relation_predictor.train()
            batch_loss = 0.0
            batch_count = 0
            for i in range(0, max(len(train_idx), 1), 16):
                batch_i = train_idx[i:i + 16]
                x = torch.from_numpy(features[batch_i]).float()
                y = torch.from_numpy(targets_arr[batch_i]).float()
                pred = self.relation_predictor(x)
                loss = criterion(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                batch_count += 1
            train_losses.append(batch_loss / max(batch_count, 1))

            self.relation_predictor.eval()
            with torch.no_grad():
                x_val = torch.from_numpy(features[val_idx]).float()
                y_val = torch.from_numpy(targets_arr[val_idx]).float()
                pred_val = self.relation_predictor(x_val)
                acc = ((pred_val >= 0.5) == (y_val >= 0.5)).float().mean().item()
                val_accs.append(acc)

            if (epoch + 1) % 10 == 0:
                logger.debug("Epoch %d/%d — train_loss=%.4f val_acc=%.3f",
                              epoch + 1, epochs, train_losses[-1], val_accs[-1])

        self.relation_predictor.eval()
        self._rp_trained = True
        logger.info("RelationPredictor training complete — final val acc: %.3f, loss: %.4f",
                     val_accs[-1] if val_accs else 0, train_losses[-1] if train_losses else 0)
        return {
            "status": "trained",
            "epochs": epochs,
            "samples": dataset_size,
            "positive": len(pair_relations),
            "negative": neg_count,
            "final_val_accuracy": round(val_accs[-1], 3) if val_accs else 0,
            "final_train_loss": round(train_losses[-1], 3) if train_losses else 0,
        }

    def process_feedback(self, subject: str, relation: str, obj: str,
                         is_correct: bool,
                         retrain: bool = False) -> Dict[str, Any]:
        """Process user feedback on a triple — core feedback loop learning.

        Every output becomes training data:
          AI answer -> User correction OR confidence check -> Update model + graph

        If the user marks a triple as incorrect:
          - Confidence is dampened in the memory system
          - The triple is stored as a hard negative example for the neural model
          - Future predictions will be less likely to produce this error

        If marked as correct:
          - Confidence is reinforced in the memory system
          - The triple is stored as a positive training example

        Args:
            subject: Subject of the triple
            relation: Relation
            obj: Object of the triple
            is_correct: True if the user confirmed, False if rejected
            retrain: If True, immediately retrain the neural model

        Returns:
            Debug dict with action taken, old/new confidence, feedback count
        """
        result: Dict[str, Any] = {
            "subject": subject, "relation": relation, "object": obj,
            "is_correct": is_correct
        }
        logger.debug("process_feedback: %s -> [%s] -> %s correct=%s",
                      subject, relation, obj, is_correct)

        if self.memory is not None:
            if is_correct:
                new_conf = self.memory._reinforce_relationship(
                    subject, relation, obj
                )
                result["action"] = "reinforced"
                result["new_confidence"] = new_conf
            else:
                cursor = self.memory.conn.execute(
                    "SELECT id, confidence FROM relationships "
                    "WHERE source_concept=? AND relation=? AND target_concept=?",
                    (subject.lower().strip(),
                     relation.lower().strip(),
                     obj.lower().strip())
                )
                row = cursor.fetchone()
                if row:
                    result["old_confidence"] = row["confidence"]
                    self.memory._dampen_relationship(row["id"])
                    cursor2 = self.memory.conn.execute(
                        "SELECT confidence FROM relationships WHERE id=?",
                        (row["id"],)
                    )
                    new_row = cursor2.fetchone()
                    result["new_confidence"] = (new_row["confidence"]
                                                if new_row else row["confidence"])
                    result["action"] = "dampened"
                else:
                    result["action"] = "not_found"
                    result["new_confidence"] = None

            # Store feedback example for neural model training
            self._feedback_negatives.append(
                (subject, relation, obj, 1.0 if is_correct else 0.0)
            )
            result["feedback_examples"] = len(self._feedback_negatives)

        if retrain and len(self._feedback_negatives) >= 2:
            train_result = self._train_on_feedback()
            result["retrain_status"] = train_result.get("status", "unknown")

        logger.debug("process_feedback result: action=%s", result.get("action"))
        return result

    def _train_on_feedback(self, epochs: int = 15,
                           lr: float = 0.001) -> Dict[str, Any]:
        """Quick retrain using only feedback examples (hard negatives).

        Called automatically when feedback accumulates and retrain=True.
        """
        if self.model is None or not HAS_TORCH:
            return {"status": "no_torch"}
        pos = [(s, r, o) for s, r, o, l in self._feedback_negatives if l > 0.5]
        neg = [(s, r, o) for s, r, o, l in self._feedback_negatives if l < 0.5]
        if not neg and not pos:
            return {"status": "skipped", "reason": "no_feedback_examples"}

        logger.debug("_train_on_feedback: %d pos, %d neg samples", len(pos), len(neg))
        features, labels = [], []
        for s, r, o in pos:
            emb = self._embed(f"{s} {r} {o}")
            features.append(emb)
            labels.append(1.0)
        for s, r, o in neg:
            emb = self._embed(f"{s} {r} {o}")
            features.append(emb)
            labels.append(0.0)

        features_arr = np.array(features)
        labels_arr = np.array(labels)

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        for epoch in range(epochs):
            self.model.train()
            for i in range(0, len(features_arr), 8):
                batch_x = torch.from_numpy(features_arr[i:i + 8]).float()
                batch_y = torch.from_numpy(labels_arr[i:i + 8]).float().unsqueeze(1)
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.model.eval()
        self._trained = True
        logger.info("Feedback training complete — %d epochs, %d samples",
                     epochs, len(features_arr))
        return {
            "status": "trained",
            "epochs": epochs,
            "samples": len(features_arr),
            "positive": len(pos),
            "negative": len(neg),
        }

    def _rule_confidence(self, text: str, subject: str,
                         relation: str, obj: str) -> float:
        """Rule-based confidence heuristic — used as baseline / ensemble component."""
        text = normalize_ligatures(text)
        score = 0.5
        # Proximity bonus: subject and object appear near each other
        subj_pos = text.lower().find(subject)
        obj_pos = text.lower().find(obj)
        if subj_pos >= 0 and obj_pos >= 0:
            distance = abs(subj_pos - obj_pos)
            if distance < 50:
                score += 0.15
            elif distance < 150:
                score += 0.05
        # Length bonus: longer sentences tend to have more reliable triples
        if len(text) > 80:
            score += 0.05
        # Relation verb present in text
        rel_patterns = {
            "is_a": r"\b(is|are)\b",
            "has": r"\b(has|have)\b",
            "can": r"\b(can|could|will)\b",
            "requires": r"\b(requires?|needs?)\b",
            "seeks": r"\b(seeks?|desires?|wants?)\b",
        }
        if relation in rel_patterns:
            if re.search(rel_patterns[relation], text, re.IGNORECASE):
                score += 0.05
        return min(1.0, score)

    def cleanup(self) -> None:
        logger.debug("Cleaning up PatternRecognizer resources")
        self.model = None
        self.relation_predictor = None
        self._trained = False
        self._rp_trained = False
