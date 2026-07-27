"""
Concept Embedding Enhancer for Offline AI Book Reader.
Refines static MiniLM embeddings using knowledge graph structure.

Instead of:
  static_embedding = MiniLM(concept_name)

You get:
  graph_embedding = GraphAggregator(static_embedding, neighbors, relations)

This makes embeddings shaped by YOUR knowledge graph — personal intelligence.

Architecture:
  - GraphAttentionLayer: attention-based neighbor aggregation + residual gate
  - ConceptEmbeddingEnhancer: wraps the model with training/inference API
  - Training: contrastive (connected concepts closer than random pairs)
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("torch not available — embedding enhancer will return base embeddings only")

RELATION_LABELS = [
    "is_a", "has", "can", "uses", "seeks", "creates", "rejects",
    "depends_on", "eats", "hunts", "lives_in", "contains", "produces",
    "requires", "controls", "benefits_from", "causes", "prevents",
]


if HAS_TORCH:

    class GraphAttentionLayer(nn.Module):
        """Attention-based neighbor aggregation for graph-enhanced embeddings.

        For concept C with base embedding e_C and neighbors N_i with relations r_i:
          1. h_i = MLP([e_Ni; rel_emb(r_i)])    — neighbor representation
          2. a_i = softmax(attn(h_i))             — attention weights
          3. h_agg = sum(a_i * h_i)               — aggregated neighborhood
          4. gate = sigmoid(W_gate([e_C; h_agg])) — learnable gate
          5. output = gate * h_agg + (1-gate) * e_C — residual blend
        """

        def __init__(self, embed_dim: int = 384, hidden_dim: int = 128,
                     num_relation_types: int = 18):
            super().__init__()
            self.embed_dim = embed_dim
            self.rel_emb = nn.Embedding(num_relation_types + 1, embed_dim)
            self.neighbor_mlp = nn.Sequential(
                nn.Linear(embed_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embed_dim),
            )
            self.attn = nn.Linear(embed_dim, 1, bias=False)
            self.gate_mlp = nn.Sequential(
                nn.Linear(embed_dim * 2, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.layer_norm = nn.LayerNorm(embed_dim)
            logger.debug("GraphAttentionLayer: embed_dim=%d hidden_dim=%d num_rels=%d",
                          embed_dim, hidden_dim, num_relation_types)

        def forward(self, concept_emb: torch.Tensor,
                    neighbor_embs: torch.Tensor,
                    relation_ids: torch.Tensor) -> torch.Tensor:
                rel_embs = self.rel_emb(relation_ids)
                combined = torch.cat([neighbor_embs, rel_embs], dim=-1)
                h = self.neighbor_mlp(combined)
                attn_scores = self.attn(h).squeeze(-1)
                attn_weights = F.softmax(attn_scores, dim=0).unsqueeze(-1)
                h_agg = (attn_weights * h).sum(dim=0)
                gate_input = torch.cat([concept_emb, h_agg], dim=-1)
                gate = torch.sigmoid(self.gate_mlp(gate_input))
                output = gate * h_agg + (1 - gate) * concept_emb
                return self.layer_norm(output)


class ConceptEmbeddingEnhancer:
    """Enhances concept embeddings using knowledge graph structure.

    Usage:
        enhancer = ConceptEmbeddingEnhancer(embedding_generator, memory_system)
        enhancer.train_from_memory()     # train on existing graph
        ref_emb = enhancer.enhance("wolf")  # get graph-aware embedding
        sim = enhancer.compute_similarity("wolf", "pack")
    """

    def __init__(self, embedding_generator, memory_system=None, vector_db=None):
        self.embedder = embedding_generator
        self.memory = memory_system
        self.vector_db = vector_db
        self._device = "cpu"
        self.model: Optional[GraphAttentionLayer] = None
        self._trained = False
        self._model_path = Path("data/embedding_enhancer.pt")

        if HAS_TORCH:
            self.model = GraphAttentionLayer(
                embed_dim=self.embedder.get_embedding_dimension(),
                num_relation_types=len(RELATION_LABELS),
            )
            if self._model_path.exists():
                try:
                    self.model.load_state_dict(torch.load(str(self._model_path), map_location="cpu"))
                    self.model.eval()
                    self._trained = True
                    logger.info("Loaded saved embedding enhancer model from %s", self._model_path)
                except Exception as e:
                    logger.warning("Failed to load saved model: %s", e)
            else:
                self.model.eval()
            logger.debug("GraphAttentionLayer initialized - dim=%d, rels=%d",
                          self.embedder.get_embedding_dimension(),
                          len(RELATION_LABELS))
        else:
            logger.debug("PyTorch unavailable - graph enhancement disabled")

    def _embed(self, text: str) -> np.ndarray:
        logger.debug("Enhancer embedding (%d chars): %.60s...", len(text), text.strip())
        return self.embedder.generate_single_text_embedding(text)

    def _relation_to_id(self, relation: str) -> int:
        if relation in RELATION_LABELS:
            return RELATION_LABELS.index(relation) + 1
        return 0

    def _get_neighbors(self, concept: str, bidirectional: bool = False) -> Tuple[List[str], List[str]]:
        """Get neighbor names and relation types from memory.

        Args:
            concept: The concept to query.
            bidirectional: If True, include both incoming and outgoing edges.
                          If False (default), only outgoing edges (matching training pair structure).

        During training, we use bidirectional=False to match the directional
        positive pairs (source -> target). During inference, bidirectional=True
        provides richer graph context.
        """
        if self.memory is None:
            return [], []
        graph = self.memory.get_concept_graph(concept)
        neighbors = []
        relations = []
        for r in graph.get("relationships", []):
            if r["direction"] == "outgoing":
                neighbors.append(r["target"])
                relations.append(r["relation"])
            elif bidirectional and r["direction"] == "incoming":
                neighbors.append(r["target"])
                relations.append(r["relation"])
        return neighbors, relations

    def enhance(self, concept: str) -> np.ndarray:
        """Get graph-enhanced embedding for a concept.

        Uses bidirectional neighbor context for richer inference.
        Falls back to base MiniLM embedding when:
          - PyTorch unavailable
          - No memory system
          - Concept has no neighbors in graph
        """
        base_emb = self._embed(concept)
        logger.debug("enhance: '%s' base_emb norm=%.4f dim=%d",
                      concept, np.linalg.norm(base_emb), len(base_emb))

        if self.model is None or self.memory is None:
            logger.debug("enhance: fallback (no model or memory)")
            return base_emb

        neighbors, relations = self._get_neighbors(concept, bidirectional=True)
        if not neighbors:
            logger.debug("enhance: fallback (no neighbors for '%s')", concept)
            return base_emb
        logger.debug("enhance: '%s' has %d neighbors", concept, len(neighbors))

        # Embed all neighbors
        neighbor_embs = []
        rel_ids = []
        for n, r in zip(neighbors, relations):
            n_emb = self._embed(n)
            neighbor_embs.append(n_emb)
            rel_ids.append(self._relation_to_id(r))

        self.model.eval()
        with torch.no_grad():
            c_emb_t = torch.from_numpy(base_emb).float()
            n_embs_t = torch.from_numpy(np.stack(neighbor_embs)).float()
            r_ids_t = torch.tensor(rel_ids, dtype=torch.long)
            enhanced = self.model(c_emb_t, n_embs_t, r_ids_t).numpy()

        logger.debug("enhance: '%s' enhanced norm=%.4f delta=%.4f",
                      concept, np.linalg.norm(enhanced),
                      np.linalg.norm(enhanced - base_emb))
        return enhanced

    def enhance_batch(self, concepts: List[str]) -> Dict[str, np.ndarray]:
        """Enhance multiple concepts in batch."""
        return {c: self.enhance(c) for c in concepts}

    def compute_similarity(self, c1: str, c2: str) -> float:
        """Cosine similarity between graph-enhanced concept embeddings."""
        e1 = self.enhance(c1)
        e2 = self.enhance(c2)
        sim = float(np.dot(e1, e2) / (
            np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8
        ))
        logger.debug("compute_similarity: '%s' <-> '%s' = %.4f", c1, c2, sim)
        return sim

    def find_similar(self, concept: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Find similar concepts using graph-enhanced embeddings."""
        logger.debug("find_similar: '%s' top_k=%d", concept, top_k)
        ref_emb = self.enhance(concept)
        if self.memory is None:
            return []
        all_concepts = self.memory.get_all_concepts()
        scored: List[Dict[str, Any]] = []
        for c in all_concepts:
            if c.lower().strip() == concept.lower().strip():
                continue
            c_emb = self.enhance(c)
            sim = float(np.dot(ref_emb, c_emb) / (
                np.linalg.norm(ref_emb) * np.linalg.norm(c_emb) + 1e-8
            ))
            scored.append({"concept": c, "similarity": sim, "enhanced": True})
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]

    def _precompute_embeddings(self, concepts: List[str]) -> Dict[str, np.ndarray]:
        """Pre-compute base embeddings for many concepts at once (batched)."""
        unique = sorted(set(concepts))
        embs = self.embedder.generate_text_embeddings(unique)
        return {c: embs[i] for i, c in enumerate(unique)}

    def _forward_cached(self, concept: str, emb_cache: Dict[str, np.ndarray],
                        grad_enabled: bool = True,
                        neighbor_cache: Any = None,
                        tensor_cache: Any = None
                        ) -> Any:
        """Like enhance() but uses cached base embeddings with gradient tracking.

        Falls back to computing fresh embedding if concept not in cache.
        NOTE: emb_cache.get(k, default) evaluates default eagerly even when
        the key is found, so we must check membership first.

        neighbor_cache: pre-computed dict mapping concept -> (neighbors, relation_ids)
                        to avoid DB queries on every forward pass during training.
        tensor_cache: pre-computed dict mapping concept ->
                      (concept_tensor, neighbor_embs_tensor, relation_ids_tensor)
                      for maximum speed. Bypasses all per-neighbor lookups.
        """
        if tensor_cache is not None:
            entry = tensor_cache.get(concept)
            if entry is not None:
                c_t, n_t, r_t = entry
                if n_t.shape[0] == 0:
                    return c_t.detach() if not grad_enabled else c_t
                if not grad_enabled:
                    with torch.no_grad():
                        return self.model(c_t, n_t, r_t)
                return self.model(c_t, n_t, r_t)

        def _cached_or_embed(key: str) -> np.ndarray:
            val = emb_cache.get(key)
            if val is not None:
                return val
            return self._embed(key)

        if self.model is None or self.memory is None:
            base = _cached_or_embed(concept)
            return torch.from_numpy(base).float().detach()

        if neighbor_cache is not None:
            neighbors, r_ids_list = neighbor_cache.get(concept, ([], []))
        else:
            neighbors_raw, relations = self._get_neighbors(concept)
            neighbors = neighbors_raw
            r_ids_list = [self._relation_to_id(r) for r in relations]

        if not neighbors:
            base = _cached_or_embed(concept)
            return torch.from_numpy(base).float().detach()

        base_emb = _cached_or_embed(concept)

        n_embs_list = []
        for n in neighbors:
            n_emb = _cached_or_embed(n)
            n_embs_list.append(n_emb)

        c_t = torch.from_numpy(base_emb).float()
        n_t = torch.from_numpy(np.stack(n_embs_list)).float()
        r_t = torch.tensor(r_ids_list, dtype=torch.long)

        if not grad_enabled:
            with torch.no_grad():
                return self.model(c_t, n_t, r_t)
        return self.model(c_t, n_t, r_t)

    def train_from_memory(self, epochs: int = 30, lr: float = 0.001,
                          margin: float = 0.3) -> Dict[str, Any]:
        """Train graph attention network using contrastive loss.

        Positive pairs: concepts connected by a relationship in the DB
        Negative pairs: random concept pairs not in the DB

        Loss: max(0, margin + sim(neg) - sim(pos))

        Uses pre-computed embeddings to avoid re-embedding every concept
        on every pair iteration.
        """
        logger.debug("train_from_memory: epochs=%d lr=%f margin=%.2f",
                      epochs, lr, margin)
        if self.model is None or not HAS_TORCH:
            return {"status": "no_torch", "error": "PyTorch unavailable"}
        if self.memory is None:
            return {"status": "no_memory", "error": "No memory system"}

        all_rels = self.memory.conn.execute(
            "SELECT source_concept, relation, target_concept FROM relationships"
        ).fetchall()
        logger.debug("train_from_memory: %d relationships found", len(all_rels))

        if len(all_rels) < 5:
            return {"status": "skipped",
                    "error": f"Only {len(all_rels)} relationships, need >= 5"}

        all_concepts = self.memory.get_all_concepts()
        concept_set = set(all_concepts)

        import random

        # Build positive pairs (subject, object)
        pos_pairs: List[Tuple[str, str]] = []
        for row in all_rels:
            s = row["source_concept"]
            o = row["target_concept"]
            pos_pairs.append((s, o))
        logger.debug("train_from_memory: %d positive pairs", len(pos_pairs))

        known_pos = set(pos_pairs)

        # --- Pre-compute all concept embeddings once ---
        all_unique = list({c for pair in pos_pairs for c in pair} | set(all_concepts))
        logger.debug("Pre-computing embeddings for %d unique concepts...", len(all_unique))
        emb_cache = self._precompute_embeddings(all_unique)
        logger.debug("Pre-computation done.")

        # --- Pre-compute neighbor tensor cache (base torch tensors, no DB/per-neighbor work during training) ---
        logger.info("Building tensor cache for %d concepts...", len(all_unique))
        tensor_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for idx, c in enumerate(all_unique):
            if idx > 0 and idx % 500 == 0:
                logger.info("Tensor cache: %d/%d concepts", idx, len(all_unique))
            neighbors, relations = self._get_neighbors(c)
            c_t = torch.from_numpy(emb_cache[c]).float()
            if neighbors:
                n_embs = np.stack([emb_cache[n] for n in neighbors])
                r_ids = [self._relation_to_id(r) for r in relations]
                n_t = torch.from_numpy(n_embs).float()
                r_t = torch.tensor(r_ids, dtype=torch.long)
            else:
                n_t = torch.empty(0, c_t.shape[0])
                r_t = torch.empty(0, dtype=torch.long)
            tensor_cache[c] = (c_t, n_t, r_t)
        logger.info("Tensor cache built (%d concepts).", len(tensor_cache))

        self.model.train()
        logger.info("Building training pairs (%d relationships, %d concepts)...",
                     len(all_rels), len(all_unique))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        losses = []

        for epoch in range(epochs):
            random.shuffle(pos_pairs)
            epoch_loss = 0.0
            batch_count = 0

            for s, o in pos_pairs:
                # Sample a random negative concept
                neg = random.choice(all_concepts)
                while neg == s or neg == o or (s, neg) in known_pos:
                    neg = random.choice(all_concepts)

                # Get enhanced embeddings (uses tensor_cache - just model forward, no Python overhead)
                e_s = self._forward_cached(s, emb_cache, grad_enabled=True,
                                           tensor_cache=tensor_cache)
                e_o = self._forward_cached(o, emb_cache, grad_enabled=True,
                                           tensor_cache=tensor_cache)
                e_n = self._forward_cached(neg, emb_cache, grad_enabled=True,
                                           tensor_cache=tensor_cache)

                # Cosine similarities
                sim_pos = F.cosine_similarity(e_s.unsqueeze(0), e_o.unsqueeze(0))
                sim_neg = F.cosine_similarity(e_s.unsqueeze(0), e_n.unsqueeze(0))

                # Margin ranking loss
                loss = F.relu(margin + sim_neg - sim_pos).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1
                if batch_count % 1000 == 0:
                    logger.info("  epoch %d: %d/%d pairs processed", epoch + 1, batch_count, len(pos_pairs))

            avg_loss = epoch_loss / max(batch_count, 1)
            losses.append(avg_loss)
            logger.info("Epoch %d/%d - loss=%.4f", epoch + 1, epochs, avg_loss)

        self.model.eval()
        self._trained = True
        logger.info("Training complete — final loss: %.4f", losses[-1] if losses else 0)

        # Persist enhanced concept embeddings and rebuild FAISS index
        embed_result = self.store_enhanced_embeddings()
        if embed_result["status"] != "no_vector_db":
            logger.info("FAISS index rebuild: status=%s stored=%d",
                        embed_result.get("rebuild", {}).get("status", "N/A"),
                        embed_result["stored"])

        return {
            "status": "trained",
            "epochs": epochs,
            "pairs": len(pos_pairs),
            "final_loss": round(losses[-1], 4) if losses else 0,
            "losses": [round(l, 4) for l in losses],
            "embeddings_stored": embed_result.get("stored", 0),
            "faiss_rebuild": embed_result.get("rebuild", {}).get("status", "no_vector_db"),
        }

    def get_enhanced_all(self) -> Dict[str, np.ndarray]:
        """Get enhanced embeddings for ALL concepts in the graph."""
        if self.memory is None:
            return {}
        concepts = self.memory.get_all_concepts()
        return self.enhance_batch(concepts)

    def store_enhanced_embeddings(self) -> Dict[str, Any]:
        """Compute enhanced embeddings for all concepts and store in vector_db.

        After training, stores the graph-enhanced embeddings for every concept
        in the knowledge graph, then triggers a FAISS index rebuild so that
        concept similarity searches use the enhanced vectors.
        """
        if self.vector_db is None:
            return {"status": "no_vector_db", "stored": 0}
        enhanced = self.get_enhanced_all()
        if not enhanced:
            return {"status": "no_concepts", "stored": 0}
        stored = 0
        for name, emb in enhanced.items():
            self.vector_db.store_concept_embedding(name, emb)
            stored += 1
        rebuild = self.vector_db.rebuild_index(
            dimension=self.embedder.get_embedding_dimension(),
            index_type="IndexFlatL2",
        )
        logger.info("Stored %d enhanced embeddings; FAISS rebuild: %s", stored, rebuild["status"])
        return {"status": "ok", "stored": stored, "rebuild": rebuild}

    def save_model(self, path: Optional[Path] = None) -> bool:
        path = path or self._model_path
        if self.model is None:
            logger.warning("No model to save")
            return False
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), str(path))
            self._trained = True
            logger.info("Model saved to %s", path)
            return True
        except Exception as e:
            logger.error("Failed to save model: %s", e)
            return False

    def cleanup(self) -> None:
        self.save_model()
        logger.debug("Cleaning up ConceptEmbeddingEnhancer")
        self.model = None
        self._trained = False
