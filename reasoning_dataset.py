"""
Reasoning Dataset — stores (problem, reasoning, answer) triples with embedding
retrieval. Lets Jafar retrieve similar reasoning patterns instead of re-deriving
strategies from scratch every time.
"""

import json
import logging
import os
import random
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)
debug_logger = logging.getLogger(f"{__name__}.debug")

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

REASONING_DOMAINS = [
    "debugging", "planning", "analysis", "root_cause",
    "decision", "optimization", "design", "learning",
]

SYNTHETIC_TEMPLATES = [
    {
        "pattern": "debug_error",
        "problem": "Code raises {error_type}: {scenario}",
        "reasoning": "Identify the specific line causing the {error_type}. Check for off-by-one, type mismatch, or unhandled edge case in {component}. Isolate the failing input and trace execution step by step.",
        "answer": "Add input validation at {component}, wrap the call in try/except, and log the actual values at each step.",
        "defaults": {"error_type": "TypeError", "scenario": "unexpected None value", "component": "the data pipeline"},
    },
    {
        "pattern": "root_cause_analysis",
        "problem": "{system} is {symptom} intermittently",
        "reasoning": "Intermittent failures suggest a race condition, resource leak, or environmental dependency. Check shared state, connection pool limits, and external service timeouts in {system}. Correlate failure timestamps with system metrics.",
        "answer": "Add distributed tracing to {system}, implement circuit breakers for external calls, and increase connection pool size.",
        "defaults": {"system": "the API server", "symptom": "returning 503 errors"},
    },
    {
        "pattern": "optimization",
        "problem": "{process} takes too long when {condition}",
        "reasoning": "Identify the bottleneck using profiling. Look for O(n^2) algorithms in hot paths, redundant I/O operations, or N+1 query patterns. Measure before optimizing to confirm the actual bottleneck.",
        "answer": "Add caching for {condition}, batch the I/O operations, and replace the nested loop with a hash lookup.",
        "defaults": {"process": "data processing", "condition": "dataset exceeds 10K rows"},
    },
    {
        "pattern": "architectural_decision",
        "problem": "Need to choose between {option_a} and {option_b} for {context}",
        "reasoning": "Compare along five axes: scalability, maintainability, team expertise, operational cost, and time to delivery. The right choice depends on which constraint is tightest. Prototype both with limited scope first.",
        "answer": "Start with {option_a} for the MVP since it minimizes time-to-delivery, then plan migration to {option_b} when {context} reaches scale.",
        "defaults": {"option_a": "a monolith", "option_b": "microservices", "context": "the new billing system"},
    },
    {
        "pattern": "customer_complaint",
        "problem": "Customer is angry about {issue}",
        "reasoning": "De-escalate first, diagnose second. The emotional state blocks rational communication. Acknowledge the frustration, then investigate the root cause systematically. Separate the symptom from the underlying need.",
        "answer": "Acknowledge the frustration, ask clarifying questions to identify the actual root cause, then propose a concrete fix with timeline.",
        "defaults": {"issue": "a missing feature they expected"},
    },
    {
        "pattern": "learning_new_tech",
        "problem": "Need to learn {technology} for {project}",
        "reasoning": "Identify the core 20% of {technology} that covers 80% of {project}'s use cases. Skip advanced features until needed. Build a small end-to-end prototype first, then layer in depth on demand.",
        "answer": "Build a minimal end-to-end prototype in {technology}, then deep-dive into the specific areas needed for {project}.",
        "defaults": {"technology": "Kubernetes", "project": "deploying a microservice"},
    },
    {
        "pattern": "debug_heisenbug",
        "problem": "Bug disappears when {condition}",
        "reasoning": "This is a classic heisenbug — the act of observing changes behavior. Likely causes: timing-dependent race condition, uninitialized memory, logging side effects, or debugger probe interference. Add logging at a lower level (e.g., kernel traces, wire captures) that doesn't alter the execution path.",
        "answer": "Replace high-level breakpoints with passive logging. Use conditional breakpoints that don't change timing. Check for undefined behavior.",
        "defaults": {"condition": "adding a print statement or attaching a debugger"},
    },
    {
        "pattern": "prioritization",
        "problem": "Multiple urgent tasks: {task_a}, {task_b}, {task_c}",
        "reasoning": "Urgency is not importance. Separate urgent-from-important using impact vs effort. Tasks that are high-impact and time-sensitive go first. Defer or delegate low-impact items regardless of urgency pressure.",
        "answer": "Rank by impact x urgency, do the highest first, defer the rest with clear expectations on timeline.",
        "defaults": {"task_a": "fix production outage", "task_b": "prepare board deck", "task_c": "review PR"},
    },
    {
        "pattern": "ambiguous_requirement",
        "problem": "Requirement '{requirement}' is ambiguous",
        "reasoning": "Ambiguity hides assumptions. Ask 'what does success look like?' and 'how would we measure it?' for each interpretation. Write concrete examples for each possible meaning. The act of writing examples usually reveals the correct interpretation.",
        "answer": "Write 3 concrete examples for the requirement, then confirm with the stakeholder which one matches their intent.",
        "defaults": {"requirement": "the system should be fast"},
    },
    {
        "pattern": "scaling_bottleneck",
        "problem": "{component} slows down linearly with {resource}",
        "reasoning": "Linear scaling suggests the algorithm is O(n) on a single dimension but the constant factor is high. Profile to find the actual hot spot. Look for accidental O(n^2) hidden in nested loops, string concatenation, or repeated allocations.",
        "answer": "Profile {component}, identify the hot loop, and apply batching, caching, or a more efficient data structure.",
        "defaults": {"component": "the search endpoint", "resource": "number of users"},
    },
]


class ReasoningDataset:
    """Stores and retrieves (problem, reasoning, answer) triples.

    Uses embedding similarity (via SentenceTransformer) to find relevant
    reasoning patterns for a given problem. Falls back to keyword search
    when embeddings are unavailable.

    State flow:
      add() -> embed() -> store in JSONL
      search() -> embed query -> cosine sim -> return top_k
    """

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        base = Path(__file__).parent / "data"
        base.mkdir(parents=True, exist_ok=True)
        self.dataset_path = dataset_path or str(base / "reasoning_dataset.jsonl")
        self.embedding_model_name = embedding_model_name

        self._embedder = None
        self._examples: List[Dict[str, Any]] = []
        self._loaded = False

    # ------------------------------------------------------------------ #
    #  Embedding
    # ------------------------------------------------------------------ #

    def _get_embedder(self):
        if self._embedder is None and HAS_SENTENCE_TRANSFORMERS:
            try:
                self._embedder = SentenceTransformer(self.embedding_model_name)
            except Exception as e:
                logger.warning("Failed to load embedder: %s", e)
        return self._embedder

    def _embed(self, text: str) -> Optional[Any]:
        embedder = self._get_embedder()
        if embedder is not None and HAS_NUMPY:
            return embedder.encode(text, normalize_embeddings=True)
        if HAS_NUMPY:
            return np.zeros(384, dtype=np.float32)
        return None

    # ------------------------------------------------------------------ #
    #  I/O
    # ------------------------------------------------------------------ #

    def load(self) -> List[Dict[str, Any]]:
        if self._loaded:
            return self._examples
        self._examples = []
        if os.path.isfile(self.dataset_path):
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self._examples.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        self._loaded = True
        logger.info("Loaded %d reasoning examples from %s", len(self._examples), self.dataset_path)
        return self._examples

    def _save(self):
        with open(self.dataset_path, "w", encoding="utf-8") as f:
            for ex in self._examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------ #
    #  CRUD
    # ------------------------------------------------------------------ #

    def add(
        self,
        problem: str,
        reasoning: str,
        answer: str,
        tags: Optional[List[str]] = None,
        domain: str = "general",
        confidence: float = 0.7,
    ) -> Dict[str, Any]:
        self.load()
        ex_id = hashlib.sha256(f"{problem}|{reasoning}".encode()).hexdigest()[:12]
        example = {
            "id": ex_id,
            "problem": problem,
            "reasoning": reasoning,
            "answer": answer,
            "tags": tags or [],
            "domain": domain,
            "confidence": confidence,
            "usage_count": 0,
            "success_count": 0,
            "created_at": time.time(),
        }
        emb = self._embed(f"{problem} {reasoning} {answer}")
        if emb is not None and HAS_NUMPY:
            example["embedding"] = emb.tolist()
        else:
            example["embedding"] = None

        existing = [e for e in self._examples if e["id"] == ex_id]
        if existing:
            existing[0]["usage_count"] += 0
            return existing[0]

        self._examples.append(example)
        self._save()
        debug_logger.debug("Added reasoning example: %s", ex_id)
        return example

    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.2,
        domain: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        self.load()
        candidates = self._examples
        if domain:
            candidates = [e for e in candidates if e.get("domain") == domain]
        if not candidates:
            return []

        query_emb = self._embed(query)
        if query_emb is not None and HAS_NUMPY:
            scored = []
            for ex in candidates:
                ex_emb = ex.get("embedding")
                if ex_emb and len(ex_emb) == len(query_emb):
                    q = query_emb / (np.linalg.norm(query_emb) + 1e-10)
                    e = np.array(ex_emb) / (np.linalg.norm(np.array(ex_emb)) + 1e-10)
                    sim = float(np.dot(q, e))
                    if sim >= threshold:
                        scored.append((sim, ex))
            scored.sort(key=lambda x: x[0], reverse=True)
            results = []
            for sim, ex in scored[:top_k]:
                r = dict(ex)
                r.pop("embedding", None)
                r["similarity"] = round(sim, 4)
                results.append(r)
            return results

        keywords = set(query.lower().split())
        scored = []
        for ex in candidates:
            text = f"{ex['problem']} {ex['reasoning']}".lower()
            matches = sum(1 for k in keywords if k in text)
            if matches > 0:
                scored.append((matches / max(len(keywords), 1), ex))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [dict(ex) for _, ex in scored[:top_k]]

    def record_usage(self, example_id: str, success: bool = True):
        self.load()
        for ex in self._examples:
            if ex["id"] == example_id:
                ex["usage_count"] = ex.get("usage_count", 0) + 1
                if success:
                    ex["success_count"] = ex.get("success_count", 0) + 1
                self._save()
                return

    def get_by_id(self, example_id: str) -> Optional[Dict[str, Any]]:
        self.load()
        for ex in self._examples:
            if ex["id"] == example_id:
                return dict(ex)
        return None

    def get_stats(self) -> Dict[str, Any]:
        self.load()
        domains = {}
        for ex in self._examples:
            d = ex.get("domain", "general")
            domains[d] = domains.get(d, 0) + 1
        total = len(self._examples)
        return {
            "total_examples": total,
            "domains": domains,
            "avg_confidence": round(
                sum(e.get("confidence", 0.5) for e in self._examples) / max(total, 1), 3
            ),
            "total_usage": sum(e.get("usage_count", 0) for e in self._examples),
            "file_path": self.dataset_path,
        }

    # ------------------------------------------------------------------ #
    #  Synthetic generation
    # ------------------------------------------------------------------ #

    def generate_synthetic(
        self,
        count: int = 100,
        domains: Optional[List[str]] = None,
        use_llm: bool = False,
        llm_generator: Optional[Any] = None,
    ) -> int:
        self.load()
        domains = domains or REASONING_DOMAINS
        added = 0
        templates = SYNTHETIC_TEMPLATES

        if use_llm and llm_generator is not None:
            return self._generate_with_llm(count, domains, llm_generator)

        target = count
        attempts = 0
        while added < target and attempts < target * 3:
            attempts += 1
            template = random.choice(templates)
            defaults = template["defaults"]
            substitutions = {}
            for key, val in defaults.items():
                if isinstance(val, str) and "{" in val:
                    substitutions[key] = val
                elif key == "error_type":
                    substitutions[key] = random.choice(["TypeError", "ValueError", "IndexError", "KeyError", "AttributeError"])
                elif key == "scenario":
                    substitutions[key] = random.choice([
                        "unexpected None value", "empty input list", "missing key in dict",
                        "divide by zero", "recursion limit exceeded", "file not found",
                    ])
                elif key == "component":
                    substitutions[key] = random.choice([
                        "the data pipeline", "the API handler", "the database layer",
                        "the user input parser", "the network client", "the cache layer",
                    ])
                elif key == "system":
                    substitutions[key] = random.choice([
                        "the API server", "the database", "the background worker",
                        "the frontend", "the auth service", "the CI pipeline",
                    ])
                elif key == "symptom":
                    substitutions[key] = random.choice([
                        "returning 503 errors", "crashing randomly", "leaking memory",
                        "returning stale data", "timing out", "producing wrong results",
                    ])
                elif key == "process":
                    substitutions[key] = random.choice([
                        "data processing", "the build pipeline", "the report generation",
                        "the search query", "the image upload", "the sync job",
                    ])
                elif key == "condition":
                    substitutions[key] = random.choice([
                        "dataset exceeds 10K rows", "traffic spikes above baseline",
                        "running on weekends", "the input is malformed",
                        "concurrent users > 50", "the cache is cold",
                    ])
                elif key == "context":
                    substitutions[key] = random.choice([
                        "the new billing system", "the recommendation engine",
                        "the notification service", "the analytics pipeline",
                    ])
                elif key == "technology":
                    substitutions[key] = random.choice([
                        "Kubernetes", "Docker", "GraphQL", "gRPC", "React", "PostgreSQL",
                    ])
                elif key == "project":
                    substitutions[key] = random.choice([
                        "deploying a microservice", "building an API", "migrating databases",
                    ])
                elif key == "issue":
                    substitutions[key] = random.choice([
                        "a missing feature they expected", "slow response times",
                        "a confusing UI", "data loss after update", "billing errors",
                    ])
                elif key == "requirement":
                    substitutions[key] = random.choice([
                        "the system should be fast", "make it scalable",
                        "improve the UX", "ensure reliability", "reduce costs",
                        "the module should be pluggable",
                    ])
                elif key == "resource":
                    substitutions[key] = random.choice([
                        "number of users", "data volume", "concurrent requests",
                        "file size", "table rows", "response time",
                    ])
                elif key in ("option_a",):
                    substitutions[key] = random.choice(["a monolith", "a REST API", "sync processing", "a single DB"])
                    substitutions["option_b"] = random.choice(["microservices", "GraphQL", "async processing", "read replicas"])
                else:
                    substitutions[key] = val.format(**{k: v for k, v in substitutions.items() if k != key})

            problem = template["problem"].format(**substitutions)
            reasoning = template["reasoning"].format(**substitutions)
            answer = template["answer"].format(**substitutions)
            domain = random.choice(domains)
            tags = [template["pattern"], domain]

            self.add(problem, reasoning, answer, tags=tags, domain=domain)
            added += 1

        logger.info("Generated %d synthetic reasoning examples", added)
        return added

    def _generate_with_llm(self, count: int, domains: List[str], llm_generator) -> int:
        added = 0
        for i in range(count):
            domain = random.choice(domains)
            prompt = (
                f"Generate a reasoning example in the '{domain}' domain. "
                f"Format:\nProblem: <short problem>\n"
                f"Reasoning: <step-by-step reasoning>\nAnswer: <concrete answer>\n"
                f"Make it realistic and specific."
            )
            try:
                text = llm_generator(prompt)
                if not text:
                    continue
                lines = text.strip().split("\n")
                problem = ""
                reasoning = ""
                answer = ""
                for line in lines:
                    if line.lower().startswith("problem:"):
                        problem = line.split(":", 1)[1].strip()
                    elif line.lower().startswith("reasoning:"):
                        reasoning = line.split(":", 1)[1].strip()
                    elif line.lower().startswith("answer:"):
                        answer = line.split(":", 1)[1].strip()
                if problem and reasoning and answer:
                    self.add(problem, reasoning, answer, tags=[domain], domain=domain)
                    added += 1
            except Exception as e:
                logger.warning("LLM generation failed at %d/%d: %s", i, count, e)
        return added

    # ------------------------------------------------------------------ #
    #  Export / Import
    # ------------------------------------------------------------------ #

    def export_json(self, path: str):
        self.load()
        with open(path, "w", encoding="utf-8") as f:
            cleaned = []
            for ex in self._examples:
                c = dict(ex)
                c.pop("embedding", None)
                cleaned.append(c)
            json.dump(cleaned, f, indent=2, ensure_ascii=False)
        logger.info("Exported %d examples to %s", len(cleaned), path)

    def import_json(self, path: str) -> int:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        count = 0
        for item in data:
            self.add(
                problem=item["problem"],
                reasoning=item["reasoning"],
                answer=item["answer"],
                tags=item.get("tags"),
                domain=item.get("domain", "general"),
                confidence=item.get("confidence", 0.7),
            )
            count += 1
        logger.info("Imported %d examples from %s", count, path)
        return count
