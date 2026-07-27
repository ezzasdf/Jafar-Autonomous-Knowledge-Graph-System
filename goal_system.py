"""
Jafar — Goal System
Manages self-directed learning objectives
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import sqlite3
import json

from memory_system import MemorySystem

logger = logging.getLogger(__name__)

GOAL_TABLE = """
    CREATE TABLE IF NOT EXISTS learning_goals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        description TEXT NOT NULL,
        focus_concept TEXT,
        status TEXT DEFAULT 'active',
        priority INTEGER DEFAULT 5,
        progress REAL DEFAULT 0.0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
"""

GOAL_MIGRATIONS = [
    ("learning_goals", "focus_concept", "TEXT"),
    ("learning_goals", "progress", "REAL DEFAULT 0.0"),
]


class GoalSystem:
    """Manages self-directed learning objectives."""

    def __init__(self, memory: MemorySystem, books_path: Optional[str] = None):
        self.memory = memory
        self.books_path = books_path
        self._ensure_table()
        logger.debug("GoalSystem initialized")

    def _ensure_table(self) -> None:
        cursor = self.memory.conn.cursor()
        cursor.execute(GOAL_TABLE)
        for table, col, col_def in GOAL_MIGRATIONS:
            try:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_def}")
            except sqlite3.OperationalError:
                pass
        self.memory.conn.commit()

    def set_goal(self, description: str, focus_concept: Optional[str] = None,
                 priority: int = 5) -> int:
        cursor = self.memory.conn.cursor()
        cursor.execute("""
            INSERT INTO learning_goals (description, focus_concept, priority)
            VALUES (?, ?, ?)
        """, (description, focus_concept, priority))
        self.memory.conn.commit()
        goal_id = cursor.lastrowid
        logger.info("Goal set: [%d] %s (focus: %s, priority: %d)",
                    goal_id, description, focus_concept or "none", priority)
        return goal_id

    def get_goals(self, status: Optional[str] = "active") -> List[Dict[str, Any]]:
        cursor = self.memory.conn.cursor()
        if status:
            cursor.execute("""
                SELECT * FROM learning_goals WHERE status = ?
                ORDER BY priority DESC, created_at DESC
            """, (status,))
        else:
            cursor.execute("""
                SELECT * FROM learning_goals ORDER BY priority DESC, created_at DESC
            """)
        return [dict(r) for r in cursor.fetchall()]

    def update_goal(self, goal_id: int, **kwargs) -> bool:
        allowed = {"status", "priority", "progress", "description", "focus_concept"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return False
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [goal_id]
        cursor = self.memory.conn.cursor()
        cursor.execute(f"""
            UPDATE learning_goals
            SET {set_clause}, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, values)
        self.memory.conn.commit()
        logger.debug("Goal %d updated: %s", goal_id, updates)
        return cursor.rowcount > 0

    def assess_knowledge_gap(self, goal_id: int) -> Dict[str, Any]:
        goals = self.get_goals()
        goal = next((g for g in goals if g["id"] == goal_id), None)
        if goal is None:
            return {"error": "Goal not found"}
        focus = goal["focus_concept"]
        if not focus:
            return {
                "goal": goal["description"],
                "note": "No focus concept — cannot assess gaps",
            }
        return self._assess_for_concept(focus)

    def _assess_for_concept(self, concept: str) -> Dict[str, Any]:
        cursor = self.memory.conn.cursor()
        graph = self.memory.get_concept_graph(concept)
        rel_count = len(graph["relationships"])

        cursor.execute("""
            SELECT COUNT(*) as cnt FROM relationships
            WHERE source_concept = ? OR target_concept = ?
        """, (concept.lower(), concept.lower()))
        total_rels = cursor.fetchone()["cnt"]
        avg_conf = 0.0
        if total_rels > 0:
            cursor.execute("""
                SELECT AVG(confidence) as avg_c FROM relationships
                WHERE source_concept = ? OR target_concept = ?
            """, (concept.lower(), concept.lower()))
            avg_conf = cursor.fetchone()["avg_c"] or 0.0

        related_concepts = set()
        for r in graph["relationships"]:
            related_concepts.add(r["target"])
        known_related = len(related_concepts)

        related_count = cursor.execute("""
            SELECT COUNT(DISTINCT source_concept || target_concept) as cnt
            FROM relationships
        """).fetchone()["cnt"]

        coverage = known_related / max(related_count, 1)

        score = min(1.0, (total_rels / 20) * 0.5 + avg_conf * 0.3 + coverage * 0.2)
        score = round(score, 2)

        return {
            "concept": concept,
            "total_relationships": total_rels,
            "avg_confidence": round(avg_conf, 2),
            "unique_connections": known_related,
            "graph_coverage": round(coverage, 2),
            "knowledge_score": score,
            "gaps": [
                f"Low relationship count ({total_rels})" if total_rels < 5 else None,
                f"Low confidence ({avg_conf:.2f})" if avg_conf < 0.5 else None,
                f"Low graph coverage ({coverage:.2f})" if coverage < 0.2 else None,
            ],
        }

    def recommend_books(self, goal_id: int,
                        available_books: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        goals = self.get_goals()
        goal = next((g for g in goals if g["id"] == goal_id), None)
        if goal is None:
            return []
        focus = goal["focus_concept"]
        if not focus:
            return []

        results: List[Dict[str, Any]] = []
        focus_lower = focus.lower()
        for book in available_books:
            title = book.get("title", "")
            score = 0
            reasons: List[str] = []
            if focus_lower in title.lower():
                score += 0.8
                reasons.append("Title matches focus concept")
            for word in focus_lower.split():
                if word in title.lower():
                    score += 0.3
                    reasons.append(f"Title contains '{word}'")
            if score > 0:
                results.append({
                    "book": book,
                    "relevance_score": round(min(1.0, score), 2),
                    "reasons": reasons,
                })

        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        logger.debug("Goal %d: found %d relevant books out of %d available",
                     goal_id, len(results), len(available_books))
        return results

    def calculate_progress(self, goal_id: int) -> float:
        assessment = self.assess_knowledge_gap(goal_id)
        if "error" in assessment:
            return 0.0
        score = assessment.get("knowledge_score", 0.0)
        self.update_goal(goal_id, progress=score)
        logger.debug("Goal %d progress: %.2f", goal_id, score)
        return score

    def update_all_progress(self) -> Dict[str, Any]:
        goals = self.get_goals(status="active")
        results = {}
        for g in goals:
            progress = self.calculate_progress(g["id"])
            results[g["id"]] = progress
        logger.info("Updated progress for %d active goals", len(goals))
        return results

    def complete_goal(self, goal_id: int, threshold: float = 0.8) -> bool:
        progress = self.calculate_progress(goal_id)
        if progress >= threshold:
            self.update_goal(goal_id, status="completed", progress=progress)
            logger.info("Goal %d completed at progress %.2f", goal_id, progress)
            return True
        logger.debug("Goal %d progress %.2f < threshold %.2f, not completing",
                     goal_id, progress, threshold)
        return False

    def get_stats(self) -> Dict[str, Any]:
        cursor = self.memory.conn.cursor()
        cursor.execute("SELECT COUNT(*) as cnt FROM learning_goals WHERE status = 'active'")
        active = cursor.fetchone()["cnt"]
        cursor.execute("SELECT COUNT(*) as cnt FROM learning_goals WHERE status = 'completed'")
        completed = cursor.fetchone()["cnt"]
        return {
            "active_goals": active,
            "completed_goals": completed,
            "total_goals": active + completed,
        }
