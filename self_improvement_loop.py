"""
Jafar -- Self-Improvement Loop

The final stage that makes Jafar self-directing:

  detect weakness
  -> prioritize
  -> generate improvement task
  -> execute task
  -> evaluate outcome
  -> update reasoning strategy
  -> record

Uses existing subsystems (ReflectionSystem, CuriosityEngine, ReasoningSystem,
ActionEngine, GoalSystem, WorldModelEngine) in a structured autonomous cycle.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

WEAKNESS_PRIORITIES = {
    "contradiction": 5,
    "isolated": 4,
    "weak": 3,
    "knowledge_gap": 2,
    "low_confidence": 1,
}


class SelfImprovementLoop:
    """Self-Improvement Loop.

    Runs autonomously to detect knowledge quality issues, generate and execute
    improvement tasks, measure outcomes, and update its own reasoning strategy.
    """

    def __init__(self, memory_system=None, reflection_system=None,
                 curiosity_engine=None, reasoning_system=None,
                 goal_system=None, action_engine=None,
                 world_model_engine=None,
                 embedding_generator=None, vector_db=None):
        self.memory = memory_system
        self.reflection = reflection_system
        self.curiosity = curiosity_engine
        self.reasoning = reasoning_system
        self.goals = goal_system
        self.action = action_engine
        self.wme = world_model_engine
        self.eg = embedding_generator
        self.vdb = vector_db
        self._cycle_count = 0
        self._ensure_tables()
        self._load_strategy()

    def _ensure_tables(self):
        """Create tables if they do not exist (delegated to memory_system)."""
        if not self.memory:
            return
        try:
            cursor = self.memory.conn.cursor()
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='improvement_cycles'
            """)
            if not cursor.fetchone():
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS improvement_cycles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        cycle_number INTEGER,
                        weaknesses_found INTEGER DEFAULT 0,
                        tasks_generated INTEGER DEFAULT 0,
                        tasks_completed INTEGER DEFAULT 0,
                        tasks_failed INTEGER DEFAULT 0,
                        strategy_updates INTEGER DEFAULT 0,
                        outcome_score REAL DEFAULT 0.0,
                        details TEXT,
                        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        completed_at TIMESTAMP
                    )
                """)
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='inference_strategy'
            """)
            if not cursor.fetchone():
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS inference_strategy (
                        rule_key TEXT PRIMARY KEY,
                        relation1 TEXT NOT NULL,
                        relation2 TEXT NOT NULL,
                        inferred_relation TEXT NOT NULL,
                        confidence_multiplier REAL DEFAULT 0.7,
                        attempts INTEGER DEFAULT 0,
                        successes INTEGER DEFAULT 0,
                        failures INTEGER DEFAULT 0,
                        avg_outcome_score REAL DEFAULT 0.0,
                        is_active INTEGER DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            self.memory.conn.commit()
        except Exception as e:
            logger.error("Failed to ensure tables: %s", e)

    def _load_strategy(self):
        """Load stored strategy into module-level INFERENCE_RULES."""
        try:
            from reasoning_system import INFERENCE_RULES

            if not self.memory:
                return

            cursor = self.memory.conn.cursor()
            cursor.execute("SELECT * FROM inference_strategy ORDER BY rule_key")
            rows = cursor.fetchall()

            if not rows:
                for rule_idx, (r1, r2, inferred, conf) in enumerate(INFERENCE_RULES):
                    key = f"rule_{rule_idx}"
                    cursor.execute("""
                        INSERT OR IGNORE INTO inference_strategy
                            (rule_key, relation1, relation2, inferred_relation, confidence_multiplier)
                        VALUES (?, ?, ?, ?, ?)
                    """, (key, r1, r2, inferred, conf))
                self.memory.conn.commit()
                return

            for i, (r1, r2, inferred, _) in enumerate(INFERENCE_RULES):
                for row in rows:
                    if (row["relation1"] == r1 and row["relation2"] == r2
                            and row["inferred_relation"] == inferred):
                        stored_mult = row["confidence_multiplier"]
                        INFERENCE_RULES[i] = (r1, r2, inferred, stored_mult)
                        break
        except Exception as e:
            logger.error("Failed to load strategy: %s", e)

    def detect_weaknesses(self) -> List[Dict[str, Any]]:
        """Run reflection + goal assessment to find all knowledge weaknesses.

        Returns prioritized list of weaknesses with type, focus, severity, reason.
        """
        weaknesses: List[Dict[str, Any]] = []

        if not self.reflection:
            return weaknesses

        try:
            report = self.reflection.run_full_reflection()

            for c in report.get("contradictions", [])[:10]:
                weaknesses.append({
                    "type": "contradiction",
                    "focus": c["source_concept"],
                    "priority": WEAKNESS_PRIORITIES["contradiction"],
                    "reason": (f"{c['source_concept']} {c['relation']} -> "
                               f"{c['targets']} ({c['target_count']} targets)"),
                    "detail": dict(c),
                })

            for w in report.get("weak_concepts", [])[:10]:
                weaknesses.append({
                    "type": "weak",
                    "focus": w["name"],
                    "priority": WEAKNESS_PRIORITIES["weak"],
                    "reason": (f"Only {w['rel_count']} rels, "
                               f"avg conf {w['avg_confidence']:.2f}"),
                    "detail": dict(w),
                })

            for name in report.get("isolated_concepts", [])[:10]:
                weaknesses.append({
                    "type": "isolated",
                    "focus": name,
                    "priority": WEAKNESS_PRIORITIES["isolated"],
                    "reason": "Concept has zero relationships",
                    "detail": {"name": name},
                })

            if self.goals:
                for g in self.goals.get_goals(status="active")[:5]:
                    fc = g.get("focus_concept")
                    if fc:
                        weaknesses.append({
                            "type": "knowledge_gap",
                            "focus": fc,
                            "priority": WEAKNESS_PRIORITIES["knowledge_gap"],
                            "reason": f"Active goal #{g['id']}: {g['description']}",
                            "detail": dict(g),
                        })
        except Exception as e:
            logger.error("Error detecting weaknesses: %s", e)

        return weaknesses

    def prioritize(self, weaknesses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort weaknesses by priority descending, stable."""
        return sorted(weaknesses, key=lambda w: w.get("priority", 0), reverse=True)

    def generate_tasks(self, weaknesses: List[Dict[str, Any]],
                       max_tasks: int = 5) -> List[Dict[str, Any]]:
        """Convert prioritized weaknesses into actionable improvement tasks."""
        tasks: List[Dict[str, Any]] = []

        for w in self.prioritize(weaknesses)[:max_tasks]:
            wtype = w["type"]
            focus = w["focus"]

            if wtype == "contradiction":
                tasks.append({
                    "type": "resolve_contradiction",
                    "focus": focus,
                    "description": f"Investigate contradictions involving {focus}",
                    "action": "ask_question",
                    "params": {
                        "question": w.get("reason", f"What is the relationship of {focus}?"),
                    },
                })

            elif wtype == "isolated":
                tasks.append({
                    "type": "explore_concept",
                    "focus": focus,
                    "description": f"Learn about {focus} and find its connections",
                    "action": "curiosity_learn",
                    "params": {"concept": focus},
                })

            elif wtype == "weak":
                tasks.append({
                    "type": "strengthen_knowledge",
                    "focus": focus,
                    "description": f"Gather more information about {focus}",
                    "action": "ask_question",
                    "params": {
                        "question": f"How does {focus} relate to other concepts?",
                    },
                })

            elif wtype == "knowledge_gap":
                tasks.append({
                    "type": "close_gap",
                    "focus": focus,
                    "description": f"Research {focus} to close knowledge gap",
                    "action": "action_loop",
                    "params": {"goal": f"Understand {focus} and its relationships"},
                })

            elif wtype == "low_confidence":
                tasks.append({
                    "type": "validate_knowledge",
                    "focus": focus,
                    "description": f"Validate and reinforce knowledge about {focus}",
                    "action": "reasoning_validate",
                    "params": {"concept": focus},
                })

        return tasks

    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single improvement task using available subsystems."""
        result: Dict[str, Any] = {
            "success": False, "task_type": task.get("type"),
            "focus": task.get("focus"), "details": {}, "error": None,
        }

        action = task.get("action")
        params = task.get("params", {})

        try:
            if action == "ask_question" and self.action:
                question = params.get("question", "")
                step = {"tool": "ask_question", "params": {"question": question},
                        "goal": f"Improve knowledge of {task.get('focus')}"}
                step_result = self.action.execute_step(step)
                insights = self.action.observe(step_result)
                self.action.update(insights, step)
                result["success"] = step_result["success"]
                result["details"]["insights"] = insights

            elif action == "curiosity_learn" and self.curiosity:
                concept = params.get("concept", "")
                cycle_result = self.curiosity.run_curiosity_cycle(max_questions=3)
                result["success"] = cycle_result.get("questions_asked", 0) > 0
                result["details"]["cycle"] = cycle_result

            elif action == "action_loop" and self.action:
                goal = params.get("goal", task.get("description", ""))
                loop_result = self.action.run(goal, max_steps=5)
                result["success"] = loop_result["steps_completed"] > 0
                result["details"]["loop"] = loop_result

            elif action == "reasoning_validate" and self.reasoning and self.vdb:
                validation = self.reasoning.cross_validate(
                    self.vdb, embedding_generator=self.eg
                )
                result["success"] = True
                result["details"]["validation"] = validation

            elif action == "reasoning_infer" and self.reasoning:
                inferred = self.reasoning.infer_all()
                result["success"] = True
                result["details"]["inferred"] = inferred

            else:
                result["error"] = f"Cannot execute action '{action}' - subsystem not available"

        except Exception as e:
            result["error"] = str(e)
            logger.error("Task execution failed: %s", e)

        return result

    def evaluate(self, before_reflection: Dict[str, Any],
                 after_reflection: Dict[str, Any]) -> Dict[str, Any]:
        """Compare before/after reflection to measure improvement."""
        score = 0.0
        changes: Dict[str, Any] = {}

        b_contra = before_reflection.get("contradiction_count", 0)
        a_contra = after_reflection.get("contradiction_count", 0)
        changes["contradictions"] = {"before": b_contra, "after": a_contra,
                                     "delta": b_contra - a_contra}
        if a_contra < b_contra:
            score += 1.0

        b_weak = before_reflection.get("weak_count", 0)
        a_weak = after_reflection.get("weak_count", 0)
        changes["weak_concepts"] = {"before": b_weak, "after": a_weak,
                                    "delta": b_weak - a_weak}
        if a_weak < b_weak:
            score += 0.5

        b_isolated = before_reflection.get("isolated_count", 0)
        a_isolated = after_reflection.get("isolated_count", 0)
        changes["isolated_concepts"] = {"before": b_isolated, "after": a_isolated,
                                        "delta": b_isolated - a_isolated}
        if a_isolated < b_isolated:
            score += 0.5

        score = min(score, 2.0)

        return {
            "outcome_score": score,
            "changes": changes,
            "improved": score > 0,
        }

    def _save_cycle(self, cycle_data: Dict[str, Any]) -> int:
        """Persist cycle data to improvement_cycles table."""
        if not self.memory:
            return -1
        try:
            cursor = self.memory.conn.cursor()
            cursor.execute("""
                INSERT INTO improvement_cycles
                    (cycle_number, weaknesses_found, tasks_generated,
                     tasks_completed, tasks_failed, strategy_updates,
                     outcome_score, details, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cycle_data.get("cycle_number", 0),
                cycle_data.get("weaknesses_found", 0),
                cycle_data.get("tasks_generated", 0),
                cycle_data.get("tasks_completed", 0),
                cycle_data.get("tasks_failed", 0),
                cycle_data.get("strategy_updates", 0),
                cycle_data.get("outcome_score", 0.0),
                json.dumps(cycle_data.get("details", {})),
                datetime.now().isoformat(),
            ))
            self.memory.conn.commit()
            return cursor.lastrowid
        except Exception as e:
            logger.error("Failed to save cycle: %s", e)
            return -1

    def update_strategy(self, outcome: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust inference rule multipliers based on task outcomes.

        Positive outcome -> nudge multiplier up (capped at 0.95).
        Negative outcome -> nudge multiplier down (floored at 0.3).
        """
        from reasoning_system import INFERENCE_RULES

        changes: List[Dict[str, Any]] = []
        outcome_score = outcome.get("outcome_score", 0.0)

        if not self.memory:
            return {"updates": 0, "changes": changes}

        cursor = self.memory.conn.cursor()
        cursor.execute("SELECT * FROM inference_strategy ORDER BY rule_key")
        rows = cursor.fetchall()

        updates = 0
        for row in rows:
            key = row["rule_key"]
            current_mult = row["confidence_multiplier"]
            attempts = row["attempts"] + 1
            new_avg = ((row["avg_outcome_score"] * row["attempts"]
                        + outcome_score) / attempts) if attempts > 0 else outcome_score

            delta = 0.0
            if outcome_score > 0.5:
                delta = 0.05
            elif outcome_score > 0:
                delta = 0.02
            elif outcome_score < -0.3:
                delta = -0.05
            elif outcome_score < 0:
                delta = -0.02

            new_mult = max(0.3, min(0.95, current_mult + delta))

            cursor.execute("""
                UPDATE inference_strategy
                SET confidence_multiplier = ?,
                    attempts = ?,
                    avg_outcome_score = ?,
                    last_updated = ?
                WHERE rule_key = ?
            """, (new_mult, attempts, new_avg, datetime.now().isoformat(), key))

            if abs(new_mult - current_mult) > 0.001:
                updates += 1
                changes.append({
                    "rule": key,
                    "relation1": row["relation1"],
                    "relation2": row["relation2"],
                    "inferred": row["inferred_relation"],
                    "old_mult": current_mult,
                    "new_mult": new_mult,
                    "delta": round(delta, 3),
                })

            for i, (r1, r2, inferred, _) in enumerate(INFERENCE_RULES):
                if (r1 == row["relation1"] and r2 == row["relation2"]
                        and inferred == row["inferred_relation"]):
                    INFERENCE_RULES[i] = (r1, r2, inferred, new_mult)
                    break

        self.memory.conn.commit()
        return {"updates": updates, "changes": changes}

    def get_strategy_summary(self) -> List[Dict[str, Any]]:
        """Show current inference rules with their stats."""
        if not self.memory:
            return []

        cursor = self.memory.conn.cursor()
        cursor.execute("""
            SELECT * FROM inference_strategy
            ORDER BY attempts DESC, rule_key ASC
        """)
        return [dict(r) for r in cursor.fetchall()]

    def run_cycle(self, max_tasks: int = 5) -> Dict[str, Any]:
        """One complete self-improvement cycle.

        Steps:
          1. Detect weaknesses via reflection
          2. Prioritize by severity
          3. Generate improvement tasks
          4. Execute each task
          5. Run reflection again to evaluate
          6. Update reasoning strategy
          7. Save cycle to DB
        """
        self._cycle_count += 1
        cycle_num = self._cycle_count
        logger.info("Starting self-improvement cycle #%d", cycle_num)

        before = self.reflection.run_full_reflection() if self.reflection else {}

        weaknesses = self.detect_weaknesses()
        tasks = self.generate_tasks(weaknesses, max_tasks=max_tasks)

        completed = 0
        failed = 0
        task_summaries: List[Dict[str, Any]] = []

        for task in tasks:
            result = self.execute_task(task)
            if result["success"]:
                completed += 1
            else:
                failed += 1
            task_summaries.append({
                "type": task["type"],
                "focus": task["focus"],
                "success": result["success"],
                "error": result.get("error"),
            })

        after = self.reflection.run_full_reflection() if self.reflection else {}

        evaluation = self.evaluate(before, after)

        strategy_result = self.update_strategy(evaluation)

        record_outcome = self.memory and self.memory.record_experience(
            event=f"self_improvement:cycle_{cycle_num}",
            result=(f"Completed {completed}/{len(tasks)} tasks, "
                    f"score {evaluation['outcome_score']:.2f}"),
            lesson=f"Self-improvement cycle {cycle_num}: {evaluation['changes']}",
            outcome_score=evaluation["outcome_score"],
            domain="self_improvement",
        ) if self.memory else None

        cycle_data = {
            "cycle_number": cycle_num,
            "weaknesses_found": len(weaknesses),
            "tasks_generated": len(tasks),
            "tasks_completed": completed,
            "tasks_failed": failed,
            "strategy_updates": strategy_result["updates"],
            "outcome_score": evaluation["outcome_score"],
            "details": {
                "weaknesses": [{"type": w["type"], "focus": w["focus"],
                                "reason": w["reason"]} for w in weaknesses[:10]],
                "tasks": task_summaries,
                "evaluation": evaluation,
                "strategy_updates": strategy_result["changes"],
            },
        }
        self._save_cycle(cycle_data)

        logger.info("Cycle #%d complete: %d/%d tasks, score %.2f",
                    cycle_num, completed, len(tasks), evaluation["outcome_score"])

        return {
            "cycle_number": cycle_num,
            "weaknesses_found": len(weaknesses),
            "tasks_generated": len(tasks),
            "tasks_completed": completed,
            "tasks_failed": failed,
            "strategy_updates": strategy_result["updates"],
            "outcome_score": evaluation["outcome_score"],
            "evaluation": evaluation,
            "task_results": task_summaries,
            "strategy_changes": strategy_result["changes"],
        }

    def run(self, cycles: int = 1, tasks_per_cycle: int = 5) -> List[Dict[str, Any]]:
        """Run multiple self-improvement cycles."""
        results = []
        for _ in range(cycles):
            result = self.run_cycle(max_tasks=tasks_per_cycle)
            results.append(result)
        return results

    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get past cycles from DB."""
        if not self.memory:
            return []
        try:
            cursor = self.memory.conn.cursor()
            cursor.execute("""
                SELECT * FROM improvement_cycles
                ORDER BY cycle_number DESC
                LIMIT ?
            """, (limit,))
            rows = []
            for r in cursor.fetchall():
                row = dict(r)
                if row.get("details"):
                    try:
                        row["details"] = json.loads(row["details"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                rows.append(row)
            return rows
        except Exception as e:
            logger.error("Failed to get history: %s", e)
            return []
