"""
Long-Horizon Autonomy Agent v2
Decomposes goals into 50+ subtasks across phases, executes with checkpoint/resume,
tracks phase-level progress, adapts plan based on results, and reports completion.

Architecture:
  Project (goal)
    → Goal decomposition (heuristic: goal → phases)
      → Phase 1: Knowledge (books, patterns, understanding)
      → Phase 2: Reasoning (inference, planning, transformer)
      → Phase 3: Active learning (curiosity, experiments, coding)
      → Phase 4: Synthesis (truth scoring, reflection, world model)
        → Milestones: "phase_name/cycle_N"
          → Actions (persistent SQLite queue)
            → Execution (delegates to LearningLoop step methods)
              → Recovery (retry_same → retry_modified → skip → escalate)
                → Plan adaptation (adjust remaining actions based on batch results)
                  → Session tracking (resume awareness, progress summaries)
"""

import json
import logging
import re
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple

from coding_loop import CodingEngine
from experiment_loop import ExperimentEngine

logger = logging.getLogger(__name__)

# Action type categories for phase construction
PHASE_DEFINITIONS = {
    "knowledge": {
        "label": "Knowledge Gathering",
        "description": "Process books, build knowledge base, detect patterns",
        "actions": {
            "process_books": 3.0,
            "deep_understanding": 2.0,
            "pattern_recognition": 2.0,
        },
    },
    "reasoning": {
        "label": "Deep Reasoning",
        "description": "Apply inference, planning, and LLM reasoning",
        "actions": {
            "reasoning": 2.0,
            "planning_reasoning": 1.5,
            "transformer_reasoning": 2.0,
            "root_cause_analysis": 0.5,
        },
    },
    "active_learning": {
        "label": "Active Learning",
        "description": "Curiosity questions, experiments, coding and geometry",
        "actions": {
            "curiosity": 2.0,
            "agentic_learning": 2.0,
            "code_engineer": 1.5,
            "code_learn": 0.5,
            "experiment": 1.0,
            "design_geometry": 0.5,
        },
    },
    "synthesis": {
        "label": "Synthesis & Quality",
        "description": "Truth scoring, reflection, world model, goal tracking",
        "actions": {
            "truth_decay": 1.0,
            "truth_system": 1.0,
            "contradiction_resolution": 1.0,
            "activation_spread": 1.0,
            "highway_prediction": 1.0,
            "reflection": 2.0,
            "goals": 0.5,
            "world_model": 2.0,
            "status_report": 0.3,
        },
    },
}

# All action types with descriptions for reporting
ALL_ACTION_DESCRIPTIONS = {
    "root_cause_analysis": "Analyze past incidents for root causes",
    "process_books": "Process unprocessed books into knowledge triples",
    "deep_understanding": "Run deep language understanding on book text",
    "pattern_recognition": "Detect cross-book relationship patterns",
    "reasoning": "Apply inference rules to derive new facts",
    "planning_reasoning": "Run planning agent reasoning cycle",
    "transformer_reasoning": "LLM-powered reasoning over knowledge graph",
    "reflection": "Find contradictions, weak knowledge, isolated concepts",
    "curiosity": "Generate and research curiosity questions",
    "truth_decay": "Apply truth decay based on recency and epistemic status",
    "truth_system": "Run truth system scoring and epistemic maintenance",
    "contradiction_resolution": "Resolve detected contradictions in knowledge",
    "activation_spread": "Spread activation through concept network",
    "highway_prediction": "Predict pathways from activated concepts",
    "goals": "Check and auto-complete goal progress",
    "world_model": "Extract causal edges from book text",
    "code_engineer": "Run a full coding cycle: write, run, debug, learn",
    "code_learn": "Mine past coding experiences for reusable patterns",
    "experiment": "Run a real-world experiment: hypothesis → test → update model",
    "design_geometry": "Generate 3D geometry (gear, drone frame) and export STL",
    "agentic_learning": "Self-directed learning via web search and hypothesis testing",
    "status_report": "Generate project status summary",
}

# Recovery strategy escalation ladder
RECOVERY_LADDER = ["retry_same", "retry_modified", "skip", "escalate"]

GOAL_PATTERNS = {
    "learn": {"knowledge": 1.5, "reasoning": 1.2, "active_learning": 0.8, "synthesis": 1.0},
    "study": {"knowledge": 1.5, "reasoning": 1.2, "active_learning": 0.8, "synthesis": 1.0},
    "research": {"knowledge": 1.5, "reasoning": 1.3, "active_learning": 0.8, "synthesis": 1.0},
    "understand": {"knowledge": 1.4, "reasoning": 1.3, "active_learning": 0.7, "synthesis": 1.0},
    "build": {"knowledge": 0.8, "reasoning": 1.0, "active_learning": 1.5, "synthesis": 0.8},
    "create": {"knowledge": 0.8, "reasoning": 1.0, "active_learning": 1.5, "synthesis": 0.8},
    "develop": {"knowledge": 0.8, "reasoning": 1.0, "active_learning": 1.5, "synthesis": 0.8},
    "code": {"knowledge": 0.5, "reasoning": 0.8, "active_learning": 2.0, "synthesis": 0.7},
    "program": {"knowledge": 0.5, "reasoning": 0.8, "active_learning": 2.0, "synthesis": 0.7},
    "design": {"knowledge": 0.8, "reasoning": 1.0, "active_learning": 1.5, "synthesis": 0.8},
    "geometry": {"knowledge": 0.5, "reasoning": 0.8, "active_learning": 1.5, "synthesis": 0.7},
    "improve": {"knowledge": 0.8, "reasoning": 1.2, "active_learning": 1.2, "synthesis": 1.0},
    "analyze": {"knowledge": 1.2, "reasoning": 1.5, "active_learning": 0.8, "synthesis": 1.2},
    "fix": {"knowledge": 0.8, "reasoning": 1.0, "active_learning": 1.5, "synthesis": 1.0},
    "optimize": {"knowledge": 0.8, "reasoning": 1.0, "active_learning": 1.5, "synthesis": 1.0},
}

# Default bias when no goal keywords match
DEFAULT_PHASE_BIAS = {"knowledge": 1.0, "reasoning": 1.0, "active_learning": 1.0, "synthesis": 1.0}


class LongHorizonAgent:
    """Executes long-horizon autonomous projects with phase decomposition."""

    def __init__(self, memory_system, learning_loop=None,
                 coding_engine=None, experiment_engine=None,
                 geometry_engine=None):
        self.ms = memory_system
        self.ll = learning_loop
        self.ce = coding_engine
        self.ee = experiment_engine
        self.ge = geometry_engine

    # ------------------------------------------------------------------ #
    #  Phase 1: Goal Decomposition
    # ------------------------------------------------------------------ #

    @staticmethod
    def _decompose_goal(goal: str) -> Dict[str, float]:
        """Analyze goal text and return phase bias multipliers."""
        goal_lower = goal.lower()
        bias = dict(DEFAULT_PHASE_BIAS)

        matched = False
        for keyword, phase_bias in GOAL_PATTERNS.items():
            if keyword in goal_lower:
                for phase, mult in phase_bias.items():
                    bias[phase] *= mult
                matched = True

        # If no keywords matched, use default (all 1.0)
        return bias

    @staticmethod
    def _build_phases(phase_bias: Dict[str, float],
                      max_actions: int) -> List[Dict[str, Any]]:
        """Build a list of phase configs with action counts proportionally allocated."""
        total_bias = sum(phase_bias.values())
        weighted = [(name, bias / total_bias) for name, bias in phase_bias.items()]

        phases = []
        allocated = 0
        for i, (name, frac) in enumerate(weighted):
            is_last = (i == len(weighted) - 1)
            if is_last:
                count = max_actions - allocated
            else:
                count = max(1, int(max_actions * frac))
            allocated += count
            phase_def = PHASE_DEFINITIONS[name]
            phases.append({
                "name": name,
                "label": phase_def["label"],
                "description": phase_def["description"],
                "action_count": count,
                "actions": dict(phase_def["actions"]),
            })

        return phases

    def _build_subtask_plan(self, phases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert phase configs into subtask descriptors with action detail.

        Distributes the remainder after floor-division to maintain total count.
        """
        subtasks = []
        for phase in phases:
            action_count = phase["action_count"]
            actions = phase["actions"]
            total_weight = sum(actions.values())

            raw: Dict[str, float] = {}
            for action_type, weight in actions.items():
                raw[action_type] = action_count * weight / total_weight

            alloc: Dict[str, int] = {}
            allocated = 0
            for action_type, val in raw.items():
                fl = int(val)
                alloc[action_type] = fl
                allocated += fl

            # Distribute remainder (actions lost to floor)
            remainder = action_count - allocated
            if remainder > 0:
                sorted_types = sorted(raw.keys(), key=lambda t: raw[t] - int(raw[t]), reverse=True)
                for i in range(remainder):
                    alloc[sorted_types[i % len(sorted_types)]] += 1

            subtask = {
                "phase": phase["name"],
                "label": phase["label"],
                "description": phase["description"],
                "action_allocation": alloc,
            }
            subtasks.append(subtask)
        return subtasks

    # ------------------------------------------------------------------ #
    #  Phase 2: Project Creation
    # ------------------------------------------------------------------ #

    def create_project(self, goal: str, max_actions: int = 1000,
                       schedule: Optional[List[tuple]] = None) -> str:
        """Create a new long-horizon project with goal-aware phase decomposition.

        Uses heuristic goal analysis to allocate actions across phases.
        If a custom schedule is provided, falls back to the original cycle-based approach.
        """
        project_id = f"proj_{uuid.uuid4().hex[:12]}"

        # Custom schedule fallback
        if schedule is not None:
            self._enqueue_schedule(project_id, schedule, max_actions, goal)
            return project_id

        # Phase-based decomposition
        phase_bias = self._decompose_goal(goal)
        phases = self._build_phases(phase_bias, max_actions)
        subtasks = self._build_subtask_plan(phases)

        total_enqueued = 0
        subtask_goals = []
        for subtask in subtasks:
            phase_name = subtask["phase"]
            milestone_base = phase_name
            subtask_actions = []
            for action_type, alloc in subtask["action_allocation"].items():
                if alloc == 0:
                    continue
                description = ALL_ACTION_DESCRIPTIONS.get(action_type, action_type)
                for cycle in range(alloc):
                    milestone = f"{milestone_base}/cycle_{cycle}"
                    action_id = self.ms.enqueue_action(
                        project_id=project_id,
                        milestone=milestone,
                        description=f"[{phase_name}] {description}",
                        action_type=action_type,
                        params={
                            "phase": phase_name,
                            "cycle": cycle,
                            "goal": goal,
                        },
                        recovery_strategy="retry_same",
                    )
                    subtask_actions.append(action_id)
                    total_enqueued += 1
            subtask_goals.append({
                "phase": phase_name,
                "actions_enqueued": len(subtask_actions),
                "action_ids": subtask_actions,
            })

        # Store project metadata
        self.ms.add_memory(
            f"Project {project_id}: {goal} — {total_enqueued} actions across "
            f"{len(subtasks)} subtask phases: "
            + ", ".join(f"{s['phase']}({s['actions_enqueued']})" for s in subtask_goals),
            source="long_horizon_agent",
            confidence=0.9
        )

        logger.info(
            "Created project %s: %d actions, %d phases — %s",
            project_id, total_enqueued, len(subtasks), goal,
        )
        for s in subtask_goals:
            logger.debug("  Subtask '%s': %d actions", s["phase"], s["actions_enqueued"])

        return project_id

    def _enqueue_schedule(self, project_id: str, schedule: List[tuple],
                          max_actions: int, goal: str) -> None:
        """Original cycle-based enqueue for custom schedules (legacy fallback)."""
        _, total_cycles = self._calculate_schedule(schedule, max_actions)
        cycle = 0
        total_enqueued = 0
        while total_enqueued < max_actions:
            for action_type, description, weight in schedule:
                if total_enqueued >= max_actions:
                    break
                for _ in range(max(1, int(weight))):
                    if total_enqueued >= max_actions:
                        break
                    milestone = f"cycle_{cycle}"
                    self.ms.enqueue_action(
                        project_id=project_id,
                        milestone=milestone,
                        description=f"[{action_type}] {description}",
                        action_type=action_type,
                        params={"cycle": cycle, "total_cycles": total_cycles, "goal": goal},
                        recovery_strategy="retry_same",
                    )
                    total_enqueued += 1
            cycle += 1
        self.ms.add_memory(
            f"Project {project_id}: {goal} — {total_enqueued} actions across {cycle} cycles",
            source="long_horizon_agent",
            confidence=0.9
        )
        logger.info("Created project %s: %d actions, %d cycles (legacy) — %s",
                     project_id, total_enqueued, cycle, goal)

    # ------------------------------------------------------------------ #
    #  Phase 3: Progress Tracking
    # ------------------------------------------------------------------ #

    def get_project_status(self, project_id: str) -> Dict[str, Any]:
        """Full status for one project including phase-level breakdown."""
        cursor = self.ms.conn.cursor()

        # Overall stats
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status = 'done' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                SUM(CASE WHEN status = 'stalled' THEN 1 ELSE 0 END) as stalled,
                SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) as running,
                SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
                MIN(created_at) as started_at,
                MAX(CASE WHEN status = 'done' THEN completed_at ELSE NULL END) as last_completed
            FROM action_queue WHERE project_id = ?
        """, (project_id,))
        row = cursor.fetchone()
        if row is None or row["total"] == 0:
            return {"project_id": project_id, "exists": False}

        total = row["total"]
        completed = row["completed"] or 0
        failed = row["failed"] or 0
        stalled = row["stalled"] or 0
        running = row["running"] or 0
        pending = row["pending"] or 0
        pct = round(completed / max(total, 1) * 100, 1)

        # Velocity and ETA
        cursor.execute("""
            SELECT COUNT(*) as cnt, MIN(completed_at) as first_done
            FROM action_queue
            WHERE project_id = ? AND status = 'done' AND completed_at IS NOT NULL
        """, (project_id,))
        done_stats = cursor.fetchone()
        done_count = done_stats["cnt"] if done_stats else 0
        first_done = done_stats["first_done"] if done_stats else None

        eta_str = None
        velocity = 0.0
        session_hint = None
        if done_count >= 5 and first_done:
            now = datetime.utcnow()
            first = datetime.strptime(first_done, "%Y-%m-%d %H:%M:%S")
            elapsed_hours = (now - first).total_seconds() / 3600
            if elapsed_hours > 0:
                velocity = done_count / elapsed_hours
                remaining = total - completed
                if velocity > 0:
                    eta_hours = remaining / velocity
                    eta_dt = now + timedelta(hours=eta_hours)
                    eta_str = eta_dt.strftime("%Y-%m-%d %H:%M")

        # Phase-level breakdown
        cursor.execute("""
            SELECT
                milestone,
                COUNT(*) as phase_total,
                SUM(CASE WHEN status = 'done' THEN 1 ELSE 0 END) as phase_completed,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as phase_failed
            FROM action_queue
            WHERE project_id = ?
            GROUP BY milestone
            ORDER BY milestone
        """, (project_id,))
        phases_raw = [dict(r) for r in cursor.fetchall()]

        # Aggregate into phase groups (milestones are "phase_name/cycle_N")
        phase_groups: Dict[str, Dict[str, Any]] = {}
        for p in phases_raw:
            phase_name = p["milestone"].split("/")[0] if "/" in p["milestone"] else p["milestone"]
            if phase_name not in phase_groups:
                phase_groups[phase_name] = {
                    "phase": phase_name,
                    "total": 0,
                    "completed": 0,
                    "failed": 0,
                }
            phase_groups[phase_name]["total"] += p["phase_total"]
            phase_groups[phase_name]["completed"] += p["phase_completed"]
            phase_groups[phase_name]["failed"] += p["phase_failed"]

        phases = list(phase_groups.values())
        for p in phases:
            p["pct"] = round(p["completed"] / max(p["total"], 1) * 100, 1)

        # Determine current phase
        current_phase = None
        if phases:
            for p in phases:
                if p["completed"] < p["total"]:
                    current_phase = p["phase"]
                    break

        # Detect session gap (multi-day awareness)
        cursor.execute("""
            SELECT MAX(completed_at) as last_done
            FROM action_queue
            WHERE project_id = ? AND status = 'done'
        """, (project_id,))
        last_row = cursor.fetchone()
        if last_row and last_row["last_done"]:
            last_done = datetime.strptime(last_row["last_done"], "%Y-%m-%d %H:%M:%S")
            gap_hours = (datetime.utcnow() - last_done).total_seconds() / 3600
            if gap_hours > 2:
                session_hint = f"Resuming after {round(gap_hours, 1)}h gap"

        return {
            "project_id": project_id,
            "exists": True,
            "total_actions": total,
            "completed": completed,
            "failed": failed,
            "stalled": stalled,
            "running": running,
            "pending": pending,
            "percent_complete": pct,
            "actions_remaining": total - completed,
            "velocity_actions_per_hour": round(velocity, 1),
            "eta": eta_str,
            "phases": phases,
            "current_phase": current_phase,
            "session_hint": session_hint,
            "phase_count": len(phases),
        }

    def list_projects(self) -> List[Dict[str, Any]]:
        """List all incomplete projects."""
        return self.ms.get_incomplete_projects()

    def get_action_history(self, project_id: str, limit: int = 50,
                           status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get action history for a project."""
        cursor = self.ms.conn.cursor()
        if status:
            cursor.execute("""
                SELECT * FROM action_queue
                WHERE project_id = ? AND status = ?
                ORDER BY id DESC LIMIT ?
            """, (project_id, status, limit))
        else:
            cursor.execute("""
                SELECT * FROM action_queue
                WHERE project_id = ?
                ORDER BY id DESC LIMIT ?
            """, (project_id, limit))
        return [dict(r) for r in cursor.fetchall()]

    # ------------------------------------------------------------------ #
    #  Phase 4: Execution
    # ------------------------------------------------------------------ #

    def execute_next_batch(self, project_id: str, batch_size: int = 10,
                           ll: Optional[Any] = None) -> Dict[str, Any]:
        """Pop and execute the next N pending actions for a project.

        After execution, triggers plan adaptation based on results.
        """
        loop = ll or self.ll
        if loop is None:
            return {"status": "error", "error": "no learning loop provided"}

        actions = self.ms.get_pending_actions(project_id=project_id, limit=batch_size)
        if not actions:
            return {"status": "done", "message": "no pending actions", "executed": 0}

        results = []
        for action in actions:
            result = self._execute_single(action, loop)
            results.append(result)

        summary = self._summarize_batch(results)

        # Plan adaptation: adjust remaining actions based on batch results
        adaptation = self._adapt_plan(project_id, results)
        if adaptation.get("adjusted", False):
            summary["plan_adapted"] = True
            summary["adaptation"] = adaptation

        return summary

    def execute_all_remaining(self, project_id: str, batch_size: int = 10,
                              ll: Optional[Any] = None,
                              progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Execute all remaining actions for a project in batches with adaptation."""
        loop = ll or self.ll
        if loop is None:
            return {"status": "error", "error": "no learning loop provided"}

        total_start = time.time()
        all_batches = []
        grand_total = {"executed": 0, "succeeded": 0, "failed": 0, "stalled": 0,
                       "adaptations": 0}

        while True:
            batch = self.execute_next_batch(project_id, batch_size=batch_size, ll=loop)
            all_batches.append(batch)
            grand_total["executed"] += batch["executed"]
            grand_total["succeeded"] += batch["succeeded"]
            grand_total["failed"] += batch["failed"]
            grand_total["stalled"] += batch.get("stalled", 0)
            if batch.get("plan_adapted"):
                grand_total["adaptations"] += 1

            status = self.get_project_status(project_id)

            if progress_callback:
                progress_callback(status)

            if batch["executed"] == 0:
                break

        total_elapsed = time.time() - total_start
        return {
            "status": "ok",
            "project_id": project_id,
            "total_batches": len(all_batches),
            **grand_total,
            "total_elapsed_seconds": round(total_elapsed, 2),
            "batches": all_batches,
        }

    def resume_incomplete(self, batch_size: int = 10,
                          ll: Optional[Any] = None) -> Dict[str, Any]:
        """Find and resume all incomplete projects with multi-day session awareness."""
        projects = self.ms.get_incomplete_projects()
        if not projects:
            return {"status": "ok", "message": "no incomplete projects", "resumed": 0}

        results = {}
        for proj in projects:
            pid = proj["project_id"]
            remaining = proj["total"] - proj["completed"]
            status = self.get_project_status(pid)
            session_info = status.get("session_hint", "")
            logger.info(
                "Resuming project %s (%d remaining) %s",
                pid, remaining, session_info,
            )
            results[pid] = self.execute_all_remaining(pid, batch_size=batch_size, ll=ll)

        return {
            "status": "ok",
            "projects_resumed": len(results),
            "results": results,
        }

    # ------------------------------------------------------------------ #
    #  Phase 5: Plan Adaptation
    # ------------------------------------------------------------------ #

    def _adapt_plan(self, project_id: str,
                    results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze batch results and adjust remaining plan.

        Checks for:
        - Consistently empty results (e.g., no unprocessed books) → skip remaining
        - High failure rates → reduce retries for that action type
        - Low-value action types → deprioritize
        """
        if not results:
            return {"adjusted": False, "reason": "no results to analyze"}

        adjustments = []
        modified = False

        # Group results by action type
        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for r in results:
            t = r.get("type", "unknown")
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(r)

        cursor = self.ms.conn.cursor()

        for action_type, type_results in by_type.items():
            done = [r for r in type_results if r.get("status") == "done"]
            failed = [r for r in type_results if r.get("status") in ("failed", "noop")]

            # Check for consistently empty results (e.g., process_books with no books)
            if done:
                # Look at summaries to detect zero-value actions
                empty_count = 0
                for r in done:
                    summary = r.get("summary", "")
                    if any(indicator in summary for indicator in
                           ["0 books", "0 new", "0 triples", "no unprocessed",
                            "no concepts", "disabled", "skipped", "noop"]):
                        empty_count += 1

                if empty_count == len(done) and len(done) >= 2:
                    # All recent results for this type are empty — skip remaining
                    cursor.execute("""
                        UPDATE action_queue
                        SET status = 'stalled',
                            result_summary = 'skipped: consistently empty results',
                            completed_at = CURRENT_TIMESTAMP
                        WHERE project_id = ?
                          AND action_type = ?
                          AND status = 'pending'
                    """, (project_id, action_type))
                    skipped = cursor.rowcount
                    if skipped > 0:
                        self.ms.conn.commit()
                        adjustments.append(f"skipped {skipped} pending '{action_type}' (empty)")
                        modified = True

            # Check for high failure rates
            if len(failed) >= 2 and len(failed) / max(len(type_results), 1) > 0.5:
                cursor.execute("""
                    UPDATE action_queue
                    SET status = 'stalled',
                        result_summary = 'skipped: high failure rate',
                        completed_at = CURRENT_TIMESTAMP
                    WHERE project_id = ?
                      AND action_type = ?
                      AND status = 'pending'
                """, (project_id, action_type))
                skipped = cursor.rowcount
                if skipped > 0:
                    self.ms.conn.commit()
                    adjustments.append(f"skipped {skipped} pending '{action_type}' (high failure)")
                    modified = True

        return {
            "adjusted": modified,
            "adjustments": adjustments,
            "reason": "; ".join(adjustments) if adjustments else "no changes needed",
        }

    # ------------------------------------------------------------------ #
    #  Internal Execution
    # ------------------------------------------------------------------ #

    def _execute_single(self, action: Dict[str, Any], loop) -> Dict[str, Any]:
        """Execute a single action and update its status."""
        action_id = action["id"]
        action_type = action["action_type"]
        retry_count = action["retry_count"]
        max_retries = action["max_retries"]

        self.ms.update_action_status(action_id, "running")

        step_method = self._resolve_step(action_type)
        if step_method is None:
            self.ms.update_action_status(action_id, "done",
                                         result_summary=f"no-op: unknown type '{action_type}'")
            return {"id": action_id, "type": action_type, "status": "noop"}

        start = time.time()
        try:
            result = step_method(loop)
            elapsed = time.time() - start
            status = "done"
            summary = self._summarize_result(action_type, result)
            logger.debug("Action %d [%s] OK (%.2fs): %s", action_id, action_type, elapsed, summary)
            self.ms.update_action_status(action_id, status, result_summary=summary)
            return {"id": action_id, "type": action_type, "status": "done",
                    "elapsed": round(elapsed, 2), "summary": summary}
        except Exception as e:
            elapsed = time.time() - start
            logger.warning("Action %d [%s] FAILED (%.2fs): %s",
                           action_id, action_type, elapsed, e)
            new_retry = retry_count + 1
            strategy = self._next_strategy(action["recovery_strategy"], new_retry, max_retries)

            if strategy == "skip":
                self.ms.update_action_status(action_id, "failed",
                                             retry_count=new_retry,
                                             result_summary=f"skip: {e}")
                return {"id": action_id, "type": action_type, "status": "failed",
                        "elapsed": round(elapsed, 2), "error": str(e)}
            else:
                self.ms.update_action_status(action_id, "pending",
                                             retry_count=new_retry,
                                             checkpoint_data=json.dumps({
                                                 "last_error": str(e),
                                                 "last_elapsed": round(elapsed, 2),
                                                 "strategy": strategy,
                                             }))
                return {"id": action_id, "type": action_type, "status": "retry",
                        "elapsed": round(elapsed, 2), "error": str(e), "strategy": strategy}

    def _resolve_step(self, action_type: str) -> Optional[Callable]:
        """Map action_type to a LearningLoop step method."""
        MAPPING = {
            "root_cause_analysis": lambda loop: loop.step_0_root_cause_analysis(),
            "process_books": lambda loop: loop.step_1_process_books(),
            "deep_understanding": lambda loop: loop.step_1b_deep_understanding(),
            "pattern_recognition": lambda loop: loop.step_2_pattern_recognition(),
            "reasoning": lambda loop: loop.step_3_reasoning(),
            "planning_reasoning": lambda loop: loop.step_3c_planning_reasoning(),
            "transformer_reasoning": lambda loop: loop.step_3b_transformer_reasoning(),
            "reflection": lambda loop: loop.step_4_reflection(),
            "curiosity": lambda loop: loop.step_5_curiosity(),
            "truth_decay": lambda loop: loop.step_6_truth_decay(),
            "truth_system": lambda loop: loop.step_6b_truth_system(),
            "contradiction_resolution": lambda loop: loop.step_6c_contradiction_resolution(),
            "activation_spread": lambda loop: loop.step_6d_activation_spread(),
            "highway_prediction": lambda loop: loop.step_6e_highway_prediction(),
            "goals": lambda loop: loop.step_7_goals(),
            "world_model": lambda loop: loop.step_8_world_model(),
            "agentic_learning": lambda loop: loop.step_9_agentic_learning(),
            "code_engineer": lambda loop: self.ce.run_cycle() if self.ce else {"status": "skipped", "reason": "CodingEngine not initialized"},
            "code_learn": lambda loop: self._run_code_learn() if self.ce else {"status": "skipped", "reason": "CodingEngine not initialized"},
            "experiment": lambda loop: self.ee.run_cycle(coding_engine=self.ce) if self.ee else {"status": "skipped", "reason": "ExperimentEngine not initialized"},
            "design_geometry": lambda loop: self._run_design_geometry(),
            "status_report": self._generate_status_report,
        }
        fn = MAPPING.get(action_type)
        if fn is None:
            return None
        if action_type in ("status_report",):
            return lambda loop: fn()
        return fn

    def _next_strategy(self, current: str, retry_count: int, max_retries: int) -> str:
        """Walk the recovery ladder based on retry count."""
        if retry_count >= max_retries:
            return "skip"
        if current == "retry_same" and retry_count >= 1:
            return "retry_modified"
        return current

    def _summarize_result(self, action_type: str, result: Dict[str, Any]) -> str:
        """Produce a one-line summary from a step result dict."""
        hints = {
            "process_books": lambda r: f"{r.get('books_processed', 0)} books, {r.get('triples_learned', 0)} triples",
            "reasoning": lambda r: f"{r.get('newly_stored', 0)} new facts",
            "reflection": lambda r: f"{r.get('contradictions', 0)} contradictions, {r.get('weak_concepts', 0)} weak",
            "curiosity": lambda r: f"{r.get('questions_asked', 0)} questions, {r.get('triples_learned', 0)} triples",
            "truth_decay": lambda r: f"{r.get('decayed', 0)} decayed",
            "truth_system": lambda r: f"{r.get('scored', 0)} scored, {r.get('promoted', 0)} promoted",
            "contradiction_resolution": lambda r: f"{r.get('resolved', 0)} resolved",
            "activation_spread": lambda r: f"{r.get('active_nodes', 0)} active",
            "highway_prediction": lambda r: f"{r.get('paths_discovered', 0)} paths",
            "goals": lambda r: f"{r.get('auto_completed', 0)} completed",
            "world_model": lambda r: f"{r.get('causal_edges_extracted', 0)} edges",
            "agentic_learning": lambda r: f"{r.get('hypotheses_tested', 0)} tested, {r.get('hypotheses_confirmed', 0)} confirmed",
            "code_engineer": lambda r: f"cycle={'completed' if r.get('code_succeeded') else 'failed'}, errors={r.get('errors', 0)}",
            "code_learn": lambda r: f"{r.get('lessons', 0)} lessons, {r.get('experiences', 0)} experiences",
            "experiment": lambda r: f"{r.get('evaluation', {}).get('outcome', 'n/a')}",
            "design_geometry": lambda r: f"type={r.get('stats',{}).get('teeth','frame')} ok={r.get('success')}",
            "root_cause_analysis": lambda r: f"{r.get('rca', {}).get('total_incidents', 0)} incidents",
            "deep_understanding": lambda r: f"{r.get('inferences_stored', 0)} inferences, {r.get('triples', 0)} triples",
            "pattern_recognition": lambda r: f"{r.get('stored', 0)} patterns",
            "planning_reasoning": lambda r: f"{r.get('plans_generated', 0)} plans",
            "transformer_reasoning": lambda r: f"{r.get('stored', 0)} stored",
        }
        fn = hints.get(action_type)
        if fn:
            try:
                return fn(result)
            except Exception:
                pass

        for v in result.values():
            if isinstance(v, (int, float)) and v > 0:
                return f"{v}"
        return "ok"

    def _run_design_geometry(self) -> Dict[str, Any]:
        """Execute a design_geometry step — delegate to GeometryEngine."""
        if self.ge is None:
            from geometry_engine import GeometryEngine
            self.ge = GeometryEngine()
        try:
            cursor = self.ms.conn.cursor()
            cursor.execute("""
                SELECT event, result FROM experiences
                WHERE domain = 'design' ORDER BY created_at DESC LIMIT 1
            """)
            row = cursor.fetchone()
            if row:
                spec_raw = row["result"] if row["result"] else row["event"]
                try:
                    import json as _json
                    spec = _json.loads(spec_raw) if isinstance(spec_raw, str) else spec_raw
                except Exception:
                    spec = {"type": "gear", "teeth": 10, "module": 1.0}
            else:
                spec = {"type": "gear", "teeth": 10, "module": 1.0}

            result = self.ge.generate_design_to_stl(spec)
            logger.info("Design geometry: %s — %s",
                         spec.get("type", "?"),
                         "OK" if result.get("success") else f"FAIL: {result.get('error')}")
            return result
        except Exception as e:
            logger.warning("_run_design_geometry failed: %s", e)
            return {"success": False, "error": str(e)}

    def _run_code_learn(self) -> Dict[str, Any]:
        """Mine past coding experiences for reusable lessons."""
        if self.ce is None:
            logger.warning("LongHorizonAgent: _run_code_learn skipped — CodingEngine not initialized")
            return {"status": "skipped", "reason": "CodingEngine not initialized"}
        try:
            cursor = self.ms.conn.cursor()
            cursor.execute("""
                SELECT event, result, lesson, outcome_score, created_at
                FROM experiences
                WHERE domain = 'coding'
                ORDER BY outcome_score DESC
                LIMIT 20
            """)
            rows = cursor.fetchall()
            logger.debug("LongHorizonAgent: _run_code_learn found %d coding experiences", len(rows))

            lessons_count = 0
            for row in rows:
                lesson_text = row["lesson"] if row["lesson"] else row["event"]
                score = 0.5 + (row["outcome_score"] or 0.5) * 0.5
                self.ce.store_lesson(row["event"], lesson_text, score)
                lessons_count += 1

            logger.info("LongHorizonAgent: _run_code_learn stored %d lessons from %d experiences",
                         lessons_count, len(rows))
            return {
                "lessons": lessons_count,
                "experiences": len(rows),
                "status": "ok",
            }
        except Exception as e:
            logger.warning("LongHorizonAgent: _run_code_learn failed: %s", e)
            return {"status": "error", "error": str(e)}

    def _generate_status_report(self) -> Dict[str, Any]:
        """Generate a global status report — standalone step."""
        try:
            stats = self.ms.get_stats() if hasattr(self.ms, 'get_stats') else {}
            pending = self.ms.count_pending_actions()
            return {
                "memories": stats.get("memories", 0),
                "relationships": stats.get("relationships", 0),
                "concepts": stats.get("concepts", 0),
                "pending_actions": pending,
                "status": "ok",
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _summarize_batch(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize a batch of action results."""
        executed = len(results)
        succeeded = sum(1 for r in results if r["status"] == "done")
        failed = sum(1 for r in results if r["status"] == "failed")
        retried = sum(1 for r in results if r["status"] == "retry")
        stalled = sum(1 for r in results if r["status"] == "stalled")
        noop = sum(1 for r in results if r["status"] == "noop")
        total_elapsed = sum(r.get("elapsed", 0) for r in results)
        return {
            "executed": executed,
            "succeeded": succeeded,
            "failed": failed,
            "retried": retried,
            "stalled": stalled,
            "noop": noop,
            "total_elapsed_seconds": round(total_elapsed, 2),
            "results": results,
        }

    @staticmethod
    def _calculate_schedule(schedule: List[tuple], max_actions: int) -> tuple:
        """Calculate how many cycles fit in max_actions."""
        actions_per_cycle = sum(max(1, int(w)) for _, _, w in schedule)
        total_cycles = max_actions // max(actions_per_cycle, 1)
        return actions_per_cycle, total_cycles
