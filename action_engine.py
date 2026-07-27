"""
Jafar -- Action Engine

The Action Loop: plan -> execute -> observe -> update

Extended with:
- Code sandbox execution (safe Python execution)
- Code memory (persistent code artifact store)
- Agentic learning loop (hypothesis-driven active learning)
- Web search integration
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from geometry_reasoner import GeometryReasoner

logger = logging.getLogger(__name__)


class ActionEngine:
    """Universal hub orchestrating all action loops.

    Maintains backward-compatible plan/execute/observe/update pipeline
    while adding code sandbox, code memory, and agentic learning.
    """

    def __init__(self, memory_system=None, qa_system=None, vector_db=None,
                 curiosity_engine=None, world_model_engine=None,
                 goal_system=None, reasoning_system=None,
                 code_sandbox=None, code_memory=None,
                 agentic_loop=None, tool_registry=None,
                 code_generator=None, planner=None,
                 truth_system=None, memory_pathways=None):
        self.memory = memory_system
        self.qa = qa_system
        self.vdb = vector_db
        self.curiosity = curiosity_engine
        self.wme = world_model_engine
        self.goals = goal_system
        self.reasoning = reasoning_system
        self.code_sandbox = code_sandbox
        self.code_memory = code_memory
        self.agentic_loop = agentic_loop
        self.tools = tool_registry
        self.code_generator = code_generator
        self.planner = planner
        self.truth_system = truth_system
        self.memory_pathways = memory_pathways
        self.geometry_reasoner = GeometryReasoner(
            picogk_tool=self.tools.get("picogk") if self.tools else None,
            memory_system=self.memory,
        )
        self._wire_geometry_reasoner()
        self._history: List[Dict[str, Any]] = []
        self._code_history: List[Dict[str, Any]] = []
        self._agentic_history: List[Dict[str, Any]] = []

    def _wire_geometry_reasoner(self) -> None:
        """Connect every available component to the GeometryReasoner."""
        gr = self.geometry_reasoner

        if self.planner and hasattr(self.planner, 'planning_agent'):
            gr.connect_planning_agent(self.planner.planning_agent)
        elif hasattr(self, 'planning_agent') and self.planning_agent:
            gr.connect_planning_agent(self.planning_agent)

        if hasattr(self, 'memory_pathways') and self.memory_pathways:
            gr.connect_pathways(self.memory_pathways)

        if hasattr(self, 'truth_system') and self.truth_system:
            gr.connect_truth_system(self.truth_system)

        if self.curiosity:
            gr.connect_curiosity(self.curiosity)

        try:
            from geometry_engine import GeometryEngine
            engine = GeometryEngine()
            gr.connect_geometry_engine(engine)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Legacy pipeline (backward compatible)
    # ------------------------------------------------------------------

    def plan(self, goal: str) -> List[Dict[str, Any]]:
        steps = []

        steps.append({
            "tool": "search_graph",
            "params": {"query": goal},
            "goal": goal,
            "description": "Search knowledge graph for concepts related to goal",
        })

        steps.append({
            "tool": "ask_question",
            "params": {"question": goal},
            "goal": goal,
            "description": "Ask a question about the goal topic",
        })

        steps.append({
            "tool": "world_model_overview",
            "params": {"goal": goal},
            "goal": goal,
            "description": "Check causal dynamics in the world model",
        })

        steps.append({
            "tool": "curiosity_explore",
            "params": {"goal": goal},
            "goal": goal,
            "description": "Find knowledge gaps via curiosity engine",
        })

        steps.append({
            "tool": "reasoning_infer",
            "params": {},
            "goal": goal,
            "description": "Infer new relationships from existing knowledge",
        })

        steps.append({
            "tool": "web_search",
            "params": {"query": goal},
            "goal": goal,
            "description": "Search the web for external information",
        })

        steps.append({
            "tool": "agentic_hypothesis",
            "params": {"goal": goal},
            "goal": goal,
            "description": "Generate and test hypotheses via agentic loop",
        })

        steps.append({
            "tool": "code_generate",
            "params": {"goal": goal},
            "goal": goal,
            "description": "Generate code for the goal via CodeGenerator",
        })

        return steps

    def execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        tool = step["tool"]
        params = step["params"]
        result: Dict[str, Any] = {"tool": tool, "success": False, "data": None, "error": None}

        try:
            if tool == "search_graph":
                query = params.get("query", "")
                if self.memory:
                    concepts = self.memory.search_concepts(query)
                    graph = self.memory.get_concept_graph(concepts[:5]) if concepts else {}
                    result["data"] = {"concepts": concepts[:10], "graph": graph}
                    result["success"] = True
                else:
                    result["error"] = "MemorySystem not initialized"

            elif tool == "ask_question":
                question = params.get("question", "")
                if self.qa:
                    answer = self.qa.answer_question(question)
                    result["data"] = answer
                    result["success"] = True
                else:
                    result["error"] = "QuestionAnsweringSystem not initialized"

            elif tool == "world_model_overview":
                if self.wme:
                    dynamics = self.wme.get_system_dynamics()
                    result["data"] = {"dynamics": dynamics}
                    result["success"] = True
                else:
                    result["error"] = "WorldModelEngine not initialized"

            elif tool == "world_model_trade_offs":
                concept = params.get("concept", "")
                if self.wme and concept:
                    to = self.wme.get_trade_offs(concept)
                    result["data"] = to
                    result["success"] = True
                else:
                    result["error"] = "WorldModelEngine not initialized or no concept"

            elif tool == "curiosity_explore":
                if self.curiosity:
                    cycle_result = self.curiosity.run_curiosity_cycle(max_questions=3)
                    result["data"] = cycle_result
                    result["success"] = True
                else:
                    result["error"] = "CuriosityEngine not initialized"

            elif tool == "reasoning_infer":
                if self.reasoning:
                    inferred = self.reasoning.infer_all()
                    result["data"] = inferred
                    result["success"] = True
                else:
                    result["error"] = "ReasoningSystem not initialized"

            elif tool == "web_search":
                if self.tools:
                    web_result = self.tools.execute("web_search",
                                                     query=params.get("query", ""))
                    result["data"] = web_result
                    result["success"] = web_result.get("success", False)
                    result["error"] = web_result.get("error")
                else:
                    result["error"] = "ToolRegistry not initialized"

            elif tool == "agentic_hypothesis":
                if self.agentic_loop:
                    ag_result = self.agentic_loop.run(
                        params.get("goal", ""), max_hypotheses=2, search_web=True)
                    result["data"] = ag_result
                    result["success"] = ag_result.get("status") == "ok"
                else:
                    result["error"] = "AgenticLoop not initialized"

            elif tool == "geometry_reason":
                q = params.get("query", params.get("goal", ""))
                gr_result = self.geometry_reasoner.reason_and_generate(q)
                result["data"] = gr_result
                result["success"] = gr_result.get("success", False)
                result["error"] = gr_result.get("error")

            elif tool == "picogk":
                if self.tools:
                    op = params.get("operation", "")
                    if not op:
                        op = "mesh_voxelize"
                    q = params.get("query", params.get("goal", ""))
                    gk_result = self.tools.execute(
                        "picogk", operation=op, query=q,
                        params={k: v for k, v in params.items()
                                if k not in ("operation", "query", "goal")})
                    result["data"] = gk_result
                    result["success"] = gk_result.get("success", False)
                    result["error"] = gk_result.get("error")
                else:
                    result["error"] = "ToolRegistry not initialized"

            elif tool == "code_generate":
                if self.code_generator:
                    gen_result = self.code_generator.generate(
                        params.get("goal", ""), store_on_success=True)
                    result["data"] = gen_result
                    result["success"] = gen_result.get("status") == "ok"
                else:
                    result["error"] = "CodeGenerator not initialized"

            else:
                result["error"] = f"Unknown tool: {tool}"

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Action step failed: {tool}: {e}")

        return result

    def observe(self, step_result: Dict[str, Any]) -> Dict[str, Any]:
        insights: Dict[str, Any] = {
            "concepts_found": 0,
            "relationships_found": 0,
            "new_triples_learned": 0,
            "has_data": False,
            "summary": "",
        }

        data = step_result.get("data")
        tool = step_result.get("tool", "")

        if not data or not step_result.get("success"):
            insights["summary"] = f"{tool}: no data returned"
            return insights

        if tool == "search_graph":
            concepts = data.get("concepts", [])
            graph = data.get("graph", {})
            insights["concepts_found"] = len(concepts)
            insights["relationships_found"] = sum(len(rels) for rels in graph.values())
            insights["has_data"] = len(concepts) > 0
            insights["summary"] = f"Found {len(concepts)} concepts, {insights['relationships_found']} relationships"

        elif tool == "ask_question":
            answer = data.get("answer", "")
            triples = data.get("triples", [])
            insights["new_triples_learned"] = len(triples)
            insights["has_data"] = bool(answer)
            insights["summary"] = f"Answer received, {len(triples)} triples learned"

        elif tool == "world_model_overview":
            dynamics = data.get("dynamics", {})
            insights["relationships_found"] = dynamics.get("total_causal_edges", 0)
            insights["has_data"] = dynamics.get("total_causal_edges", 0) > 0
            insights["summary"] = f"{dynamics.get('total_causal_edges', 0)} causal edges in world model"

        elif tool == "world_model_trade_offs":
            insights["relationships_found"] = len(data.get("increases", [])) + len(data.get("decreases", []))
            insights["has_data"] = insights["relationships_found"] > 0
            insights["summary"] = (f"Trade-offs: {len(data.get('increases', []))} increases, "
                                   f"{len(data.get('decreases', []))} decreases")

        elif tool == "curiosity_explore":
            questions_asked = data.get("questions_asked", 0)
            total_learned = sum(
                q.get("triples_learned", 0) for q in data.get("results", [])
            )
            insights["new_triples_learned"] = total_learned
            insights["has_data"] = questions_asked > 0
            insights["summary"] = f"Asked {questions_asked} questions, learned {total_learned} triples"

        elif tool == "reasoning_infer":
            insights["relationships_found"] = data.get("newly_stored", 0)
            insights["has_data"] = data.get("newly_stored", 0) > 0
            insights["summary"] = f"Inferred {data.get('newly_stored', 0)} new relationships"

        elif tool == "web_search":
            content = data.get("content", "")
            total = data.get("total_results", 0)
            insights["has_data"] = bool(content)
            insights["summary"] = f"Web search returned {total} results"

        elif tool == "agentic_hypothesis":
            tested = data.get("hypotheses_tested", 0)
            confirmed = data.get("confirmed", 0)
            delta = data.get("total_confidence_delta", 0)
            insights["has_data"] = tested > 0
            insights["summary"] = (f"Agentic: {tested} hypotheses, "
                                   f"{confirmed} confirmed, Δ={delta:+.3f}")

        elif tool == "geometry_reason":
            op = data.get("operation", "unknown")
            filepath = data.get("filepath", "")
            stats = data.get("stats", {})
            verts = stats.get("vertices", 0)
            insights["has_data"] = data.get("success", False)
            insights["relationships_found"] = 2
            insights["summary"] = (f"GeometryReason: {op} - {verts} vertices "
                                   f"-> {os.path.basename(filepath) if filepath else 'none'}")

        elif tool == "picogk":
            op = data.get("operation", "unknown")
            filepath = data.get("filepath", "")
            stats = data.get("stats", {})
            verts = stats.get("vertices", 0)
            faces = stats.get("faces", 0)
            insights["has_data"] = data.get("success", False)
            insights["relationships_found"] = 1
            insights["summary"] = (f"PicoGK: {op} - {verts} vertices, "
                                   f"{faces} faces -> {filepath}")

        elif tool == "code_generate":
            status = data.get("status", "")
            name = data.get("name", "")
            attempts = data.get("attempts", 0)
            insights["has_data"] = status == "ok"
            insights["summary"] = (f"CodeGen: {status} — '{name}' "
                                   f"in {attempts} attempt(s)")

        return insights

    def update(self, insights: Dict[str, Any], step: Dict[str, Any]) -> bool:
        if not self.memory:
            return False

        try:
            outcome_score = 0.0
            if insights.get("concepts_found", 0) > 0:
                outcome_score += 0.2
            if insights.get("relationships_found", 0) > 0:
                outcome_score += 0.3
            if insights.get("new_triples_learned", 0) > 0:
                outcome_score += 0.5
            if not insights.get("has_data"):
                outcome_score -= 0.1

            self.memory.record_experience(
                event=f"action:{step['tool']}",
                result=insights.get("summary", ""),
                lesson=(f"{step['goal']}: {insights['summary']}"
                        if insights.get("summary") else None),
                outcome_score=outcome_score,
                domain="action_loop",
            )
            return True
        except Exception as e:
            logger.error(f"Failed to record experience: {e}")
            return False

    def run(self, goal: str, max_steps: int = 10) -> Dict[str, Any]:
        start_time = datetime.now().isoformat()
        steps = self.plan(goal)[:max_steps]

        results: List[Dict[str, Any]] = []
        total_insights: Dict[str, int] = {
            "concepts_found": 0,
            "relationships_found": 0,
            "new_triples_learned": 0,
        }

        for i, step in enumerate(steps):
            step_result = self.execute_step(step)
            insights = self.observe(step_result)
            self.update(insights, step)

            total_insights["concepts_found"] += insights.get("concepts_found", 0)
            total_insights["relationships_found"] += insights.get("relationships_found", 0)
            total_insights["new_triples_learned"] += insights.get("new_triples_learned", 0)

            results.append({
                "step": i + 1,
                "tool": step["tool"],
                "description": step.get("description", ""),
                "status": "completed" if step_result["success"] else "failed",
                "error": step_result.get("error"),
                "insights": insights,
            })

        summary: Dict[str, Any] = {
            "goal": goal,
            "steps_planned": len(steps),
            "steps_completed": sum(1 for r in results if r["status"] == "completed"),
            "steps_failed": sum(1 for r in results if r["status"] == "failed"),
            "total_concepts_found": total_insights["concepts_found"],
            "total_relationships_found": total_insights["relationships_found"],
            "total_triples_learned": total_insights["new_triples_learned"],
            "results": results,
            "started_at": start_time,
            "completed_at": datetime.now().isoformat(),
        }

        self._history.append(summary)
        return summary

    # ------------------------------------------------------------------
    # Code sandbox operations
    # ------------------------------------------------------------------

    def run_code(self, code: str, timeout: float = 5.0) -> Dict[str, Any]:
        if not self.code_sandbox:
            return {"success": False, "error": "CodeSandbox not initialized"}
        result = self.code_sandbox.run_code(code, timeout=timeout)
        self._code_history.append({
            "code_length": len(code),
            "success": result.get("success", False),
            "execution_time": result.get("execution_time", 0),
        })
        return result

    def run_function(self, func_source: str, func_name: str,
                     args: Optional[List[Any]] = None,
                     kwargs: Optional[Dict[str, Any]] = None,
                     timeout: float = 5.0) -> Dict[str, Any]:
        if not self.code_sandbox:
            return {"success": False, "error": "CodeSandbox not initialized"}
        return self.code_sandbox.run_function(
            func_source, func_name, args=args, kwargs=kwargs, timeout=timeout)

    # ------------------------------------------------------------------
    # Code memory operations
    # ------------------------------------------------------------------

    def scan_code(self, file_path: str) -> List[Dict[str, Any]]:
        if not self.code_memory:
            return []
        artifacts = self.code_memory.scan_file(file_path)
        return [{"name": a.name, "kind": a.kind, "signature": a.signature,
                 "file_path": a.file_path, "line_start": a.line_start}
                for a in artifacts]

    def search_code(self, query: str, top_k: int = 5,
                    kind: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self.code_memory:
            return []
        artifacts = self.code_memory.search(query, top_k=top_k, kind=kind)
        return [{"name": a.name, "kind": a.kind, "signature": a.signature,
                 "docstring": a.docstring[:100] if a.docstring else "",
                 "file_path": a.file_path, "usage_count": a.usage_count,
                 "success_rate": (a.success_count / max(a.usage_count, 1))}
                for a in artifacts]

    def store_code(self, name: str, kind: str, source: str,
                   signature: str = "", docstring: str = "",
                   tags: Optional[List[str]] = None,
                   file_path: str = "") -> Dict[str, Any]:
        if not self.code_memory:
            return {"success": False, "error": "CodeMemory not initialized"}
        art = self.code_memory.store(name, kind, source, signature=signature,
                                     docstring=docstring, tags=tags,
                                     file_path=file_path)
        return {
            "success": True,
            "artifact_id": art.artifact_id,
            "name": art.name,
            "kind": art.kind,
        }

    def get_code_stats(self) -> Dict[str, Any]:
        if not self.code_memory:
            return {"total": 0}
        all_arts = self.code_memory.get_all(limit=10000)
        by_kind: Dict[str, int] = {}
        for a in all_arts:
            by_kind[a.kind] = by_kind.get(a.kind, 0) + 1
        return {
            "total": len(all_arts),
            "by_kind": by_kind,
        }

    # ------------------------------------------------------------------
    # Code generation
    # ------------------------------------------------------------------

    def generate_code(self, goal: str, max_attempts: int = 5,
                      store_on_success: bool = True) -> Dict[str, Any]:
        if not self.code_generator:
            return {"status": "error", "goal": goal,
                    "error": "CodeGenerator not initialized"}
        result = self.code_generator.generate(
            goal, max_attempts=max_attempts,
            store_on_success=store_on_success)
        self._code_history.append({
            "code_length": len(result.get("code", "")),
            "success": result.get("status") == "ok",
        })
        return result

    # ------------------------------------------------------------------
    # Agentic learning loop
    # ------------------------------------------------------------------

    def run_agentic_cycle(self, goal: str, max_hypotheses: int = 3,
                          search_web: bool = True) -> Dict[str, Any]:
        if not self.agentic_loop:
            return {"goal": goal, "status": "no_agentic_loop"}
        result = self.agentic_loop.run(
            goal, max_hypotheses=max_hypotheses, search_web=search_web)
        self._agentic_history.append(result)
        return result

    def run_agentic_with_code(self, goal: str, code_snippet: str,
                              timeout: float = 10.0) -> Dict[str, Any]:
        if not self.agentic_loop:
            return {"goal": goal, "status": "no_agentic_loop"}
        return self.agentic_loop.run_with_code(goal, code_snippet, timeout=timeout)

    # ------------------------------------------------------------------
    # Planner-based execution
    # ------------------------------------------------------------------

    def plan_and_execute(self, goal: str) -> Dict[str, Any]:
        if not self.planner:
            return {"goal": goal, "status": "no_planner",
                    "error": "Planner not initialized"}
        plan = self.planner.create_plan(goal)
        plan = self.planner.execute_plan(plan)
        result = plan.to_dict()
        result["status"] = plan.status
        return result

    # ------------------------------------------------------------------
    # Web search
    # ------------------------------------------------------------------

    def web_search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        if not self.tools:
            return {"success": False, "error": "ToolRegistry not initialized"}
        return self.tools.execute("web_search", query=query, max_results=max_results)

    # ------------------------------------------------------------------
    # History & status
    # ------------------------------------------------------------------

    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self._history[-limit:]

    def get_code_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self._code_history[-limit:]

    def get_agentic_history(self, limit: int = 5) -> List[Dict[str, Any]]:
        return self._agentic_history[-limit:]

    def get_status(self) -> Dict[str, Any]:
        total = len(self._history)
        completed = sum(1 for h in self._history if h.get("steps_failed", 0) == 0)
        total_rels = sum(h.get("total_relationships_found", 0) for h in self._history)

        agentic_total = len(self._agentic_history)
        agentic_confirmed = sum(
            h.get("confirmed", 0) for h in self._agentic_history)

        return {
            "total_runs": total,
            "successful_runs": completed,
            "total_relationships_found": total_rels,
            "recent_goals": [h.get("goal", "") for h in self._history[-5:]],
            "code_executions": len(self._code_history),
            "agentic_cycles": agentic_total,
            "agentic_hypotheses_confirmed": agentic_confirmed,
        }
