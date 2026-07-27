"""
Jafar — Planner
Decomposes goals into subtasks, executes them with tools,
evaluates results, retries failures, and learns from outcomes.
Replaces the hardcoded 8-step pipeline with dynamic plan generation.
"""
import json
from typing import Dict, List, Any, Optional, Callable
import logging
from datetime import datetime, timezone
import re

logger = logging.getLogger(__name__)

# ── Plan Templates ──────────────────────────────────────────────────
# Matched against goal keywords to produce a structured decomposition.

PLAN_TEMPLATES: List[Dict] = [
    {
        "keywords": ["flask", "website", "web app", "web application",
                     "django", "fastapi"],
        "type": "web_app",
        "subtasks": [
            {"description": "Create project structure",
             "tool": "code_generate",
             "params": {"goal": "create project structure"},
             "expected_outcome": "Project directory and files created"},
            {"description": "Build backend",
             "tool": "code_generate",
             "params": {"goal": "implement backend routes and logic"},
             "expected_outcome": "Backend code written"},
            {"description": "Create frontend templates",
             "tool": "code_generate",
             "params": {"goal": "create templates and views"},
             "expected_outcome": "Templates created"},
            {"description": "Add styling",
             "tool": "code_generate",
             "params": {"goal": "add CSS and styling"},
             "expected_outcome": "Styles applied"},
            {"description": "Test and fix errors",
             "tool": "code_generate",
             "params": {"goal": "test the application and fix errors"},
             "expected_outcome": "Tests pass"},
        ],
    },
    {
        "keywords": ["3d", "3d model", "geometry", "mesh", "voxel",
                     "stl", "obj", "cad", "lattice", "boolean",
                     "picogk", "offset mesh", "smooth mesh",
                     "gearbox", "gear ", "bracket", "phone stand",
                     "robot arm", "joint", "housing", "printable"],
        "type": "geometry",
        "subtasks": [
            {"description": "Search knowledge graph for relevant geometry concepts",
             "tool": "search_graph",
             "params": {"query": ""},
             "expected_outcome": "Existing geometry knowledge found"},
            {"description": "Reason about geometry and generate 3D model via PicoGK",
             "tool": "geometry_reason",
             "params": {},
             "expected_outcome": "3D model generated from knowledge-grounded parameters"},
            {"description": "Store geometry result in knowledge graph",
             "tool": "search_graph",
             "params": {"query": ""},
             "expected_outcome": "Geometry metadata stored in knowledge graph"},
        ],
    },
    {
        "keywords": ["learn", "research", "what is", "explain",
                     "understand", "how does", "tell me about"],
        "type": "research",
        "subtasks": [
            {"description": "Search knowledge graph",
             "tool": "search_graph",
             "params": {"query": ""},
             "expected_outcome": "Existing knowledge found"},
            {"description": "Search web for information",
             "tool": "web_search",
             "params": {"query": ""},
             "expected_outcome": "Web results gathered"},
            {"description": "Extract concepts and store",
             "tool": "curiosity_explore",
             "params": {"goal": ""},
             "expected_outcome": "New triples stored"},
        ],
    },
    {
        "keywords": ["code", "function", "implement", "write",
                     "generate", "script", "program"],
        "type": "code",
        "subtasks": [
            {"description": "Analyze request",
             "tool": "reasoning_infer",
             "params": {},
             "expected_outcome": "Requirements understood"},
            {"description": "Generate code",
             "tool": "code_generate",
             "params": {"goal": ""},
             "expected_outcome": "Code generated and tested"},
            {"description": "Verify correctness",
             "tool": "search_graph",
             "params": {"query": ""},
             "expected_outcome": "Code verified"},
        ],
    },
    {
        "keywords": ["debug", "fix", "error", "bug", "issue",
                     "not working", "broken", "failing"],
        "type": "debug",
        "subtasks": [
            {"description": "Search for known solutions",
             "tool": "search_graph",
             "params": {"query": ""},
             "expected_outcome": "Known solutions found"},
            {"description": "Search web for error",
             "tool": "web_search",
             "params": {"query": ""},
             "expected_outcome": "Web solutions found"},
            {"description": "Generate fix",
             "tool": "code_generate",
             "params": {"goal": ""},
             "expected_outcome": "Fix generated"},
        ],
    },
]


class Subtask:
    """A single step within a plan."""

    def __init__(self, description: str, tool: str, params: Dict,
                 expected_outcome: str = ""):
        self.description = description
        self.tool = tool
        self.params = dict(params)
        self.expected_outcome = expected_outcome
        self.status = "pending"
        self.result: Optional[Dict] = None
        self.error: Optional[str] = None
        self.attempts = 0
        self.max_retries = 3
        self.duration: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "description": self.description,
            "tool": self.tool,
            "params": self.params,
            "expected_outcome": self.expected_outcome,
            "status": self.status,
            "error": self.error,
            "attempts": self.attempts,
            "duration": self.duration,
        }

    def __repr__(self) -> str:
        return f"Subtask({self.description}, tool={self.tool}, status={self.status})"


class Plan:
    """A decomposition of a goal into ordered subtasks."""

    def __init__(self, goal: str, subtasks: List[Subtask],
                 task_type: str = "general"):
        self.goal = goal
        self.subtasks = subtasks
        self.task_type = task_type
        self.status = "pending"
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.completed_at: Optional[str] = None
        self.duration: float = 0.0

    @property
    def success_rate(self) -> float:
        if not self.subtasks:
            return 1.0
        succeeded = sum(1 for s in self.subtasks if s.status == "success")
        return succeeded / len(self.subtasks)

    def to_dict(self) -> Dict:
        return {
            "goal": self.goal,
            "task_type": self.task_type,
            "status": self.status,
            "success_rate": self.success_rate,
            "duration": self.duration,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "subtasks": [s.to_dict() for s in self.subtasks],
        }

    def __repr__(self) -> str:
        n = len(self.subtasks)
        done = sum(1 for s in self.subtasks if s.status in ("success", "failed"))
        return f"Plan({self.status}, {done}/{n} subtasks, type={self.task_type})"


class Planner:
    """Decomposes goals into plans, executes subtasks, evaluates results.

    Planner is stateless: plans are created fresh. All persistence and
    learning happens through the optional ExperienceMemory.
    """

    def __init__(self, tool_executor: Optional[Callable] = None,
                 experience_memory=None, code_generator=None,
                 strategic_reasoner=None):
        self.executor = tool_executor or self._default_executor
        self.experience_memory = experience_memory
        self.code_generator = code_generator
        self.reasoner = strategic_reasoner

    # ── Plan Creation ───────────────────────────────────────────────

    def create_plan(self, goal: str) -> Plan:
        """Decompose a goal into a plan using templates or generic fallback."""
        goal_lower = goal.lower()

        # ── 0. Strategic analysis before planning ──
        if self.reasoner:
            strategic = self.reasoner.before_plan(goal)
            if strategic.get("obstacles"):
                logger.info("Anticipated %d obstacle(s) for '%s'",
                            len(strategic["obstacles"]), goal)

        # ── 1. Check experience memory for similar tasks ──
        similar = []
        if self.experience_memory:
            similar = self.experience_memory.query(goal, limit=3)

        # ── 2. Template matching ──
        template = self._match_template(goal_lower)
        if template:
            subtasks = self._build_from_template(template, goal)
            plan = Plan(goal=goal, subtasks=subtasks, task_type=template["type"])
            self._apply_lessons(plan, similar)
            return plan

        # ── 3. Geometry goal detection (before generic "create") ──
        if any(kw in goal_lower for kw in ["3d", "geometry", "mesh", "voxel",
                                            "stl", "cad", "lattice", "picogk",
                                            "offset", "smooth mesh",
                                            "gearbox", "gear ", "bracket",
                                            "phone stand", "robot arm",
                                            "joint", "housing", "printable"]):
            subtasks = self._generate_geometry_plan(goal)
            return Plan(goal=goal, subtasks=subtasks, task_type="geometry")

        # ── 4. Code goal detection ──
        if any(kw in goal_lower for kw in ["function", "implement", "create",
                                            "build", "write", "generate",
                                            "script", "code"]):
            subtasks = self._generate_code_plan(goal)
            return Plan(goal=goal, subtasks=subtasks, task_type="code")

        # ── 5. Research goal detection ──
        if any(kw in goal_lower for kw in ["what is", "who is", "explain",
                                            "research", "learn", "tell me",
                                            "how does"]):
            subtasks = self._generate_research_plan(goal)
            return Plan(goal=goal, subtasks=subtasks, task_type="research")

        # ── 6. Generic fallback ──
        subtasks = self._generate_generic_plan(goal)
        plan = Plan(goal=goal, subtasks=subtasks, task_type="general")
        self._apply_lessons(plan, similar)
        return plan

    def _match_template(self, goal_lower: str) -> Optional[Dict]:
        for tpl in PLAN_TEMPLATES:
            for kw in tpl["keywords"]:
                if kw in goal_lower:
                    return tpl
        return None

    def _build_from_template(self, template: Dict, goal: str) -> List[Subtask]:
        subtasks = []
        for st in template["subtasks"]:
            params = dict(st["params"])
            if params.get("goal") == "":
                params["goal"] = goal
            if params.get("query") == "":
                params["query"] = goal
            subtasks.append(Subtask(
                description=st["description"],
                tool=st["tool"],
                params=params,
                expected_outcome=st.get("expected_outcome", ""),
            ))
        return subtasks

    def _generate_code_plan(self, goal: str) -> List[Subtask]:
        return [
            Subtask("Analyze requirements", "reasoning_infer", {},
                    "Requirements understood"),
            Subtask(f"Generate code: {goal}", "code_generate",
                    {"goal": goal}, "Code generated"),
            Subtask("Verify correctness", "search_graph",
                    {"query": goal}, "Code verified"),
        ]

    def _generate_research_plan(self, goal: str) -> List[Subtask]:
        return [
            Subtask(f"Search internal knowledge: {goal}", "search_graph",
                    {"query": goal}, "Internal knowledge found"),
            Subtask(f"Search web: {goal}", "web_search",
                    {"query": goal}, "Web results gathered"),
            Subtask(f"Extract concepts", "curiosity_explore",
                    {"goal": goal}, "New knowledge stored"),
        ]

    def _generate_geometry_plan(self, goal: str) -> List[Subtask]:
        subtasks = [
            Subtask(f"Search knowledge for geometry: {goal}", "search_graph",
                    {"query": goal},
                    "Existing geometry knowledge retrieved"),
            Subtask(f"Reason and generate: {goal}", "geometry_reason",
                    {"query": goal},
                    "3D model generated from knowledge-grounded parameters"),
            Subtask("Store geometry metadata", "search_graph",
                    {"query": goal}, "Geometry metadata stored in knowledge graph"),
        ]
        return subtasks

    def _generate_generic_plan(self, goal: str) -> List[Subtask]:
        subtasks = [
            Subtask(f"Search for information: {goal}", "web_search",
                    {"query": goal}, "Information found"),
            Subtask(f"Analyze and reason", "reasoning_infer", {},
                    "Analysis complete"),
        ]
        if any(kw in goal.lower() for kw in ["code", "function", "script",
                                               "program", "implement"]):
            subtasks.append(
                Subtask(f"Generate code: {goal}", "code_generate",
                        {"goal": goal}, "Code generated"),
            )
        subtasks.append(
            Subtask("Store findings", "search_graph",
                    {"query": goal}, "Findings stored"),
        )
        return subtasks

    def _apply_lessons(self, plan: Plan, similar: List[Dict]):
        """Enrich plan with lessons from similar past experiences."""
        if not similar:
            return
        for exp in similar:
            lessons_raw = exp.get("lessons", "[]")
            try:
                lessons = json.loads(lessons_raw) if isinstance(lessons_raw, str) else []
            except (json.JSONDecodeError, TypeError):
                continue
            for lesson in lessons:
                if any(kw in lesson.lower() for kw in ["install", "dependency",
                                                       "prerequisite"]):
                    pre = Subtask(
                        f"Install dependencies: {lesson}", "reasoning_infer",
                        {}, lesson,
                    )
                    plan.subtasks.insert(0, pre)
                if any(kw in lesson.lower() for kw in ["test", "verify",
                                                       "validate"]):
                    plan.subtasks.append(
                        Subtask(f"Verify: {lesson}", "search_graph",
                                {"query": plan.goal}, lesson),
                    )
            # Copy max_retries from failed attempts
            failure_subtasks = self._get_failure_subtasks(exp)
            for fs in failure_subtasks:
                for ps in plan.subtasks:
                    if fs.get("description", "").lower() in ps.description.lower():
                        ps.max_retries = max(ps.max_retries, fs.get("attempts", 1) + 1)

    @staticmethod
    def _get_failure_subtasks(exp: Dict) -> List[Dict]:
        """Extract subtask results from an experience dict."""
        return exp.get("subtask_results", exp.get("subtasks", []))

    # ── Plan Execution ──────────────────────────────────────────────

    def execute_plan(self, plan: Plan) -> Plan:
        """Execute each subtask in order, evaluating and retrying as needed."""
        logger.info("Executing plan for: %s (%d subtasks)",
                    plan.goal, len(plan.subtasks))
        plan.status = "running"
        start = datetime.now(timezone.utc)

        for subtask in plan.subtasks:
            self._execute_subtask(subtask, plan)

        plan.status = "completed" if plan.success_rate > 0 else "failed"
        plan.completed_at = datetime.now(timezone.utc).isoformat()
        duration = (datetime.now(timezone.utc) - start).total_seconds()
        plan.duration = round(duration, 2)

        # ── Record strategic consequences ──
        if self.reasoner:
            self.reasoner.after_plan(plan.to_dict())

        # ── Store in experience memory ──
        self._store_experience(plan)
        return plan

    def _execute_subtask(self, subtask: Subtask, plan: Plan):
        subtask.status = "running"
        st_start = datetime.now(timezone.utc)
        goal_type = plan.task_type

        # Strategic prediction before execution
        if self.reasoner:
            prediction = self.reasoner.before_subtask(subtask.tool, goal_type)
            if prediction.get("warnings"):
                logger.info("  Known risks for %s: %s",
                            subtask.tool, prediction["warnings"])

        while subtask.attempts < subtask.max_retries:
            subtask.attempts += 1
            logger.debug("  [%s] attempt %d/%d: %s",
                         subtask.tool, subtask.attempts,
                         subtask.max_retries, subtask.description)

            # Inject plan goal into params that need it
            params = dict(subtask.params)
            for key, val in params.items():
                if isinstance(val, str) and val == "" and key in ("goal", "query"):
                    params[key] = plan.goal

            # Execute
            try:
                result = self.executor(subtask.tool, **params)
            except Exception as e:
                result = {"success": False, "error": str(e)}

            # Evaluate
            subtask.duration = round((datetime.now(timezone.utc) - st_start).total_seconds(), 2)
            if self._is_success(result):
                subtask.status = "success"
                subtask.result = result
                logger.info("  + %s", subtask.description)
                # Record successful consequence
                if self.reasoner:
                    self.reasoner.after_subtask(
                        subtask.tool, goal_type, True, subtask.duration,
                        description=subtask.description,
                    )
                return
            else:
                subtask.error = result.get("error", "Unknown error")
                logger.warning("  - %s (attempt %d): %s",
                               subtask.description, subtask.attempts,
                               subtask.error)

        subtask.status = "failed"
        logger.error("  - %s failed after %d attempts",
                     subtask.description, subtask.max_retries)
        # Record failure consequence
        if self.reasoner:
            self.reasoner.after_subtask(
                subtask.tool, goal_type, False, subtask.duration,
                error=subtask.error, description=subtask.description,
            )

    def _is_success(self, result: Dict) -> bool:
        """Determine if a tool execution result indicates success."""
        if isinstance(result, dict):
            success = result.get("success", False)
            error = result.get("error", "")
            if success and not error:
                return True
            if result.get("status") == "ok":
                return True
        return False

    def _store_experience(self, plan: Plan):
        if not self.experience_memory:
            return
        lessons = self._extract_lessons(plan)
        self.experience_memory.store(
            task=plan.goal,
            subtasks=[s.to_dict() for s in plan.subtasks],
            success_rate=plan.success_rate,
            duration=plan.duration,
            lessons=lessons,
            task_type=plan.task_type,
        )

    def _extract_lessons(self, plan: Plan) -> List[str]:
        lessons = []
        for s in plan.subtasks:
            if s.status == "failed":
                lessons.append(
                    f"{s.description} failed: {s.error}"
                )
            elif s.status == "success" and s.attempts > 1:
                lessons.append(
                    f"{s.description} required {s.attempts} attempts"
                )
        return lessons

    # ── Default Executor (standalone use, no ActionEngine) ──────────

    @staticmethod
    def _default_executor(tool: str, **params) -> Dict:
        return {"success": True, "tool": tool, "data": "default",
                "error": "", "status": "ok"}
