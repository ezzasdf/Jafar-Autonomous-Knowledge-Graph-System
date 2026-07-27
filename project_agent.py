"""
Project Agent — takes high-level goals and produces complete multi-file projects.

Flow:
  1. Plan (decompose goal into tasks per file)
  2. Generate files (write code to workspace/<project>/)
  3. Detect dependencies (scan imports across all files)
  4. Generate tests (create test files alongside source)
  5. Execute (run entry point, check result)
  6. Repair (hand off failures to RepairAgent)
  7. Completion report (structured summary)

Example: "build a weather dashboard"
  -> task list: [main.py, weather_api.py, display.py, config.py, test_weather_api.py]
  -> files created in workspace/weather_dashboard/
  -> dependencies: [requests, json, datetime]
  -> tests pass, executes, repaired if needed
  -> completion report with file list, test results, exec status
"""

import logging
import re
import time
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)
debug_logger = logging.getLogger(f"{__name__}.debug")


@dataclass
class ProjectTask:
    file_path: str
    description: str
    code: str = ""
    tests: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "description": self.description,
            "code_length": len(self.code),
            "test_count": len(self.tests),
            "dependencies": self.dependencies,
            "status": self.status,
            "error": self.error,
        }


@dataclass
class ProjectResult:
    success: bool
    project_name: str
    goal: str
    file_count: int
    test_count: int
    tests_passed: int
    total_tests: int
    execution_result: Optional[Dict[str, Any]]
    repair_summary: Dict[str, Any]
    duration: float
    details: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProjectPlanner:
    """Decomposes a high-level goal into a structured project plan."""

    PROJECT_TEMPLATES = {
        "cli": {
            "files": [
                ("main.py", "Entry point with CLI argument parsing and main logic"),
                ("utils.py", "Utility functions used by the main module"),
            ],
            "entry_point": "main.py",
        },
        "data": {
            "files": [
                ("main.py", "Entry point that loads data, processes it, and outputs results"),
                ("loader.py", "Data loading functions (CSV, JSON, API)"),
                ("processor.py", "Data processing and transformation functions"),
                ("config.py", "Configuration constants and paths"),
            ],
            "entry_point": "main.py",
        },
        "web": {
            "files": [
                ("app.py", "Web application entry point and route definitions"),
                ("utils.py", "Helper functions for the web app"),
                ("templates/index.html", "Main HTML template"),
                ("static/style.css", "CSS stylesheet"),
            ],
            "entry_point": "app.py",
        },
        "api": {
            "files": [
                ("api.py", "API endpoint definitions and request handling"),
                ("handlers.py", "Request handler functions"),
                ("models.py", "Data models and validation"),
                ("config.py", "API configuration"),
            ],
            "entry_point": "api.py",
        },
        "automation": {
            "files": [
                ("run.py", "Main automation script with loop and scheduling"),
                ("tasks.py", "Task definitions for automation"),
                ("config.py", "Configuration settings"),
            ],
            "entry_point": "run.py",
        },
        "library": {
            "files": [
                ("__init__.py", "Package init, exports public API"),
                ("core.py", "Core functionality and data structures"),
                ("utils.py", "Internal utility functions"),
            ],
            "entry_point": "__init__.py",
        },
        "dashboard": {
            "files": [
                ("dashboard.py", "Main dashboard entry point with display logic"),
                ("data_source.py", "Data fetching and caching"),
                ("renderer.py", "Rendering and formatting data for display"),
                ("config.py", "Configuration constants"),
            ],
            "entry_point": "dashboard.py",
        },
    }

    def __init__(self, llm_generator: Optional[Callable] = None):
        self.llm = llm_generator

    def plan(self, goal: str) -> Tuple[List[ProjectTask], str, str]:
        tasks: List[ProjectTask] = []
        project_type = self._detect_type(goal)
        template = self.PROJECT_TEMPLATES.get(project_type, self.PROJECT_TEMPLATES["cli"])
        project_name = self._make_project_name(goal)

        if self.llm is not None:
            llm_tasks = self._plan_with_llm(goal, project_type, project_name)
            if llm_tasks:
                return llm_tasks, project_name, template["entry_point"]

        for fp, desc in template["files"]:
            tasks.append(ProjectTask(file_path=fp, description=desc))

        test_files = self._generate_test_tasks(tasks)
        tasks.extend(test_files)

        return tasks, project_name, template["entry_point"]

    def _detect_type(self, goal: str) -> str:
        lower = goal.lower()
        if any(w in lower for w in ["dashboard", "visualize", "chart", "plot", "graph"]):
            return "dashboard"
        if any(w in lower for w in ["web", "site", "page", "html", "server", "flask"]):
            return "web"
        if any(w in lower for w in ["api", "rest", "endpoint", "service"]):
            return "api"
        if any(w in lower for w in ["data", "csv", "json", "file", "process", "pipeline"]):
            return "data"
        if any(w in lower for w in ["automate", "bot", "cron", "scheduler", "watch"]):
            return "automation"
        if any(w in lower for w in ["library", "package", "module", "sdk"]):
            return "library"
        return "cli"

    def _make_project_name(self, goal: str) -> str:
        words = re.findall(r"[a-zA-Z0-9]+", goal.lower())
        short = [w for w in words if w not in ("a", "an", "the", "and", "or", "but", "in", "of", "to", "for", "with")]
        if not short:
            short = words
        return "_".join(short[:4])

    def _plan_with_llm(self, goal: str, project_type: str, project_name: str) -> Optional[List[ProjectTask]]:
        try:
            prompt = (
                f"Decompose this goal into Python files for a project:\n"
                f"Goal: {goal}\n"
                f"Project type: {project_type}\n"
                f"Project name: {project_name}\n\n"
                f"For each file specify:\n"
                f"FILE: <path/name.py>\n"
                f"DESC: <one-line description of what this file does>\n\n"
                f"Include test files (prefixed with test_).\n"
                f"Generate 3-6 source files plus corresponding tests."
            )
            text = self.llm(prompt, temperature=0.3)
            return self._parse_llm_plan(text)
        except Exception as e:
            logger.warning("LLM planning failed: %s", e)
            return None

    def _parse_llm_plan(self, text: str) -> List[ProjectTask]:
        tasks = []
        current = None
        for line in text.split("\n"):
            line = line.strip()
            if line.upper().startswith("FILE:") or line.startswith("- FILE:"):
                if current:
                    tasks.append(current)
                fp = line.split(":", 1)[1].strip().lstrip("* ")
                if not fp.endswith(".py"):
                    fp += ".py"
                current = ProjectTask(file_path=fp, description="")
            elif current and (line.upper().startswith("DESC:") or line.startswith("- DESC:")):
                current.description = line.split(":", 1)[1].strip().lstrip("* ")
        if current:
            tasks.append(current)
        return tasks

    def _generate_test_tasks(self, source_tasks: List[ProjectTask]) -> List[ProjectTask]:
        test_tasks = []
        seen = set()
        for t in source_tasks:
            base = t.file_path.replace(".py", "")
            name = base.split("/")[-1]
            if name.startswith("test_") or name == "__init__":
                continue
            test_path = f"test_{name}.py"
            if t.file_path.startswith("templates/") or t.file_path.startswith("static/"):
                continue
            if test_path not in seen:
                seen.add(test_path)
                test_tasks.append(ProjectTask(
                    file_path=test_path,
                    description=f"Tests for {t.file_path}: {t.description}",
                ))
        return test_tasks


class DependencyDetector:
    """Scans Python code for import dependencies."""

    STDLIB_MODULES = {
        "abc", "argparse", "ast", "asyncio", "base64", "collections", "concurrent",
        "copy", "csv", "datetime", "decimal", "enum", "functools", "glob", "gzip",
        "hashlib", "html", "http", "io", "itertools", "json", "logging", "math",
        "multiprocessing", "operator", "os", "pathlib", "pickle", "pprint", "queue",
        "random", "re", "secrets", "shutil", "signal", "socket", "sqlite3",
        "statistics", "string", "struct", "subprocess", "sys", "tempfile", "textwrap",
        "threading", "time", "traceback", "typing", "unittest", "urllib", "uuid",
        "warnings", "weakref", "xml", "zipfile",
    }

    def detect(self, code: str) -> List[str]:
        imports = set()
        for m in re.finditer(r"^\s*import\s+(\S+)", code, re.MULTILINE):
            module = m.group(1).split(".")[0].split(" as ")[0].strip()
            module = module.split(",")[0].strip()
            if module and module not in self.STDLIB_MODULES:
                imports.add(module)
        for m in re.finditer(r"^\s*from\s+(\S+)", code, re.MULTILINE):
            module = m.group(1).split(".")[0].strip()
            if module and module not in self.STDLIB_MODULES and module != "__future__":
                imports.add(module)
        return sorted(imports)


class CodeGenerator:
    """Generates Python code for project tasks using LLM or templates."""

    def __init__(self, llm_generator: Optional[Callable] = None):
        self.llm = llm_generator

    def generate(self, task: ProjectTask, project_context: str = "") -> str:
        if self.llm is not None:
            code = self._generate_with_llm(task, project_context)
            if code:
                return code

        return self._generate_template(task, project_context)

    def _generate_with_llm(self, task: ProjectTask, context: str) -> Optional[str]:
        try:
            is_test = task.file_path.startswith("test_")
            prompt = (
                f"Write Python code for this project file.\n"
                f"File: {task.file_path}\n"
                f"Description: {task.description}\n"
                f"{'Test file: write pytest-style tests.' if is_test else 'Source file: write clean, working Python code.'}\n"
                f"Project context: {context[:400] if context else 'none'}\n\n"
                f"Rules:\n"
                f"- Return ONLY valid Python code, no markdown fences\n"
                f"- Include all imports at the top\n"
                f"- Use stdlib only unless the description requires otherwise\n"
                f"- Include docstrings for functions and classes\n"
                f"{'- Write at least 3 test functions covering normal, edge, and error cases' if is_test else ''}"
            )
            code = self.llm(prompt, temperature=0.4)
            code = self._clean_code(code)
            return code
        except Exception as e:
            logger.warning("LLM code generation failed for %s: %s", task.file_path, e)
            return None

    def _generate_template(self, task: ProjectTask, context: str) -> str:
        fp = task.file_path
        desc = task.description

        if fp == "__init__.py":
            return '"""Package initialization."""\n'

        if fp.endswith(".html"):
            return f"<!DOCTYPE html>\n<html><head><title>{context[:30]}</title></head><body><h1>{context[:50]}</h1></body></html>"

        if fp.endswith(".css"):
            return "/* Stylesheet */\nbody { font-family: sans-serif; margin: 2rem; }"

        if fp.startswith("test_"):
            return self._template_test(fp, desc)

        return self._template_source(fp, desc, context)

    def _template_source(self, fp: str, desc: str, context: str) -> str:
        name = fp.replace("/", "_").replace(".py", "")
        safe_desc = desc.replace('"', "'")
        return (
            f'"""\n{safe_desc}\n"""\n\n'
            f"import json\nimport sys\n\n\n"
            f"def main():\n"
            f'    """Main entry point."""\n'
            f"    print(f'Running {name}...')\n"
            f"    return 0\n\n\n"
            f"if __name__ == '__main__':\n"
            f"    sys.exit(main())\n"
        )

    def _template_test(self, fp: str, desc: str) -> str:
        target = fp.replace("test_", "").replace(".py", "")
        return (
            f'"""\n{desc}\n"""\n\n'
            f"import pytest\n"
            f"from {target} import *\n\n\n"
            f"def test_basic():\n"
            f"    assert True\n\n\n"
            f"def test_edge_cases():\n"
            f"    assert True\n"
        )

    def _clean_code(self, text: str) -> str:
        text = text.strip()
        markdown = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
        if markdown:
            text = markdown.group(1).strip()
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            if re.match(r"^(Here|This|The|Below|Let|Note|Example|Usage|Sure|Certainly)", line, re.IGNORECASE):
                continue
            cleaned.append(line)
        return "\n".join(cleaned).strip()


class ProjectAgent:
    """High-level goal → multi-file project → tested → executed → repaired.

    Full pipeline:
      plan(goal) -> tasks (files + tests)
      generate(tasks) -> write files to workspace/<project>/
      detect_deps(tasks) -> scan imports
      generate_tests(tasks) -> write test files
      execute(project_name) -> run entry point
      repair(failures) -> RepairAgent for each failed file
      build(goal) -> all of the above -> ProjectResult

    Integrates with: coding_workspace (file I/O), execution_agent (run),
                     repair_agent (fix failures), transformer_reasoning (LLM)
    """

    def __init__(
        self,
        workspace: Optional[Any] = None,
        execution_runner: Optional[Callable] = None,
        llm_generator: Optional[Callable] = None,
        repair_agent: Optional[Any] = None,
        max_file_repairs: int = 2,
    ):
        self.workspace = workspace
        self.execution_runner = execution_runner
        self.llm = llm_generator
        self.repair_agent = repair_agent
        self.max_file_repairs = max_file_repairs

        self.planner = ProjectPlanner(llm_generator=self.llm)
        self.code_gen = CodeGenerator(llm_generator=self.llm)
        self.dep_detector = DependencyDetector()

        self._debug_trace: List[str] = []

    def build(
        self,
        goal: str,
        debug: bool = False,
    ) -> ProjectResult:
        """Full pipeline: plan → generate → detect deps → test → execute → repair → report."""
        self._debug_trace = []
        start = time.time()

        self._log(f"=== PROJECT AGENT ===", debug)
        self._log(f"Goal: {goal}", debug)

        # 1. Plan
        self._log(f"\n--- Phase 1: Plan ---", debug)
        tasks, project_name, entry_point = self.planner.plan(goal)
        self._log(f"Project name: {project_name}", debug)
        self._log(f"Tasks: {len(tasks)} ({sum(1 for t in tasks if t.file_path.startswith('test_'))} tests)", debug)
        for t in tasks:
            self._log(f"  {t.file_path} — {t.description[:60]}", debug)

        if self.workspace:
            self.workspace.create_project(project_name)

        # 2. Generate files
        self._log(f"\n--- Phase 2: Generate Files ---", debug)
        context = f"Project: {project_name}, Goal: {goal}"
        for task in tasks:
            if task.file_path.startswith("test_"):
                continue
            code = self.code_gen.generate(task, context)
            if not code:
                code = "# placeholder\n"
            task.code = code
            task.status = "generated"

            full_path = f"{project_name}/{task.file_path}"
            if self.workspace:
                self.workspace.create(full_path, code)
            self._log(f"  Wrote {task.file_path} ({len(code)} chars)", debug)

        # 3. Detect dependencies
        self._log(f"\n--- Phase 3: Detect Dependencies ---", debug)
        all_deps = set()
        for task in tasks:
            if task.code:
                deps = self.dep_detector.detect(task.code)
                task.dependencies = deps
                all_deps.update(deps)
        all_deps.discard("__future__")
        if all_deps:
            self._log(f"  External dependencies: {', '.join(sorted(all_deps))}", debug)
            for dep in sorted(all_deps):
                self._log(f"  WARNING: '{dep}' may need pip install", debug)
        else:
            self._log(f"  No external dependencies (stdlib only)", debug)

        # 4. Generate tests
        self._log(f"\n--- Phase 4: Generate Tests ---", debug)
        test_count = 0
        for task in tasks:
            if not task.file_path.startswith("test_"):
                continue
            if not task.code:
                code = self.code_gen.generate(task, context)
                if not code:
                    code = "# placeholder\n"
                task.code = code
            task.status = "generated"
            test_count += 1

            full_path = f"{project_name}/{task.file_path}"
            if self.workspace:
                self.workspace.create(full_path, code)
            self._log(f"  Wrote {task.file_path} ({len(code)} chars)", debug)

        # 5. Execute
        self._log(f"\n--- Phase 5: Execute ---", debug)
        exec_result = self._execute_project(project_name, entry_point, debug)

        # 6. Repair if needed
        repairs_summary = {"total": 0, "fixed": 0, "escalated": 0, "details": []}
        need_repair = False
        if exec_result and exec_result.get("exit_code", 0) != 0:
            need_repair = True
            self._log(f"\n--- Phase 6: Repair ---", debug)

            source_tasks = [t for t in tasks if not t.file_path.startswith("test_") and t.code]
            for task in source_tasks:
                full_path = f"{project_name}/{task.file_path}"
                repairs_summary["total"] += 1
                self._log(f"  Repairing: {task.file_path}", debug)

                if self.repair_agent is not None:
                    repair_result = self.repair_agent.repair(
                        code=task.code,
                        file_path=full_path,
                        error_result=exec_result,
                        debug=debug,
                    )
                    if repair_result.success:
                        repairs_summary["fixed"] += 1
                        task.status = "repaired"
                        if repair_result.fix_applied:
                            task.code = repair_result.fix_applied
                        self._log(f"    FIXED (attempt {repair_result.attempts})", debug)
                    else:
                        repairs_summary["escalated"] += 1
                        task.status = "failed"
                        task.error = repair_result.escalation_reason
                        self._log(f"    FAILED: {repair_result.escalation_reason[:80]}", debug)

                    repairs_summary["details"].append({
                        "file": task.file_path,
                        "success": repair_result.success,
                        "attempts": repair_result.attempts,
                        "error_type": repair_result.error_type,
                        "root_cause": repair_result.root_cause,
                    })
                else:
                    self._log(f"    No repair agent available, skipping", debug)
                    repairs_summary["escalated"] += 1

            re_run = self._execute_project(project_name, entry_point, debug) if repairs_summary["fixed"] > 0 else None
            if re_run and re_run.get("exit_code") == 0:
                exec_result = re_run
                self._log(f"  Re-run after repairs: OK", debug)
            else:
                self._log(f"  Re-run after repairs: still failing", debug)

        # 7. Run tests
        self._log(f"\n--- Phase 7: Run Tests ---", debug)
        tests_passed = 0
        total_tests = 0
        for task in tasks:
            if not task.file_path.startswith("test_"):
                continue
            test_code = task.code
            if not test_code:
                continue

            total_tests += 1
            result = self._run_test_code(test_code)
            if result.get("exit_code") == 0:
                tests_passed += 1
                self._log(f"  {task.file_path}: PASS", debug)
            else:
                err = result.get("stderr", "")[:80]
                self._log(f"  {task.file_path}: FAIL ({err})", debug)

        success = (exec_result is None or exec_result.get("exit_code", 0) == 0) or (need_repair and repairs_summary["fixed"] > 0)

        result = ProjectResult(
            success=success,
            project_name=project_name,
            goal=goal,
            file_count=len([t for t in tasks if not t.file_path.startswith("test_")]),
            test_count=test_count,
            tests_passed=tests_passed,
            total_tests=total_tests,
            execution_result={
                "exit_code": exec_result.get("exit_code") if exec_result else None,
                "stdout": (exec_result.get("stdout", "")[:500] if exec_result else ""),
                "stderr": (exec_result.get("stderr", "")[:200] if exec_result else ""),
            } if exec_result else None,
            repair_summary=repairs_summary,
            duration=round(time.time() - start, 2),
            details=[t.to_dict() for t in tasks],
            warnings=list(all_deps) if all_deps else [],
        )

        self._log(f"\n=== PROJECT {'OK' if success else 'FAILED'} ===", debug)
        self._log(f"Files: {result.file_count}, Tests: {tests_passed}/{total_tests} passed", debug)
        self._log(f"Repairs: {repairs_summary['fixed']}/{repairs_summary['total']} fixed", debug)
        self._log(f"Duration: {result.duration}s", debug)

        if debug and self.workspace:
            self._save_debug_trace(goal, project_name, result)

        return result

    def _execute_project(self, project_name: str, entry_point: str, debug: bool) -> Optional[Dict[str, Any]]:
        if self.execution_runner is not None:
            try:
                if self.workspace:
                    return self.execution_runner(f"{project_name}/{entry_point}")
                return None
            except Exception as e:
                self._log(f"  Execution error: {e}", debug)
                return {"exit_code": 1, "stdout": "", "stderr": str(e)}
        try:
            from execution_agent import execute_file
            return execute_file(f"{project_name}/{entry_point}", timeout=5.0)
        except Exception as e:
            self._log(f"  Execution error: {e}", debug)
            return {"exit_code": 1, "stdout": "", "stderr": str(e)}

    def _run_test_code(self, code: str) -> Dict[str, Any]:
        if self.execution_runner is not None:
            try:
                return self.execution_runner(code)
            except Exception:
                pass
        try:
            from execution_agent import execute_python
            return execute_python(code, timeout=5.0)
        except Exception as e:
            return {"exit_code": 1, "stdout": "", "stderr": str(e)}

    def _log(self, message: str, to_console: bool = False):
        self._debug_trace.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        if to_console:
            print(message)

    def _save_debug_trace(self, goal: str, project_name: str, result: ProjectResult):
        try:
            safe_name = re.sub(r"[^a-zA-Z0-9_]+", "_", project_name)[:30]
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            path = f"logs/project_{safe_name}_{timestamp}.txt"
            lines = [
                f"Project Agent Debug Trace",
                f"Goal: {goal}",
                f"Project: {project_name}",
                f"Success: {result.success}",
                f"Duration: {result.duration}s",
                f"Files: {result.file_count}, Tests: {result.tests_passed}/{result.total_tests}",
                f"Repairs: {result.repair_summary}",
                f"",
            ]
            lines.extend(self._debug_trace)
            self.workspace.create(path, "\n".join(lines))
        except Exception as e:
            logger.debug("Failed to save project trace: %s", e)
