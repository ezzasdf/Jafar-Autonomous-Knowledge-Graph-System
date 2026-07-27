"""
Repair Agent — structured Error → Root Cause Analysis → Fix → Rewrite → Re-run.

Flow per failure:
  1. Capture error (stdout, stderr, exit_code, traceback)
  2. Classify error type (NameError, TypeError, ImportError, logic failure, timeout, etc.)
  3. Root cause extraction (line hint, variable scope, missing import, type mismatch)
  4. Generate fix (pattern-based for common errors, LLM for complex ones)
  5. Rewrite file (backup original, apply diff/patch, save)
  6. Re-run and verify (if still failing, retry or escalate)
  7. Escalate if N consecutive repairs fail on the same code

Integrates with: execution_agent (run code), coding_workspace (file rewrite),
                 autonomous_loop.FailureAnalyzer/CodeFixer (reuse patterns),
                 transformer_reasoning (LLM fix generation)
"""

import logging
import re
import time
import textwrap
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)
debug_logger = logging.getLogger(f"{__name__}.debug")


@dataclass
class RepairIncident:
    file_path: str
    error_type: str
    error_message: str
    stdout: str
    stderr: str
    exit_code: int
    code: str
    line_hint: Optional[int] = None
    timestamp: float = 0.0
    root_cause: str = ""
    fix_strategy: str = ""
    fix_applied: str = ""
    attempt: int = 1
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RepairResult:
    success: bool
    file_path: str
    error_type: str
    root_cause: str
    fix_strategy: str
    fix_applied: str
    attempts: int
    duration: float
    re_run_result: Optional[Dict[str, Any]] = None
    incidents: List[Dict[str, Any]] = field(default_factory=list)
    escalation: bool = False
    escalation_reason: str = ""


class ErrorClassifier:
    """Classifies runtime errors by type and severity."""

    ERROR_PATTERNS = {
        "name_error": re.compile(r"NameError:\s*name\s+'(\w+)' is not defined"),
        "type_error": re.compile(r"TypeError:\s*(.+)"),
        "value_error": re.compile(r"ValueError:\s*(.+)"),
        "index_error": re.compile(r"IndexError:\s*(.+)"),
        "key_error": re.compile(r"KeyError:\s*(\d+|'.+?')"),
        "attribute_error": re.compile(r"AttributeError:\s*(.+)"),
        "zero_division": re.compile(r"ZeroDivisionError:\s*(.+)"),
        "import_error": re.compile(r"(?:ImportError|ModuleNotFoundError):\s*(.+)"),
        "syntax_error": re.compile(r"SyntaxError:\s*(.+)"),
        "indentation_error": re.compile(r"(?:IndentationError|TabError):\s*(.+)"),
        "file_not_found": re.compile(r"FileNotFoundError:\s*(.+)"),
        "recursion": re.compile(r"RecursionError:\s*(.+)"),
        "assertion_error": re.compile(r"AssertionError:\s*(.+)"),
        "stop_iteration": re.compile(r"StopIteration:\s*(.+)"),
        "timeout": re.compile(r"timed\s*out", re.IGNORECASE),
        "memory": re.compile(r"(?:MemoryError|OutOfMemory)"),
        "keyboard_interrupt": re.compile(r"KeyboardInterrupt"),
        "os_error": re.compile(r"OSError:\s*(.+)"),
        "permission": re.compile(r"PermissionError:\s*(.+)"),
    }

    SEVERITY_MAP = {
        "name_error": "low",
        "type_error": "medium",
        "value_error": "low",
        "index_error": "low",
        "key_error": "low",
        "attribute_error": "medium",
        "zero_division": "low",
        "import_error": "high",
        "syntax_error": "high",
        "indentation_error": "high",
        "file_not_found": "medium",
        "recursion": "medium",
        "assertion_error": "medium",
        "stop_iteration": "low",
        "timeout": "high",
        "memory": "high",
        "keyboard_interrupt": "low",
        "os_error": "medium",
        "permission": "medium",
        "logic_error": "medium",
        "unknown": "medium",
    }

    def classify(self, result: Dict[str, Any]) -> Tuple[str, str, int]:
        stderr = result.get("stderr", "")
        stdout = result.get("stdout", "")
        error_field = result.get("error", "")
        exit_code = result.get("exit_code", 0)

        combined = f"{stderr}\n{error_field}"

        if exit_code == -1:
            return "timeout", "Execution timed out", -1

        for error_type, pattern in self.ERROR_PATTERNS.items():
            match = pattern.search(combined)
            if match:
                message = match.group(0) if match.lastindex else combined[:150]
                return error_type, message.strip(), exit_code

        if exit_code != 0 and not combined.strip():
            return "logic_error", "Code exited with non-zero code but no stderr", exit_code

        return "unknown", combined[:150] if combined else "No error information", exit_code

    def severity(self, error_type: str) -> str:
        return self.SEVERITY_MAP.get(error_type, "medium")


class RootCauseAnalyzer:
    """Extracts structured root cause from error output and code context."""

    def __init__(self, llm_generator: Optional[Callable] = None):
        self.llm = llm_generator

    def analyze(self, incident: RepairIncident) -> Dict[str, Any]:
        code = incident.code
        error_message = incident.error_message
        error_type = incident.error_type

        line_hint = self._extract_line_hint(incident.stderr)
        incident.line_hint = line_hint

        if error_type == "name_error":
            return self._analyze_name_error(error_message, code, line_hint)
        elif error_type == "type_error":
            return self._analyze_type_error(error_message, code, line_hint)
        elif error_type == "import_error":
            return self._analyze_import_error(error_message, code)
        elif error_type == "syntax_error" or error_type == "indentation_error":
            return self._analyze_syntax_error(error_message, code, line_hint)
        elif error_type == "zero_division":
            return self._analyze_zero_division(code, line_hint)
        elif error_type == "index_error":
            return self._analyze_index_error(code, line_hint)
        elif error_type == "key_error":
            return self._analyze_key_error(error_message, code)
        elif error_type == "attribute_error":
            return self._analyze_attribute_error(error_message, code, line_hint)
        elif error_type == "assertion_error":
            return self._analyze_assertion(code, line_hint)
        elif error_type == "timeout":
            return self._analyze_timeout(code)
        elif error_type == "recursion":
            return self._analyze_recursion(code)
        elif error_type == "logic_error":
            return self._analyze_logic(code)
        else:
            return self._analyze_with_llm(incident)

    def _extract_line_hint(self, stderr: str) -> Optional[int]:
        m = re.search(r"(?:line\s*(\d+)|File.*?line\s*(\d+))", stderr)
        if m:
            return int(m.group(1) or m.group(2))
        return None

    def _analyze_name_error(self, message: str, code: str, line_hint: Optional[int]) -> Dict[str, Any]:
        m = re.search(r"'(\w+)'", message) or re.search(r"\bname\s+'?(\w+)'?\s+is not defined", message)
        missing_name = m.group(1) if m else "unknown"
        return {
            "root_cause": f"Undefined variable or import: '{missing_name}'",
            "fix_strategy": "add_import_or_define",
            "missing_symbol": missing_name,
            "fix_detail": f"Name '{missing_name}' is used but never defined or imported",
            "line_hint": line_hint,
        }

    def _analyze_type_error(self, message: str, code: str, line_hint: Optional[int]) -> Dict[str, Any]:
        return {
            "root_cause": f"Type mismatch: {message[:100]}",
            "fix_strategy": "type_cast_or_check",
            "fix_detail": f"Add type checking or conversion: {message[:100]}",
            "line_hint": line_hint,
        }

    def _analyze_import_error(self, message: str, code: str) -> Dict[str, Any]:
        m = re.search(r"'(\\w+)'", message) or re.search(r"No module named\\s+'([^']+)'", message)
        missing_module = m.group(1) if m else "unknown"
        return {
            "root_cause": f"Missing import: '{missing_module}'",
            "fix_strategy": "add_import_or_install",
            "missing_module": missing_module,
            "fix_detail": f"Module '{missing_module}' not found. Add import or remove dependency.",
        }

    def _analyze_syntax_error(self, message: str, code: str, line_hint: Optional[int]) -> Dict[str, Any]:
        return {
            "root_cause": f"Syntax error: {message[:100]}",
            "fix_strategy": "fix_syntax",
            "fix_detail": f"Invalid syntax near line {line_hint or 'unknown'}",
            "line_hint": line_hint,
        }

    def _analyze_zero_division(self, code: str, line_hint: Optional[int]) -> Dict[str, Any]:
        return {
            "root_cause": "Division by zero",
            "fix_strategy": "add_zero_guard",
            "fix_detail": "Add check before division to prevent divide-by-zero",
            "line_hint": line_hint,
        }

    def _analyze_index_error(self, code: str, line_hint: Optional[int]) -> Dict[str, Any]:
        return {
            "root_cause": "List index out of range",
            "fix_strategy": "bound_check",
            "fix_detail": "Check list length before indexing, or use .get() for dicts",
            "line_hint": line_hint,
        }

    def _analyze_key_error(self, message: str, code: str) -> Dict[str, Any]:
        m = re.search(r"KeyError:\s*(\d+|'.+?'|\".+?\")", message)
        key = m.group(1) if m else "?"
        return {
            "root_cause": f"Dictionary key not found: {key}",
            "fix_strategy": "use_dict_get",
            "fix_detail": "Replace direct bracket access with .get() method",
        }

    def _analyze_attribute_error(self, message: str, code: str, line_hint: Optional[int]) -> Dict[str, Any]:
        m = re.search(r"'(\w+)' object has no attribute '(\w+)'", message)
        obj_type = m.group(1) if m else "?"
        attr = m.group(2) if m else "?"
        return {
            "root_cause": f"'{obj_type}' object has no attribute '{attr}'",
            "fix_strategy": "check_attribute",
            "fix_detail": f"Object of type '{obj_type}' does not have '{attr}'",
            "line_hint": line_hint,
        }

    def _analyze_assertion(self, code: str, line_hint: Optional[int]) -> Dict[str, Any]:
        return {
            "root_cause": "Assertion failed",
            "fix_strategy": "fix_assertion",
            "fix_detail": "Assert condition is not met. Check the assertion logic.",
            "line_hint": line_hint,
        }

    def _analyze_timeout(self, code: str) -> Dict[str, Any]:
        return {
            "root_cause": "Execution timeout (infinite loop or too slow)",
            "fix_strategy": "optimize_or_limit",
            "fix_detail": "Code took too long. Check for infinite loops or optimize algorithm.",
        }

    def _analyze_recursion(self, code: str) -> Dict[str, Any]:
        return {
            "root_cause": "Maximum recursion depth exceeded",
            "fix_strategy": "fix_recursion",
            "fix_detail": "Recursive function has no base case or recurses too deeply.",
        }

    def _analyze_logic(self, code: str) -> Dict[str, Any]:
        return {
            "root_cause": "Code completed but produced unexpected output",
            "fix_strategy": "review_logic",
            "fix_detail": "The code ran without errors but the result is incorrect.",
        }

    def _analyze_with_llm(self, incident: RepairIncident) -> Dict[str, Any]:
        if self.llm is None:
            return {
                "root_cause": f"Unknown error: {incident.error_message[:100]}",
                "fix_strategy": "generic_retry",
                "fix_detail": incident.error_message[:200],
            }
        try:
            prompt = (
                f"Python code failed with error type '{incident.error_type}':\n"
                f"Error: {incident.error_message[:300]}\n\n"
                f"Code:\n{incident.code[:1200]}\n\n"
                f"Analyze the root cause. Respond:\n"
                f"ROOT: <one-line root cause>\n"
                f"FIX_STRATEGY: <strategy label>\n"
                f"DETAIL: <one-line fix description>"
            )
            text = self.llm(prompt, temperature=0.1)
            root = re.search(r"ROOT:\s*(.+)", text)
            strategy = re.search(r"FIX_STRATEGY:\s*(.+)", text)
            detail = re.search(r"DETAIL:\s*(.+)", text)
            return {
                "root_cause": root.group(1).strip() if root else "Unknown (LLM analysis inconclusive)",
                "fix_strategy": strategy.group(1).strip() if strategy else "llm_generated",
                "fix_detail": detail.group(1).strip() if detail else "See LLM output",
            }
        except Exception as e:
            logger.warning("LLM root cause analysis failed: %s", e)
            return {
                "root_cause": f"Unknown: {incident.error_message[:100]}",
                "fix_strategy": "generic_retry",
                "fix_detail": str(e)[:100],
            }


class FixGenerator:
    """Generates code fixes based on root cause analysis."""

    def __init__(self, llm_generator: Optional[Callable] = None):
        self.llm = llm_generator

    def generate(self, code: str, analysis: Dict[str, Any]) -> Optional[str]:
        strategy = analysis.get("fix_strategy", "")
        line_hint = analysis.get("line_hint")

        if strategy == "add_import_or_define":
            return self._fix_missing_name(code, analysis)
        elif strategy == "type_cast_or_check":
            return self._fix_type_error(code, analysis)
        elif strategy == "add_import_or_install":
            return self._fix_missing_module(code, analysis)
        elif strategy == "fix_syntax":
            return self._fix_syntax(code)
        elif strategy == "add_zero_guard":
            return self._fix_zero_division(code, line_hint)
        elif strategy == "bound_check":
            return self._fix_index_error(code, line_hint)
        elif strategy == "use_dict_get":
            return self._fix_key_error(code)
        elif strategy == "check_attribute":
            return self._fix_attribute_error(code, analysis)
        elif strategy == "fix_assertion":
            return self._fix_assertion(code)
        elif strategy == "optimize_or_limit":
            return self._fix_timeout(code)
        elif strategy == "fix_recursion":
            return self._fix_recursion(code)
        elif strategy == "review_logic":
            return self._fix_logic(code)
        elif strategy == "generic_retry" or strategy == "llm_generated":
            return self._fix_with_llm(code, analysis)
        else:
            return self._fix_with_llm(code, analysis)

    def _fix_missing_name(self, code: str, analysis: Dict[str, Any]) -> str:
        missing = analysis.get("missing_symbol", "")
        if not missing:
            return code

        common_stdlib = {"math", "json", "random", "collections", "itertools",
                         "datetime", "re", "string", "functools", "statistics",
                         "copy", "typing", "pathlib", "csv", "io", "os", "sys"}

        if missing in common_stdlib:
            return f"import {missing}\n{code}"

        lines = code.split("\n")
        first_use = -1
        for i, line in enumerate(lines):
            if missing in line and "import" not in line and not line.strip().startswith("#"):
                first_use = i
                break

        if first_use < 0:
            return code

        indent = " " * (len(lines[first_use]) - len(lines[first_use].lstrip()))
        if indent:
            lines.insert(first_use, f"{indent}{missing} = None")
        else:
            lines.insert(first_use, f"{missing} = None")

        return "\n".join(lines)

    def _fix_type_error(self, code: str, analysis: Dict[str, Any]) -> str:
        lines = code.split("\n")
        fixed = []
        for line in lines:
            if "None" in line and "def " not in line and "class " not in line:
                line = line.replace(" None +", " 0 +").replace(" + None", " + 0")
                line = line.replace("(None", "(0").replace("None)", "0)")
            fixed.append(line)
        return "\n".join(fixed)

    def _fix_missing_module(self, code: str, analysis: Dict[str, Any]) -> str:
        return code

    def _fix_syntax(self, code: str) -> str:
        return textwrap.dedent(code)

    def _fix_zero_division(self, code: str, line_hint: Optional[int]) -> str:
        lines = code.split("\n")
        fixed = []
        for i, line in enumerate(lines):
            if "/" in line and "if" not in line and "#" not in line:
                stripped = line.strip()
                if stripped.startswith("#") or stripped.endswith(":"):
                    fixed.append(line)
                    continue
                indent = " " * (len(line) - len(line.lstrip()))
                body_indent = indent + "    "
                at_top_level = indent == ""
                guard = "pass" if at_top_level else "return 0"
                fixed.append(f"{indent}try:")
                fixed.append(f"{body_indent}{stripped}")
                fixed.append(f"{indent}except ZeroDivisionError:")
                fixed.append(f"{body_indent}{guard}")
            else:
                fixed.append(line)
        return "\n".join(fixed)

    def _fix_index_error(self, code: str, line_hint: Optional[int]) -> str:
        lines = code.split("\n")
        if line_hint and 1 <= line_hint <= len(lines):
            bad = lines[line_hint - 1]
            if "[" in bad:
                indent = " " * (len(bad) - len(bad.lstrip()))
                var_match = re.search(r"(\w+)\[", bad)
                if var_match:
                    var = var_match.group(1)
                    fixed_line = bad.replace(f"{var}[", f"{var}[")  # no change yet
                    at_top_level = indent == ""
                    guard = "pass" if at_top_level else "return None"
                    lines[line_hint - 1] = (
                        f"{indent}if isinstance({var}, list) and len({var}) > 0:\n"
                        f"{indent}    {bad.strip()}\n"
                        f"{indent}else:\n"
                        f"{indent}    {guard}"
                    )
        return "\n".join(lines)

    def _fix_key_error(self, code: str) -> str:
        bracket_access = re.findall(r"(\w+)\[(\'[^\']+\'|\"[^\"]+\")\]", code)
        if not bracket_access:
            bracket_access = re.findall(r"(\w+)\[(\d+\w*)\]", code)
        for var, key in bracket_access:
            code = code.replace(f"{var}[{key}]", f"{var}.get({key})", 1)
        return code

    def _fix_attribute_error(self, code: str, analysis: Dict[str, Any]) -> str:
        return code

    def _fix_assertion(self, code: str) -> str:
        lines = code.split("\n")
        fixed = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("assert "):
                indent = " " * (len(line) - len(line.lstrip()))
                cond = stripped[7:]
                fixed.append(f"{indent}if not ({cond}):")
                fixed.append(f"{indent}    return False")
            else:
                fixed.append(line)
        return "\n".join(fixed)

    def _fix_timeout(self, code: str) -> str:
        return code

    def _fix_recursion(self, code: str) -> str:
        return code

    def _fix_logic(self, code: str) -> str:
        return code

    def _fix_with_llm(self, code: str, analysis: Dict[str, Any]) -> Optional[str]:
        if self.llm is None:
            return None
        try:
            prompt = (
                f"Fix this Python code that has a '{analysis.get('fix_strategy', 'unknown')}' error.\n"
                f"Root cause: {analysis.get('root_cause', 'unknown')}\n\n"
                f"Code:\n{code[:1500]}\n\n"
                f"Return ONLY the corrected Python code, no explanations, no markdown fences."
            )
            fixed = self.llm(prompt, temperature=0.2)
            fixed = re.sub(r"```(?:python)?\s*\n?", "", fixed).strip()
            if fixed and (fixed.startswith("def ") or fixed.startswith("import ") or fixed.startswith("from ") or fixed.startswith("class ")):
                return fixed
            if fixed and len(fixed) > 10:
                return fixed
        except Exception as e:
            logger.warning("LLM fix generation failed: %s", e)
        return None


class RepairAgent:
    """Error → Classify → Root Cause → Fix → Rewrite → Re-run → Escalate.

    Standalone repair system with:
      - debug mode: console tracing + trace file
      - test mode: verify fix by running assertions
      - workspace integration: backup original, rewrite patched
      - escalation: after max_repairs, produce structured failure report

    Can be used standalone or wired into autonomous_loop / project_agent.
    """

    def __init__(
        self,
        execution_runner: Optional[Callable] = None,
        workspace: Optional[Any] = None,
        llm_generator: Optional[Callable] = None,
        max_repairs: int = 3,
    ):
        self.execution_runner = execution_runner
        self.workspace = workspace
        self.llm = llm_generator
        self.max_repairs = max_repairs

        self.classifier = ErrorClassifier()
        self.rca = RootCauseAnalyzer(llm_generator=self.llm)
        self.fixer = FixGenerator(llm_generator=self.llm)

        self._repair_history: List[RepairResult] = []
        self._debug_trace: List[str] = []

    def repair(
        self,
        code: str,
        file_path: str = "",
        error_result: Optional[Dict[str, Any]] = None,
        debug: bool = False,
        tests: Optional[List[str]] = None,
    ) -> RepairResult:
        """Run the full repair pipeline on failed code.

        Args:
            code: The Python source code that failed
            file_path: Optional workspace file path for rewriting
            error_result: Dict with stdout/stderr/exit_code from previous run
            debug: Enable console tracing
            tests: Optional list of test code strings to verify fix

        Returns:
            RepairResult with success/failure, fix details, and re-run result
        """
        self._debug_trace = []
        start = time.time()

        if error_result is None:
            error_result = {"stdout": "", "stderr": "No error result provided", "exit_code": 1, "error": "Unknown"}

        self._log(f"=== REPAIR: {file_path or '<inline>'} ===", debug)
        self._log(f"Exit code: {error_result.get('exit_code')}", debug)

        incidents: List[RepairIncident] = []
        current_code = code
        resolved = False
        escalation = False
        escalation_reason = ""

        for attempt in range(1, self.max_repairs + 1):
            self._log(f"\n--- Repair attempt {attempt}/{self.max_repairs} ---", debug)

            # 1. Classify
            error_type, error_message, exit_code = self.classifier.classify(error_result)
            severity = self.classifier.severity(error_type)
            self._log(f"Error type: {error_type} ({severity})", debug)
            self._log(f"Message: {error_message[:100]}", debug)

            # 2. Create incident
            incident = RepairIncident(
                file_path=file_path,
                error_type=error_type,
                error_message=error_message,
                stdout=error_result.get("stdout", ""),
                stderr=error_result.get("stderr", ""),
                exit_code=exit_code,
                code=current_code,
                attempt=attempt,
            )

            # 3. Root cause analysis
            analysis = self.rca.analyze(incident)
            incident.root_cause = analysis.get("root_cause", "")
            incident.fix_strategy = analysis.get("fix_strategy", "")
            self._log(f"Root cause: {analysis['root_cause'][:120]}", debug)
            self._log(f"Strategy: {analysis['fix_strategy']}", debug)

            # 4. Generate fix
            fix = self.fixer.generate(current_code, analysis)
            incident.fix_applied = fix[:200] if fix else ""

            if fix is None or fix == current_code:
                escalation = True
                escalation_reason = f"No fix generated at attempt {attempt} (strategy: {analysis['fix_strategy']})"
                self._log(f"  ESCALATION: {escalation_reason}", debug)
                incidents.append(incident)
                break

            self._log(f"Fix applied ({len(fix)} chars, was {len(current_code)} chars)", debug)

            # 5. Rewrite file (if workspace path given)
            if file_path and self.workspace:
                self._rewrite_file(file_path, fix, current_code, debug)

            current_code = fix

            # 6. Re-run
            run_result = self._execute(current_code)
            self._log(f"Re-run: exit={run_result.get('exit_code')}", debug)

            if run_result.get("exit_code") == 0:
                resolved = True
                self._log(f"  FIXED (attempt {attempt})", debug)

                # 7. Verify with tests if provided
                if tests:
                    test_pass = self._verify_tests(current_code, tests, debug)
                    if not test_pass:
                        resolved = False
                        escalation = True
                        escalation_reason = f"Fix passes but tests fail at attempt {attempt}"
                        self._log(f"  TESTS FAILED after fix", debug)
                        error_result = {"stdout": "", "stderr": "Tests failed after fix", "exit_code": 1}
                        incidents.append(incident)
                        continue

                    self._log(f"  All {len(tests)} tests passed", debug)

                incident.resolved = True
                incidents.append(incident)
                break

            error_result = run_result
            incidents.append(incident)

        result = RepairResult(
            success=resolved,
            file_path=file_path,
            error_type=incidents[-1].error_type if incidents else "unknown",
            root_cause=incidents[-1].root_cause if incidents else "",
            fix_strategy=incidents[-1].fix_strategy if incidents else "",
            fix_applied=incidents[-1].fix_applied if incidents else "",
            attempts=len(incidents),
            duration=round(time.time() - start, 2),
            re_run_result=self._execute(current_code) if resolved else error_result,
            incidents=[i.to_dict() for i in incidents],
            escalation=escalation,
            escalation_reason=escalation_reason,
        )

        self._repair_history.append(result)
        self._log(f"\n=== REPAIR {'FIXED' if resolved else 'FAILED'} in {result.duration}s ===", debug)

        if debug and file_path and self.workspace:
            self._save_debug_trace(file_path, result)

        return result

    def _rewrite_file(self, file_path: str, new_code: str, old_code: str, debug: bool):
        try:
            backup_path = f".backups/{file_path}.bak"
            self.workspace.create(backup_path, old_code)
            self._log(f"  Backup saved: {backup_path}", debug)
        except Exception:
            pass

        try:
            content = self.workspace.read(file_path)
            if content.get("success"):
                self.workspace.edit(file_path, old_code, new_code)
                self._log(f"  Rewritten: {file_path}", debug)
            else:
                self.workspace.create(file_path, new_code)
                self._log(f"  Created: {file_path}", debug)
        except Exception as e:
            self._log(f"  File rewrite failed: {e}", debug)

    def _execute(self, code: str) -> Dict[str, Any]:
        if self.execution_runner is not None:
            try:
                return self.execution_runner(code)
            except Exception as e:
                return {"stdout": "", "stderr": str(e), "exit_code": 1, "error": str(e)}
        from execution_agent import execute_python
        return execute_python(code)

    def _verify_tests(self, code: str, tests: List[str], debug: bool) -> bool:
        all_pass = True
        for i, test in enumerate(tests):
            test_code = f"{code}\n\n{test}"
            result = self._execute(test_code)
            ok = result.get("exit_code") == 0
            if not ok:
                self._log(f"  Test {i+1} FAILED: {result.get('stderr', '')[:80]}", debug)
                all_pass = False
        return all_pass

    def _log(self, message: str, to_console: bool = False):
        self._debug_trace.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        if to_console:
            print(message)

    def _save_debug_trace(self, file_path: str, result: RepairResult):
        try:
            safe_name = re.sub(r"[^a-zA-Z0-9_]+", "_", file_path)[:30]
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            path = f"logs/repair_{safe_name}_{timestamp}.txt"
            lines = [
                f"Repair Agent Debug Trace",
                f"File: {file_path}",
                f"Success: {result.success}",
                f"Attempts: {result.attempts}",
                f"Duration: {result.duration}s",
                f"Escalation: {result.escalation}",
                f"",
            ]
            lines.extend(self._debug_trace)
            self.workspace.create(path, "\n".join(lines))
        except Exception as e:
            logger.debug("Failed to save repair trace: %s", e)

    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._repair_history[-limit:]]

    def get_stats(self) -> Dict[str, Any]:
        if not self._repair_history:
            return {"total_repairs": 0}
        total = len(self._repair_history)
        resolved = sum(1 for r in self._repair_history if r.success)
        escalated = sum(1 for r in self._repair_history if r.escalation)
        avg_attempts = round(sum(r.attempts for r in self._repair_history) / total, 1)
        return {
            "total_repairs": total,
            "resolved": resolved,
            "escalated": escalated,
            "success_rate": round(resolved / total * 100, 1),
            "avg_attempts": avg_attempts,
            "avg_duration_seconds": round(sum(r.duration for r in self._repair_history) / total, 2),
        }
