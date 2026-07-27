"""
Execution Agent — runs Python code and returns structured results.

Wraps the existing code_sandbox for safe execution. The primary entry
point is execute_python(code) which returns {stdout, stderr, exit_code}.

Also supports:
  - execute_file(path) — run a file from the coding workspace
  - execute_project(name, entry_point) — run a project's main script
"""

import logging
import sys
import traceback
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)
debug_logger = logging.getLogger(f"{__name__}.debug")

try:
    from code_sandbox import run_code as sandbox_run
    HAS_SANDBOX = True
except ImportError:
    HAS_SANDBOX = False


def execute_python(code: str, timeout: float = 10.0) -> Dict[str, Any]:
    """Execute Python code and return stdout, stderr, exit_code.

    Args:
        code: Python source code to execute
        timeout: Max execution time in seconds

    Returns:
        Dict with keys:
          stdout:     Captured standard output text
          stderr:     Captured standard error text
          exit_code:  0 for success, 1 for runtime error, -1 for timeout
          error:      Error type and message (empty on success)
          duration:   Execution time in seconds
    """
    if not code or not code.strip():
        return {
            "stdout": "",
            "stderr": "No code provided",
            "exit_code": 1,
            "error": "EmptyCode: nothing to execute",
            "duration": 0.0,
        }

    result = {"stdout": "", "stderr": "", "exit_code": 0, "error": "", "duration": 0.0}

    if HAS_SANDBOX:
        try:
            sandbox_result = sandbox_run(code, timeout=timeout)
            result["stdout"] = sandbox_result.get("output", "")
            result["stderr"] = sandbox_result.get("error", "")
            result["exit_code"] = 0 if sandbox_result.get("success") else 1
            result["error"] = sandbox_result.get("error", "")
            result["duration"] = sandbox_result.get("execution_time", 0.0)

            if not sandbox_result.get("success"):
                if "timed out" in sandbox_result.get("error", "").lower():
                    result["exit_code"] = -1
                else:
                    result["exit_code"] = 1
        except Exception as e:
            result["stderr"] = str(e)
            result["exit_code"] = 1
            result["error"] = f"ExecutionAgentError: {e}"
    else:
        result["stderr"] = "Sandbox unavailable (code_sandbox not found)"
        result["exit_code"] = 1
        result["error"] = "SandboxUnavailable"

    debug_logger.debug(
        "execute_python: exit=%d, stdout=%d chars, stderr=%d chars, duration=%.2fs",
        result["exit_code"], len(result["stdout"]), len(result["stderr"]), result["duration"],
    )
    return result


def execute_file(file_path: str, timeout: float = 10.0) -> Dict[str, Any]:
    """Read and execute a Python file from disk.

    The file can be an absolute path or relative to the workspace.
    """
    from pathlib import Path

    path = Path(file_path)
    if not path.is_absolute():
        base = Path(__file__).parent / "workspace"
        path = (base / file_path).resolve()

    if not path.is_file():
        return {
            "stdout": "",
            "stderr": f"File not found: {file_path}",
            "exit_code": 1,
            "error": f"FileNotFound: {file_path}",
            "duration": 0.0,
        }

    try:
        code = path.read_text(encoding="utf-8")
    except Exception as e:
        return {
            "stdout": "",
            "stderr": f"Cannot read file: {e}",
            "exit_code": 1,
            "error": f"ReadError: {e}",
            "duration": 0.0,
        }

    return execute_python(code, timeout=timeout)


def execute_project(
    project_name: str,
    entry_point: str = "main.py",
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """Run a project's entry point script from the workspace.

    Projects are stored under workspace/<project_name>/.
    """
    from pathlib import Path

    base = Path(__file__).parent / "workspace" / project_name
    if not base.is_dir():
        return {
            "stdout": "",
            "stderr": f"Project not found: {project_name}",
            "exit_code": 1,
            "error": f"ProjectNotFound: {project_name}",
            "duration": 0.0,
        }

    return execute_file(str(base / entry_point), timeout=timeout)


def check_syntax(code: str) -> Dict[str, Any]:
    """Check Python syntax without executing."""
    try:
        compile(code.strip(), "<check>", "exec")
        return {"valid": True, "error": ""}
    except SyntaxError as e:
        return {
            "valid": False,
            "error": f"{e.msg} at line {e.lineno}",
            "lineno": e.lineno,
        }
