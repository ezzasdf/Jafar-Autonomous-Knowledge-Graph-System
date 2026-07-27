"""
Jafar CLI — unified interface for workspace, execution, loop, repair, and project agents.

Usage:
  python jafar_cli.py workspace list
  python jafar_cli.py workspace read <path>
  python jafar_cli.py workspace write <path> --content "..."
  python jafar_cli.py workspace delete <path>

  python jafar_cli.py execute <code> [--timeout N]
  python jafar_cli.py execute-file <path> [--timeout N]

  python jafar_cli.py loop <goal> [--debug] [--test] [--max-retries N]

  python jafar_cli.py repair <file_path> [--max-repairs N] [--debug]

  python jafar_cli.py project <goal> [--debug] [--repair] [--no-tests]
  python jafar_cli.py project list
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import click
import time
import logging
from pathlib import Path
from typing import Optional, Callable

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

from config import (
    CODING_WORKSPACE_CONFIG, EXECUTION_AGENT_CONFIG,
    REPAIR_AGENT_CONFIG, PROJECT_AGENT_CONFIG,
    AUTONOMOUS_LOOP_CONFIG, TRANSFORMER_REASONING_CONFIG,
    GGUF_CONFIG, LLM_CONFIG, LLAMA_SERVER_CONFIG,
)
from coding_workspace import CodingWorkspace
from execution_agent import execute_python, execute_file
from repair_agent import RepairAgent, FixGenerator
from project_agent import ProjectAgent
from autonomous_loop import AutonomousLoop
from rule_engine import RuleEngine

# ------------------------------------------------------------------
# RuleEngine gate — tries deterministic rules first, then Qwen
# ------------------------------------------------------------------
_llm_engine = None
_rule_engine = None

def _get_llm_generator() -> Optional[Callable]:
    global _llm_engine, _rule_engine
    if _rule_engine is None:
        raw_llm = _create_raw_llm()
        _rule_engine = RuleEngine(llm_generator=raw_llm, confidence_threshold=0.65)

    def _generate(prompt: str, temperature: float = 0.4, max_tokens: int = 256) -> str:
        return _rule_engine.decide(prompt, temperature=temperature, max_tokens=max_tokens)

    return _generate

def _create_raw_llm() -> Optional[Callable]:
    global _llm_engine
    if _llm_engine is None:
        try:
            from transformer_reasoning import TransformerReasoningEngine
            _llm_engine = TransformerReasoningEngine(
                memory_system=None,
                model_name=GGUF_CONFIG.get("model_path", ""),
                max_new_tokens=256,
                temperature=0.4,
                use_gguf=True,
            )
            _llm_engine._load()
        except Exception as e:
            click.echo(f"  (LLM not available: {e})", err=True)
            return None

    def _generate(prompt: str, temperature: float = 0.4, max_tokens: int = 256) -> str:
        old_temp = _llm_engine.temperature
        old_tokens = _llm_engine.max_new_tokens
        _llm_engine.temperature = temperature
        _llm_engine.max_new_tokens = max_tokens
        try:
            result = _llm_engine._generate(prompt)
            return result or ""
        except Exception as e:
            return ""
        finally:
            _llm_engine.temperature = old_temp
            _llm_engine.max_new_tokens = old_tokens

    return _generate


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

def get_workspace() -> CodingWorkspace:
    return CodingWorkspace()


def _print_result(result, indent=0):
    prefix = " " * indent
    if isinstance(result, dict):
        for k, v in result.items():
            if isinstance(v, dict):
                click.echo(f"{prefix}{k}:")
                _print_result(v, indent + 2)
            elif isinstance(v, list):
                click.echo(f"{prefix}{k}: [{len(v)} items]")
                for i, item in enumerate(v[:5]):
                    if isinstance(item, dict):
                        click.echo(f"{prefix}  [{i}]")
                        _print_result(item, indent + 4)
                    else:
                        click.echo(f"{prefix}  [{i}] {item}")
                if len(v) > 5:
                    click.echo(f"{prefix}  ... ({len(v) - 5} more)")
            else:
                click.echo(f"{prefix}{k}: {v}")
    else:
        click.echo(f"{prefix}{result}")


# ------------------------------------------------------------------
# Main CLI group
# ------------------------------------------------------------------

@click.group()
def cli():
    """Jafar CLI — coding workspace, execution, autonomous loop, repair, and project agents."""


# ==================================================================
#  workspace
# ==================================================================

@cli.group()
def workspace():
    """Manage coding workspace files."""

@workspace.command("list")
def ws_list():
    """List all files in the workspace."""
    ws = get_workspace()
    files = ws.list_files()
    if not files:
        click.echo("(empty workspace)")
        return
    click.echo(f"Workspace: {CODING_WORKSPACE_CONFIG['workspace_dir']}")
    click.echo(f"Files: {len(files)}")
    for f in files:
        size = f.get("size", 0)
        click.echo(f"  {f['path']} ({size} chars)")

@workspace.command("read")
@click.argument("path")
def ws_read(path):
    """Read a workspace file."""
    ws = get_workspace()
    r = ws.read(path)
    if r is None or not r.get("success"):
        click.echo(f"File not found: {path}", err=True)
        sys.exit(1)
    click.echo(r["content"])

@workspace.command("write")
@click.argument("path")
@click.option("--content", "-c", default="", help="File content (use --content or pipe via stdin)")
@click.option("--stdin", "from_stdin", is_flag=True, help="Read content from stdin")
def ws_write(path, content, from_stdin):
    """Write content to a workspace file."""
    if from_stdin:
        content = click.get_text_stream("stdin").read()
    ws = get_workspace()
    ok = ws.write(path, content)
    if ok.get("success"):
        click.echo(f"Written: {path} ({len(content)} chars)")
    else:
        click.echo(f"Failed to write: {path} ({ok.get('error', 'unknown')})", err=True)
        sys.exit(1)

@workspace.command("delete")
@click.argument("path")
def ws_delete(path):
    """Delete a workspace file."""
    ws = get_workspace()
    ok = ws.delete(path)
    if ok:
        click.echo(f"Deleted: {path}")
    else:
        click.echo(f"Failed to delete: {path}", err=True)
        sys.exit(1)


# ==================================================================
#  execute
# ==================================================================

@cli.command()
@click.argument("code", required=False)
@click.option("--timeout", "-t", default=10.0, help="Execution timeout in seconds")
@click.option("--stdin", "from_stdin", is_flag=True, help="Read code from stdin")
def execute(code, timeout, from_stdin):
    """Execute Python code and show results."""
    if from_stdin:
        code = click.get_text_stream("stdin").read()
    if not code:
        click.echo("No code provided. Use --stdin or pass code as argument.", err=True)
        sys.exit(1)

    result = execute_python(code, timeout=timeout)
    click.echo(f"Exit code: {result.get('exit_code', '?')}")
    click.echo(f"Duration: {result.get('duration', 0):.2f}s")
    if result.get("stdout", "").strip():
        click.echo(f"--- stdout ---")
        click.echo(result["stdout"])
    if result.get("stderr", "").strip():
        click.echo(f"--- stderr ---")
        click.echo(result["stderr"])
    if result.get("error", ""):
        click.echo(f"Error: {result['error']}")

@cli.command()
@click.argument("path")
@click.option("--timeout", "-t", default=10.0, help="Execution timeout in seconds")
def execute_file_cmd(path, timeout):
    """Execute a file from the workspace."""
    result = execute_file(path, timeout=timeout)
    click.echo(f"Exit code: {result.get('exit_code', '?')}")
    click.echo(f"Duration: {result.get('duration', 0):.2f}s")
    if result.get("stdout", "").strip():
        click.echo(f"--- stdout ---")
        click.echo(result["stdout"])
    if result.get("stderr", "").strip():
        click.echo(f"--- stderr ---")
        click.echo(result["stderr"])


# ==================================================================
#  loop (autonomous)
# ==================================================================

@cli.command()
@click.argument("goal")
@click.option("--debug", "-d", is_flag=True, help="Show debug trace")
@click.option("--test", "-t", "run_tests", is_flag=True, help="Auto-generate and run tests")
@click.option("--max-retries", default=3, help="Max retries per task")
def loop(goal, debug, run_tests, max_retries):
    """Run the autonomous coding loop on a goal."""
    ws = get_workspace()

    llm = _get_llm_generator()
    dataset = None
    try:
        from reasoning_dataset import ReasoningDataset
        dataset = ReasoningDataset()
    except Exception:
        pass

    loop_engine = AutonomousLoop(
        llm_generator=llm,
        workspace=ws,
        reasoning_dataset=dataset,
    )

    click.echo(f"Goal: {goal}")
    click.echo(f"LLM: {'available' if llm else 'unavailable (template fallback)'}")
    click.echo(f"Dataset: {'available' if dataset else 'unavailable'}")
    click.echo(f"Max retries: {max_retries}")
    click.echo()

    start = time.time()
    result = loop_engine.run(
        goal,
        max_retries_per_task=max_retries,
        debug=debug,
        test=run_tests,
    )
    elapsed = round(time.time() - start, 2)

    click.echo(f"Status: {result.get('status', '?')}")
    click.echo(f"Tasks: {result.get('successful', 0)} ok / {result.get('failed', 0)} failed")
    click.echo(f"Duration: {result.get('duration', elapsed)}s")

    if result.get("tests"):
        for t in result["tests"]:
            click.echo(f"  Tests for {t['task']}: {t['passed']}/{t['total']} passed")

    if debug and result.get("debug_trace"):
        click.echo(f"\nDebug trace ({len(result['debug_trace'])} lines):")
        for line in result["debug_trace"][-20:]:
            click.echo(f"  {line}")


# ==================================================================
#  repair
# ==================================================================

@cli.command()
@click.argument("file_path")
@click.option("--max-repairs", "-m", default=3, help="Max repair attempts")
@click.option("--debug", "-d", is_flag=True, help="Show debug output")
def repair(file_path, max_repairs, debug):
    """Analyze and repair a Python file."""
    ws = get_workspace()
    read_result = ws.read(file_path)
    if read_result is None or not read_result.get("success"):
        click.echo(f"File not found: {file_path}", err=True)
        sys.exit(1)
    code: str = read_result["content"]

    def exec_runner(code_str):
        return execute_python(code_str, timeout=10.0)

    agent = RepairAgent(
        execution_runner=exec_runner,
        max_repairs=max_repairs,
        workspace=ws,
    )

    click.echo(f"Repairing: {file_path}")
    click.echo(f"Code: {len(code)} chars")
    click.echo()

    initial_result = execute_python(code, timeout=10.0)

    result = agent.repair(code, file_path=file_path, error_result=initial_result, debug=debug)

    click.echo(f"Success: {result.success}")
    click.echo(f"Error type: {result.error_type}")
    click.echo(f"Root cause: {result.root_cause}")
    click.echo(f"Strategy: {result.fix_strategy}")
    click.echo(f"Escalation: {result.escalation}")
    if result.fix_applied:
        click.echo(f"Fix: {result.fix_applied[:100]}...")


# ==================================================================
#  project
# ==================================================================

@cli.group()
def project():
    """Manage coding projects."""

@project.command("build")
@click.argument("goal")
@click.option("--debug", "-d", is_flag=True, help="Show debug output")
@click.option("--repair", "-r", "use_repair", is_flag=True, help="Use repair agent on failures")
@click.option("--no-llm", is_flag=True, help="Skip LLM, use templates only")
def project_build(goal, debug, use_repair, no_llm):
    """Build a multi-file project from a natural language goal."""
    ws = get_workspace()

    llm = None
    if not no_llm:
        click.echo("Loading LLM (Qwen GGUF)...")
        llm = _get_llm_generator()
        if llm:
            click.echo("  LLM ready.")
        else:
            click.echo("  (template fallback only)")

    repair_agent = None
    if use_repair:
        def exec_runner(code_str):
            return execute_python(code_str, timeout=10.0)
        repair_agent = RepairAgent(
            execution_runner=exec_runner,
            max_repairs=REPAIR_AGENT_CONFIG.get("max_repairs", 3),
            workspace=ws,
        )

    click.echo(f"Building project: {goal}")
    click.echo(f"Mode: {'LLM' if llm else 'Template'} | Repair: {'yes' if use_repair else 'no'}")
    click.echo()

    pa = ProjectAgent(
        llm_generator=llm,
        workspace=ws,
        repair_agent=repair_agent,
    )

    result = pa.build(goal, debug=debug)

    click.echo(f"Success: {result.success}")
    click.echo(f"Project: {result.project_name}")
    click.echo(f"Files: {result.file_count}")
    click.echo(f"Tests passed: {result.tests_passed}/{result.total_tests}")
    click.echo(f"Duration: {result.duration}s")

    if result.warnings:
        for w in result.warnings:
            click.echo(f"  Warning: {w}")

    if result.repair_summary and any(result.repair_summary.values()):
        rs = result.repair_summary
        click.echo(f"Repairs: {rs.get('fixed', 0)}/{rs.get('total', 0)} fixed, {rs.get('escalated', 0)} escalated")

    if result.file_count > 0:
        click.echo(f"\nFiles generated: {result.file_count}")
        for d in result.details:
            if d.get("file"):
                click.echo(f"  {d['file']}")
            elif d.get("file_path"):
                click.echo(f"  {d['file_path']}")

@project.command("list")
def project_list():
    """List built projects in the workspace."""
    ws = get_workspace()
    projects = ws.list_projects()
    if not projects:
        click.echo("No projects built yet.")
        return
    click.echo(f"Projects ({len(projects)}):")
    for p in projects:
        files = ws.list_files(p)
        click.echo(f"  {p}/ ({len(files)} files)")

@project.command("read")
@click.argument("project_name")
@click.argument("file_path", required=False, default="")
def project_read(project_name, file_path):
    """Read a project file."""
    ws = get_workspace()
    if not file_path:
        files = ws.list_files(project_name)
        if not files:
            click.echo(f"No files in project: {project_name}", err=True)
            sys.exit(1)
        click.echo(f"Project: {project_name}")
        for f in files:
            click.echo(f"  {f['path']}")
        return
    full_path = f"{project_name}/{file_path}"
    r = ws.read(full_path)
    if r is None or not r.get("success"):
        click.echo(f"File not found: {full_path}", err=True)
        sys.exit(1)
    click.echo(r["content"])


@project.command("clean")
@click.argument("project_name", required=False)
def project_clean(project_name):
    """Delete project files from workspace."""
    ws = get_workspace()
    if project_name:
        for f in ws.list_files(project_name):
            ws.delete(f["path"])
        click.echo(f"Cleaned project: {project_name}")
    else:
        for p in ws.list_projects():
            for f in ws.list_files(p):
                ws.delete(f["path"])
        click.echo("Cleaned all projects")


# ==================================================================
#  benchmark
# ==================================================================

@cli.command()
@click.option("--prompt", "-p", default="Write a short sentence about AI.", help="Benchmark prompt")
@click.option("--threads", "-t", "thread_counts", default=None, help="Comma-separated thread counts to test (default: 4,6,8)")
def benchmark(prompt, thread_counts):
    """Benchmark llama-server thread scaling without editing config."""
    from llama_server import LlamaServer
    server = LlamaServer()

    counts = None
    if thread_counts:
        counts = [int(x.strip()) for x in thread_counts.split(",")]

    click.echo("Benchmarking thread scaling...")
    click.echo(f"Model: {os.path.basename(server.model_path)}")
    click.echo(f"Prompt: {prompt[:60]}...")
    click.echo()

    result = server.benchmark_threads(prompt=prompt, counts=counts)

    click.echo(f"{'Threads':<10} {'Tokens':<10} {'Time (s)':<12} {'TPS':<10}")
    click.echo("-" * 42)
    for n, r in sorted(result["results"].items()):
        if r.get("ok"):
            click.echo(f"{n:<10} {r['tokens']:<10} {r['elapsed']:<12} {r['tps']:<10}")
        else:
            click.echo(f"{n:<10} {'FAILED':<10} {r.get('error', ''):<12}")

    best_n = result["best_threads"]
    best_tps = result["results"][best_n]["tps"]
    click.echo()
    click.echo(f"Best thread count: {best_n} ({best_tps} t/s)")

    server.stop()


# ==================================================================
#  Entry point
# ==================================================================

if __name__ == "__main__":
    cli()
