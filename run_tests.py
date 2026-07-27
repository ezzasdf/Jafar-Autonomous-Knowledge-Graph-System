"""Test runner for Jafar. Discovers and runs all test_*.py files."""
import sys
import os
import subprocess
import unittest
import time


def _find_standalone_tests():
    """Find test_*.py files that use check()-based module-level testing (no TestCase)."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    standalone = []
    for f in sorted(os.listdir(test_dir)):
        if f.startswith('test_') and f.endswith('.py'):
            with open(os.path.join(test_dir, f), encoding='utf-8') as fh:
                content = fh.read()
                if 'check(' in content and 'unittest.TestCase' not in content:
                    standalone.append(f)
    return standalone


def run_standalone_tests():
    """Run standalone test files as subprocesses and return (pass_count, fail_count)."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    standalone = _find_standalone_tests()
    standalones_passed = 0
    standalones_failed = 0
    for filename in standalone:
        filepath = os.path.join(test_dir, filename)
        try:
            proc = subprocess.run(
                [sys.executable, filepath],
                capture_output=True, text=True, timeout=300, cwd=test_dir
            )
        except subprocess.TimeoutExpired:
            print(f"\n--- {filename} TIMEOUT (300s) ---")
            standalones_failed += 1
            continue
        print(proc.stdout)
        if proc.stderr.strip():
            print(proc.stderr)
        if proc.returncode == 0:
            standalones_passed += 1
        else:
            standalones_failed += 1
    return standalones_passed, standalones_failed


def run_all_tests():
    start = time.time()
    test_dir = os.path.dirname(os.path.abspath(__file__))
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir,
                            pattern='test_*.py',
                            top_level_dir=test_dir)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    standalones_passed, standalones_failed = run_standalone_tests()

    elapsed = time.time() - start
    total_failures = len(result.failures) + len(result.errors) + standalones_failed
    print(f"\n{'=' * 60}")
    print(f"Ran in {elapsed:.2f}s — "
          f"{result.testsRun} unittest tests, "
          f"{len(result.failures)} failures, "
          f"{len(result.errors)} errors"
          + (f", {len(result.skipped)} skipped" if result.skipped else "")
          + f" | standalone: {standalones_passed} passed, {standalones_failed} failed")
    print(f"{'=' * 60}")

    return result.wasSuccessful() and standalones_failed == 0


def run_quick():
    """Run only core tests (reasoning, reflection, goals). Skip NN-dependent tests."""
    loader = unittest.TestLoader()
    core_patterns = ['test_reasoning', 'test_reflection', 'test_goals']
    suite = unittest.TestSuite()
    for pattern in core_patterns:
        try:
            tests = loader.discover(os.path.dirname(os.path.abspath(__file__)),
                                    pattern=f'{pattern}.py',
                                    top_level_dir=os.path.dirname(os.path.abspath(__file__)))
            suite.addTests(tests)
        except Exception:
            pass

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if mode == 'quick':
        success = run_quick()
    else:
        success = run_all_tests()
    sys.exit(0 if success else 1)
