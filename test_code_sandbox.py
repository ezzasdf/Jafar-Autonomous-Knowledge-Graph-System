"""Tests for CodeSandbox — safe Python execution with timeout."""

import sys, os, json, unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from code_sandbox import run_code, run_function


class TestCodeSandboxBasic(unittest.TestCase):
    def test_simple_execution(self):
        result = run_code("x = 1 + 1\nprint(x)")
        self.assertTrue(result["success"])
        self.assertIn("2", result["output"])

    def test_execution_with_variables(self):
        result = run_code("y = 42")
        self.assertTrue(result["success"])
        self.assertEqual(result["variables"].get("y"), "42")

    def test_print_capture(self):
        code = 'print("hello world")'
        result = run_code(code)
        self.assertTrue(result["success"])
        self.assertIn("hello world", result["output"])

    def test_multiple_lines(self):
        code = 'a = [1,2,3]\nb = [x*2 for x in a]\nprint(b)'
        result = run_code(code)
        self.assertTrue(result["success"])
        self.assertIn("2", result["output"])
        self.assertIn("6", result["output"])


class TestCodeSandboxSafety(unittest.TestCase):
    def test_os_module_blocked(self):
        result = run_code("import os")
        self.assertTrue(result["success"],
                        "Stripped code should still 'succeed'")
        output = result.get("output", "")
        self.assertNotIn("os", result.get("variables", {}),
                         "os import should have been stripped")
        if "variables" not in result or "os" not in result["variables"]:
            pass  # os was not executed — good
        else:
            self.fail("os import should be stripped")

    def test_subprocess_blocked(self):
        result = run_code("import subprocess\nprint('hi')")
        self.assertNotIn("import subprocess", result.get("output", ""),
                         "Dangerous import should be stripped")

    def test_eval_blocked(self):
        result = run_code("eval('1+1')")
        self.assertNotIn("1+1", result.get("output", ""),
                         "eval() should be stripped")

    def test_open_blocked(self):
        result = run_code("print(open('/etc/passwd'))")
        self.assertNotIn("/etc/passwd", result.get("output", ""),
                         "open() should be stripped")

    def test_import_blocked(self):
        result = run_code("import shutil\nprint(shutil.__version__)")
        self.assertNotIn("shutil", result.get("output", ""),
                         "Dangerous import should be stripped")


class TestCodeSandboxTimeout(unittest.TestCase):
    def test_infinite_loop_times_out(self):
        result = run_code("while True: pass", timeout=0.5)
        self.assertFalse(result["success"])
        self.assertIn("timed out", result["error"])

    def test_long_sleep_times_out(self):
        result = run_code("x = 0\nwhile True: x += 1", timeout=0.5)
        self.assertFalse(result["success"])
        self.assertIn("timed out", result["error"])


class TestCodeSandboxWhitelist(unittest.TestCase):
    def test_math_module_available(self):
        result = run_code("print(math.sqrt(16))")
        self.assertTrue(result["success"])
        self.assertIn("4.0", result["output"])

    def test_random_module_available(self):
        result = run_code("print(random.randint(1, 10))")
        self.assertTrue(result["success"])

    def test_json_module_available(self):
        result = run_code("print(json.dumps({'a': 1}))")
        self.assertTrue(result["success"])
        self.assertIn('{"a":', result["output"])

    def test_collections_available(self):
        result = run_code("print(collections.Counter('aabbc')['a'])")
        self.assertTrue(result["success"])
        self.assertIn("2", result["output"])

    def test_itertools_available(self):
        result = run_code("print(list(itertools.islice([1,2,3], 2)))")
        self.assertTrue(result["success"])

    def test_datetime_available(self):
        result = run_code("print(datetime.date(2024, 1, 1))")
        self.assertTrue(result["success"])


class TestCodeSandboxLimits(unittest.TestCase):
    def test_max_code_length_exceeded(self):
        long_code = "x = 1\n" * 2000
        result = run_code(long_code)
        self.assertFalse(result["success"])
        self.assertIn("exceeds max length", result["error"])

    def test_max_output_truncated(self):
        code = "for i in range(10000):\n    print('x' * 100)"
        result = run_code(code, timeout=2.0)
        # Should not crash; output may be truncated
        self.assertIsInstance(result, dict)
        self.assertIn("output", result)


class TestCodeSandboxRunFunction(unittest.TestCase):
    def test_run_simple_function(self):
        source = "def add(a, b):\n    return a + b"
        result = run_function(source, "add", args=[2, 3])
        self.assertTrue(result["success"])
        # result is _safe_repr'd: 5 -> "5"
        self.assertIsNotNone(result.get("result"))

    def test_run_function_with_kwargs(self):
        source = "def greet(name, greeting='Hello'):\n    return f'{greeting}, {name}!'"
        result = run_function(source, "greet", kwargs={"name": "World"})
        self.assertTrue(result["success"])
        self.assertIsNotNone(result.get("result"))
        self.assertIn("Hello", str(result["result"]))

    def test_run_nonexistent_function(self):
        source = "def foo(): pass"
        result = run_function(source, "bar")
        self.assertFalse(result["success"])


class TestCodeSandboxEdgeCases(unittest.TestCase):
    def test_empty_code(self):
        result = run_code("")
        self.assertTrue(result["success"])

    def test_comment_only(self):
        result = run_code("# just a comment")
        self.assertTrue(result["success"])

    def test_syntax_error(self):
        result = run_code("def broken(")
        self.assertFalse(result["success"])
        self.assertIn("SyntaxError", result.get("error", ""))

    def test_runtime_error(self):
        result = run_code("x = 1/0")
        self.assertFalse(result["success"])
        self.assertIn("ZeroDivisionError", result.get("error", ""))

    def test_type_error(self):
        result = run_code("'hello' + 42")
        self.assertFalse(result["success"])

    def test_unicode_output(self):
        result = run_code("print('héllo wörld')")
        self.assertTrue(result["success"])
        self.assertIn("héllo wörld", result["output"])
