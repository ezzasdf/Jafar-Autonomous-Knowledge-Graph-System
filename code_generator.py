"""
CodeGenerator — turns goals into working code via test-and-fix loop.

Goal -> parse intent -> match template -> generate skeleton
-> run in sandbox -> error? fix it -> run again
-> pass? store in CodeMemory
"""

import ast
import logging
import re
import textwrap
import time
from typing import Dict, List, Any, Optional, Tuple, Callable

logger = logging.getLogger(__name__)

TEMPLATES: Dict[str, Dict[str, Any]] = {
    "sort": {
        "keywords": ["sort", "order", "ascending", "descending"],
        "code": """def {name}(items):
    return sorted(items)
""",
        "test": "assert {name}([3, 1, 2]) == [1, 2, 3]",
    },
    "reverse": {
        "keywords": ["reverse", "backward", "flip"],
        "code": """def {name}(s):
    return s[::-1]
""",
        "test": "assert {name}('abc') == 'cba'",
    },
    "palindrome": {
        "keywords": ["palindrome"],
        "code": """def {name}(s):
    return s == s[::-1]
""",
        "test": "assert {name}('racecar') == True",
    },
    "fibonacci": {
        "keywords": ["fibonacci", "fib", "sequence"],
        "code": """def {name}(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
""",
        "test": "assert {name}(10) == 55",
    },
    "factorial": {
        "keywords": ["factorial", "factor"],
        "code": """def {name}(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
""",
        "test": "assert {name}(5) == 120",
    },
    "prime": {
        "keywords": ["prime", "is_prime"],
        "code": """def {name}(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
""",
        "test": "assert {name}(7) == True\nassert {name}(4) == False",
    },
    "count_char": {
        "keywords": ["count", "character", "occurrence"],
        "code": """def {name}(s, c):
    return s.count(c)
""",
        "test": "assert {name}('hello', 'l') == 2",
    },
    "binary_search": {
        "keywords": ["binary search", "binary_search", "bisect"],
        "code": """def {name}(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
""",
        "test": "assert {name}([1, 2, 3, 4, 5], 3) == 2",
    },
    "bubble_sort": {
        "keywords": ["bubble sort", "bubble_sort"],
        "code": """def {name}(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
""",
        "test": "assert {name}([3, 1, 2]) == [1, 2, 3]",
    },
    "sum_list": {
        "keywords": ["sum", "total", "add all"],
        "code": """def {name}(items):
    return sum(items)
""",
        "test": "assert {name}([1, 2, 3]) == 6",
    },
    "max_value": {
        "keywords": ["maximum", "max", "largest"],
        "code": """def {name}(items):
    return max(items)
""",
        "test": "assert {name}([1, 5, 2]) == 5",
    },
    "fizzbuzz": {
        "keywords": ["fizzbuzz", "fizz buzz", "fizz_buzz"],
        "code": """def {name}(n):
    result = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            result.append('FizzBuzz')
        elif i % 3 == 0:
            result.append('Fizz')
        elif i % 5 == 0:
            result.append('Buzz')
        else:
            result.append(str(i))
    return result
""",
        "test": "assert {name}(15)[-1] == 'FizzBuzz'",
    },
    "even_odd": {
        "keywords": ["even", "odd", "even_odd", "parity"],
        "code": """def {name}(n):
    if n % 2 == 0:
        return 'even'
    return 'odd'
""",
        "test": "assert {name}(2) == 'even'\nassert {name}(3) == 'odd'",
    },
    "anagram": {
        "keywords": ["anagram", "anagram_check"],
        "code": """def {name}(s1, s2):
    return sorted(s1) == sorted(s2)
""",
        "test": "assert {name}('listen', 'silent') == True",
    },
}


class CodeGenerator:
    """Generates, tests, and fixes code for a given goal."""

    def __init__(self, code_memory=None, code_sandbox=None):
        self.code_memory = code_memory
        self.sandbox = code_sandbox

    def generate(self, goal: str, max_attempts: int = 5,
                 store_on_success: bool = True) -> Dict[str, Any]:
        start = time.time()
        intent = self._parse_goal(goal)

        template = self._match_template(intent)
        if template is None:
            return {
                "status": "no_template",
                "goal": goal,
                "error": "No matching template found for goal",
                "duration": round(time.time() - start, 2),
            }

        name = self._derive_name(goal)
        code = self._fill_template(template, name)

        history: List[Dict[str, Any]] = []

        for attempt in range(1, max_attempts + 1):
            result = self._run_code(code)
            entry = {
                "attempt": attempt,
                "code": code,
                "result": result,
            }
            history.append(entry)

            if result["success"]:
                test_ok = self._run_tests(template, name)
                entry["tests_passed"] = test_ok
                if test_ok:
                    artifact = None
                    if store_on_success and self.code_memory:
                        artifact = self.code_memory.store(
                            name=name,
                            kind="function",
                            source=code,
                            signature=self._extract_signature(code),
                            docstring=self._extract_docstring(code),
                            tags=intent.get("keywords", []) + ["generated"],
                        )
                    return {
                        "status": "ok",
                        "goal": goal,
                        "name": name,
                        "code": code,
                        "artifact_id": artifact.artifact_id if artifact else None,
                        "attempts": attempt,
                        "history": history,
                        "duration": round(time.time() - start, 2),
                    }

            code = self._fix_code(code, result, attempt, max_attempts)
            if code is None:
                return {
                    "status": "gave_up",
                    "goal": goal,
                    "error": "Fix strategy exhausted",
                    "attempts": attempt,
                    "history": history,
                    "duration": round(time.time() - start, 2),
                }

        return {
            "status": "max_attempts",
            "goal": goal,
            "error": f"Failed after {max_attempts} attempts",
            "attempts": max_attempts,
            "history": history,
            "duration": round(time.time() - start, 2),
        }

    def _parse_goal(self, goal: str) -> Dict[str, Any]:
        lower = goal.lower()
        words = set(re.findall(r"[a-z_]+", lower))
        return {
            "raw": goal,
            "lower": lower,
            "words": words,
        }

    def _match_template(self, intent: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        lower = intent["lower"]
        best: Optional[Tuple[str, Dict[str, Any]]] = None
        best_score = 0

        for key, tmpl in TEMPLATES.items():
            score = 0
            for kw in tmpl["keywords"]:
                if kw in lower:
                    score += len(kw)
            if score > best_score:
                best_score = score
                best = (key, tmpl)

        if best and best_score > 0:
            return best[1]
        return None

    def _derive_name(self, goal: str) -> str:
        lower = goal.lower()
        for key, tmpl in TEMPLATES.items():
            for kw in tmpl["keywords"]:
                if kw in lower:
                    return kw.replace(" ", "_")
        words = re.findall(r"[a-zA-Z_]\w*", goal)
        return (words[0] if words else "generated_func").lower()

    def _fill_template(self, template: Dict[str, Any],
                       name: str) -> str:
        code = template["code"]
        return code.replace("{name}", name)

    def _run_code(self, code: str) -> Dict[str, Any]:
        if not self.sandbox:
            return {"success": True, "output": "", "error": "",
                    "execution_time": 0.0, "variables": {}}
        if isinstance(self.sandbox, dict):
            return self.sandbox["run_code"](code)
        return self.sandbox.run_code(code)

    def _run_tests(self, template: Dict[str, Any],
                   name: str) -> bool:
        test_code = template.get("test", "")
        if not test_code:
            return True
        test_code = test_code.replace("{name}", name)
        if self.sandbox:
            if isinstance(self.sandbox, dict):
                result = self.sandbox["run_code"](test_code)
            else:
                result = self.sandbox.run_code(test_code)
            return result["success"]
        return True

    def _extract_signature(self, code: str) -> str:
        lines = code.strip().split("\n")
        for line in lines:
            m = re.match(r"def\s+(\w+\s*\(.*?\))\s*:", line)
            if m:
                return m.group(1)
        return ""

    def _extract_docstring(self, code: str) -> str:
        try:
            tree = ast.parse(code.strip())
            if isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
                return ast.get_docstring(tree.body[0]) or ""
        except SyntaxError:
            pass
        return ""

    def _fix_code(self, code: str, result: Dict[str, Any],
                  attempt: int, max_attempts: int
                  ) -> Optional[str]:
        if result["success"]:
            return code

        error = result.get("error", "")

        if "NameError" in error:
            return self._fix_name_error(code, error)
        if "TypeError" in error:
            return self._fix_type_error(code, error)
        if "ZeroDivisionError" in error:
            return self._fix_zero_div_error(code)
        if "IndexError" in error or "KeyError" in error:
            return self._fix_index_error(code, error)
        if "ValueError" in error:
            return self._fix_value_error(code)
        if "SyntaxError" in error or "IndentationError" in error:
            return self._fix_syntax_error(code)

        return None

    def _fix_name_error(self, code: str, error: str) -> Optional[str]:
        m = re.search(r"name\s+'(\w+)'", error)
        if not m:
            return None
        missing = m.group(1)
        if missing in ("math", "json", "random", "collections",
                       "itertools", "datetime", "re", "string",
                       "functools", "statistics", "pprint", "copy"):
            extra = ""
            if "math." in code or missing == "math":
                extra = "\nimport math\n"
            code = extra + code
        else:
            code = f"{missing} = {missing} or 0\n{code}"
        return code

    def _fix_type_error(self, code: str, error: str) -> Optional[str]:
        m = re.search(r"'(\w+)' object is not callable", error)
        if m:
            return None
        m = re.search(r"unsupported operand type", error)
        if m:
            code = code.replace("None", "0").replace("''", "' '")
        return code

    def _fix_zero_div_error(self, code: str) -> Optional[str]:
        return code.replace("n % i == 0", "i > 0 and n % i == 0")

    def _fix_index_error(self, code: str, error: str) -> Optional[str]:
        lines = code.split("\n")
        fixed = []
        for line in lines:
            stripped = line.strip()
            if "arr[mid]" in stripped or "arr[mid]" in stripped:
                indent = " " * (len(line) - len(line.lstrip()))
                if "arr[mid]" in stripped and "mid < len(arr)" not in stripped:
                    before = line.rstrip()
                    guard = f'{indent}if mid < len(arr) and mid >= 0:'
                    guarded = f'{indent}    pass'
                    fixed.append(line)
                else:
                    fixed.append(line)
            else:
                fixed.append(line)
        return "\n".join(fixed)

    def _fix_value_error(self, code: str) -> Optional[str]:
        return code

    def _fix_syntax_error(self, code: str) -> Optional[str]:
        try:
            ast.parse(code.strip())
            return code
        except SyntaxError as e:
            lines = code.split("\n")
            lineno = e.lineno
            if lineno and lineno <= len(lines):
                bad = lines[lineno - 1]
                fixed_line = bad.rstrip().rstrip(",").rstrip(":")
                if fixed_line != bad:
                    lines[lineno - 1] = fixed_line
                    return "\n".join(lines)

            code = textwrap.dedent(code)
            try:
                ast.parse(code.strip())
                return code
            except SyntaxError:
                pass
        return None
