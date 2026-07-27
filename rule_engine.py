"""
RuleEngine — gates LLM calls behind deterministic rule matching.

Pattern: Rule → Confidence ≥ threshold? → Return result / Fall through to Qwen.

Reduces Qwen invocations by handling known patterns (code generation,
error fixes, simple queries) with template-based rules first.
"""

import ast
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
debug_logger = logging.getLogger(f"{__name__}.debug")

RuleHandler = Callable[[str], Optional[Tuple[float, str]]]

_QA_ANSWERS = {
    "what is python": "Python is a high-level, general-purpose programming language known for readability and extensive standard library.",
    "what is a variable": "A variable is a named storage location in memory that holds a value which can change during program execution.",
    "what is a function": "A function is a reusable block of code that performs a specific task when called.",
    "what is a list": "A list is an ordered, mutable collection in Python that can hold items of different types.",
    "what is a dictionary": "A dictionary is a mapping type in Python that stores key-value pairs with O(1) average lookup.",
    "what is a set": "A set is an unordered collection of unique elements supporting fast membership testing.",
    "what is a tuple": "A tuple is an ordered, immutable collection in Python, typically used for heterogeneous data.",
    "what is recursion": "Recursion is a technique where a function calls itself to solve smaller instances of the same problem.",
    "what is oop": "OOP (Object-Oriented Programming) organizes code into objects combining data and behavior.",
    "what is an algorithm": "An algorithm is a step-by-step procedure for solving a problem or accomplishing a task.",
    "what is a database": "A database is an organized collection of structured data, typically stored electronically.",
    "what is an api": "An API (Application Programming Interface) defines how software components should interact.",
    "what is json": "JSON (JavaScript Object Notation) is a lightweight data-interchange format using key-value pairs.",
    "what is a decorator": "A decorator is a function that takes another function and extends its behavior without modifying it.",
    "what is a generator": "A generator is a function that yields values lazily using 'yield', producing an iterator.",
    "what is pip": "pip is the package installer for Python, used to install and manage third-party libraries.",
    "what is a lambda": "A lambda is an anonymous inline function defined with the 'lambda' keyword in Python.",
}

_CODE_TEMPLATES = {
    "sort": {
        "keywords": ["sort", "order", "ascending", "descending"],
        "code": "def {name}(items):\n    return sorted(items)\n",
    },
    "reverse": {
        "keywords": ["reverse", "backward", "flip"],
        "code": "def {name}(s):\n    return s[::-1]\n",
    },
    "palindrome": {
        "keywords": ["palindrome"],
        "code": "def {name}(s):\n    return s == s[::-1]\n",
    },
    "fibonacci": {
        "keywords": ["fibonacci", "fib", "sequence"],
        "code": "def {name}(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
    },
    "factorial": {
        "keywords": ["factorial", "factor"],
        "code": "def {name}(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n",
    },
    "prime": {
        "keywords": ["prime", "is_prime"],
        "code": "def {name}(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n",
    },
    "count_char": {
        "keywords": ["count", "character", "occurrence"],
        "code": "def {name}(s, c):\n    return s.count(c)\n",
    },
    "binary_search": {
        "keywords": ["binary search", "binary_search", "bisect"],
        "code": "def {name}(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n",
    },
    "bubble_sort": {
        "keywords": ["bubble sort", "bubble_sort"],
        "code": "def {name}(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n - i - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr\n",
    },
    "sum_list": {
        "keywords": ["sum", "total", "add all"],
        "code": "def {name}(items):\n    return sum(items)\n",
    },
    "max_value": {
        "keywords": ["maximum", "max", "largest"],
        "code": "def {name}(items):\n    return max(items)\n",
    },
    "fizzbuzz": {
        "keywords": ["fizzbuzz", "fizz buzz", "fizz_buzz"],
        "code": "def {name}(n):\n    result = []\n    for i in range(1, n + 1):\n        if i % 15 == 0:\n            result.append('FizzBuzz')\n        elif i % 3 == 0:\n            result.append('Fizz')\n        elif i % 5 == 0:\n            result.append('Buzz')\n        else:\n            result.append(str(i))\n    return result\n",
    },
    "even_odd": {
        "keywords": ["even", "odd", "even_odd", "parity"],
        "code": "def {name}(n):\n    if n % 2 == 0:\n        return 'even'\n    return 'odd'\n",
    },
    "anagram": {
        "keywords": ["anagram", "anagram_check"],
        "code": "def {name}(s1, s2):\n    return sorted(s1) == sorted(s2)\n",
    },
}

_PLACEHOLDER_NAME = "solution"


class RuleEngine:
    """Gate that tries deterministic rules before delegating to the LLM.

    Each rule is a (match_fn, handler_fn, threshold) triple.
    - match_fn(prompt) → confidence 0.0-1.0 (0 = no match)
    - handler_fn(prompt) → response string (or None to fall through)
    - threshold: minimum confidence to short-circuit the LLM
    """

    def __init__(self, llm_generator: Optional[Callable] = None, confidence_threshold: float = 0.65):
        self.llm = llm_generator
        self.threshold = confidence_threshold
        self._rules: List[Tuple[Callable[[str], float], Callable[[str], Optional[str]], str]] = []
        self._register_defaults()

    def register(self, match_fn: Callable[[str], float], handler_fn: Callable[[str], Optional[str]], name: str = ""):
        self._rules.append((match_fn, handler_fn, name or f"rule_{len(self._rules)}"))

    def decide(self, prompt: str, temperature: float = 0.4, max_tokens: int = 256) -> str:
        best_conf = 0.0
        best_result = None
        best_name = ""

        for match_fn, handler_fn, name in self._rules:
            conf = match_fn(prompt)
            if conf > best_conf:
                try:
                    result = handler_fn(prompt)
                    if result is not None:
                        best_conf = conf
                        best_result = result
                        best_name = name
                except Exception:
                    continue

        if best_conf >= self.threshold and best_result is not None:
            debug_logger.debug("RuleEngine: handled by '%s' (conf=%.2f) — LLM skipped", best_name, best_conf)
            return best_result

        if self.llm is not None:
            debug_logger.debug("RuleEngine: no rule matched (best=%.2f, '%s') — delegating to LLM", best_conf, best_name)
            return self.llm(prompt, temperature=temperature, max_tokens=max_tokens)

        return ""

    # ------------------------------------------------------------------
    #  Default rules
    # ------------------------------------------------------------------

    def _register_defaults(self):
        self._rules = [
            (self._match_error_reply, self._handle_error_reply, "error_reply"),
            (self._match_assert_test, self._handle_assert_test, "assert_test_template"),
            (self._match_code_template, self._handle_code_template, "code_template"),
            (self._match_qa, self._handle_qa, "qa_answer"),
            (self._match_math, self._handle_math, "math_eval"),
            (self._match_string_op, self._handle_string_op, "string_op"),
            (self._match_list_op, self._handle_list_op, "list_op"),
            (self._match_file_op, self._handle_file_op, "file_op"),
        ]

    @staticmethod
    def _match_error_reply(prompt: str) -> float:
        p = prompt.lower()
        if any(x in p for x in ["error:", "fix this", "the error is", "nameerror", "typeerror",
                                 "valueerror", "indexerror", "attributeerror", "zerodivisionerror",
                                 "syntaxerror", "indentationerror", "importerror", "modulenotfounderror"]):
            return 0.7
        return 0.0

    @staticmethod
    def _handle_error_reply(prompt: str) -> Optional[str]:
        error_patterns = {
            "nameerror": "The variable or function name is misspelled or not defined. Check the name for typos and ensure it has been assigned before use.",
            "typeerror": "There is a type mismatch. Ensure you are using the correct types for the operation (e.g., don't concatenate str + int).",
            "valueerror": "The value is not within the expected range or format. Verify the input constraints.",
            "indexerror": "The index is out of range. Check the list/sequence length before accessing elements.",
            "keyerror": "The dictionary key does not exist. Use .get() with a default or check with 'in' first.",
            "attributeerror": "The object does not have this attribute. Check the object type and attribute name spelling.",
            "zerodivisionerror": "Division by zero. Add a guard: if denominator != 0 before dividing.",
            "import error": "The module is not installed or not in the Python path. Install it or check the import path.",
            "modulenotfounderror": "The module is not installed. Run pip install <module-name>.",
            "syntaxerror": "There is a syntax error. Check for missing colons, parentheses, or quotes.",
            "indentationerror": "The indentation is inconsistent. Use consistent spaces (4 per level) throughout.",
            "filenotfounderror": "The file does not exist at the given path. Check the file path and current working directory.",
            "permissionerror": "No permission to access the file or resource. Check file permissions.",
            "timeouterror": "The operation timed out. The code may be too slow or stuck in an infinite loop.",
            "recursionerror": "Maximum recursion depth exceeded. Check for missing base case in recursion.",
            "stopiteration": "The iterator is exhausted. Wrap in a try/except or use a default.",
            "assertionerror": "An assertion failed. Check the condition being asserted.",
            "memoryerror": "Out of memory. Reduce data size or use streaming/incremental processing.",
            "oserror": "Operating system error. Check file paths and system resources.",
        }
        # Also detect error messages by natural language patterns
        natlang_patterns = {
            r"name\s+['\"]?\w+['\"]?\s+is not defined": "The variable or function name is misspelled or not defined. Check the name for typos and ensure it has been assigned before use.",
            r"cannot\s+(find|import)\s+module": "The module is not installed or not in the Python path. Install it or check the import path.",
            r"division\s+by\s+zero": "Division by zero. Add a guard: if denominator != 0 before dividing.",
            r"index\s+out\s+of\s+range": "The index is out of range. Check the list/sequence length before accessing elements.",
            r"list\s+index\s+out\s+of\s+range": "The index is out of range. Check the list/sequence length before accessing elements.",
            r"key\s+['\"]?\w+['\"]?\s+not\s+found": "The dictionary key does not exist. Use .get() with a default or check with 'in' first.",
            r"out\s+of\s+(memory|bounds)": "Out of memory. Reduce data size or use streaming/incremental processing.",
            r"maximum.*recursion": "Maximum recursion depth exceeded. Check for missing base case in recursion.",
            r"permission\s+denied": "No permission to access the file or resource. Check file permissions.",
            r"no\s+such\s+file|file.*not\s+found|cannot\s+find\s+path": "The file does not exist at the given path. Check the file path and current working directory.",
            r"invalid\s+syntax": "There is a syntax error. Check for missing colons, parentheses, or quotes.",
            r"unexpected\s+(indent|dedent)|inconsistent.*indent": "The indentation is inconsistent. Use consistent spaces (4 per level) throughout.",
            r"connection\s+(refused|reset|timed?\s*out)": "Connection error. Check that the server is running and reachable.",
        }
        p_lower = prompt.lower()
        for keyword, tip in error_patterns.items():
            if keyword in p_lower:
                return f"Fix: {tip}"
        for pattern, tip in natlang_patterns.items():
            if re.search(pattern, p_lower):
                return f"Fix: {tip}"
        return None

    @staticmethod
    def _match_assert_test(prompt: str) -> float:
        p = prompt.lower()
        if "assert" in p and ("test" in p or "check" in p or "verify" in p):
            return 0.75
        return 0.0

    @staticmethod
    def _handle_assert_test(prompt: str) -> Optional[str]:
        lines = prompt.strip().split("\n")
        test_lines = [l for l in lines if l.strip().startswith("assert ")]
        if test_lines:
            return "\n".join(test_lines)
        return None

    @staticmethod
    def _match_code_template(prompt: str) -> float:
        p = prompt.lower()
        for name, tmpl in _CODE_TEMPLATES.items():
            if any(kw in p for kw in tmpl["keywords"]):
                return 0.85
        # Stopwords like "function" alone don't trigger templates;
        # require code-intent keywords
        if any(x in p for x in ["def ", "write code", "generate", "create a", "implement"]):
            return 0.4
        return 0.0

    @staticmethod
    def _handle_code_template(prompt: str) -> Optional[str]:
        p = prompt.lower()
        best_name = None
        best_score = 0
        for name, tmpl in _CODE_TEMPLATES.items():
            score = sum(1 for kw in tmpl["keywords"] if kw in p)
            if score > best_score:
                best_score = score
                best_name = name
        if best_name is None or best_score == 0:
            return None
        func_name = _PLACEHOLDER_NAME
        match = re.search(r"\b(function|method|helper)\s+(\w+)", prompt.lower())
        if match:
            func_name = match.group(2)
        return _CODE_TEMPLATES[best_name]["code"].format(name=func_name)

    # ------------------------------------------------------------------
    #  Q&A rules
    # ------------------------------------------------------------------

    @staticmethod
    def _match_qa(prompt: str) -> float:
        p = prompt.lower().strip().rstrip("?")
        for question in _QA_ANSWERS:
            if question in p:
                return 0.9
        if p.startswith(("what is ", "what are ", "what does ", "define ", "explain ")):
            return 0.8  # Handled by fallback topic extraction
        return 0.0

    @staticmethod
    def _handle_qa(prompt: str) -> Optional[str]:
        p = prompt.lower().strip().rstrip("?").rstrip(".").rstrip("!")
        best_match = None
        best_len = 0
        for question, answer in _QA_ANSWERS.items():
            if question in p and len(question) > best_len:
                best_match = answer
                best_len = len(question)
        if best_match:
            return best_match
        # Fallback for "define X" / "explain X" — pull the key noun
        match = re.search(r"(?:define|explain|what\s+is|what\s+are)\s+(a\s+|an\s+)?(\w+(?:\s+\w+){0,3})", p)
        if match:
            topic = match.group(2)
            # Try full match first, then progressively shorter suffixes
            words = topic.split()
            for end in range(len(words), 0, -1):
                sub = " ".join(words[:end])
                for question, answer in _QA_ANSWERS.items():
                    if sub in question:
                        return answer
        return None

    # ------------------------------------------------------------------
    #  Math evaluation rules
    # ------------------------------------------------------------------

    @staticmethod
    def _match_math(prompt: str) -> float:
        p = prompt.lower().strip()
        triggers = ["calculate", "what is", "compute", "evaluate", "solve", "="]
        if any(p.startswith(t) or f" {t} " in p for t in triggers):
            if re.search(r"[\d+\-*/()%=]", p):
                return 0.85
        if re.search(r"^\d+\s*[\+\-\*/]\s*\d+", p):
            return 0.7
        return 0.0

    @staticmethod
    def _handle_math(prompt: str) -> Optional[str]:
        import ast
        p = prompt.lower().strip()
        # Remove common prefixes to isolate the expression
        for prefix in ["calculate", "what is", "compute", "evaluate", "solve"]:
            if p.startswith(prefix):
                p = p[len(prefix):].strip()
                break
        p = p.strip("= ").strip()
        # Find the first evaluable expression (handles parens, **, etc.)
        expr_candidates = re.findall(r"\(?[-]?\d+(?:\s*[\+\-\*/]\s*\d+(?:\.\d+)?)+\)?(?:\s*[\+\-\*/]\s*\(?[-]?\d+(?:\s*[\+\-\*/]\s*\d+)*\)?)*", p)
        if not expr_candidates:
            # Try ** (power)
            expr_candidates = re.findall(r"\d+\s*\*\*\s*\d+", p)
        if not expr_candidates:
            # Try pow( 
            pow_match = re.search(r"pow\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", p)
            if pow_match:
                expr_candidates = [f"{pow_match.group(1)}**{pow_match.group(2)}"]
        for expr in expr_candidates:
            expr = expr.replace(" ", "")
            try:
                tree = ast.parse(expr, mode="eval")
                allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                           ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
                           ast.Pow, ast.Mod, ast.USub, ast.UAdd)
                for node in ast.walk(tree):
                    if not isinstance(node, allowed):
                        if not isinstance(node, ast.Expression):
                            return None
                result = eval(expr, {"__builtins__": {}}, {})
                return str(result)
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------
    #  String operation rules
    # ------------------------------------------------------------------

    _STRING_OPS = {
        "upper": str.upper,
        "uppercase": str.upper,
        "lower": str.lower,
        "lowercase": str.lower,
        "capitalize": str.capitalize,
        "title case": str.title,
        "titlecase": str.title,
        "reverse": lambda s: s[::-1],
        "strip": str.strip,
        "trim": str.strip,
    }

    @classmethod
    def _match_string_op(cls, prompt: str) -> float:
        p = prompt.lower().strip()
        if any(op in p for op in cls._STRING_OPS):
            if re.search(r"['\"].+['\"]", p):
                return 0.95  # Higher than code_template (0.85) to win ties
        return 0.0

    @classmethod
    def _handle_string_op(cls, prompt: str) -> Optional[str]:
        p = prompt.lower().strip()
        op_match = re.search(r"(upper|uppercase|lower|lowercase|capitalize|title\s*case|titlecase|reverse|strip|trim)", p)
        if not op_match:
            return None
        op_name = op_match.group(1).replace(" ", "_").replace("title_case", "titlecase")
        str_match = re.search(r"['\"]([^'\"]+)['\"]", prompt)
        if not str_match:
            return None
        s = str_match.group(1)
        op_fn = cls._STRING_OPS.get(op_name)
        if op_fn:
            try:
                return op_fn(s)
            except Exception:
                return None
        return None

    # ------------------------------------------------------------------
    #  List operation rules
    # ------------------------------------------------------------------

    _LIST_OPERATIONS = {
        "filter": {"keywords": ["filter", "where", "only", "keep"], "template": "filter"},
        "map": {"keywords": ["map", "transform", "convert each", "square", "double", "negate"], "template": "map"},
        "sum": {"keywords": ["sum", "total", "add all"], "template": "sum"},
        "sort": {"keywords": ["sort", "order", "sorted"], "template": "sort"},
        "reverse": {"keywords": ["reverse", "backward", "flip"], "template": "reverse"},
    }

    @classmethod
    def _match_list_op(cls, prompt: str) -> float:
        p = prompt.lower().strip()
        if not re.search(r"\[.*\]", p):
            return 0.0
        for op_name, op_info in cls._LIST_OPERATIONS.items():
            if any(kw in p for kw in op_info["keywords"]):
                return 0.95  # Higher than code_template (0.85) to win ties
        return 0.0

    @classmethod
    def _handle_list_op(cls, prompt: str) -> Optional[str]:
        import ast
        p = prompt.lower().strip()
        list_match = re.search(r"\[([^\]]+)\]", prompt)
        if not list_match:
            return None
        try:
            items = ast.literal_eval(f"[{list_match.group(1)}]")
            if not isinstance(items, list):
                return None
        except Exception:
            return None

        op_name = None
        for name, info in cls._LIST_OPERATIONS.items():
            if any(kw in p for kw in info["keywords"]):
                op_name = name
                break
        if op_name is None:
            return None

        if op_name == "sum":
            return str(sum(items))
        elif op_name == "sort":
            return str(sorted(items))
        elif op_name == "reverse":
            return str(list(reversed(items)))
        elif op_name == "filter":
            if "even" in p or "divisible by 2" in p:
                return str([x for x in items if isinstance(x, (int, float)) and x % 2 == 0])
            elif "odd" in p:
                return str([x for x in items if isinstance(x, (int, float)) and x % 2 == 1])
            elif "positive" in p or "> 0" in p or "greater than 0" in p:
                return str([x for x in items if isinstance(x, (int, float)) and x > 0])
            elif "negative" in p or "< 0" in p or "less than 0" in p:
                return str([x for x in items if isinstance(x, (int, float)) and x < 0])
            elif "string" in p or "str" in p:
                return str([x for x in items if isinstance(x, str)])
            elif "number" in p or "int" in p or "float" in p:
                return str([x for x in items if isinstance(x, (int, float))])
            else:
                return str(items)
        elif op_name == "map":
            if "square" in p:
                return str([x * x if isinstance(x, (int, float)) else x for x in items])
            elif "double" in p:
                return str([x * 2 if isinstance(x, (int, float)) else x for x in items])
            elif "negate" in p:
                return str([-x if isinstance(x, (int, float)) else x for x in items])
            else:
                return str(items)
        return None

    # ------------------------------------------------------------------
    #  File operation rules
    # ------------------------------------------------------------------

    @staticmethod
    def _match_file_op(prompt: str) -> float:
        p = prompt.lower().strip()
        if any(x in p for x in ["read file", "write file", "list files", "open file",
                                 "create file", "delete file", "read from", "write to",
                                 "ls ", "directory"]):
            if re.search(r"['\"][^'\"]+['\"]", p) or "list files" in p or "list directory" in p:
                return 0.7
        return 0.0

    @staticmethod
    def _handle_file_op(prompt: str) -> Optional[str]:
        p = prompt.lower().strip()
        path_match = re.search(r"['\"]([^'\"]+\.\w+)['\"]", prompt)
        path = path_match.group(1) if path_match else None
        if "list files" in p or "list directory" in p or "ls " in p:
            path = path_match.group(1) if path_match else "."
            try:
                import os
                files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
                if files:
                    return "\n".join(sorted(files))
                return "(empty directory)"
            except Exception as e:
                return f"Error listing directory: {e}"
        if path and ("read file" in p or "open file" in p or "read from" in p):
            try:
                import os
                if not os.path.isfile(path):
                    return f"File not found: {path}"
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read(2000)
                return content
            except Exception as e:
                return f"Error reading file: {e}"
        if path and ("write file" in p or "write to" in p or "create file" in p):
            return "File writing not supported through RuleEngine. Use a dedicated tool."
        return None
