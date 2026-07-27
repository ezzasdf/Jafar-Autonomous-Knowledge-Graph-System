"""
Jafar Evolution Engine — mathematical discovery via evolutionary mutation.

Pipeline:
  1. Seed: pull known math/algorithm code from the knowledge graph
  2. Mutate: Qwen-2.5-7B purposefully breaks/mutates the equation
  3. Evaluate: sandbox tests mutation against thousands of numeric inputs
  4. Promote: if valid and faster, commit the discovery to the graph
"""

import json
import logging
import random
import time
import traceback
from typing import Dict, List, Any, Optional, Tuple

from tqdm import tqdm
from collections import deque

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
#  Rich progress helpers
# ------------------------------------------------------------------

class MutationProgress:
    """Tracks and renders live mutation progress with stats."""

    def __init__(self):
        self.seed_name: str = ""
        self.iteration: int = 0
        self.total_iterations: int = 0
        self.latest_pass_rate: float = 0.0
        self.latest_speedup: float = 0.0
        self.cumulative_attempted: int = 0
        self.cumulative_passed: int = 0
        self.cumulative_discoveries: int = 0
        self.baseline_time_ms: float = 0.0
        self._history: deque = deque(maxlen=20)

    def update(self, iteration: int, attempted: int, passed: int,
               pass_rate: float = 0.0, speedup: float = 0.0,
               discoveries: int = 0):
        self.iteration = iteration
        self.cumulative_attempted = attempted
        self.cumulative_passed = passed
        self.latest_pass_rate = pass_rate
        self.latest_speedup = speedup
        self.cumulative_discoveries = discoveries
        self._history.append({
            "i": iteration, "pr": pass_rate, "sp": speedup,
        })

    def format_desc(self) -> str:
        pr = f"{self.latest_pass_rate:.2f}" if self.latest_pass_rate else "?"
        sp = f"{self.latest_speedup:.2f}x" if self.latest_speedup else "?"
        return (
            f"{self.seed_name} | "
            f"pass {pr} speedup {sp} | "
            f"{self.cumulative_passed}/{self.cumulative_attempted} ok | "
            f"{self.cumulative_discoveries} discoveries"
        )

MUTATION_TEMPLATE = """You are a mathematical code optimizer. Rewrite the given Python function using exactly ONE of these strategies.

Pick the strategy best suited to the code. Study the function carefully before choosing.

STRATEGIES:

1. LOOP UNROLLING (for loops with fixed/small iteration count):
   - If the loop body is simple and iteration count is known (e.g. range(4), range(8)), fully unroll: write each iteration inline.
   - If count is moderate (e.g. range(16), range(32)), partially unroll by factor 4 or 8: batch iterations together, keep a remainder loop.
   - Replace `for i in range(n): result += arr[i]` with `for i in range(0, n, 4): result += arr[i] + arr[i+1] + arr[i+2] + arr[i+3]` plus a tail loop.

2. CONSTANT FOLDING & HOISTING (for loop-invariant computation):
   - Move sub-expressions that don't change across loop iterations OUTSIDE the loop.
   - Pre-compute `limit = int(n**0.5) + 1` before the loop instead of recomputing `int(n**0.5) + 1` each iteration.
   - If a divisor, multiplier, or threshold is loop-invariant, compute it once.

3. ALGEBRAIC SUBSTITUTION (replace expensive ops with cheap identities):
   - `x ** 2` → `x * x`
   - `x ** 3` → `x * x * x`
   - `x / 2` → `x * 0.5` (multiply by reciprocal)
   - `x / y` → `x * (1.0 / y)` if y is invariant
   - `sqrt(x)` → `x ** 0.5` (or keep if stdlib is fast)
   - `math.exp(x)` → use `pow(E, x)` or keep as-is
   - Replace `pow(x, y)` with `x**y` if exponent is small integer

4. STRENGTH REDUCTION (replace expensive ops with cheaper ones in loops):
   - Replace `i * k` inside a loop with an accumulator: `val = 0; for i: val += k`
   - Replace `i ** 2` inside a loop with additive odd-number accumulation (n^2 = sum of first n odds).
   - Replace repeated `len(arr)` calls with a local variable.

5. REDUNDANCY ELIMINATION (cache repeated calculations):
   - If the same sub-expression appears multiple times (e.g., `(a + b) * (a + b)`), compute once: `s = a + b; result = s * s`.
   - If a function result is used multiple times, store in local variable.
   - If a list/dict lookup is repeated, store reference locally.

6. GUARD CLUSTERING & BRANCHLESS (replace branches with arithmetic):
   - Replace `if a < low: a = low; if a > high: a = high` with `a = max(low, min(high, a))`.
   - Replace `if x >= 0: return x; else: return -x` with `return x if x >= 0 else -x` (or `abs(x)`).
   - Replace multiple `if x in {a, b, c}` with a bitmask range check when possible.
   - Use `(cond) * val_true + (not cond) * val_false` for simple numeric branches.

7. MEMOIZATION (for recursive or repeated-call functions):
   - Add `functools.lru_cache(maxsize=None)` decorator.
   - For non-hashable args, create a manual dict-based memo table.
   - For DP-like recursion, switch to bottom-up iterative with a table.

8. PREFETCH & DATA-ORIENTED (for loops over collections):
   - If iterating over a list of dicts/objects, extract fields into parallel lists first.
   - Restructure nested loops: loop over the outermost invariant first (loop interchange).
   - Use `map()` / list comprehension instead of manual `for` with `append`.

9. BIT MANIPULATION (for integer-heavy code):
   - Replace `n % 2 == 0` with `n & 1 == 0`.
   - Replace `n * 2` with `n << 1`.
   - Replace `n // 2` with `n >> 1`.
   - Replace `n % (2**k)` with `n & (2**k - 1)`.
   - Replace `is_power_of_two`: `n > 0 and (n & (n - 1)) == 0`.

Rules:
- Apply exactly ONE strategy per rewrite
- The function must remain mathematically equivalent for ALL valid inputs
- Preserve the exact function name and signature
- Output ONLY the rewritten Python code between ```python markers
- NO explanation, NO commentary outside the code block

Original function:
```python
{code}
```

Rewritten function (applying one optimization strategy):
```python"""

DISCOVERY_SOURCE = "jafar_synthetic_discovery"


class EvolutionEngine:
    def __init__(
        self,
        memory_system: Any,
        transformer_engine: Any,
        code_sandbox: Dict[str, Any],
        truth_system: Any,
        population_size: int = 10,
        mutation_temperature: float = 0.9,
        sandbox_timeout: float = 5.0,
        test_case_count: int = 1000,
    ):
        self.ms = memory_system
        self.tre = transformer_engine
        self.sandbox = code_sandbox
        self.ts = truth_system
        self.population_size = population_size
        self.mutation_temperature = mutation_temperature
        self.sandbox_timeout = sandbox_timeout
        self.test_case_count = test_case_count

    def run_evolution_cycle(
        self,
        population_size: Optional[int] = None,
        test_case_count: Optional[int] = None,
        iterations: int = 5,
        use_random_mutator: bool = False,
    ) -> Dict[str, Any]:
        pop = population_size or self.population_size
        tcc = test_case_count or self.test_case_count

        report = {
            "seeds_found": 0,
            "mutations_attempted": 0,
            "mutations_passed": 0,
            "discoveries": [],
            "errors": [],
            "elapsed_seconds": 0.0,
            "mutator": "random" if use_random_mutator else "qwen",
        }
        t_start = time.time()

        seeds = self._get_algorithm_seeds()
        if not seeds:
            report["errors"].append("No algorithm seeds found in knowledge graph")
            report["elapsed_seconds"] = round(time.time() - t_start, 2)
            return report
        report["seeds_found"] = len(seeds)

        seed_list = seeds[:max(1, min(pop, 5))]
        mprog = MutationProgress()

        seed_pbar = tqdm(seed_list, desc="Seeds", unit="seed", leave=True)
        for seed in seed_pbar:
            seed_code = seed.get("code", "") or self._code_from_triples(seed)
            if not seed_code:
                continue
            seed_name = seed.get("name", seed.get("concept", "unknown"))
            mprog.seed_name = seed_name
            mprog.total_iterations = iterations

            seed_pbar.set_description(f"Evolving: {seed_name}")

            test_cases = self._generate_test_cases(seed_code, seed_name, count=tcc)
            if not test_cases:
                tqdm.write(f"  No valid test cases for {seed_name} — skipping")
                continue

            baseline = self._benchmark(seed_code, test_cases)
            if not baseline.get("success"):
                tqdm.write(f"  Baseline failed for {seed_name} — skipping")
                continue
            baseline_time = baseline.get("avg_time_ms", float("inf"))
            mprog.baseline_time_ms = baseline_time

            mutate_fn = self._random_mutate_code if use_random_mutator else self._mutate_code

            mut_pbar = tqdm(range(iterations), desc=f"  Mutations", unit="mut",
                            leave=False, colour="cyan")
            for i in mut_pbar:
                mut_pbar.set_description(mprog.format_desc())
                mutated = mutate_fn(seed_code, seed_name)
                if not mutated:
                    mprog.update(i + 1, mprog.cumulative_attempted, mprog.cumulative_passed,
                                 discoveries=len(report["discoveries"]))
                    mut_pbar.set_description(mprog.format_desc())
                    continue
                report["mutations_attempted"] += 1

                eval_result = self._evaluate_mutation(mutated, test_cases, seed_code)
                if not eval_result.get("success"):
                    mprog.update(i + 1, report["mutations_attempted"],
                                 report["mutations_passed"],
                                 discoveries=len(report["discoveries"]))
                    mut_pbar.set_description(mprog.format_desc())
                    continue

                report["mutations_passed"] += 1
                speedup = baseline_time / max(eval_result.get("avg_time_ms", 1e-9), 1e-9) if baseline_time > 0 else 1.0
                pass_rate = eval_result.get("pass_rate", 0)

                mprog.update(
                    i + 1, report["mutations_attempted"],
                    report["mutations_passed"],
                    pass_rate=pass_rate, speedup=speedup,
                    discoveries=len(report["discoveries"]),
                )
                mut_pbar.set_description(mprog.format_desc())

                is_discovery = (
                    eval_result["pass_rate"] >= 0.99
                    and speedup > 1.05
                )

                if is_discovery:
                    self._promote_discovery(
                        seed_name=seed_name,
                        original_code=seed_code,
                        mutated_code=mutated,
                        pass_rate=eval_result["pass_rate"],
                        speedup=speedup,
                        avg_time_ms=eval_result.get("avg_time_ms", 0),
                        baseline_time_ms=baseline_time,
                    )
                    report["discoveries"].append({
                        "seed": seed_name,
                        "speedup": round(speedup, 3),
                        "pass_rate": round(eval_result["pass_rate"], 4),
                        "iteration": i + 1,
                    })
                    mut_pbar.set_postfix(DISCOVERY=f"{speedup:.2f}x", refresh=True)
                    logger.info(
                        "DISCOVERY: %s — speedup %.3fx (%.2fms vs %.2fms)",
                        seed_name, speedup,
                        eval_result.get("avg_time_ms", 0), baseline_time,
                    )
            mut_pbar.close()

        report["elapsed_seconds"] = round(time.time() - t_start, 2)
        return report

    def _get_algorithm_seeds(self) -> List[Dict[str, Any]]:
        seeds = []
        seen_concepts = set()

        # Phase 0: book-backed seeds — query knowledge graph for concepts
        # that were derived from actual book processing (highest quality).
        book_terms = [
            "algorithm", "function", "computation", "matrix", "vector",
            "sort", "search", "optimization", "regression", "gradient",
            "fibonacci", "prime", "factorial", "convolution", "transform",
            "probability", "statistics", "linear algebra", "calculus",
        ]
        for term in book_terms:
            try:
                graph = self.ms.get_concept_graph(term)
                rels = graph.get("relationships", [])
                if not rels:
                    continue
                code = self._try_extract_code_from_rels(rels)
                source_types = {r.get("source_type", "") for r in rels}
                has_book_source = any(
                    st in ("book_learning", "book", "book_source", "learning")
                    for st in source_types
                )
                if code and has_book_source:
                    name = term
                    if name not in seen_concepts:
                        seen_concepts.add(name)
                        seeds.append({
                            "name": name, "concept": name,
                            "code": code, "relationships": rels,
                            "source_quality": "book",
                            "book_backed": True,
                        })
            except Exception:
                continue

        # Phase 1: query knowledge graph for math/algorithm concepts by name
        math_concepts = [
            "matrix multiplication", "matrix", "fibonacci", "sort",
            "linear algebra", "polynomial", "prime", "factorial",
            "convolution", "transform", "gradient descent",
            "differential equation", "integration", "derivative",
            "eigenvalue", "vector", "dot product", "norm",
            "interpolation", "optimization", "regression",
            "probability distribution", "entropy", "fourier",
        ]
        for concept in math_concepts:
            if concept in seen_concepts:
                continue
            try:
                graph = self.ms.get_concept_graph(concept)
                rels = graph.get("relationships", [])
                if rels:
                    code = self._try_extract_code_from_rels(rels)
                    if code:
                        seen_concepts.add(concept)
                        seeds.append({
                            "name": concept, "concept": concept,
                            "code": code, "relationships": rels,
                            "source_quality": "graph",
                            "book_backed": False,
                        })
            except Exception:
                continue

        # Phase 1b: search for algorithm-like concepts in the KG
        try:
            existing = self.ms.search_concepts("algorithm", limit=20)
            for c in existing:
                cname = c.get("name", "")
                if cname in seen_concepts:
                    continue
                graph = self.ms.get_concept_graph(cname)
                rels = graph.get("relationships", [])
                code = self._try_extract_code_from_rels(rels)
                if code:
                    seen_concepts.add(cname)
                    seeds.append({
                        "name": cname, "concept": cname,
                        "code": code, "relationships": rels,
                        "source_quality": "graph",
                        "book_backed": False,
                    })
        except Exception:
            pass

        # Phase 2: scan graph for any relationship containing code-like context
        if len(seeds) < 3:
            for broad_term in ["function", "compute", "calculate", "evaluate",
                               "method", "procedure", "operation"]:
                if len(seeds) >= 10:
                    break
                try:
                    hits = self.ms.search_concepts(broad_term, limit=10)
                    for c in hits:
                        cname = c.get("name", "")
                        if cname in seen_concepts:
                            continue
                        graph = self.ms.get_concept_graph(cname)
                        rels = graph.get("relationships", [])
                        code = self._try_extract_code_from_rels(rels)
                        if code:
                            seen_concepts.add(cname)
                            seeds.append({
                                "name": cname, "concept": cname,
                                "code": code, "relationships": rels,
                                "source_quality": "graph_fallback",
                                "book_backed": False,
                            })
                except Exception:
                    continue

        # Phase 3: scan book-source relationships for code snippets
        if len(seeds) < 2:
            for src_type in ["book_learning", "book", "book_source"]:
                try:
                    rels = self.ms.get_relationships_by_source_type(src_type, limit=100)
                    code = self._try_extract_code_from_rels(rels)
                    if code:
                        key = f"extracted_{src_type}"
                        if key not in seen_concepts:
                            seen_concepts.add(key)
                            seeds.append({
                                "name": key, "concept": key,
                                "code": code, "relationships": rels,
                                "source_quality": "book_raw",
                                "book_backed": True,
                            })
                except Exception:
                    continue

        # Sort: book-backed first, then graph, then fallback
        quality_order = {"book": 0, "graph": 1, "graph_fallback": 2, "book_raw": 3}
        seeds.sort(key=lambda s: quality_order.get(s.get("source_quality", ""), 99))

        if not seeds:
            seeds = self._fallback_seeds()
        return seeds

    def _fallback_seeds(self) -> List[Dict[str, Any]]:
        fallbacks = [
            ("fibonacci", "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a"),
            ("is_prime", "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0: return False\n    return True"),
            ("factorial", "def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result"),
            ("dot_product", "def dot_product(a, b):\n    return sum(x * y for x, y in zip(a, b))"),
            ("matrix_multiply", "def matrix_multiply(A, B):\n    n = len(A)\n    C = [[0]*n for _ in range(n)]\n    for i in range(n):\n        for k in range(n):\n            aik = A[i][k]\n            for j in range(n):\n                C[i][j] += aik * B[k][j]\n    return C"),
        ]
        return [
            {"name": name, "concept": name, "code": code, "relationships": []}
            for name, code in fallbacks
        ]

    def _try_extract_code_from_rels(self, rels: List[Dict[str, Any]]) -> str:
        for r in rels:
            if r.get("context") and ("def " in r["context"] or "lambda" in r["context"]):
                return r["context"]
        return ""

    def _code_from_triples(self, seed: Dict[str, Any]) -> str:
        name = seed.get("name", "function")
        rels = seed.get("relationships", [])
        parts = [f"def {name.replace(' ', '_')}(x):"]
        for r in rels[:5]:
            rel = r.get("relation", "related_to")
            target = r.get("target", "?")
            parts.append(f"    # --[{rel}]--> {target}")
        parts.append("    return x")
        return "\n".join(parts)

    def _generate_test_cases(self, code: str, func_name: str, count: int = 500) -> List[Tuple[Any, Any]]:
        test_cases = []
        actual_name = self._get_func_name(code) or func_name.replace(" ", "_").replace("-", "_")
        params_needed = self._count_params(code)
        is_multi_arg = params_needed > 1
        needs_matrices = "len(A)" in code or "len(a)" in code or "[[0]*n" in code or "matrix" in func_name.lower()
        matrix_n = 3
        for _ in range(count):
            if params_needed <= 1:
                inp = random.randint(0, 100)
            elif params_needed == 2:
                if needs_matrices:
                    inp = ([[random.randint(0, 10) for _ in range(matrix_n)] for _ in range(matrix_n)],
                           [[random.randint(0, 10) for _ in range(matrix_n)] for _ in range(matrix_n)])
                else:
                    size = random.randint(1, 10)
                    inp = ([random.randint(0, 20) for _ in range(size)],
                           [random.randint(0, 20) for _ in range(size)])
            else:
                n = random.randint(1, 8)
                inp = ([[random.randint(0, 10) for _ in range(n)] for _ in range(n)],
                       [[random.randint(0, 10) for _ in range(n)] for _ in range(n)])
            test_cases.append((inp, None))
        call_str = f"{actual_name}(*x)" if is_multi_arg else f"{actual_name}(x)"
        inputs_repr = repr([tc[0] for tc in test_cases])
        wrapper = f"{code}\n\ninputs_list = {inputs_repr}\nresults_list = [{call_str} for x in inputs_list]"
        while len(wrapper) > 4000 and count > 10:
            count = count // 2
            test_cases = test_cases[:count]
            inputs_repr = repr([tc[0] for tc in test_cases])
            wrapper = f"{code}\n\ninputs_list = {inputs_repr}\nresults_list = [{call_str} for x in inputs_list]"
        result = self.sandbox["run_code"](wrapper, timeout=self.sandbox_timeout, safe_mode=False)
        if result.get("success") and result.get("variables"):
            vars_dict = result["variables"]
            results_raw = vars_dict.get("results_list")
            if results_raw and isinstance(results_raw, str):
                try:
                    results_raw = eval(results_raw)
                except Exception:
                    return test_cases
            if results_raw and len(results_raw) == count:
                test_cases = [(inp, out) for (inp, _), out in zip(test_cases, results_raw)]
        return test_cases

    def _count_params(self, code: str) -> int:
        import re
        m = re.search(r"def\s+\w+\s*\(([^)]*)\)", code)
        if not m:
            return 1
        args = [a.strip() for a in m.group(1).split(",") if a.strip() and a.strip() != "self"]
        return len(args)

    def _benchmark(self, code: str, test_cases: List[Tuple[Any, Any]]) -> Dict[str, Any]:
        if not test_cases:
            return {"success": False, "error": "no test cases"}
        sample = test_cases[:min(100, len(test_cases))]
        func_name = self._get_func_name(code)
        wrapper = self._build_benchmark_wrapper(code, func_name, sample)
        while len(wrapper) > 4000 and len(sample) > 5:
            sample = sample[:len(sample)//2]
            wrapper = self._build_benchmark_wrapper(code, func_name, sample)
        result = self.sandbox["run_code"](wrapper, timeout=self.sandbox_timeout, safe_mode=False)
        if not result.get("success"):
            return {"success": False, "error": result.get("error", "unknown")}
        total_time = result.get("execution_time", 0.5)
        return {
            "success": True,
            "avg_time_ms": round((total_time / max(1, len(sample))) * 1000, 4),
            "total_time": total_time,
            "samples": len(sample),
        }

    def _build_benchmark_wrapper(self, code: str, func_name: str, sample: List[Tuple[Any, Any]]) -> str:
        is_multi_arg = self._count_params(code) > 1
        call_str = f"{func_name}(*x)" if is_multi_arg else f"{func_name}(x)"
        inputs_repr = repr([tc[0] for tc in sample])
        return f"""import time
{code}

inputs_list = {inputs_repr}
times_list = []
t0 = time.perf_counter()
for x in inputs_list:
    times_list.append(time.perf_counter())
    result_val = {call_str}
    times_list[-1] = time.perf_counter() - times_list[-1]
elapsed = time.perf_counter() - t0
avg_ms = (sum(times_list) / len(times_list)) * 1000 if times_list else 0"""

    def _mutate_code(self, code: str, seed_name: str) -> Optional[str]:
        old_temp = self.tre.temperature
        self.tre.temperature = self.mutation_temperature
        try:
            prompt = MUTATION_TEMPLATE.format(code=code)
            raw = self.tre._generate(prompt)
            if not raw:
                return None
            mutated = self._extract_python_code(raw)
            if not mutated or mutated.strip() == code.strip():
                return None
            return mutated
        except Exception as e:
            logger.debug("Mutation failed for %s: %s", seed_name, e)
            return None
        finally:
            self.tre.temperature = old_temp

    def _random_mutate_code(self, code: str, seed_name: str) -> Optional[str]:
        import re
        transforms = [
            ("swap_add_mul", lambda c: c.replace(" + ", " * ") if " + " in c else None),
            ("swap_mul_add", lambda c: c.replace(" * ", " + ") if " * " in c else None),
            ("gt_to_lt", lambda c: _replace_compare(c, " > ", " < ") if " > " in c else None),
            ("lt_to_gt", lambda c: _replace_compare(c, " < ", " > ") if " < " in c else None),
            ("eq_to_neq", lambda c: c.replace(" == ", " != ") if " == " in c else None),
            ("neq_to_eq", lambda c: c.replace(" != ", " == ") if " != " in c else None),
            ("plus_one", lambda c: _add_to_return(c, 1)),
            ("minus_one", lambda c: _add_to_return(c, -1)),
            ("swap_init", lambda c: c.replace("a, b = 0, 1", "a, b = 1, 0") if "a, b = 0, 1" in c else None),
            ("reverse_range", lambda c: c.replace("range(n)", "range(n, 0, -1)") if "range(n)" in c else None),
            ("swap_binop", lambda c: re.sub(r"(\w+)\s*\+\s*(\w+)", r"\2 + \1", c) if " + " in c else None),
        ]

        def _replace_compare(c: str, old: str, new: str) -> str:
            return c.replace(old, new)

        def _add_to_return(c: str, delta: int) -> Optional[str]:
            m = re.search(r"return\s+(\w+)", c)
            if m:
                var = m.group(1)
                sign = "+" if delta >= 0 else "-"
                return c.replace(f"return {var}", f"return {var} {sign} {abs(delta)}")
            return None

        available = []
        for name, fn in transforms:
            try:
                result = fn(code)
                if result is not None and result != code:
                    available.append(result)
            except Exception:
                continue

        if not available:
            return None
        return random.choice(available)

    def show_seeds(self, population_size: Optional[int] = None) -> None:
        import click
        pop = population_size or self.population_size
        seeds = self._get_algorithm_seeds()
        if not seeds:
            click.echo("  No algorithm seeds found.")
            return
        click.echo(f"  Available seeds ({len(seeds)}):")
        click.echo()
        for seed in seeds[:max(1, min(pop, len(seeds)))]:
            seed_code = seed.get("code", "") or self._code_from_triples(seed)
            seed_name = seed.get("name", seed.get("concept", "unknown"))
            code_preview = seed_code.split("\n")[0] if seed_code else "(no code)"
            quality = seed.get("source_quality", "unknown")
            book_mark = " [BOOK]" if seed.get("book_backed") else ""
            badge = {"book": "green", "graph": "blue", "fallback": "yellow"}.get(quality, "default")
            click.echo(f"    [{seed_name}]  {code_preview}")
            click.echo(f"           source: {quality}{book_mark}  badge={badge}")
        click.echo()

    def _extract_python_code(self, text: str) -> Optional[str]:
        if "```python" in text:
            start = text.index("```python") + 10
            end = text.index("```", start) if "```" in text[start:] else len(text)
            return text[start:end].strip()
        if "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start) if "```" in text[start:] else len(text)
            return text[start:end].strip()
        if "def " in text:
            return text.strip()
        return None

    def _evaluate_mutation(
        self,
        mutated_code: str,
        test_cases: List[Tuple[Any, Any]],
        original_code: str,
    ) -> Dict[str, Any]:
        if not test_cases:
            return {"success": False, "error": "no test cases"}
        passed = 0
        failed = 0
        total_time = 0.0
        sample = test_cases[:min(self.test_case_count, len(test_cases))]
        func_name = self._get_func_name(mutated_code)
        for inp, expected in sample:
            fn_args = list(inp) if isinstance(inp, (list, tuple)) else [inp]
            result = self.sandbox["run_function"](
                mutated_code, func_name,
                args=fn_args,
                timeout=min(self.sandbox_timeout, 1.0),
            )
            elapsed = result.get("execution_time", 0)
            total_time += elapsed
            if result.get("success") and result.get("result") is not None:
                actual = result["result"]
                if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
                    if abs(actual - expected) / max(abs(expected), 1e-9) < 1e-6:
                        passed += 1
                    else:
                        failed += 1
                elif actual == expected:
                    passed += 1
                else:
                    failed += 1
            else:
                failed += 1
        total = passed + failed
        return {
            "success": total > 0,
            "pass_rate": passed / max(total, 1),
            "passed": passed,
            "failed": failed,
            "total": total,
            "avg_time_ms": round((total_time / max(total, 1)) * 1000, 4) if total > 0 else 0,
        }

    def _get_func_name(self, code: str) -> str:
        import re
        m = re.search(r"def\s+(\w+)\s*\(", code)
        return m.group(1) if m else "evolved_function"

    def _promote_discovery(
        self,
        seed_name: str,
        original_code: str,
        mutated_code: str,
        pass_rate: float,
        speedup: float,
        avg_time_ms: float,
        baseline_time_ms: float,
    ) -> None:
        existing = self.ms.search_concepts(f"discovery_{seed_name}", limit=1)
        if existing:
            logger.info("Discovery already exists for %s — updating confidence", seed_name)
        discovery_name = f"synthetic_discovery_{seed_name.replace(' ', '_')}"
        self.ms.add_concept(discovery_name, source_type=DISCOVERY_SOURCE)
        self.ms.add_concept("synthetic_discovery", source_type=DISCOVERY_SOURCE)
        tags = {
            "type": "algorithmic_discovery",
            "speedup": str(round(speedup, 4)),
            "pass_rate": str(round(pass_rate, 4)),
            "avg_time_ms": str(round(avg_time_ms, 4)),
            "baseline_time_ms": str(round(baseline_time_ms, 4)),
        }
        self.ms.add_fact_triple(
            subject=discovery_name,
            relation="derives_from",
            obj=seed_name,
            confidence=min(1.0, speedup / 10),
            source_type=DISCOVERY_SOURCE,
            tags=tags,
            truth_confidence=min(0.95, pass_rate),
            source_quality="synthetic_verified",
            evidence=f"Pass rate: {pass_rate:.4f}, Speedup: {speedup:.3f}x",
            domain="mathematical_discovery",
        )
        self.ms.add_fact_triple(
            subject=discovery_name,
            relation="is_discovery",
            obj="synthetic_discovery",
            confidence=min(1.0, speedup / 10),
            source_type=DISCOVERY_SOURCE,
            tags=tags,
            truth_confidence=min(0.95, pass_rate),
            domain="mathematical_discovery",
        )
        self.ms.add_fact_triple(
            subject=seed_name,
            relation="has_synthetic_variant",
            obj=discovery_name,
            confidence=min(1.0, speedup / 10),
            source_type=DISCOVERY_SOURCE,
            tags=tags,
            truth_confidence=min(0.95, pass_rate),
            domain="mathematical_discovery",
        )
        self._store_discovery_record(
            seed_name=seed_name,
            discovery_name=discovery_name,
            original_code=original_code,
            mutated_code=mutated_code,
            pass_rate=pass_rate,
            speedup=speedup,
            avg_time_ms=avg_time_ms,
        )

    def _store_discovery_record(
        self,
        seed_name: str,
        discovery_name: str,
        original_code: str,
        mutated_code: str,
        pass_rate: float,
        speedup: float,
        avg_time_ms: float,
    ) -> None:
        try:
            cursor = self.ms.conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS synthetic_discoveries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    discovery_name TEXT NOT NULL,
                    seed_name TEXT NOT NULL,
                    original_code TEXT,
                    mutated_code TEXT,
                    pass_rate REAL,
                    speedup REAL,
                    avg_time_ms REAL,
                    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                INSERT INTO synthetic_discoveries
                    (discovery_name, seed_name, original_code, mutated_code,
                     pass_rate, speedup, avg_time_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                discovery_name, seed_name, original_code, mutated_code,
                pass_rate, speedup, avg_time_ms,
            ))
            self.ms.conn.commit()
        except Exception as e:
            logger.warning("Failed to store discovery record: %s", e)

    def get_discovery_stats(self) -> Dict[str, Any]:
        try:
            cursor = self.ms.conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS synthetic_discoveries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    discovery_name TEXT NOT NULL,
                    seed_name TEXT NOT NULL,
                    original_code TEXT,
                    mutated_code TEXT,
                    pass_rate REAL,
                    speedup REAL,
                    avg_time_ms REAL,
                    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("SELECT COUNT(*) as cnt FROM synthetic_discoveries")
            count = cursor.fetchone()["cnt"]
            cursor.execute("""
                SELECT * FROM synthetic_discoveries ORDER BY discovered_at DESC LIMIT 50
            """)
            rows = [dict(r) for r in cursor.fetchall()]
            return {"total": count, "discoveries": rows}
        except Exception as e:
            return {"total": 0, "discoveries": [], "error": str(e)}

    # ------------------------------------------------------------------
    #  Level 2 — Hypothesis Generation (cross-correlate knowledge graph)
    #
    #  Given two concepts from books (e.g. "X improves memory" and
    #  "Y improves attention"), generate: "Combining X and Y may improve
    #  learning speed."  Most will be wrong. Some will be valuable.
    # ------------------------------------------------------------------

    def generate_hypotheses(self, count: int = 5) -> List[Dict[str, Any]]:
        hypotheses = []
        seeds = self._get_algorithm_seeds()
        if len(seeds) < 2:
            # Try Level 3 as fallback — may discover something from sparse data
            return self._mine_abc_correlations(count=count)

        # Gather all concept names from the knowledge graph for context
        try:
            concepts = self.ms.search_concepts("", limit=100)
        except Exception:
            concepts = []
        concept_names = [c["name"] for c in concepts if c.get("name")]

        # Also mine Level 3 ABC correlations in parallel
        abc_patterns = self._mine_abc_correlations(count=max(2, count // 2))
        hypotheses.extend(abc_patterns)

        seen_patterns: set = set()
        # Prioritize book-backed seeds for cross-pollination
        book_seeds = [s for s in seeds if s.get("book_backed")]
        graph_seeds = [s for s in seeds if not s.get("book_backed")]
        ordered_seeds = book_seeds + graph_seeds

        for a in ordered_seeds[:15]:
            for b in ordered_seeds[1:16]:
                if a["concept"] == b["concept"]:
                    continue
                pattern_key = tuple(sorted([a["concept"], b["concept"]]))
                if pattern_key in seen_patterns:
                    continue
                seen_patterns.add(pattern_key)

                # Score graph connectivity between A and B
                graph_strength = self._score_graph_connectivity(
                    a["concept"], b["concept"]
                )

                hypothesis = self._try_llm_hypothesis(a, b, concept_names)
                if hypothesis:
                    hypothesis["graph_strength"] = graph_strength
                    hypothesis["confidence"] = min(1.0, 0.3 + graph_strength * 0.4)
                    hypothesis["book_backed"] = a.get("book_backed", False) or b.get("book_backed", False)
                    hypotheses.append(hypothesis)
                    if len(hypotheses) >= count:
                        return self._store_hypotheses(hypotheses)

                fallback = self._rule_hypothesis(a, b)
                if fallback:
                    fallback["generator"] = "rule"
                    fallback["graph_strength"] = graph_strength
                    fallback["confidence"] = min(1.0, fallback.get("confidence", 0.2) + graph_strength * 0.3)
                    fallback["book_backed"] = a.get("book_backed", False) or b.get("book_backed", False)
                    if not any(h["text"] == fallback["text"] for h in hypotheses):
                        hypotheses.append(fallback)
                        if len(hypotheses) >= count:
                            return self._store_hypotheses(hypotheses)

        return self._store_hypotheses(hypotheses)

    def _score_graph_connectivity(self, concept_a: str, concept_b: str) -> float:
        """Score how well-connected two concepts are in the knowledge graph.
        Returns 0.0 (no connection) to 1.0 (strongly connected)."""
        try:
            ga = self.ms.get_concept_graph(concept_a)
            gb = self.ms.get_concept_graph(concept_b)
            rels_a = {r.get("target", "").lower() for r in ga.get("relationships", [])}
            rels_b = {r.get("target", "").lower() for r in gb.get("relationships", [])}
            shared = rels_a & rels_b
            if not shared:
                return 0.0
            overlap = len(shared)
            total = len(rels_a | rels_b)
            jaccard = overlap / max(total, 1)
            return min(1.0, jaccard * 2.0)  # amplify weak signal
        except Exception:
            return 0.0

    def _try_llm_hypothesis(
        self, a: Dict[str, Any], b: Dict[str, Any],
        all_concepts: List[str],
    ) -> Optional[Dict[str, Any]]:
        if not hasattr(self, 'tre') or not self.tre:
            return None
        sample = random.sample(all_concepts, min(5, len(all_concepts)))
        a_label = "book-derived" if a.get("book_backed") else "known"
        b_label = "book-derived" if b.get("book_backed") else "known"
        prompt = (
            "You are a mathematical discovery engine. Given two algorithm concepts, "
            "suggest a novel combined approach that might outperform them individually.\n\n"
            f"Concept A ({a_label}): {a['concept']}\nCode A:\n{a.get('code', '')[:300]}\n\n"
            f"Concept B ({b_label}): {b['concept']}\nCode B:\n{b.get('code', '')[:300]}\n\n"
            f"Other known concepts: {', '.join(sample)}\n\n"
            "Propose a hypothesis of the form: \"Combining the [aspect] of A with the [aspect] of B "
            "may produce a faster algorithm because [reason].\"\n"
            "Output ONLY the hypothesis text, no preamble."
        )
        try:
            old_temp = self.tre.temperature
            self.tre.temperature = 0.8
            raw = self.tre._generate(prompt)
            self.tre.temperature = old_temp
            if raw and len(raw) > 20:
                return {
                    "text": raw.strip(),
                    "concepts": [a["concept"], b["concept"]],
                    "generator": "llm",
                    "confidence": 0.3,
                    "seed_a_code": a.get("code", ""),
                    "seed_b_code": b.get("code", ""),
                }
        except Exception:
            pass
        return None

    def _rule_hypothesis(self, a: Dict[str, Any], b: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        cross_patterns = [
            (["sort", "matrix"], "Applying the partitioning strategy from {A} to {B} "
             "may reduce the asymptotic complexity by splitting the work into independent subproblems."),
            (["fibonacci", "matrix"], "Applying the state-transition pattern from {A} to {B} "
             "may enable logarithmic-step computation via exponentiation of a recurrence matrix."),
            (["prime", "fibonacci"], "The number-theoretic structure underlying {A} and {B} "
             "shares a modular recurrence that could be computed with a single combined pass."),
            (["dot_product", "matrix"], "The reduction pattern in {A} can be extended to {B} "
             "by expressing the tile operation as a vector inner product."),
            (["factorial", "prime"], "The multiplicative accumulation in {A} can skip composite factors "
             "using the primality test from {B}, reducing the number of multiplication steps."),
            (["gradient", "matrix"], "Applying the iterative refinement pattern from {A} to {B} "
             "may converge faster by warm-starting the matrix operation with a gradient step."),
            (["convolution", "sort"], "The sliding-window pattern in {A} can be accelerated by "
             "pre-sorting the data with {B}, enabling branchless accumulation."),
            (["entropy", "optimization"], "The uncertainty measure from {A} can guide the search "
             "in {B} by focusing exploration on high-information regions."),
            (["regression", "interpolation"], "The error-minimization approach of {A} combined with "
             "the adaptive sampling of {B} may produce a hybrid that generalizes better from sparse data."),
        ]
        a_name = a["concept"].lower()
        b_name = b["concept"].lower()
        for keywords, template in cross_patterns:
            if any(k in a_name for k in keywords) and any(k in b_name for k in keywords):
                return {
                    "text": template.format(A=a["concept"], B=b["concept"]),
                    "concepts": [a["concept"], b["concept"]],
                    "generator": "rule",
                    "confidence": 0.5,
                    "seed_a_code": a.get("code", ""),
                    "seed_b_code": b.get("code", ""),
                }
        if a_name != b_name:
            return {
                "text": f"Applying the optimization strategy of {a['concept']} to {b['concept']} "
                        f"may uncover a faster variant by rewriting the core loop structure.",
                "concepts": [a["concept"], b["concept"]],
                "generator": "rule",
                "confidence": 0.2,
                "seed_a_code": a.get("code", ""),
                "seed_b_code": b.get("code", ""),
            }
        return None

    # ------------------------------------------------------------------
    #  Level 3 — ABC Correlation Mining (discovering unknown patterns)
    #
    #  Given: knowledge graph facts like "A improves B", "B improves C"
    #  Predict: "A may influence C" (even though no direct test exists)
    #
    #  This mirrors how drug-discovery systems find novel candidates:
    #  correlate A→B and B→C, then hypothesize A→C.
    # ------------------------------------------------------------------

    def _mine_abc_correlations(self, count: int = 3) -> List[Dict[str, Any]]:
        """Mine the knowledge graph for A→B→C chains where A correlates
        with B and B correlates with C, but A vs C has never been checked.
        Returns hypothesis dicts with confidence scored by chain strength."""
        hypotheses = []
        try:
            concepts = self.ms.get_all_concepts()
        except Exception:
            return hypotheses

        # Build a concept adjacency list from relationships
        adj: Dict[str, List[Dict[str, Any]]] = {}
        for c in concepts[:100]:
            try:
                graph = self.ms.get_concept_graph(c)
                for r in graph.get("relationships", []):
                    src = r.get("source_concept", "").lower().strip() or c.lower()
                    tgt = r.get("target_concept", "").lower().strip() or r.get("target", "").lower().strip()
                    rel = r.get("relation", "").lower().strip()
                    conf = max(r.get("confidence", 0.5), r.get("truth_confidence", 0.5))
                    if src and tgt and rel:
                        adj.setdefault(src, []).append({
                            "target": tgt, "relation": rel, "confidence": conf,
                        })
            except Exception:
                continue

        # Direct causal/influence relations to chain
        influence_rels = {"improves", "increases", "enhances", "boosts", "accelerates",
                          "reduces", "decreases", "amplifies", "strengthens",
                          "triggers", "promotes", "produces", "generates"}

        chained: set = set()
        for a, edges_a in adj.items():
            for e1 in edges_a:
                if e1["relation"] not in influence_rels:
                    continue
                b = e1["target"]
                if b not in adj:
                    continue
                for e2 in adj[b]:
                    if e2["relation"] not in influence_rels:
                        continue
                    c = e2["target"]
                    if a == c or b == c:
                        continue
                    chain_key = tuple(sorted([a, c]))
                    if chain_key in chained:
                        continue
                    chained.add(chain_key)

                    # Check if A→C already exists in graph
                    already_known = any(
                        e["target"] == c and e["relation"] in influence_rels
                        for e in edges_a
                    )

                    chain_confidence = e1["confidence"] * e2["confidence"]
                    novelty_bonus = 0.4 if not already_known else 0.0
                    overall_conf = min(1.0, chain_confidence + novelty_bonus)

                    if already_known:
                        continue  # not novel

                    hypotheses.append({
                        "text": (
                            f"Since {a} is known to {e1['relation']} {b}, "
                            f"and {b} is known to {e2['relation']} {c}, "
                            f"it follows that {a} may indirectly {e1['relation']} {c} "
                            f"through the {b} pathway. This relationship has not been "
                            f"directly verified."
                        ),
                        "concepts": [a, b, c],
                        "generator": "abc_correlation",
                        "confidence": round(overall_conf, 3),
                        "chain": f"{a} --[{e1['relation']}]--> {b} --[{e2['relation']}]--> {c}",
                        "novel": True,
                        "graph_strength": round(e1["confidence"] * e2["confidence"], 3),
                        "seed_a_code": "",
                        "seed_b_code": "",
                    })
                    if len(hypotheses) >= count:
                        break
            if len(hypotheses) >= count:
                break

        return hypotheses

    def discover_correlations(self, count: int = 5) -> List[Dict[str, Any]]:
        """Level 3 public API — discover unknown A→C correlations through
        intermediate B.  Returns hypotheses with novelty flags."""
        return self._mine_abc_correlations(count=count)

    def _store_hypotheses(self, hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            cursor = self.ms.conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hypotheses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hypothesis_text TEXT NOT NULL,
                    concepts_involved TEXT,
                    generator TEXT,
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            for h in hypotheses:
                cursor.execute("""
                    INSERT INTO hypotheses (hypothesis_text, concepts_involved, generator, confidence)
                    VALUES (?, ?, ?, ?)
                """, (
                    h["text"],
                    json.dumps(h.get("concepts", [])),
                    h.get("generator", "unknown"),
                    h.get("confidence", 0.0),
                ))
            self.ms.conn.commit()
            cursor.execute("SELECT last_insert_rowid()")
            first_id = cursor.fetchone()[0]
            for i, h in enumerate(hypotheses):
                h["id"] = first_id + i
        except Exception as e:
            logger.warning("Failed to store hypotheses: %s", e)
        return hypotheses

    def get_hypothesis_stats(self) -> Dict[str, Any]:
        try:
            cursor = self.ms.conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hypotheses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hypothesis_text TEXT NOT NULL,
                    concepts_involved TEXT,
                    generator TEXT,
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("SELECT COUNT(*) as cnt FROM hypotheses")
            count = cursor.fetchone()["cnt"]
            cursor.execute("SELECT * FROM hypotheses ORDER BY created_at DESC LIMIT 50")
            rows = [dict(r) for r in cursor.fetchall()]
            return {"total": count, "hypotheses": rows}
        except Exception as e:
            return {"total": 0, "hypotheses": [], "error": str(e)}

    # ------------------------------------------------------------------
    #  Level 3 — Experiment Runner
    # ------------------------------------------------------------------

    def run_experiment(
        self,
        hypothesis: Dict[str, Any],
        num_candidates: int = 5,
        test_case_count: Optional[int] = None,
        use_random_mutator: bool = False,
    ) -> Dict[str, Any]:
        tcc = test_case_count or self.test_case_count
        result: Dict[str, Any] = {
            "hypothesis": hypothesis.get("text", ""),
            "concepts": hypothesis.get("concepts", []),
            "candidates_tried": 0,
            "candidates_passed": 0,
            "best_speedup": 0.0,
            "best_candidate": None,
            "best_code": None,
            "seed_code": hypothesis.get("seed_a_code", ""),
            "seed2_code": hypothesis.get("seed_b_code", ""),
            "all_results": [],
        }

        seed_code = result["seed_code"]
        if not seed_code:
            seeds = self._get_algorithm_seeds()
            for s in seeds:
                if s["concept"] == hypothesis.get("concepts", [None])[0]:
                    seed_code = s.get("code", "")
                    result["seed_code"] = seed_code
                    break

        if not seed_code:
            result["error"] = "No seed code available for hypothesis"
            return result

        seed_name = hypothesis.get("concepts", ["unknown"])[0]
        test_cases = self._generate_test_cases(seed_code, seed_name, count=tcc)
        if not test_cases:
            result["error"] = "Failed to generate test cases"
            return result

        baseline = self._benchmark(seed_code, test_cases)
        if not baseline.get("success"):
            result["error"] = f"Baseline failed: {baseline.get('error', 'unknown')}"
            return result
        baseline_time = baseline.get("avg_time_ms", float("inf"))
        result["baseline_time_ms"] = baseline_time

        mutate_fn = self._random_mutate_code if use_random_mutator else self._mutate_code

        mprog = MutationProgress()
        mprog.seed_name = f"exp:{seed_name}"
        mprog.total_iterations = num_candidates
        mprog.baseline_time_ms = baseline_time

        cand_pbar = tqdm(range(num_candidates), desc="  Candidates", unit="cand",
                         leave=False, colour="green")
        for i in cand_pbar:
            cand_pbar.set_description(mprog.format_desc())
            candidate = mutate_fn(seed_code, seed_name)
            if not candidate:
                mprog.update(i + 1, result["candidates_tried"],
                             result["candidates_passed"],
                             discoveries=1 if result["best_speedup"] > 1.05 else 0)
                cand_pbar.set_description(mprog.format_desc())
                continue
            result["candidates_tried"] += 1

            eval_result = self._evaluate_mutation(candidate, test_cases, seed_code)
            if not eval_result.get("success"):
                mprog.update(i + 1, result["candidates_tried"],
                             result["candidates_passed"],
                             discoveries=1 if result["best_speedup"] > 1.05 else 0)
                cand_pbar.set_description(mprog.format_desc())
                continue
            result["candidates_passed"] += 1

            speedup = baseline_time / max(eval_result.get("avg_time_ms", 1e-9), 1e-9) if baseline_time > 0 else 1.0
            pass_rate = eval_result.get("pass_rate", 0)

            mprog.update(i + 1, result["candidates_tried"],
                         result["candidates_passed"],
                         pass_rate=pass_rate, speedup=speedup,
                         discoveries=1 if result["best_speedup"] > 1.05 else 0)
            cand_pbar.set_description(mprog.format_desc())

            entry = {
                "iteration": i + 1,
                "pass_rate": pass_rate,
                "avg_time_ms": eval_result.get("avg_time_ms", 0),
                "speedup": round(speedup, 3),
                "code": candidate,
            }
            result["all_results"].append(entry)

            is_passing = eval_result["pass_rate"] >= 0.99
            if is_passing and speedup > result["best_speedup"]:
                result["best_speedup"] = round(speedup, 3)
                result["best_candidate"] = entry
                result["best_code"] = candidate
                cand_pbar.set_postfix(BEST=f"{speedup:.2f}x", refresh=True)
        cand_pbar.close()

        if result["best_speedup"] > 1.05:
            self._promote_discovery(
                seed_name=seed_name,
                original_code=seed_code,
                mutated_code=result["best_code"],
                pass_rate=result["best_candidate"]["pass_rate"],
                speedup=result["best_speedup"],
                avg_time_ms=result["best_candidate"]["avg_time_ms"],
                baseline_time_ms=baseline_time,
            )

        self._store_experiment_result(hypothesis, result, seed_code, baseline_time)
        return result

    def _store_experiment_result(
        self,
        hypothesis: Dict[str, Any],
        result: Dict[str, Any],
        seed_code: str,
        baseline_time: float,
    ) -> None:
        try:
            cursor = self.ms.conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hypothesis_text TEXT,
                    concepts TEXT,
                    candidates_tried INTEGER,
                    candidates_passed INTEGER,
                    best_speedup REAL,
                    best_code TEXT,
                    seed_code TEXT,
                    baseline_time_ms REAL,
                    full_results TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                INSERT INTO experiments
                    (hypothesis_text, concepts, candidates_tried, candidates_passed,
                     best_speedup, best_code, seed_code, baseline_time_ms, full_results)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                hypothesis.get("text", ""),
                json.dumps(hypothesis.get("concepts", [])),
                result.get("candidates_tried", 0),
                result.get("candidates_passed", 0),
                result.get("best_speedup", 0.0),
                result.get("best_code", ""),
                seed_code,
                baseline_time,
                json.dumps(result.get("all_results", []), default=str),
            ))
            self.ms.conn.commit()
        except Exception as e:
            logger.warning("Failed to store experiment result: %s", e)

    def get_experiment_stats(self) -> Dict[str, Any]:
        try:
            cursor = self.ms.conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hypothesis_text TEXT,
                    concepts TEXT,
                    candidates_tried INTEGER,
                    candidates_passed INTEGER,
                    best_speedup REAL,
                    best_code TEXT,
                    seed_code TEXT,
                    baseline_time_ms REAL,
                    full_results TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("SELECT COUNT(*) as cnt FROM experiments")
            count = cursor.fetchone()["cnt"]
            cursor.execute("SELECT * FROM experiments ORDER BY created_at DESC LIMIT 50")
            rows = [dict(r) for r in cursor.fetchall()]
            return {"total": count, "experiments": rows}
        except Exception as e:
            return {"total": 0, "experiments": [], "error": str(e)}

    def cleanup(self):
        try:
            if hasattr(self, 'ms') and self.ms:
                pass
        except Exception:
            pass
