"""
CPython Internals — Part 7: Hands-On C Extensions

WHAT THIS IS:
  Jafar writes REAL C extensions (not ctypes simulation).
  4 modules live in c_extensions/, built with setuptools.

  ┌─────────────────────────────────────────────────────────────┐
  │  BUILD:  cd c_extensions && python setup.py build_ext --inplace  │
  │  IMPORT: import jafar_hello, jafar_data, jafar_iter, jafar_perf │
  └─────────────────────────────────────────────────────────────┘

MODULES:
  1. jafar_hello — Calling conventions:
       METH_NOARGS, METH_O, METH_VARARGS, METH_KEYWORDS
       PyArg_ParseTuple format codes, PyErr_SetString
  2. jafar_data  — Custom type (Point3D):
       tp_new, tp_init, tp_dealloc, tp_str, tp_repr
       tp_richcompare (==,<,>,etc), tp_hash
       tp_getset (property-like getters/setters)
       Type registration via PyModule_AddObject
  3. jafar_iter  — Container with sequence protocol:
       sq_length, sq_item, sq_ass_item
       tp_iter, tp_iternext (custom iterator type)
       PySequence_Fast for input conversion
  4. jafar_perf  — CPU-bound benchmarks:
       Fibonacci (iterative O(n))
       Prime sieve (Eratosthenes)
       Fast sum, dot_product, matmul
       benchmark() harness

REFERENCE: Python/C API docs
  https://docs.python.org/3/c-api/
  https://docs.python.org/3/extending/

TEST: Run this file directly to test all 4 modules.
"""

import importlib
import importlib.util
import os
import sys
import time

C_EXT_DIR = os.path.join(os.path.dirname(__file__), "c_extensions")


def _ensure_built():
    """Check if .pyd files exist; tell user how to build if not."""
    needed = ["jafar_hello", "jafar_data", "jafar_iter", "jafar_perf"]
    missing = []
    for mod in needed:
        spec = importlib.util.find_spec(mod)
        if spec is None:
            missing.append(mod)

    if missing:
        build_cmd = (
            f"cd /d \"{C_EXT_DIR}\" && python setup.py build_ext --inplace"
        )
        print(f"  MISSING: {', '.join(missing)}")
        print(f"  BUILD:   {build_cmd}")
        print()
        return False
    return True


# ═══════════════════════════════════════════════════════════════════
# 1. jafar_hello — Calling Convention Patterns
# ═══════════════════════════════════════════════════════════════════
# hellomodule.c shows 6 methods with different METH_* flags:
#
#   hi()        METH_NOARGS         — no args, returns None
#   greet(s)    METH_O              — single str arg
#   add(a,b)    METH_VARARGS        — positional tuple
#   repeat(...) METH_KEYWORDS       — positional + keyword
#   describe()  METH_VARARGS        — mixed types (str, int, float)
#   countdown() METH_VARARGS        — void return (prints)
#
# KEY INSIGHT: METH_O is faster than METH_VARARGS for single-arg
# functions because it skips tuple creation.
#

def test_hello():
    print("=" * 60)
    print("1. jafar_hello — Calling Convention Patterns")
    print("=" * 60)

    import jafar_hello as hello

    # METH_NOARGS
    print("\n  hi() → prints from C")
    hello.hi()

    # METH_O
    result = hello.greet("Jafar")
    print(f"  greet('Jafar') → {result!r}")
    assert isinstance(result, str)
    assert "Jafar" in result
    assert "C" in result

    # METH_VARARGS
    result = hello.add(40, 2)
    print(f"  add(40, 2) → {result}")
    assert result == 42

    # METH_KEYWORDS
    result = hello.repeat("ha", 3)
    print(f"  repeat('ha', 3) → {result!r}")
    assert result == "ha, ha, ha"

    # Default keyword arg
    result = hello.repeat("boo")
    print(f"  repeat('boo') (default count=1) → {result!r}")
    assert result == "boo"

    # Mixed types
    result = hello.describe("Alice", 30, 1.75)
    print(f"  describe('Alice', 30, 1.75) → {result!r}")
    assert "Alice" in result and "30" in result

    # Error handling — wrong type
    try:
        hello.add("not", "numbers")
        assert False, "should have raised TypeError"
    except TypeError:
        print("  add('not', 'numbers') → TypeError (correct)")

    # Error handling — negative count
    try:
        hello.countdown(-1)
        assert False, "should have raised ValueError"
    except ValueError:
        print("  countdown(-1) → ValueError (correct)")

    print("\n  ✅ jafar_hello: all tests pass\n")


# ═══════════════════════════════════════════════════════════════════
# 2. jafar_data — Custom Type (Point3D)
# ═══════════════════════════════════════════════════════════════════
# datamodule.c defines Point3D with:
#
#   tp_new       — allocates struct, sets defaults (0,0,0)
#   tp_init      — parses args (x, y, z all optional)
#   tp_dealloc   — frees (no GC tracking needed for 3 doubles)
#   tp_str       — "(1.00, 2.00, 3.00)"
#   tp_repr      — "Point3D(1.00, 2.00, 3.00)"
#   tp_richcompare — compares by origin-distance (LT/LE/GT/GE),
#                    exact equality for EQ/NE
#   tp_hash      — XOR of 3 double hashes
#   tp_getset    — .x, .y, .z as property-like attributes
#
# Module-level:
#   distance(a, b) — Euclidean distance
#   from_list(seq) — create Point3D from [x, y, z]
#

def test_data():
    print("=" * 60)
    print("2. jafar_data — Custom Type: Point3D")
    print("=" * 60)

    import jafar_data as data

    # Default constructor
    p0 = data.Point3D()
    print(f"\n  Point3D() → {p0!r}")
    assert str(p0) == "(0.00, 0.00, 0.00)"

    # Positional args
    p1 = data.Point3D(1.0, 2.0, 3.0)
    print(f"  Point3D(1, 2, 3) → {p1!r}")
    assert "1.00" in repr(p1)

    # Keyword args
    p2 = data.Point3D(z=5.0)
    print(f"  Point3D(z=5) → {p2!r}")
    assert "5.00" in str(p2)

    # Properties (tp_getset)
    p1.x = 10.0
    print(f"  p1.x = 10 → {p1!r}")
    assert p1.x == 10.0

    p1.y = 20.0
    assert p1.y == 20.0

    p1.z = 30.0
    assert p1.z == 30.0

    # Rich comparison
    a = data.Point3D(1, 2, 3)
    b = data.Point3D(1, 2, 3)
    c = data.Point3D(4, 5, 6)
    assert (a == b) == True
    assert (a != c) == True
    assert (a <  c) == True
    assert (c >  a) == True
    print("  a == b: True, a != c: True, a < c: True")

    # Distance
    d = data.distance(p1, p0)
    print(f"  distance(p1, origin) → {d:.2f}")
    # sqrt(10^2 + 20^2 + 30^2) = sqrt(1400) ≈ 37.42
    assert abs(d - 37.42) < 0.01

    # from_list
    p3 = data.from_list([7, 8, 9])
    print(f"  from_list([7,8,9]) → {p3!r}")
    assert p3.x == 7.0 and p3.y == 8.0 and p3.z == 9.0

    # Hashable (can put in set/dict)
    s = {a, b, c}
    print(f"  set of 3 points (two equal) → {len(s)} items")
    assert len(s) == 2  # a and b are equal

    print("\n  ✅ jafar_data: all tests pass\n")


# ═══════════════════════════════════════════════════════════════════
# 3. jafar_iter — Custom Container (IntList)
# ═══════════════════════════════════════════════════════════════════
# itermodule.c defines IntList:
#
#   sq_length    — len()
#   sq_item      — obj[i]
#   sq_ass_item  — obj[i] = val
#   tp_iter      — returns IntListIterator
#   tp_iternext  — next value or StopIteration
#
# IntList stores longs in a flat C array (no PyObject* overhead).
# That means it's more memory-efficient than Python list for integers.
#
# Module-level:
#   append(list, value) — append long to IntList
#   sum(list)           — sum all elements (C loop)
#

def test_iter():
    print("=" * 60)
    print("3. jafar_iter — Custom Container: IntList")
    print("=" * 60)

    import jafar_iter as iter_mod

    # Empty constructor
    lst = iter_mod.IntList()
    print(f"\n  IntList() → {lst}")
    assert len(lst) == 0

    # From iterable
    lst = iter_mod.IntList([1, 2, 3, 4, 5])
    print(f"  IntList([1,2,3,4,5]) → {lst}")
    assert len(lst) == 5

    # Sequence protocol: __getitem__
    assert lst[0] == 1
    assert lst[4] == 5
    assert lst[-1] == 5
    print("  lst[0]=1, lst[4]=5, lst[-1]=5")

    # Sequence protocol: __setitem__
    lst[0] = 99
    assert lst[0] == 99
    print(f"  lst[0] = 99 → {lst}")

    # Iterator protocol
    total = 0
    for v in lst:
        total += v
    print(f"  Sum via Python for-loop: {total}")
    assert total == 99 + 2 + 3 + 4 + 5

    # Module-level sum (C loop)
    lst2 = iter_mod.IntList(range(100))
    c_sum = iter_mod.sum(lst2)
    py_sum = sum(range(100))
    assert c_sum == py_sum
    print(f"  C sum(range(100)) = {c_sum} (matches Python: {py_sum})")

    # Module-level append
    iter_mod.append(lst, 100)
    assert lst[-1] == 100
    print(f"  After append(lst, 100): last = {lst[-1]}")

    # Error: out of bounds
    try:
        _ = lst[999]
        assert False, "should have raised IndexError"
    except IndexError:
        print("  lst[999] → IndexError (correct)")

    print("\n  ✅ jafar_iter: all tests pass\n")


# ═══════════════════════════════════════════════════════════════════
# 4. jafar_perf — CPU-Bound Benchmarks
# ═══════════════════════════════════════════════════════════════════
# perfmodule.c provides C implementations of common algorithms:
#
#   fib(n)        — O(n) Fibonacci, 1000x faster than Python recursion
#   primes(n)     — Sieve of Eratosthenes, ~50x faster
#   fast_sum(n)   — Loop sum, ~100x faster than Python for-loop
#   dot_product   — List dot product, ~10x faster
#   benchmark()   — Time any Python callable (CLOCKS_PER_SEC)
#   matmul(n)     — NxN matrix multiply, measures seconds
#
# KEY INSIGHT: C extensions avoid Python interpreter overhead for
# each loop iteration. A simple `for i in range(n): total += i` in
# Python does N bytecode dispatches. In C, it's one CMP+JMP.
#

def test_perf():
    print("=" * 60)
    print("4. jafar_perf — CPU-Bound Benchmarks")
    print("=" * 60)

    import jafar_perf as perf

    # Fibonacci
    print("\n  Fibonacci (n=40):")
    start = time.perf_counter()
    c_fib = perf.fib(40)
    c_time = time.perf_counter() - start

    def py_fib(n):
        if n <= 1: return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b

    start = time.perf_counter()
    py_result = py_fib(40)
    py_time = time.perf_counter() - start

    print(f"    C fib(40) = {c_fib} ({c_time*1000:.2f}ms)")
    print(f"    Python    = {py_result} ({py_time*1000:.2f}ms)")
    print(f"    Speedup:  {py_time/c_time:.1f}x")
    assert c_fib == 102334155

    # Fast sum
    n = 10_000_000
    print(f"\n  Fast sum(0..{n-1}):")
    start = time.perf_counter()
    c_sum = perf.fast_sum(n)
    c_time = time.perf_counter() - start
    print(f"    C: {c_sum} ({c_time*1000:.2f}ms)")
    assert c_sum == n * (n - 1) // 2

    # Dot product
    size = 1_000_000
    a = [float(i) for i in range(size)]
    b = [float(i * 2) for i in range(size)]
    print(f"\n  Dot product ({size} elements):")
    start = time.perf_counter()
    c_dot = perf.dot_product(a, b)
    c_time = time.perf_counter() - start

    def py_dot(a, b):
        total = 0.0
        for i in range(len(a)):
            total += a[i] * b[i]
        return total

    start = time.perf_counter()
    py_result = py_dot(a, b)
    py_time = time.perf_counter() - start

    print(f"    C: {c_dot:.0f} ({c_time*1000:.2f}ms)")
    print(f"    Python: {py_result:.0f} ({py_time*1000:.2f}ms)")
    print(f"    Speedup: {py_time/c_time:.1f}x")
    assert abs(c_dot - py_result) < 1.0

    print("\n  ✅ jafar_perf: all tests pass\n")


# ═══════════════════════════════════════════════════════════════════
# 5. CodeObject Inspection of C Extension Functions
# ═══════════════════════════════════════════════════════════════════
# C extension functions have NO bytecode — they're native C calls.
# Their __code__ is None. They're implemented as PyCFunction.

def inspect_c_functions():
    print("=" * 60)
    print("5. C Extension Function Internals")
    print("=" * 60)

    import jafar_hello as hello

    def inspect(obj, name):
        print(f"\n  {name}:")
        print(f"    type:       {type(obj)}")
        print(f"    __module__: {getattr(obj, '__module__', 'N/A')}")
        print(f"    __name__:   {getattr(obj, '__name__', 'N/A')}")
        print(f"    __code__:   {getattr(obj, '__code__', 'N/A')}")
        print(f"    __doc__:    {obj.__doc__!r}")

    inspect(hello.hi, "hello.hi (METH_NOARGS)")
    inspect(hello.greet, "hello.greet (METH_O)")
    inspect(hello.add, "hello.add (METH_VARARGS)")
    inspect(hello.repeat, "hello.repeat (METH_KEYWORDS)")

    print()
    print("  KEY INSIGHT: C functions have __code__ = None")
    print("  because they are NOT Python functions — they are")
    print("  PyCFunction objects backed by native machine code.")
    print("  This is why dis.dis() shows nothing for them.")
    print("  Only dis.dis(module) shows the method table.")
    print()

    import dis
    print("  dis.dis(jafar_hello):")
    try:
        dis.dis(hello)
    except Exception as e:
        print(f"    (dis not available: {e})")
    print()

    # Inspect Point3D type
    import jafar_data as data
    print("  Point3D type slots (from Python):")
    for slot in ['__new__', '__init__', '__str__', '__repr__',
                 '__hash__', '__eq__', '__lt__', '__gt__']:
        print(f"    {slot:>15}: {hasattr(data.Point3D, slot)}")


# ═══════════════════════════════════════════════════════════════════
# 6. Key Insights (stored as memories for Jafar)
# ═══════════════════════════════════════════════════════════════════
# These would be stored with confidence = 0.85, source = "cpython_handson"
#
# 1. C extension pattern: every module needs 4 things:
#    a) C functions with signature PyObject*(PyObject *self, PyObject *args)
#    b) PyMethodDef array (NULL-terminated)
#    c) PyModuleDef struct with PyModuleDef_HEAD_INIT
#    d) PyInit_<name> init function
#
# 2. Four calling conventions: METH_NOARGS (fast, no args),
#    METH_O (single arg, skip tuple), METH_VARARGS (positional tuple),
#    METH_KEYWORDS (adds PyObject *kwargs). METH_FASTCALL (3.7+) for C array.
#
# 3. PyArg_ParseTuple format: "s" (str), "i" (int), "d" (double),
#    "O" (PyObject*, borrowed ref), "|" (optional args follow),
#    "O!" (type-checked PyObject*), "l" (long), "K" (unsigned long long).
#    Always check return value — return NULL on failure.
#
# 4. Custom types need: struct with PyObject_HEAD, tp_new (allocate),
#    tp_init (initialize), tp_dealloc (free), tp_str/tp_repr,
#    tp_richcompare (6-comparison dispatch), tp_hash.
#    Register via PyType_Ready() + PyModule_AddObject().
#
# 5. Sequence protocol: fill tp_as_sequence with sq_length, sq_item,
#    sq_ass_item. Iterator: tp_iter returns a separate iterator object
#    with its own tp_iternext.
#
# 6. C extensions are 10-100x faster for tight loops because:
#    no bytecode dispatch, no type checking at runtime, no boxing
#    (ints stored as raw C longs, not PyObject*).
#
# 7. C extension functions have __code__ = None — they're not Python.
#    They can't be traced by sys.settrace (they appear as 'c_call').
#
# 8. Memory management: PyObject_Malloc for allocations, explicit
#    free() for C-level arrays, Py_DECREF for Python objects.
#    A C extension must not leak references or double-free.
#
# 9. tp_getset enables property-like attributes from C (getter + setter
#    function pointers). More efficient than Python @property.
#
# 10. Build process: setuptools.Extension + setup() → compiler finds
#     Python include dirs and libs automatically. Output is a .pyd
#     (Python Dynamic Library = DLL with Python exports).


# ═══════════════════════════════════════════════════════════════════
# Main: Run all tests
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("JAFAR — Phase 4, Part 7: Hands-On C Extensions")
    print("=" * 60)
    print()

    if not _ensure_built():
        print("Run the build command above, then re-run this file.")
        return

    test_hello()
    test_data()
    test_iter()
    test_perf()
    inspect_c_functions()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    print()
    print("Jafar has now written REAL C extensions and used them.")
    print("Next: Phase 5 (concurrency) or deeper C extension mastery.")


if __name__ == "__main__":
    main()
