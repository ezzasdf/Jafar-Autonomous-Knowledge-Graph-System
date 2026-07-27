"""
CPython Internals — Part 6: C Extensions

Covers:
  • PyModuleDef — defining a module from C
  • PyMethodDef — exposing C functions to Python
  • PyType_Spec — defining new types from C (since Python 3.2)
  • Building extensions: setuptools, distutils, Meson (Python 3.12+)
  • ctypes — calling C libraries from Python
  • cffi — alternative C foreign function interface
  • Cython — writing C extensions in a Python-like language
  • Embedding Python in C applications (Py_Initialize, PyRun_SimpleString)
  • Calling conventions: METH_VARARGS, METH_KEYWORDS, METH_NOARGS, METH_O
  • Error handling: PyErr_SetString, return NULL pattern
  • Reference counting in extensions: borrowing vs stealing
  • PyPy vs Cython vs CPython implementation notes

Reference:
  - CPython Internals (Anthony Shaw) — Chapters 6–7
  - Python/C API docs: https://docs.python.org/3/c-api/
  - Extending Python with C: https://docs.python.org/3/extending/
  - Embedding Python: https://docs.python.org/3/extending/embedding.html
  - PEP 384 (stable ABI), PEP 632 (deprecate distutils)
  - Meson build: https://mesonbuild.com/Python-module.html
"""

import ctypes
import sys
import os
import struct


# ═══════════════════════════════════════════════════════════════════
# 1. PyModuleDef + PyMethodDef — The Pattern
# ═══════════════════════════════════════════════════════════════════
#
# Every C extension module follows this pattern:
#
#   // 1. Define methods
#   static PyObject* my_func(PyObject *self, PyObject *args) {
#       // ...
#   }
#
#   // 2. Method table (NULL-terminated)
#   static PyMethodDef MyMethods[] = {
#       {"func_name", my_func, METH_VARARGS, "doc string"},
#       {NULL, NULL, 0, NULL}  // sentinel
#   };
#
#   // 3. Module definition
#   static struct PyModuleDef mymodule = {
#       PyModuleDef_HEAD_INIT,
#       "mymodule",     // m_name
#       "doc string",   // m_doc
#       -1,             // m_size (-1 = no per-module state)
#       MyMethods       // m_methods
#   };
#
#   // 4. Module init function
#   PyMODINIT_FUNC PyInit_mymodule(void) {
#       return PyModule_Create(&mymodule);
#   }
#

def simulate_module_def():
    """Simulate the C extension module pattern in Python."""
    print("=== C Extension Module Pattern ===")

    # This is what a PyMethodDef array looks like conceptually
    MethodDef = type('MethodDef', (), {
        'ml_name': 'func_name',
        'ml_meth': lambda self, args: None,
        'ml_flags': 0x0001,  # METH_VARARGS
        'ml_doc': 'doc string'
    })

    # METH_ flag constants
    flags = {
        "METH_VARARGS": 0x0001,
        "METH_KEYWORDS": 0x0002,
        "METH_NOARGS": 0x0004,
        "METH_O": 0x0008,
        "METH_CLASS": 0x0010,
        "METH_STATIC": 0x0020,
        "METH_COEXIST": 0x0040,
        "METH_FASTCALL": 0x0080,
    }

    print(f"  {'Flag':>20} {'Value':>10}")
    print(f"  {'-'*20} {'-'*10}")
    for name, val in flags.items():
        print(f"  {name:>20} 0x{val:04x}")
    print()

simulate_module_def()


# ═══════════════════════════════════════════════════════════════════
# 2. PyArg_ParseTuple — Parsing Python Arguments in C
# ═══════════════════════════════════════════════════════════════════
#
# In a C extension, Python args come as a single tuple.
# PyArg_ParseTuple parses it based on a format string:
#
#   "s"   → const char* (UTF-8, borrowed reference)
#   "s#"  → const char* + Py_ssize_t (with length)
#   "z"   → const char* or NULL (None → NULL)
#   "i"   → int
#   "I"   → unsigned int
#   "l"   → long
#   "d"   → double
#   "f"   → float
#   "O"   → PyObject* (borrowed reference)
#   "O&"  → converter function
#   "|"   → following args are optional
#   ":"   → error message separator
#   ";"   → full error message

def simulate_parse_tuple():
    """Simulate PyArg_ParseTuple format parsing."""
    print("=== PyArg_ParseTuple format codes ===")

    format_codes = [
        ("s", "const char* (UTF-8, null-terminated)"),
        ("s#", "const char* + length"),
        ("z", "const char* or NULL (None converts to NULL)"),
        ("i", "int"),
        ("I", "unsigned int"),
        ("l", "long"),
        ("d", "double"),
        ("f", "float"),
        ("O", "PyObject* (borrowed reference)"),
        ("O!", "type check + PyObject*"),
        ("O&", "converter function"),
        ("|", "following arguments are optional"),
        (":", "error message format (rest of string)"),
        (";", "full error message (rest of string)"),
        ("(sii)", "nested tuple (unpacked)"),
        ("es", "encoding-aware string"),
        ("et", "encoding + fallback"),
    ]

    print(f"  {'Code':>8} {'Meaning'}")
    print(f"  {'-'*8} {'-'*40}")
    for code, meaning in format_codes:
        print(f"  {code:>8} {meaning}")
    print()

simulate_parse_tuple()


# ═══════════════════════════════════════════════════════════════════
# 3. ctypes — Calling C Libraries from Python
# ═══════════════════════════════════════════════════════════════════
#
# ctypes is the built-in way to call C functions from Python.
# It handles: loading shared libs, marshalling arguments, error codes.

def demo_ctypes():
    """Full ctypes demonstration."""
    print("=== ctypes: Calling C from Python ===")

    # Load the C runtime
    try:
        libc = ctypes.CDLL("msvcrt.dll")  # Windows
    except OSError:
        try:
            libc = ctypes.CDLL("libc.so.6")  # Linux
        except OSError:
            try:
                libc = ctypes.CDLL("libc.dylib")  # macOS
            except OSError:
                print("  (no system C library found)")
                return

    # strlen — standard function
    libc.strlen.argtypes = [ctypes.c_char_p]
    libc.strlen.restype = ctypes.c_size_t
    text = b"hello, ctypes!"
    result = libc.strlen(text)
    print(f"  strlen({text!r}) = {result}")

    # atoi — string to int
    libc.atoi.argtypes = [ctypes.c_char_p]
    libc.atoi.restype = ctypes.c_int
    print(f"  atoi(b'42') = {libc.atoi(b'42')}")

    # qsort — sorting via callback
    print()
    print("  qsort with Python callback:")

    # Create a CMPFUNC callback
    CMPFUNC = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)

    def py_cmp(a, b):
        x = ctypes.cast(a, ctypes.POINTER(ctypes.c_int))[0]
        y = ctypes.cast(b, ctypes.POINTER(ctypes.c_int))[0]
        return x - y

    cmp_cb = CMPFUNC(py_cmp)

    arr = (ctypes.c_int * 5)(5, 3, 1, 4, 2)
    libc.qsort(arr, len(arr), ctypes.sizeof(ctypes.c_int), cmp_cb)
    sorted_arr = [arr[i] for i in range(len(arr))]
    print(f"    sorted: {sorted_arr}")
    print()

demo_ctypes()


# ═══════════════════════════════════════════════════════════════════
# 4. Simulating PyType_Spec for Custom Types
# ═══════════════════════════════════════════════════════════════════
#
# In Python 3.2+, the recommended way to create types from C is
# via PyType_Spec + PyType_FromSpec:
#
#   PyType_Spec spec = {
#       .name = "mymodule.MyType",
#       .basicsize = sizeof(MyTypeObject),
#       .itemsize = 0,
#       .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
#       .slots = { ... }  // typed slots array
#   };
#   PyObject *type = PyType_FromSpec(&spec);
#
# This is higher-level than manually filling PyTypeObject fields.
# We'll simulate the slot mechanism:

def simulate_type_spec():
    """Simulate PyType_Spec and typed slots."""
    print("=== PyType_Spec simulation ===")

    # Slot IDs (from Include/compile.h for type specs)
    slots = {
        "Py_tp_dealloc": 0,
        "Py_tp_repr": 1,
        "Py_tp_hash": 2,
        "Py_tp_call": 3,
        "Py_tp_str": 4,
        "Py_tp_getattro": 5,
        "Py_tp_setattro": 6,
        "Py_tp_doc": 7,
        "Py_tp_base": 8,
        "Py_tp_init": 9,
        "Py_tp_new": 10,
        "Py_tp_iter": 11,
        "Py_tp_iternext": 12,
        "Py_tp_descr_get": 13,
        "Py_tp_descr_set": 14,
        "Py_tp_richcompare": 15,
    }

    print(f"  {'Slot':>20} {'ID':>5}")
    print(f"  {'-'*20} {'-'*5}")
    for name, slot_id in slots.items():
        print(f"  {name:>20} {slot_id:>5}")

    print()
    print("  A PyType_Spec with typed slots looks like:")
    print("""
    static PyType_Slot MyType_slots[] = {
        {Py_tp_init, MyType_init},
        {Py_tp_new,  MyType_new},
        {Py_tp_str,  MyType_str},
        {Py_tp_repr, MyType_repr},
        {0, NULL},  // sentinel
    };

    static PyType_Spec MyType_spec = {
        .name = "mymodule.MyType",
        .basicsize = sizeof(MyTypeObject),
        .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        .slots = MyType_slots,
    };

    // Later:
    PyObject *type = PyType_FromSpec(&MyType_spec);
    """)
    print()

simulate_type_spec()


# ═══════════════════════════════════════════════════════════════════
# 5. Stable ABI (PEP 384)
# ═══════════════════════════════════════════════════════════════════
#
# The stable ABI (limited API) guarantees that a compiled extension
# works across Python 3.x versions without recompilation.
#
# Key:
#   - Py_LIMITED_API macro
#   - Limited set of functions (python3.dll, not python3XX.dll)
#   - Notable exclusions: type struct member access, some inlines
#   - PyType_FromSpec is part of the stable ABI
#   - PyMemberDef access is NOT part of the stable ABI

def check_stable_abi():
    """Check if limited API is relevant."""
    print("=== Stable ABI (PEP 384) ===")

    # In Python 3.13+, we can check the ABI flags
    print(f"  Python version: {sys.version}")
    abiflags = getattr(sys, 'abiflags', '')
    print(f"  ABI flags:      {abiflags}")

    # Abi tag
    from sysconfig import get_config_var
    abi_tag = get_config_var('EXT_SUFFIX')
    print(f"  Extension suffix: {abi_tag}")
    print(f"  (Stable ABI uses '.abi3' suffix to work across versions)")
    print()

check_stable_abi()


# ═══════════════════════════════════════════════════════════════════
# 6. Building Extensions — setuptools vs Meson
# ═══════════════════════════════════════════════════════════════════
#
# Traditional (setuptools/distutils) — setup.py:
#
#   from setuptools import setup, Extension
#   module = Extension('mymodule', sources=['mymodule.c'])
#   setup(name='mymodule', ext_modules=[module])
#
# Modern (Meson, Python 3.12+) — meson.build:
#
#   project('mymodule', 'c')
#   py = import('python')
#   py.extension_module('mymodule', 'mymodule.c')
#
# For pure Python modules, there's also:
#   from setuptools import setup
#   setup(name='mymodule', py_modules=['mymodule'])
#

def simulate_build_config():
    """Show extension build configurations."""
    print("=== Extension build configurations ===")

    setup_py_template = """
from setuptools import setup, Extension

module = Extension(
    'mymodule',
    sources=['mymodule.c'],
    include_dirs=['/path/to/python/Include'],
    libraries=['python3'],
    define_macros=[('Py_LIMITED_API', '0x030C0000')],  # stable ABI
)

setup(
    name='mymodule',
    version='1.0',
    description='A minimal C extension',
    ext_modules=[module],
)
"""

    meson_template = """
project('mymodule', 'c', version: '1.0')

py = import('python').find_installation()
py.extension_module(
    'mymodule',
    'mymodule.c',
    subdir: 'mymodule',
    install: true,
)
"""

    print("  setup.py (setuptools):")
    for line in setup_py_template.strip().split("\n"):
        print(f"    {line}")
    print()
    print("  meson.build (Meson, Python 3.12+):")
    for line in meson_template.strip().split("\n"):
        print(f"    {line}")
    print()

simulate_build_config()


# ═══════════════════════════════════════════════════════════════════
# 7. Cython — Writing Extensions in Python-like Language
# ═══════════════════════════════════════════════════════════════════
#
# Cython compiles a Python-like language to C and then builds a
# C extension. Key features:
#   - Static type declarations (cdef, cpdef)
#   - Direct C function calls
#   - Automatic refcounting
#   - Parallel loop support (prange)

def demo_cython_syntax():
    """Show Cython syntax equivalents."""
    print("=== Cython syntax ===")

    cython_code = """
# mymodule.pyx

# C-level integer (stack-allocated, no Python overhead)
cdef int add_c(int a, int b):
    return a + b

# Python-callable wrapper
def add(a, b):
    cdef int result
    result = add_c(a, b)
    return result

# Fast loop with typed variable
def sum_range(int n):
    cdef int i
    cdef long total = 0
    for i in range(n):
        total += i
    return total

# C-level class declaration
cdef class Point:
    cdef public int x, y

    def __init__(self, int x, int y):
        self.x = x
        self.y = y

    cpdef double distance(self):
        # cpdef = C + Python callable
        return (self.x ** 2 + self.y ** 2) ** 0.5
"""

    for line in cython_code.strip().split("\n"):
        print(f"    {line}")
    print()

demo_cython_syntax()


# ═══════════════════════════════════════════════════════════════════
# 8. Embedding Python in C Applications
# ═══════════════════════════════════════════════════════════════════
#
# Embedding: a C program that starts Python and calls Python code.
#
#   #include <Python.h>
#
#   int main(int argc, char *argv[]) {
#       Py_Initialize();              // Start interpreter
#       PyRun_SimpleString(
#           "print('Hello from embedded Python!')"
#       );                            // Run Python code
#       Py_Finalize();                // Shut down
#       return 0;
#   }
#
# More advanced: PyImport_ImportModule, PyObject_CallMethod, etc.
#
# Embedding uses the same API as extensions but from the C side.
# Key functions:
#   Py_Initialize()           — Start interpreter
#   Py_InitializeFromConfig() — Start with config (3.10+)
#   PyRun_SimpleString()      — Run a string of Python
#   PyRun_SimpleFile()        — Run a .py file
#   PyImport_ImportModule()   — Import a Python module
#   PyObject_CallMethod()     — Call a Python method
#   Py_Finalize()             — Shut down interpreter

def simulate_embedding():
    """Simulate embedding Python in C."""
    print("=== Embedding Python in C ===")
    print()
    print("  C code: compile with -I<include> -L<lib> -lpython3.XX")
    print()
    print("""
    #include <Python.h>

    int main(int argc, char *argv[]) {
        // Setup
        Py_Initialize();

        // Run simple Python
        PyRun_SimpleString("import sys");
        PyRun_SimpleString("print(f'Python {sys.version}')");

        // Import a module and call a function
        PyObject *mod = PyImport_ImportModule("json");
        if (mod) {
            PyObject *result = PyObject_CallMethod(
                mod, "dumps", "(O)", PyDict_New()
            );
            if (result) {
                printf("JSON: %s\\n", PyUnicode_AsUTF8(result));
                Py_DECREF(result);
            }
            Py_DECREF(mod);
        }

        // Cleanup
        if (Py_FinalizeEx() < 0) {
            return 120;  // error
        }
        return 0;
    }
    """)
    print()

    # Actually demonstrate embedding-like behavior in Python
    print("  Embedded Python execution (simulated):")
    ns = {}
    exec("import sys; print(f'  Python {sys.version}')", ns)
    import json
    result = json.dumps({"hello": "from embedded Python"})
    print(f"  JSON: {result}")
    print()

simulate_embedding()


# ═══════════════════════════════════════════════════════════════════
# 9. Reference Counting Discipline in Extensions
# ═══════════════════════════════════════════════════════════════════
#
# Rules for C extensions:
#
#   1. Functions that return a new reference:
#      - Py_BuildValue, PyLong_FromLong, PyUnicode_FromString
#      - Caller owns the reference → must Py_DECREF when done
#
#   2. Functions that return a borrowed reference:
#      - PyList_GetItem, PyTuple_GetItem, PyDict_GetItem
#      - Caller does NOT own → must Py_INCREF before storing
#
#   3. PyArg_ParseTuple with "O" → borrowed reference
#   4. PyArg_ParseTuple with "O!" → borrowed reference (with type check)
#   5. PyArg_ParseTuple with "N" → NEW reference (steals ownership)
#
#   6. Return NULL to indicate an exception was set
#   7. Use PyErr_SetString, PyErr_Format, or PyErr_SetObject
#   8. Always check for NULL returns from API functions

def simulate_refcounting():
    """Simulate refcounting rules for C extensions."""
    print("=== Reference counting rules for extensions ===")

    scenarios = [
        ("PyLong_FromLong(42)", "New reference -> caller must Py_DECREF"),
        ("PyUnicode_FromString('hello')", "New reference -> caller must Py_DECREF"),
        ("PyList_GetItem(list, 0)", "Borrowed reference -> Py_INCREF before storing"),
        ("PyTuple_GetItem(tuple, 0)", "Borrowed reference -> Py_INCREF before storing"),
        ("PyDict_GetItem(dict, key)", "Borrowed reference -> Py_INCREF before storing"),
        ("Py_BuildValue('(i)', 42)", "New reference -> caller must Py_DECREF"),
        ("PyArg_ParseTuple('O')", "Borrowed reference (args)"),
        ("PyArg_ParseTuple('N')", "Steals reference -> caller relinquishes"),
        ("PyObject_GetAttr(obj, name)", "New reference -> caller must Py_DECREF"),
        ("PyObject_Str(obj)", "New reference -> caller must Py_DECREF"),
        ("sys.getrefcount(obj)", "Returns Py_ssize_t (always new, borrowed in C)"),
    ]

    print(f"  {'API call':>40} {'Discipline':>40}")
    print(f"  {'-'*40} {'-'*40}")
    for api, rule in scenarios:
        print(f"  {api:>40} {rule:>40}")
    print()

simulate_refcounting()


# ═══════════════════════════════════════════════════════════════════
# 10. Building the Simplest Possible C Extension
# ═══════════════════════════════════════════════════════════════════
#
# A minimal C extension source file:
#
#   // myhello.c
#   #define PY_SSIZE_T_CLEAN
#   #include <Python.h>
#
#   static PyObject* hello(PyObject *self, PyObject *args) {
#       const char *name;
#       if (!PyArg_ParseTuple(args, "s", &name))
#           return NULL;
#       printf("Hello, %s!\n", name);
#       Py_RETURN_NONE;
#   }
#
#   static PyMethodDef HelloMethods[] = {
#       {"hello", hello, METH_VARARGS, "Say hello"},
#       {NULL, NULL, 0, NULL}
#   };
#
#   static struct PyModuleDef hello_module = {
#       PyModuleDef_HEAD_INIT,
#       "myhello",
#       "A minimal hello module",
#       -1,
#       HelloMethods
#   };
#
#   PyMODINIT_FUNC PyInit_myhello(void) {
#       return PyModule_Create(&hello_module);
#   }

# Let's simulate loading our own "extension" using pure Python
# to show what the API would look like if built and imported:

def simulate_c_extension():
    """Simulate importing a C extension."""
    print("=== Simulated C extension usage ===")

    # In real usage, after `pip install .` or `python setup.py build_ext`:
    #   import myhello
    #   myhello.hello("world")

    # Since we can't compile a real C extension here, we simulate
    # the same behavior with a Python module that mirrors the API:
    class SimulatedCModule:
        @staticmethod
        def hello(name: str) -> None:
            """Say hello from C. (Simulated)"""
            print(f"    Hello, {name}!")
            return None

        @staticmethod
        def add(a: int, b: int) -> int:
            """Add two integers. (Simulated C implementation)"""
            return a + b

        @staticmethod
        def sum_ints(n: int) -> int:
            """Fast sum loop (like a C-level for loop)."""
            total = 0
            for i in range(n):
                total += i
            return total

    mod = SimulatedCModule()
    print("  >>> import myhello")
    print("  >>> myhello.hello('world')")
    mod.hello("world")
    print("  >>> myhello.add(40, 2)")
    print(f"  {mod.add(40, 2)}")
    print("  >>> myhello.sum_ints(100000)")
    print(f"  {mod.sum_ints(100000)}")
    print()

    # Compare Python vs C speed (simulated)
    print("  Performance comparison (add, 1M iterations):")
    import time

    def py_add(a, b):
        return a + b

    start = time.perf_counter()
    for _ in range(1_000_000):
        _ = py_add(1, 2)
    py_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(1_000_000):
        _ = mod.add(1, 2)  # "C" call
    c_time = time.perf_counter() - start

    print(f"    Python: {py_time:.4f}s")
    print(f"    C (sim): {c_time:.4f}s")
    print(f"    Note: In a real C extension, the C call would be")
    print(f"    10-100x faster due to no interpreter overhead.")
    print()

simulate_c_extension()


# ═══════════════════════════════════════════════════════════════════
# 11. Implementation Comparison: CPython vs PyPy vs Cython
# ═══════════════════════════════════════════════════════════════════
#
# CPython: Reference implementation. JIT-free. GIL. C extensions.
# PyPy:    JIT compiler. ~4x faster for pure Python. No GIL (STM).
#           Limited C extension support (CPyExt).
# Cython:  Python-to-C compiler. Static typing. Super-fast loops.
#           Used for wrapping C libraries.
# Numba:   JIT for numerical code. LLVM backend. Best for numpy.

def implementation_comparison():
    """Compare Python implementations."""
    print("=== Python implementation comparison ===")

    # Our implementation
    impl = sys.implementation
    print(f"  Current: {impl.name} {impl.version}")

    # Key differences
    comparisons = [
        ("Compiler", "Bytecode interpreter", "JIT (tracing)"),
        ("Speed", "Baseline", "~4x faster"),
        ("GIL", "Yes (optional in 3.13+)", "No (STM)"),
        ("C Extensions", "Native support", "Limited (CPyExt)"),
        ("Memory", "Moderate", "Higher (~2x)"),
        ("Startup", "Fast", "Slower (JIT warmup)"),
        ("GC", "Refcount + generational", "Generational only"),
        ("Best for", "Compatibility, C extensions", "Pure Python CPU-bound"),
    ]

    print()
    print(f"  {'Aspect':>20} {'CPython':>20} {'PyPy':>20}")
    print(f"  {'-'*20} {'-'*20} {'-'*20}")
    for aspect, cp, pypy in comparisons:
        print(f"  {aspect:>20} {cp:>20} {pypy:>20}")
    print()

implementation_comparison()


# ═══════════════════════════════════════════════════════════════════
# 12. Full C Extension Lifecycle
# ═══════════════════════════════════════════════════════════════════
#
# 1. Write the .c file (PyModuleDef, PyMethodDef, functions)
# 2. Write setup.py or meson.build
# 3. Build: pip install .  or  python setup.py build_ext --inplace
# 4. Load: import mymodule → calls PyInit_mymodule
# 5. Use: mymodule.hello("world")
# 6. On interpreter shutdown: tp_dealloc, module cleanup

def extension_lifecycle():
    """Show the full C extension lifecycle."""
    print("=== C Extension Lifecycle ===")
    print()

    phases = [
        ("1. Write Source",
         "mymodule.c with PyModuleDef, PyMethodDef, and functions"),
        ("2. Setup Script",
         "setup.py with Extension('mymodule', sources=['mymodule.c'])"),
        ("3. Build",
         "python setup.py build_ext --inplace  ->  mymodule.cp3XX-win_amd64.pyd"),
        ("4. Import",
         "import mymodule  ->  Python calls PyInit_mymodule()"),
        ("5. Usage",
         "mymodule.hello('world')  ->  calls C hello() via METH_VARARGS"),
        ("6. Cleanup",
         "Py_Finalize() -> calls tp_dealloc, Py_DECREF module"),
    ]

    for phase, desc in phases:
        print(f"  {phase}")
        print(f"    {desc}")
        print()


extension_lifecycle()


# ═══════════════════════════════════════════════════════════════════
# 13. Key Insights (memories for Jafar)
# ═══════════════════════════════════════════════════════════════════
#
# 1. C extension pattern: PyMethodDef array → PyModuleDef →
#    PyInit_<name> init function. PyModuleDef_HEAD_INIT starts the
#    module definition struct.
#
# 2. Method flags: METH_VARARGS (positional tuple), METH_KEYWORDS
#    (adds keyword dict), METH_NOARGS (no args), METH_O (single arg),
#    METH_FASTCALL (C array + nargs, Python 3.7+).
#
# 3. PyArg_ParseTuple format codes: "s" (str), "i" (int), "d" (double),
#    "O" (PyObject*, borrowed), "|" (optional), ":" (error format).
#    Returns 0 on failure, 1 on success (with exception set).
#
# 4. Reference counting rules: new refs (PyLong_FromLong, etc.) require
#    Py_DECREF by caller. Borrowed refs (PyList_GetItem, etc.) require
#    Py_INCREF before storing. Return NULL to signal exception.
#
# 5. PyType_Spec + PyType_FromSpec (Python 3.2+) is the modern way to
#    create types in C extension modules. Uses typed slots array
#    instead of directly filling PyTypeObject fields.
#
# 6. Stable ABI (PEP 384): compile with Py_LIMITED_API to get .abi3
#    suffix. Works across Python 3.x versions without recompile.
#    Some API functions are excluded from the limited API.
#
# 7. Build tools: setuptools (legacy), Meson (Python 3.12+, modern).
#    Meson is now the default for CPython itself.
#
# 8. Cython compiles Python-like code to C for fast extensions.
#    Key: cdef (C-level), cpdef (C + Python), typed variables,
#    fast loops, prange (parallel).
#
# 9. Embedding: Py_Initialize → PyRun_SimpleString/PyImport_ImportModule
#    → PyObject_CallMethod → Py_FinalizeEx. Python runs inside a
#    C application (not the other way around).
#
# 10. ctypes and cffi are alternatives to writing C extensions.
#     ctypes is built-in but slower. cffi is faster but third-party.
#     Both avoid C compilation step.
#

if __name__ == "__main__":
    pass
