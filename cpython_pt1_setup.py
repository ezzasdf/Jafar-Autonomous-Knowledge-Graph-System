"""
CPython Internals — Part 1: Setup, Directory Map, and Execution Pipeline

Covers:
  • CPython source directory layout and what each directory does
  • Building CPython from source (configure/make/pcbuild)
  • Execution pipeline overview (source → tokens → AST → bytecode → runtime)
  • Simple C extension module (the "hello from C" pattern)
  • The Python/C API basics (PyObject, refcount, module init)

Reference:
  - CPython Internals (Anthony Shaw, Real Python 2021) — Chapters 1–2
  - https://realpython.com/cpython-source-code-guide/
  - https://devguide.python.org/getting-started/setup-building/
  - https://github.com/python/cpython/blob/main/InternalDocs/structure.md
"""

# ═══════════════════════════════════════════════════════════════════
# 1. CPython Source Directory Layout
# ═══════════════════════════════════════════════════════════════════
#
# The CPython repository (https://github.com/python/cpython) is organized:
#
#   Lib/              — Pure Python stdlib (*.py files)
#   Modules/          — C extension modules (e.g. _socket, _ssl, _sqlite3)
#   Objects/          — Core object type implementations (int, str, list, dict, etc.)
#   Python/           — Interpreter core: compiler, ceval.c, sysmodule, marshal
#   Parser/           — Tokenizer (tokenize), PEG parser generator, parser
#   Programs/         — Entry points (python.c, pythonw.c)
#   Include/          — C API headers (Include/ and Include/internal/)
#   PCbuild/          — Visual Studio project files for Windows build
#   Doc/              — Sphinx documentation source
#   InternalDocs/     — Developer docs for CPython internals
#   Tools/            — Build utilities, JIT, scripts
#   Grammar/          — PEG grammar definition (python.gram)
#
# Key files at a glance:
#
#   Python/ceval.c        — The bytecode evaluation loop (heart of the VM)
#   Python/compile.c      — AST → bytecode compiler
#   Python/symtable.c     — Symbol table pass (scope analysis)
#   Parser/tokenize.c     — Tokenizer (source → tokens)
#   Parser/parser.c       — PEG parser (tokens → CST → AST)
#   Objects/object.c      — PyObject base, refcount, type comparison
#   Objects/typeobject.c  — PyTypeObject (metaclass machinery)
#   Objects/longobject.c  — int type (arbitrary-precision)
#   Objects/unicodeobject.c — str type (Unicode)
#   Objects/listobject.c  — list type
#   Objects/dictobject.c  — dict type
#   Modules/gcmodule.c    — Garbage collector
#   Python/pystate.c      — Interpreter/thread state (GIL, PyGILState)
#   Python/memory.c       — Memory allocators (PyMem_*, PyObject_*)
#

# ═══════════════════════════════════════════════════════════════════
# 2. Building CPython from Source
# ═══════════════════════════════════════════════════════════════════
#
# Unix/macOS:
#   git clone https://github.com/python/cpython
#   cd cpython
#   ./configure --enable-optimizations --with-lto
#   make -j$(nproc)
#   make test            # optional: run the full test suite
#   ./python             # you now have a custom build!
#
# Windows (Visual Studio):
#   git clone https://github.com/python/cpython
#   cd cpython
#   PCbuild\build.bat    # builds with VS 2022
#   PCbuild\amd64\python.exe
#
# Key configure flags:
#   --enable-optimizations   Enable PGO (profile-guided optimization)
#   --with-lto               Enable link-time optimization
#   --with-pydebug           Debug build (extra assertions, Py_DEBUG)
#   --with-trace-refs        Track all refcounts (find leaks)
#

# ═══════════════════════════════════════════════════════════════════
# 3. Execution Pipeline Overview
# ═══════════════════════════════════════════════════════════════════
#
# When you run python foo.py, the pipeline is:
#
#   Source code (.py)
#       │
#       ▼
#   [Tokenization]         Parser/tokenize.c    — break chars into tokens
#       │
#       ▼
#   [PEG Parsing]          Parser/parser.c      — tokens → Concrete Syntax Tree
#       │
#       ▼
#   [AST Construction]     Python/ast.c         — CST → Abstract Syntax Tree
#       │
#       ▼
#   [Symbol Table]         Python/symtable.c    — scope/variable resolution
#       │
#       ▼
#   [Compilation]          Python/compile.c     — AST → CodeObject (bytecode)
#       │
#       ▼
#   [Execution]            Python/ceval.c       — eval loop runs bytecode
#
# A CodeObject contains:
#   - co_code:         Raw bytecode (bytes)
#   - co_consts:       Tuple of constants
#   - co_names:        Tuple of global names
#   - co_varnames:     Tuple of local variable names
#   - co_stacksize:    Maximum stack depth
#   - co_flags:        Flags (generator, coroutine, etc.)
#

# Let's inspect CodeObject attributes on a real function

def inspect_code_object():
    def sample(a, b):
        x = a + b
        y = x * 2
        return y

    code = sample.__code__
    print("=== CodeObject inspection ===")
    print(f"  co_code (raw bytes):      {code.co_code[:20]}...")
    print(f"  co_consts:                {code.co_consts}")
    print(f"  co_names:                 {code.co_names}")
    print(f"  co_varnames:              {code.co_varnames}")
    print(f"  co_stacksize:             {code.co_stacksize}")
    print(f"  co_nlocals:               {code.co_nlocals}")
    print(f"  co_filename:              {code.co_filename}")
    print(f"  co_name:                  {code.co_name}")
    print(f"  co_argcount:              {code.co_argcount}")
    print(f"  co_flags:                 {code.co_flags:#x}")
    print(f"  co_lnotab (line -> byte): {code.co_lnotab!r}")
    print()

inspect_code_object()


# ═══════════════════════════════════════════════════════════════════
# 4. Disassembling Bytecode
# ═══════════════════════════════════════════════════════════════════
#
# The `dis` module lets you see what bytecode your function compiles to.
# Each instruction is one or two bytes: opcode + optional argument.
# The eval loop in ceval.c uses a massive switch(opcode) dispatch.

import dis

def show_bytecode():
    def compute(n):
        total = 0
        for i in range(n):
            total += i * i
        return total

    print("=== Bytecode disassembly ===")
    dis.dis(compute)
    print()

    # Also show the raw bytes with dis.Bytecode
    bc = dis.Bytecode(compute)
    print("=== Bytecode instruction list ===")
    for instr in bc:
        lineno = instr.starts_line if instr.starts_line else ''
        print(f"  {str(lineno):>4}  {instr.offset:>4}  {instr.opname:<20} {instr.argrepr}")
    print()

show_bytecode()


# ═══════════════════════════════════════════════════════════════════
# 5. Python/C API Basics
# ═══════════════════════════════════════════════════════════════════
#
# Everything in Python is a PyObject*. The C API is defined in Include/.
#
#   PyObject (Include/object.h):
#       struct _object {
#           Py_ssize_t ob_refcnt;     // Reference count
#           PyTypeObject *ob_type;    // Pointer to type
#       };
#
#   Every type extends PyObject by putting it as the first field:
#       struct { PyObject_VAR_HEAD; ... }
#
#   Key C API functions:
#       Py_INCREF(op)           — Increment refcount
#       Py_DECREF(op)           — Decrement refcount, free at 0
#       PyObject_TypeCheck(op, type) — Check type (like isinstance)
#       PyArg_ParseTuple(args, fmt, ...) — Parse Python args to C
#       Py_BuildValue(fmt, ...) — Build Python object from C data
#
#   The GIL (Global Interpreter Lock):
#       - Only one thread runs Python bytecode at a time
#       - Released during I/O or in C extensions that call
#         Py_BEGIN_ALLOW_THREADS / Py_END_ALLOW_THREADS
#       - In Python 3.13+: can be disabled via --disable-gil
#         (free-threaded mode, PEP 703)
#

# 5a. Simulating refcount behavior in pure Python

def demo_refcount():
    import sys

    a = "hello"
    print(f"=== Reference count simulation ===")
    print(f"  refcount of 'hello': {sys.getrefcount(a)}")

    b = a
    print(f"  after b = a:         {sys.getrefcount(a)}")

    c = [a]
    print(f"  after c = [a]:       {sys.getrefcount(a)}")

    del b
    print(f"  after del b:         {sys.getrefcount(a)}")
    print()

demo_refcount()


# 5b. Using the marshal module to examine CodeObject serialization
#     (marshal.c is how .pyc files are written)

def demo_marshal_code():
    import marshal, types

    def foo():
        return 42

    # CodeObject serialization format (used by .pyc files)
    serialized = marshal.dumps(foo.__code__)
    restored = marshal.loads(serialized)
    restored_func = types.FunctionType(restored, globals(), "restored_foo")

    print("=== Marshal CodeObject round-trip ===")
    print(f"  original:   foo() = {foo()}")
    print(f"  restored:   {restored_func.__name__}() = {restored_func()}")
    print(f"  bytecode identical: {foo.__code__.co_code == restored.co_code}")
    print()

demo_marshal_code()


# ═══════════════════════════════════════════════════════════════════
# 6. Writing a C Extension Module (simulated in ctypes)
# ═══════════════════════════════════════════════════════════════════
#
# A real C extension looks like:
#
#   // mymodule.c
#   #include <Python.h>
#
#   static PyObject* my_hello(PyObject *self, PyObject *args) {
#       const char *name;
#       if (!PyArg_ParseTuple(args, "s", &name))
#           return NULL;
#       printf("Hello from C, %s!\n", name);
#       Py_RETURN_NONE;
#   }
#
#   static PyMethodDef MyMethods[] = {
#       {"hello", my_hello, METH_VARARGS, "Say hello from C"},
#       {NULL, NULL, 0, NULL}  // sentinel
#   };
#
#   static struct PyModuleDef mymodule = {
#       PyModuleDef_HEAD_INIT, "mymodule", NULL, -1, MyMethods
#   };
#
#   PyMODINIT_FUNC PyInit_mymodule(void) {
#       return PyModule_Create(&mymodule);
#   }
#
# On Windows, you'd also need a .def file or use dllexport.
#
# For now, let's build a minimal equivalent using ctypes to simulate
# calling into a shared library.

def demo_ctypes_module():
    import ctypes, sys

    print("=== Simulated C extension via ctypes ===")

    # On Windows, we can load ucrtbase or kernel32 to show the mechanics
    try:
        libc = ctypes.CDLL("msvcrt.dll")  # Windows
    except OSError:
        try:
            libc = ctypes.CDLL(None)  # macOS/Linux
        except OSError:
            print("  (no system lib available for demo)")
            return

    libc.printf(b"  Hello from 'C' (via ctypes), Jafar!\n")

    # The real Python/C API uses PyArg_ParseTuple with format codes:
    #   "s"   -> const char* (UTF-8)
    #   "i"   -> int
    #   "d"   -> double
    #   "O"   -> PyObject*
    #   "O&"  -> converter function
    #   "|"   -> following args are optional
    #   ":"   -> error message separator
    #   ";"   -> error message (rest of string)
    print()
    print("  PyArg_ParseTuple format codes:")
    for code, meaning in [("s", "const char* (UTF-8)"),
                           ("i", "int"),
                           ("d", "double"),
                           ("O", "PyObject*"),
                           ("O&", "converter function"),
                           ("|", "optional args follow"),
                           (":", "error message separator")]:
        print(f"    {code:<6} {meaning}")
    print()

demo_ctypes_module()


# ═══════════════════════════════════════════════════════════════════
# 7. Key Insights (stored as memories)
# ═══════════════════════════════════════════════════════════════════
#
# These would be stored with confidence=0.80, source="cpython_internals"
#
# 1. CPython source layout: Lib/ (Python stdlib), Objects/ (type impls),
#    Python/ (compiler+ceval), Parser/ (tokenizer+PEG), Programs/ (entry)
#    → Key files: ceval.c (VM loop), compile.c (bytecode gen), object.c
#
# 2. Execution pipeline: source → tokenizer → PEG parser → AST →
#    symbol table → compiler → CodeObject → ceval.c eval loop
#    Each phase lives in a separate C file under Parser/ or Python/
#
# 3. CodeObject fields: co_code (bytes), co_consts, co_names, co_varnames,
#    co_stacksize, co_flags, co_lnotab — all inspectable from Python
#
# 4. dis module reveals bytecode: each instruction is opcode + arg,
#    executed by the eval loop switch in ceval.c
#
# 5. Python/C API: PyObject has ob_refcnt + ob_type; Py_INCREF/Py_DECREF
#    manage lifetimes; PyArg_ParseTuple/BuildValue convert between C and Python
#
# 6. GIL: only one thread runs bytecode at a time; C extensions can release
#    it during blocking operations; Python 3.13+ has free-threaded mode
#
# 7. .pyc files use marshal to serialize CodeObjects; marshal format is
#    a simple tag-length-value format (see Python/marshal.c)
#
# 8. C extensions use PyMethodDef + PyModuleDef + PyInit_<name> pattern;
#    METH_VARARGS is the standard calling convention
#

if __name__ == "__main__":
    pass
