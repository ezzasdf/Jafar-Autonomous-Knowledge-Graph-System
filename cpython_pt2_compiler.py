"""
CPython Internals — Part 2: The Compiler Pipeline

Covers:
  • Full compile pipeline: source → tokens → AST → symbol table → bytecode
  • tokenize module and how Python breaks source into tokens
  • AST construction and manipulation (ast module)
  • Symbol table (symtable module)
  • bytecode generation (compile builtin + __code__ objects)
  • dis module for disassembly
  • .pyc file format and __pycache__
  • compile() built-in and exec/eval modes
  • CodeObject internals (co_code, co_consts, co_names, lnotab)

Reference:
  - CPython Internals (Anthony Shaw) — Chapters 2–3
  - Python/compile.c, Parser/tokenize.c, Parser/parser.c, Python/symtable.c
  - https://github.com/python/cpython/blob/main/InternalDocs/compiler.md
"""

import ast
import dis
import sys
import symtable
import tokenize
import io
import marshal
import struct
import time

# ═══════════════════════════════════════════════════════════════════
# 1. Tokenization — Source Code → Tokens
# ═══════════════════════════════════════════════════════════════════
#
# The tokenizer (Parser/tokenize.c) reads raw source bytes and emits
# tokens. Each token has a type (NAME, NUMBER, STRING, NEWLINE, etc.),
# a string value, and a start/end position.
#
# In CPython, the C tokenizer is the primary one used by the compiler.
# The Python tokenize module mirrors its behavior for debugging.

def demo_tokenization():
    source = """
def greet(name):
    msg = "Hello, " + name
    return msg
"""

    print("=== Tokenization ===")
    print(f"Source:\n{source}")

    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    for tok in tokens:
        print(f"  {tok.type:>20} {tok.string!r:>15}  line {tok.start[0]}:{tok.start[1]}")
    print()

demo_tokenization()


# ═══════════════════════════════════════════════════════════════════
# 2. Parsing — Tokens → AST
# ═══════════════════════════════════════════════════════════════════
#
# CPython uses a PEG parser (Parser/parser.c) since Python 3.9.
# The grammar is defined in Grammar/python.gram and compiled into
# parser tables by Tools/peg_generator/pegen.
#
# The parser produces a Concrete Syntax Tree (CST), then Python/ast.c
# converts it to an Abstract Syntax Tree (AST).
#
# The `ast` module lets us inspect and even *modify* ASTs programmatically.

def demo_ast():
    source = "def greet(name):\n    return f'Hello, {name}!'"
    tree = ast.parse(source)

    print("=== AST (dumped) ===")
    print(ast.dump(tree, indent=2))
    print()

    # Walk the AST manually
    print("=== AST node walk ===")
    for node in ast.walk(tree):
        print(f"  {type(node).__name__:>20}", end="")
        if isinstance(node, ast.Name):
            print(f"  id={node.id!r}", end="")
        elif isinstance(node, ast.Constant):
            print(f"  value={node.value!r}", end="")
        elif isinstance(node, ast.FunctionDef):
            print(f"  name={node.name!r}", end="")
        print()
    print()

demo_ast()


# ═══════════════════════════════════════════════════════════════════
# 3. AST Manipulation — Programmatic Code Generation
# ═══════════════════════════════════════════════════════════════════
#
# Because we can inspect the AST, we can also *modify* it or create
# new ASTs from scratch, then compile + execute them.
# This is how Python code generation frameworks work (e.g., Hy, Coconut).

def ast_transform_example():
    # Build an AST for:  lambda x: x + 1
    lambda_node = ast.Lambda(
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="x")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=ast.BinOp(
            left=ast.Name(id="x", ctx=ast.Load()),
            op=ast.Add(),
            right=ast.Constant(value=1),
        ),
    )

    # Compile and execute
    expr_node = ast.Expression(body=lambda_node)
    code = compile(ast.fix_missing_locations(expr_node), "<ast>", "eval")
    func = eval(code)

    print("=== AST-driven code generation ===")
    print(f"  lambda x: x + 1  ->  func(5) = {func(5)}")
    print()

ast_transform_example()


# ═══════════════════════════════════════════════════════════════════
# 4. Symbol Table — Scope Analysis
# ═══════════════════════════════════════════════════════════════════
#
# After parsing, CPython builds a symbol table (Python/symtable.c).
# This pass resolves:
#   - Which names are local vs global vs free vs cell
#   - Nested scopes (closures)
#   - Early detection of syntax errors (e.g., "return" outside function)
#
# The `symtable` module exposes this information.

def demo_symbol_table():
    source = """
x = 10  # global
def outer(a):
    b = 20
    def inner():
        return a + b + x
    return inner()
"""

    table = symtable.symtable(source, "<demo>", "exec")

    def dump_table(tab, indent=0):
        prefix = "  " * indent
        print(f"{prefix}SymbolTable: name={tab.get_name()}, type={tab.get_type()}")
        for sym in tab.get_symbols():
            flags = []
            if sym.is_local():
                flags.append("local")
            if sym.is_global():
                flags.append("global")
            if sym.is_free():
                flags.append("free")
            if sym.is_assigned():
                flags.append("assigned")
            if sym.is_referenced():
                flags.append("referenced")
            if sym.is_parameter():
                flags.append("param")
            print(f"{prefix}  {sym.get_name():>8}: {', '.join(flags)}")
        for child in tab.get_children():
            dump_table(child, indent + 1)

    print("=== Symbol table ===")
    dump_table(table)
    print()

demo_symbol_table()


# ═══════════════════════════════════════════════════════════════════
# 5. Compilation — AST → CodeObject (Bytecode)
# ═══════════════════════════════════════════════════════════════════
#
# The compiler (Python/compile.c) takes the AST and produces a
# CodeObject containing bytecode and metadata.
#
# Key CodeObject fields:
#   co_code      — Raw bytecode (bytes)
#   co_consts    — Tuple of constants used by this code
#   co_names     — Tuple of global names
#   co_varnames  — Tuple of local variable names
#   co_filename  — Source file path
#   co_name      — Function/module name
#   co_argcount  — Number of positional args (excluding *args, **kwargs)
#   co_nlocals   — Number of local variables
#   co_stacksize — Maximum stack depth needed
#   co_flags     — Bitmask (0x02=generator, 0x08=coroutine, etc.)
#   co_lnotab    — Line number table (byte offset → line number)
#   co_freevars  — Free variables (closure variables from outer scope)
#   co_cellvars  — Cell variables (variables referenced by inner scope)
#   co_code      — Raw bytecode bytes
#

def inspect_codeobject_deep():
    def complex_fn(a, b, c):
        total = 0
        for i in range(a):
            total += i * b + c
        return total

    code = complex_fn.__code__

    print("=== Deep CodeObject inspection ===")
    for attr in ["co_argcount", "co_nlocals", "co_stacksize", "co_flags",
                  "co_code", "co_consts", "co_names", "co_varnames",
                  "co_filename", "co_name", "co_lnotab", "co_freevars",
                  "co_cellvars"]:
        val = getattr(code, attr)
        # Truncate long byte strings
        if isinstance(val, bytes) and len(val) > 30:
            val = val[:30] + b"..."
        print(f"  {attr:>15} = {val!r}")
    print()

inspect_codeobject_deep()


# ═══════════════════════════════════════════════════════════════════
# 6. Bytecode Disassembly with `dis`
# ═══════════════════════════════════════════════════════════════════
#
# The `dis` module disassembles bytecode back into human-readable
# instructions. Each instruction is one of:
#   - 2 bytes: opcode + argument (most instructions)
#   - 1 byte:  opcode only (e.g., RETURN_VALUE, NOP)
#
# The eval loop in ceval.c does a giant switch(opcode) on these bytes.

def demo_disassembly():
    def factorial(n):
        if n <= 1:
            return 1
        return n * factorial(n - 1)

    print("=== Bytecode disassembly (factorial) ===")
    dis.dis(factorial)
    print()

    # Show raw bytecode offsets and argument resolution
    bc = dis.Bytecode(factorial)
    print("=== Detailed instruction list ===")
    print(f"  {'offset':>6} {'opname':>20} {'arg':>5} {'argrepr':>20}")
    for instr in bc:
        argrepr = instr.argrepr if instr.argrepr else ""
        print(f"  {instr.offset:>6} {instr.opname:>20} {instr.arg if instr.arg is not None else '':>5} {argrepr:>20}")
    print()

    # Show raw bytes
    print(f"  Raw co_code: {factorial.__code__.co_code.hex()}")
    print()

demo_disassembly()


# ═══════════════════════════════════════════════════════════════════
# 7. Compile Modes — exec, eval, single
# ═══════════════════════════════════════════════════════════════════
#
# The compile() builtin accepts a mode parameter:
#   'exec'   — module-level code (statements, defs, classes)
#   'eval'   — a single expression (returns a value)
#   'single' — interactive statement (like REPL, prints results)

def demo_compile_modes():
    print("=== compile() modes ===")

    # exec mode
    exec_code = compile("x = [1, 2, 3]; y = sum(x)", "<exec>", "exec")
    ns = {}
    exec(exec_code, ns)
    print(f"  exec:   x = {ns['x']}, y = {ns['y']}")

    # eval mode
    eval_code = compile("sum(range(100))", "<eval>", "eval")
    result = eval(eval_code)
    print(f"  eval:   sum(range(100)) = {result}")

    # single mode (like REPL — prints result)
    single_code = compile("42 + 1", "<single>", "single")
    print("  single: ", end="")
    exec(single_code)
    print()

demo_compile_modes()


# ═══════════════════════════════════════════════════════════════════
# 8. .pyc File Format
# ═══════════════════════════════════════════════════════════════════
#
# When you import a module, CPython caches the compiled bytecode in
# __pycache__/<module>.cpython-3XX.pyc
#
# Format (Python 3.3+):
#   4 bytes — magic number (per Python version)
#   4 bytes — flags (bit 0 = source hash, not mtime)
#   4 bytes — source timestamp (or hash if flag set)
#   4 bytes — source size
#   rest    — marshalled CodeObject (marshal format)
#
# The marshal format is a tag-length-value serialization defined in
# Python/marshal.c. Tags include:
#   TYPE_CODE     = 'c' (0x63)    — CodeObject
#   TYPE_STRING   = 's' (0x73)
#   TYPE_TUPLE    = '(' (0x28)
#   TYPE_INT      = 'i' (0x69)
#   TYPE_SMALL_TUPLE = ')' (0x29)
#   TYPE_NONE     = 'N' (0x4e)
#   TYPE_TRUE     = 'T' (0x54)
#   TYPE_FALSE    = 'F' (0x46)
#   TYPE_SHORT_ASCII_INTERNED = 'z' (0x7a)
#

def demo_pyc_format():
    print("=== .pyc file format ===")

    # Get the magic number for this Python version
    import importlib.util
    magic = importlib.util.MAGIC_NUMBER
    print(f"  Magic number:         {magic.hex()} ({sys.version})")

    # Verify: write a minimal .pyc manually
    source = "x = 42"
    code = compile(source, "<demo>", "exec")

    pyc_data = bytearray()
    pyc_data.extend(magic)                 # magic
    pyc_data.extend(struct.pack("<I", 0))  # flags
    pyc_data.extend(struct.pack("<I", 0))  # timestamp
    pyc_data.extend(struct.pack("<I", len(source.encode())))  # source size
    pyc_data.extend(marshal.dumps(code))   # marshalled code object

    print(f"  Source:               {source!r}")
    print(f"  .pyc size:            {len(pyc_data)} bytes")
    print(f"  Marshalled code size: {len(pyc_data) - 16} bytes")
    print()

    # Verify round-trip
    loaded = marshal.loads(pyc_data[16:])  # skip header
    assert type(loaded).__name__ == "code"
    ns = {}
    exec(loaded, ns)
    assert ns["x"] == 42
    print("  Round-trip verified! x =", ns["x"])
    print()

demo_pyc_format()


# ═══════════════════════════════════════════════════════════════════
# 9. Bytecode Optimization — Peephole
# ═══════════════════════════════════════════════════════════════════
#
# CPython's compiler includes a simple peephole optimizer
# (Python/peephole.c in 3.12-, replaced by more advanced optimization
# in Python/flowgraph.c in 3.13+).
#
# Examples of peephole optimizations:
#   - Constant folding:  2 + 3  →  5
#   - Tuple/list unpacking of constants
#   - Removing unreachable code after return/raise
#   - JUMP_ABSOLUTE → NOP elimination

def demo_constant_folding():
    print("=== Constant folding ===")

    def with_folding():
        return 2 + 3 * 4

    def without_folding(a, b, c):
        return a + b * c

    print("  Constant expression (2 + 3 * 4):")
    dis.dis(with_folding)
    print()
    print("  Variable expression (a + b * c):")
    dis.dis(without_folding)
    print()
    print("  Notice: the constant version computes 14 at compile time!")
    print()

demo_constant_folding()


# ═══════════════════════════════════════════════════════════════════
# 10. Co_qualname and Nested Functions
# ═══════════════════════════════════════════════════════════════════
#
# Nested functions create separate CodeObjects linked via co_consts.
# The inner function's code object is stored as a constant in the outer.

def demo_nested_codeobjects():
    print("=== Nested function CodeObjects ===")

    def outer(x):
        def inner(y):
            return x + y
        return inner(10)

    outer_code = outer.__code__
    print(f"  outer co_consts: {outer_code.co_consts}")

    # Find the inner code object in co_consts
    for const in outer_code.co_consts:
        if hasattr(const, 'co_code'):  # it's a code object
            print(f"  inner code object found!")
            print(f"    inner co_name:     {const.co_name}")
            print(f"    inner co_varnames: {const.co_varnames}")
            print(f"    inner co_freevars: {const.co_freevars}")
            print()

            print("  inner disassembly:")
            dis.dis(const)
    print()

demo_nested_codeobjects()


# ═══════════════════════════════════════════════════════════════════
# 11. Line Number Table (co_lnotab)
# ═══════════════════════════════════════════════════════════════════
#
# co_lnotab maps bytecode offsets to source line numbers.
# Format: pairs of (byte_increment, line_increment).
# Byte offsets and line numbers are computed cumulatively.
#
# In Python 3.10+, co_lnotab is superseded by co_linetable
# which uses a more compact encoding.

def demo_lnotab():
    def multi_line(a, b, c):
        x = a + b
        y = x * c
        z = y - a
        return z

    code = multi_line.__code__
    print("=== Line number table ===")

    # co_lnotab (pre-3.10) or co_linetable (3.10+)
    if hasattr(code, 'co_linetable'):
        print(f"  co_linetable (3.10+ format): {code.co_linetable[:40]!r}...")
    if hasattr(code, 'co_lnotab'):
        print(f"  co_lnotab (pre-3.10 format): {code.co_lnotab!r}")

    # Use dis to show lines
    print()
    print("  Line mapping:")
    bc = dis.Bytecode(multi_line)
    for instr in bc:
        lineno = instr.starts_line if instr.starts_line else ''
        print(f"    line {str(lineno):>4}  offset {instr.offset:>4}  {instr.opname}")
    print()

demo_lnotab()


# ═══════════════════════════════════════════════════════════════════
# 12. Compiler Flags
# ═══════════════════════════════════════════════════════════════════
#
# co_flags is a bitmask. Key bits:
#   0x0001  CO_OPTIMIZED     — uses LOAD_FAST/STORE_FAST
#   0x0002  CO_NEWLOCALS     — has a local scope (functions)
#   0x0004  CO_VARARGS       — has *args
#   0x0008  CO_VARKEYWORDS   — has **kwargs
#   0x0010  CO_NESTED        — is a nested function
#   0x0020  CO_GENERATOR     — is a generator
#   0x0040  CO_NOFREE        — no free/cell variables
#   0x0080  CO_COROUTINE     — is a coroutine (async def)
#   0x0100  CO_ITERABLE_COROUTINE
#   0x0200  CO_ASYNC_GENERATOR
#

def demo_code_flags():
    print("=== co_flags breakdown ===")

    def plain():
        pass

    def with_args(*args, **kwargs):
        pass

    async def coro():
        pass

    def gen():
        yield 1

    for name, fn in [("plain", plain), ("*args/**kwargs", with_args),
                      ("coroutine", coro), ("generator", gen)]:
        flags = fn.__code__.co_flags
        bits = []
        if flags & 0x0001: bits.append("OPTIMIZED")
        if flags & 0x0002: bits.append("NEWLOCALS")
        if flags & 0x0004: bits.append("VARARGS")
        if flags & 0x0008: bits.append("VARKEYWORDS")
        if flags & 0x0010: bits.append("NESTED")
        if flags & 0x0020: bits.append("GENERATOR")
        if flags & 0x0080: bits.append("COROUTINE")
        print(f"  {name:>20}: 0x{flags:04x}  {'|'.join(bits)}")
    print()

demo_code_flags()


# ═══════════════════════════════════════════════════════════════════
# 13. The compileall & __pycache__ in action
# ═══════════════════════════════════════════════════════════════════
#
# When importlib loads a module, it:
#   1. Looks for .pyc in __pycache__ matching source mtime
#   2. If found and valid, loads the marshalled CodeObject
#   3. Otherwise, compiles source and writes new .pyc

def demo_import_bytecode_cache():
    import importlib, importlib.util, importlib._bootstrap_external

    print("=== Import cache mechanics ===")
    spec = importlib.util.find_spec("json")
    if spec and spec.origin:
        print(f"  json origin:  {spec.origin}")
        print(f"  json cached:  {spec.cached}")
    else:
        print("  (json spec not available)")

    # Show the __pycache__ for the current module
    import os
    pycache = os.path.join(os.path.dirname(__file__) or ".", "__pycache__")
    if os.path.isdir(pycache):
        pyc_files = [f for f in os.listdir(pycache) if f.endswith(".pyc")]
        print(f"  __pycache__ files: {pyc_files[:5]}")
    else:
        print(f"  (no __pycache__ at {pycache})")
    print()

demo_import_bytecode_cache()


# ═══════════════════════════════════════════════════════════════════
# 14. Full Pipeline Demonstration
# ═══════════════════════════════════════════════════════════════════
#
# Trace a single piece of code through the entire pipeline

def trace_full_pipeline():
    source = """def square(n):
    return n * n
"""

    print("=" * 60)
    print("FULL COMPILER PIPELINE TRACE")
    print("=" * 60)
    print()

    # Step 1: Tokenization
    print("--- 1. Tokenization ---")
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    for tok in tokens:
        print(f"  {tok.start}–{tok.end}  {tokenize.tok_name[tok.type]:>10}  {tok.string!r}")
    print()

    # Step 2: AST
    print("--- 2. Abstract Syntax Tree ---")
    tree = ast.parse(source)
    print(ast.dump(tree, indent=2))
    print()

    # Step 3: Symbol table
    print("--- 3. Symbol Table ---")
    table = symtable.symtable(source, "<trace>", "exec")
    for sym in table.get_symbols():
        print(f"  {sym.get_name()}: local={sym.is_local()} assigned={sym.is_assigned()}")
    for child in table.get_children():
        for sym in child.get_symbols():
            print(f"  [nested] {sym.get_name()}: local={sym.is_local()} param={sym.is_parameter()}")
    print()

    # Step 4: Compilation
    print("--- 4. Compilation → Bytecode ---")
    code = compile(source, "<trace>", "exec")

    # Find the inner function's code object
    for const in code.co_consts:
        if hasattr(const, 'co_code'):
            dis.dis(const)
            break
    print()

    # Step 5: Execution
    print("--- 5. Execution ---")
    ns = {}
    exec(code, ns)
    result = ns["square"](5)
    print(f"  square(5) = {result}")
    print()

trace_full_pipeline()


# ═══════════════════════════════════════════════════════════════════
# 15. Key Insights (memories for Jafar)
# ═══════════════════════════════════════════════════════════════════
#
# 1. The 4-stage compile pipeline: tokenize → parse → symtable → compile.
#    Each stage has its own C file (tokenize.c, parser.c, symtable.c, compile.c).
#
# 2. The AST is the bridge between parsing and compilation.
#    ast.parse() shows the tree; ast.dump() serializes it.
#    You can create/modify ASTs programmatically, then compile+exec them.
#
# 3. The symbol table resolves scope. symtable.symtable() shows
#    which names are local/global/free/assigned/referenced per scope.
#    The compiler uses this to decide LOAD_FAST vs LOAD_GLOBAL vs LOAD_DEREF.
#
# 4. CodeObject stores everything: bytecode (co_code), constants (co_consts),
#    names (co_names), locals (co_varnames), line map (co_lnotab/co_linetable).
#    All fields are accessible from Python.
#
# 5. dis shows bytecode as human-readable instructions.
#    Bytecode is opcode + optional arg; the eval loop in ceval.c dispatches.
#
# 6. compile() has three modes: 'exec' (statements), 'eval' (expressions),
#    'single' (interactive REPL). Different modes produce different bytecode.
#
# 7. .pyc files are: 16-byte header + marshalled CodeObject.
#    marshal.dumps(code) serializes; importlib caches in __pycache__.
#
# 8. Nested functions create separate CodeObjects stored as constants
#    in the outer function's co_consts. Free variables (closure) are
#    tracked in co_freevars.
#
# 9. Peephole optimization folds constants (2+3 → 5 at compile time).
#    You can see this by comparing disassembly of constant vs variable expressions.
#
# 10. co_flags is a bitmask encoding features: optimized, generator,
#     coroutine, varargs, nested. Each bit corresponds to a CO_* flag.
#

if __name__ == "__main__":
    pass
