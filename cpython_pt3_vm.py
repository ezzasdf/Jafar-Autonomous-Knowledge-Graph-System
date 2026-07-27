"""
CPython Internals — Part 3: The Evaluation Loop (ceval.c)

Covers:
  • ceval.c main loop architecture — the "big switch" dispatch
  • Frame objects and the frame stack
  • Stack-based execution model
  • Key opcodes: LOAD_FAST, LOAD_CONST, BINARY_OP, CALL, RETURN
  • Function call protocol (CALL_FUNCTION → _PyEval_EvalFrame)
  • Generator and coroutine execution (YIELD_VALUE, SEND, RESUME)
  • Exception handling bytecode (SETUP_FINALLY, PUSH_EXC_INFO, etc.)
  • sys._getframe() for runtime frame inspection
  • Tracing and profiling (sys.settrace, sys.setprofile)

Reference:
  - CPython Internals (Anthony Shaw) — Chapter 3
  - Python/ceval.c (the heart of the VM)
  - Include/frameobject.h, Include/cpython/frameobject.h
  - https://github.com/python/cpython/blob/main/InternalDocs/eval.md
"""

import dis
import sys
import types


# ═══════════════════════════════════════════════════════════════════
# 1. Frame Objects — The Execution Context
# ═══════════════════════════════════════════════════════════════════
#
# A frame (frameobject.h) represents one execution scope:
#   - f_locals     — local namespace (dict)
#   - f_globals    — global namespace (dict)
#   - f_builtins   — builtins namespace
#   - f_code       — the CodeObject being executed
#   - f_lineno     — current source line number
#   - f_lasti      — last bytecode instruction index
#   - f_back       — previous frame (caller)
#   - f_valuestack — the evaluation stack
#   - f_stackdepth — current stack depth
#   - f_state      — frame state (running, suspended, etc.)
#
# The eval loop in ceval.c pushes a new frame for each function call
# and pops it on return.

def inspect_current_frame():
    def inner():
        return sys._getframe()

    def outer():
        return inner()

    frame = outer()
    print("=== Current frame (inspect frame chain) ===")
    f = frame
    depth = 0
    while f:
        print(f"  depth {depth}: {f.f_code.co_name} at {os.path.basename(f.f_code.co_filename)}:{f.f_lineno}")
        print(f"    f_locals keys:  {list(f.f_locals.keys())}")
        print(f"    f_lasti:        {f.f_lasti}")
        f = f.f_back
        depth += 1
    print()

import os
inspect_current_frame()


# ═══════════════════════════════════════════════════════════════════
# 2. Simulating the Stack Machine
# ═══════════════════════════════════════════════════════════════════
#
# CPython is a stack-based VM. Most instructions push/pop from the
# evaluation stack. For example:
#
#   LOAD_CONST 1     # push co_consts[1] onto stack
#   LOAD_FAST 0      # push local var 0 onto stack
#   BINARY_OP +      # pop two, add, push result
#   RETURN_VALUE     # pop one, return it
#
# Let's simulate this for a simple expression.

class SimpleVM:
    """Minimal simulation of CPython's stack machine."""

    def __init__(self, code):
        self.code = code
        self.stack = []
        self.locals = {}
        self.globals = {}
        self.builtins = __builtins__

    def LOAD_CONST(self, arg):
        self.stack.append(self.code.co_consts[arg])

    def LOAD_FAST(self, arg):
        name = self.code.co_varnames[arg]
        self.stack.append(self.locals[name])

    def LOAD_FAST_LOAD_FAST(self, arg):
        name0 = self.code.co_varnames[arg >> 4]
        name1 = self.code.co_varnames[arg & 0xF]
        self.stack.append(self.locals[name0])
        self.stack.append(self.locals[name1])

    def STORE_FAST(self, arg):
        name = self.code.co_varnames[arg]
        self.locals[name] = self.stack.pop()

    def LOAD_GLOBAL(self, arg):
        name = self.code.co_names[arg]
        self.stack.append(self.globals.get(name, getattr(self.builtins, name, None)))

    def RESUME(self, arg=None):
        pass

    def BINARY_OP(self, op):
        right = self.stack.pop()
        left = self.stack.pop()
        ops = {0: lambda a, b: a + b, 1: lambda a, b: a - b,
               2: lambda a, b: a * b, 3: lambda a, b: a / b}
        self.stack.append(ops.get(op, lambda a, b: a + b)(left, right))

    def RETURN_VALUE(self):
        pass  # value stays on stack; run() pops it at the end

    def run(self):
        bc = dis.Bytecode(self.code)
        for instr in bc:
            method = getattr(self, instr.opname, None)
            if method:
                if instr.arg is not None:
                    method(instr.arg)
                else:
                    method()
        return self.stack.pop() if self.stack else None

def demo_simple_vm():
    print("=== Simple VM simulation ===")

    # Add 2 numbers
    src = "lambda x, y: x + y"
    code = compile(src, "<sim>", "eval")

    # Extract the inner code object
    inner_code = code.co_consts[0]  # the lambda's code
    vm = SimpleVM(inner_code)
    vm.locals = {"x": 10, "y": 20}
    vm.globals = {}

    print(f"  Source: {src}")
    print(f"  Bytecode:")
    dis.dis(inner_code)
    result = vm.run()
    print(f"  Result: {result}")
    print()

demo_simple_vm()


# ═══════════════════════════════════════════════════════════════════
# 3. Function Call Protocol
# ═══════════════════════════════════════════════════════════════════
#
# When Python calls a function, the bytecode sequence is:
#
#   LOAD_GLOBAL  fn     # push function object
#   LOAD_CONST   arg1   # push args
#   ...
#   CALL_FUNCTION nargs  # pop nargs + function, call, push result
#
# CALL_FUNCTION in ceval.c:
#   1. Reads nargs from oparg
#   2. Pops function, args from stack
#   3. Creates a new frame (or uses _PyEval_EvalFrame)
#   4. Pushes return value (or raises exception)
#
# In CPython 3.11+, CALL_FUNCTION was replaced by PRECALL + CALL opcodes
# for better specialization (inline caching).

def inspect_function_call():
    print("=== Function call bytecode ===")

    def caller():
        def callee(a, b):
            return a + b
        return callee(10, 20)

    print("  caller bytecode:")
    dis.dis(caller)
    print()

    # Trace through the call using sys.settrace
    trace_log = []

    def trace_calls(frame, event, arg):
        if event == 'call':
            trace_log.append(f"CALL   {frame.f_code.co_name}")
            return trace_lines
        return trace_lines

    def trace_lines(frame, event, arg):
        if event == 'line':
            pass  # would log every line
        elif event == 'return':
            trace_log.append(f"RETURN {frame.f_code.co_name} → {arg!r}")
        elif event == 'exception':
            trace_log.append(f"EXC    {frame.f_code.co_name}: {arg[1]}")
        return trace_lines

    old_trace = sys.settrace(trace_calls)
    try:
        result = caller()
    finally:
        sys.settrace(old_trace)

    print("  Trace log:")
    for entry in trace_log:
        print(f"    {entry}")
    print(f"  Result: {result}")
    print()

inspect_function_call()


# ═══════════════════════════════════════════════════════════════════
# 4. Generator Execution — Suspend/Resume
# ═══════════════════════════════════════════════════════════════════
#
# Generators are special: they suspend execution with YIELD_VALUE
# and resume with SEND (or by calling next()).
#
# Frame state for generators:
#   - Frame is NOT popped on yield; it stays on the frame stack
#   - f_state is set to FRAME_SUSPENDED (since 3.11)
#   - next() / send() resume at the instruction after YIELD_VALUE
#
# Internally (ceval.c):
#   YIELD_VALUE:
#       - Pops the top-of-stack as the yield value
#       - Returns to caller (but frame stays alive)
#   RESUME (Python 3.10+):
#       - Restores the generator frame
#       - Continues execution from f_lasti + 2
#

def inspect_generator_state():
    print("=== Generator frame inspection ===")

    def gen():
        yield 1
        yield 2
        yield 3

    g = gen()
    print(f"  After creation: gi_frame = {g.gi_frame is not None}")

    val = next(g)
    print(f"  After next(g):  value={val}, gi_frame = {g.gi_frame is not None}")
    if g.gi_frame:
        print(f"    f_lasti = {g.gi_frame.f_lasti}")
        print(f"    f_lineno = {g.gi_frame.f_lineno}")

    val = next(g)
    print(f"  After 2nd next: value={val}, gi_frame = {g.gi_frame is not None}")

    val = next(g)
    print(f"  After 3rd next: value={val}")

    try:
        next(g)
    except StopIteration:
        print(f"  After exhaustion: gi_frame = {g.gi_frame is not None}")
    print()

import os
inspect_generator_state()


# ═══════════════════════════════════════════════════════════════════
# 5. Exception Handling Bytecode
# ═══════════════════════════════════════════════════════════════════
#
# Before Python 3.11, try/except used SETUP_FINALLY + POP_BLOCK +
# JUMP_FORWARD in a complex pattern.
#
# Since Python 3.11 (PEP 523), exception handling tables are stored
# separately in co_exceptiontable — the bytecode itself is linearized.
#
# Let's look at the traditional approach:

def inspect_exception_bytecode():
    print("=== Exception handling bytecode ===")

    def try_except(x):
        try:
            result = 10 / x
        except ZeroDivisionError:
            result = -1
        return result

    print("  try/except bytecode:")
    dis.dis(try_except)
    print()

    # Show the exception table (Python 3.11+)
    code = try_except.__code__
    if hasattr(code, 'co_exceptiontable') and code.co_exceptiontable:
        print(f"  co_exceptiontable: {code.co_exceptiontable.hex()}")
    print()

inspect_exception_bytecode()


# ═══════════════════════════════════════════════════════════════════
# 6. sys._getframe() — Runtime Frame Walking
# ═══════════════════════════════════════════════════════════════════
#
# sys._getframe(depth) returns the frame at the given depth.
# Frame objects are linked via f_back, forming a chain.

def frame_walk_example():
    print("=== sys._getframe() frame walking ===")

    def level3():
        f = sys._getframe()
        print("  Walking frame chain from level3:")
        depth = 0
        while f:
            name = f.f_code.co_name
            fname = os.path.basename(f.f_code.co_filename)
            lineno = f.f_lineno
            locals_list = list(f.f_locals.keys())
            print(f"    depth {depth}: {name} at {fname}:{lineno}, locals={locals_list}")
            f = f.f_back
            depth += 1
        return depth

    def level2():
        return level3()

    def level1():
        x = 42
        return level2()

    total_depth = level1()
    print(f"  Total frame depth: {total_depth}")
    print()

frame_walk_example()


# ═══════════════════════════════════════════════════════════════════
# 7. Tracing with sys.settrace
# ═══════════════════════════════════════════════════════════════════
#
# sys.settrace installs a per-thread trace function.
# Events:
#   'call'       — function call (frame, 'call', None)
#   'line'       — new source line (frame, 'line', None)
#   'return'     — function return (frame, 'return', value)
#   'exception'  — exception raised (frame, 'exception', (exc, val, tb))
#   'opcode'     — (3.12+) each bytecode instruction

def trace_example():
    print("=== sys.settrace line tracing ===")

    lines = []

    def my_trace(frame, event, arg):
        if event in ('line', 'call', 'return'):
            name = frame.f_code.co_name
            lineno = frame.f_lineno
            lines.append(f"    {event:>10}  {name}:{lineno}")
        return my_trace

    def fib(n):
        if n <= 1:
            return n
        return fib(n - 1) + fib(n - 2)

    old_trace = sys.settrace(my_trace)
    try:
        result = fib(3)
    finally:
        sys.settrace(old_trace)

    print("  Trace (fib(3)):")
    for line in lines[:20]:
        print(line)
    print(f"  Result: {result}")
    print()

trace_example()


# ═══════════════════════════════════════════════════════════════════
# 8. opcode-level Tracing (Python 3.12+)
# ═══════════════════════════════════════════════════════════════════
#
# Python 3.12 added sys.monitoring (PEP 669) for low-overhead
# bytecode-level tracing. We'll use the traditional approach for
# cross-version compatibility.

def opcode_trace():
    print("=== Per-opcode execution trace ===")
    print("  (using dis.Bytecode to show what each instruction does)")

    def compute(n):
        total = 0
        for i in range(n):
            total += i
        return total

    # Show the full instruction sequence with offsets
    bc = dis.Bytecode(compute)
    print(f"  {'offset':>6} {'opname':>20} {'arg':>5} {'argval':>8}")
    for instr in bc:
        print(f"  {instr.offset:>6} {instr.opname:>20} {instr.arg if instr.arg is not None else '':>5} {str(instr.argval):>8}")
    print()

    # In Python 3.13+ we could use sys.monitoring
    if hasattr(sys, 'monitoring'):
        print("  Python 3.13+ sys.monitoring available!")
        print("  sys.monitoring provides low-overhead bytecode events:")
        print("    - PY_START/PY_RETURN for function calls")
        print("    - LINE for line events")
        print("    - INSTRUCTION for each opcode")
    else:
        print("  (sys.monitoring not available in this Python version)")
    print()

opcode_trace()


# ═══════════════════════════════════════════════════════════════════
# 9. Adaptive Execution (Python 3.11+)
# ═══════════════════════════════════════════════════════════════════
#
# Python 3.11 (PEP 659) introduced "specializing adaptive interpreter."
# Hot bytecode instructions get replaced with faster versions:
#   LOAD_FAST  → LOAD_FAST_CHECK + LOAD_FAST__LOAD_FAST (pair)
#   LOAD_CONST → LOAD_CONST__LOAD_FAST (pair)
#   BINARY_OP  → more specific versions
#
# This is visible via dis output showing "specialized" opcodes.

def check_adaptive_specialization():
    print("=== Adaptive specialization ===")

    # In Python 3.11+, dis shows specialized opcodes
    def hot_loop(n):
        total = 0
        for i in range(n):
            total += i * 2 + 1
        return total

    # Run it a bunch to trigger specialization
    for _ in range(1000):
        hot_loop(100)

    print("  After 1000 iterations (specialized):")
    dis.dis(hot_loop)
    print()

    # Check if we have 3.11+ adaptive instructions
    import opcode
    has_specialized = hasattr(opcode, '_specialized_instructions')
    print(f"  Specialized instructions in opcode: {has_specialized}")
    print()

check_adaptive_specialization()


# ═══════════════════════════════════════════════════════════════════
# 10. Key Insights (memories for Jafar)
# ═══════════════════════════════════════════════════════════════════
#
# 1. The eval loop (ceval.c) is a giant switch(opcode) that reads
#    bytecode from co_code, executing each instruction.
#    Frame objects (frameobject.h) hold the execution context.
#
# 2. CPython is a stack-machine: LOAD_* pushes, STORE_* pops,
#    BINARY_OP pops two and pushes one, RETURN_VALUE pops and returns.
#    Stack depth is precomputed at compile time (co_stacksize).
#
# 3. Function calls push a new frame onto the frame stack.
#    CALL_FUNCTION pops nargs + function, creates frame, runs eval loop.
#    callee's RETURN_VALUE pops the frame and returns to caller.
#
# 4. Generators suspend without destroying the frame.
#    f_state = FRAME_SUSPENDED, and the generator object holds
#    the live frame. next()/send() resume via RESUME opcode.
#
# 5. Exception handling: pre-3.11 uses SETUP_FINALLY + POP_BLOCK
#    bytecode pattern. 3.11+ uses co_exceptiontable for linear bytecode
#    with exception handling metadata stored separately.
#
# 6. sys._getframe(depth) walks the call stack. Each frame is linked
#    via f_back. Useful for debugging, logging, and introspection.
#
# 7. sys.settrace() installs a per-thread callback called on
#    call/line/return/exception events. Used by debuggers (pdb) and
#    coverage tools (coverage.py).
#
# 8. sys.monitoring (Python 3.13+, PEP 669) is a lower-overhead
#    alternative to sys.settrace for bytecode-level events.
#
# 9. Python 3.11+ adaptive specialization (PEP 659) replaces hot
#    bytecode instructions with faster specialized versions.
#    LOAD_FAST, LOAD_CONST, BINARY_OP, CALL all have fast paths.
#

if __name__ == "__main__":
    pass
