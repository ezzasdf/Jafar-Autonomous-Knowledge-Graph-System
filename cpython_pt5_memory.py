"""
CPython Internals — Part 5: Memory Management and the GIL

Covers:
  • 3-tier memory allocator: arena → pool → block (obmalloc.c)
  • PyMem_* vs PyObject_* allocator families
  • The GIL (Global Interpreter Lock) — ceval_gil.h
  • GIL release/acquire protocol (Py_BEGIN_ALLOW_THREADS)
  • PyThreadState and PyInterpreterState
  • Subinterpreters (PEP 554) — isolated interpreters
  • Free-threaded CPython (PEP 703, Python 3.13+ —disable-gil)
  • Buffer protocol (memoryview, Py_buffer)
  • Weak references (weakrefmodule.c)
  • sys.getsizeof() and object size accounting

Reference:
  - CPython Internals (Anthony Shaw) — Chapter 4 (memory) + Chapter 5 (GIL)
  - Objects/obmalloc.c — The allocator
  - Python/ceval_gil.h — GIL implementation
  - Python/pystate.c — Thread/interpreter state
  - Modules/gcmodule.c — GC (covered in Part 4)
  - Include/object.h — PyMem_* / PyObject_* macros
  - PEP 703 (free-threaded), PEP 554 (subinterpreters)
"""

import sys
import threading
import time
import ctypes
import weakref
import _thread


# ═══════════════════════════════════════════════════════════════════
# 1. The 3-Tier Memory Allocator (obmalloc.c)
# ═══════════════════════════════════════════════════════════════════
#
# CPython's memory allocator (obmalloc.c) sits on top of the system
# malloc (or mmap) and uses three tiers:
#
#   Arena  (256 KB)     — Large regions mapped from the OS
#     Pool (4 KB)       — Divided from arenas, page-aligned
#       Block (8..N)    — Individual allocations (8-byte aligned)
#
# Blocks are grouped by size class:
#   Size class index = (size + 8 - 1) >> 3   (for 8-byte alignment)
#   Each pool holds blocks of exactly one size class.
#
# This avoids:
#   - Fragmentation (same-size blocks are recycled efficiently)
#   - System call overhead (arenas are pooled)
#
# The allocator has 3 API families:
#   PyMem_RawMalloc / PyMem_RawFree      — raw, no GIL needed
#   PyMem_Malloc / PyMem_Free            — Python memory (with GIL)
#   PyObject_Malloc / PyObject_Free       — object memory (fast path)
#
# PyObject_Malloc is the fast path used by PyType_GenericAlloc.
# It uses thread-local free lists for small objects.

def inspect_sizeof():
    print("=== Memory allocator: objects size ===")

    # sys.getsizeof only returns the object's direct memory
    # (does NOT include contained objects' memory)
    objects = [
        ("None", None),
        ("int 0", 0),
        ("int 2**63", 2**63),
        ("float 1.0", 1.0),
        ("bool True", True),
        ("str ''", ""),
        ("str 'hello'", "hello"),
        ("str * 1000", "x" * 1000),
        ("bytes", b"hello"),
        ("tuple ()", ()),
        ("tuple (1,2,3)", (1, 2, 3)),
        ("list []", []),
        ("list [1,2,3]", [1, 2, 3]),
        ("dict {}", {}),
        ("dict {1:1}", {1: 1}),
        ("set()", set()),
        ("object()", object()),
    ]

    print(f"  {'Object':>20} {'Size (bytes)':>15} {'Overhead'}")
    print(f"  {'-'*20} {'-'*15} {'-'*10}")
    for label, obj in objects:
        size = sys.getsizeof(obj)
        base = 16  # PyObject header (refcnt + type ptr on 64-bit)
        overhead = size - base
        print(f"  {label:>20} {size:>10}     +{overhead}")
    print()

inspect_sizeof()


# ═══════════════════════════════════════════════════════════════════
# 2. PyMem_* Allocator Families
# ═══════════════════════════════════════════════════════════════════
#
# From Include/pymem.h:
#
#   // Raw — no GIL, no GC tracking, direct system call
#   void* PyMem_RawMalloc(size_t n)
#   void* PyMem_RawCalloc(size_t n)
#   void* PyMem_RawRealloc(void *p, size_t n)
#   void  PyMem_RawFree(void *p)
#
#   // Python memory — with GIL, GC tracked
#   void* PyMem_Malloc(size_t n)
#   void* PyMem_Calloc(size_t n)
#   void* PyMem_Realloc(void *p, size_t n)
#   void  PyMem_Free(void *p)
#
#   // Object memory — fast path, uses obmalloc pools
#   void* PyObject_Malloc(size_t n)
#   void* PyObject_Calloc(size_t n)
#   void* PyObject_Realloc(void *p, size_t n)
#   void  PyObject_Free(void *p)
#
# The allocator can be overridden (PyMem_SetupDebugHooks adds
# debug checks, Py_SetAllocator replaces the allocator entirely).

def simulate_allocator():
    """Simulate the obmalloc block size class calculation."""
    print("=== obmalloc block size classes ===")

    # Size class formula: idx = (size - 1) >> 3 (for 8-byte aligned)
    # But minimum size is 8 (one pointer), maximum is 512 (pool size)
    sizes = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 96, 128,
             160, 192, 224, 256, 320, 384, 448, 512]

    print(f"  {'Class':>6} {'Size':>6} {'Next':>6}")
    for i, size in enumerate(sizes):
        next_size = sizes[i + 1] if i + 1 < len(sizes) else size
        print(f"  {i:>6} {size:>6} {next_size:>6}")

    print()
    print("  Key: each pool stores blocks of exactly one size class.")
    print("  A request for 33 bytes gets a 40-byte block (class 4).")
    print()

simulate_allocator()


# ═══════════════════════════════════════════════════════════════════
# 3. The GIL — Global Interpreter Lock
# ═══════════════════════════════════════════════════════════════════
#
# The GIL (Python/ceval_gil.h) is a mutex that ensures only one
# thread executes Python bytecode at a time. It's released:
#   - During blocking I/O (read/write/select/sleep)
#   - In C extensions with Py_BEGIN_ALLOW_THREADS
#   - Every 5ms (sys.getswitchinterval()) for thread switching
#
# GIL switching protocol:
#   1. The running thread checks _Py_atomic_load(gil->locked)
#   2. If another thread is waiting, it releases after 5ms
#   3. The waiting thread signals via gil->switch_condition
#
# The GIL is NOT released during CPU-bound Python code.
# This is why threading doesn't help with CPU-bound tasks.

def demo_gil_effect():
    """Demonstrate the GIL effect on CPU-bound threads."""
    print("=== GIL effect demonstration ===")

    def count_to(n):
        start = time.perf_counter()
        total = 0
        for i in range(n):
            total += i
        elapsed = time.perf_counter() - start
        return total, elapsed

    # Sequential
    start = time.perf_counter()
    r1, _ = count_to(50_000_000)
    r2, _ = count_to(50_000_000)
    sequential_time = time.perf_counter() - start

    # Parallel (threaded — GIL prevents true parallelism)
    results = [None, None]

    def worker(idx, n):
        total, t = count_to(n)
        results[idx] = (total, t)

    start = time.perf_counter()
    t1 = threading.Thread(target=worker, args=(0, 50_000_000))
    t2 = threading.Thread(target=worker, args=(1, 50_000_000))
    t1.start(); t2.start()
    t1.join(); t2.join()
    parallel_time = time.perf_counter() - start

    print(f"  Sequential: {sequential_time:.3f}s")
    print(f"  Parallel:   {parallel_time:.3f}s")
    print(f"  Ratio:      {parallel_time / sequential_time:.2f}x")
    print(f"  (With the GIL, parallel ~= sequential or worse for CPU-bound)")
    print()

demo_gil_effect()


# ═══════════════════════════════════════════════════════════════════
# 4. GIL Release in C Extensions
# ═══════════════════════════════════════════════════════════════════
#
# C extensions can release the GIL during blocking operations:
#
#   Py_BEGIN_ALLOW_THREADS
#       result = some_blocking_io(...);
#   Py_END_ALLOW_THREADS
#
# This is equivalent to:
#   PyThreadState *_save = PyEval_SaveThread();
#   // ... blocking work ...
#   PyEval_RestoreThread(_save);
#
# While the GIL is released, other Python threads can run.

def simulate_gil_release():
    """Simulate the effect of GIL release (I/O bound)."""
    print("=== GIL release (I/O bound) ===")

    def io_simulated():
        # Simulating I/O that releases the GIL
        time.sleep(0.05)  # sleep() releases the GIL
        return 42

    def cpu_work():
        total = 0
        for i in range(500_000):
            total += i
        return total

    # Sequential
    start = time.perf_counter()
    for _ in range(10):
        io_simulated()
        cpu_work()
    sequential = time.perf_counter() - start

    # Parallel (I/O overlaps with CPU due to GIL release)
    def mixed_worker():
        io_simulated()
        cpu_work()

    start = time.perf_counter()
    threads = [threading.Thread(target=mixed_worker) for _ in range(10)]
    for t in threads: t.start()
    for t in threads: t.join()
    parallel = time.perf_counter() - start

    print(f"  Sequential: {sequential:.3f}s")
    print(f"  Parallel:   {parallel:.3f}s")
    print(f"  Speedup:    {sequential / parallel:.2f}x")
    print(f"  (I/O releases the GIL, so threads overlap effectively)")
    print()

simulate_gil_release()


# ═══════════════════════════════════════════════════════════════════
# 5. Thread State and Interpreter State
# ═══════════════════════════════════════════════════════════════════
#
# PyThreadState (Include/pystate.h):
#   - One per thread
#   - Holds: frame (current), recursion depth, exception info,
#            tracing function, async_exc (pending exception)
#   - Linked list via PyThreadState.next
#
# PyInterpreterState:
#   - One per interpreter (main interpreter + any subinterpreters)
#   - Holds: module dict, import hooks, sys.path copies
#   - Free-threaded config (--disable-gil)
#
# These are managed by Python/pystate.c.

def inspect_thread_state():
    """Inspect current thread state via Python API."""
    print("=== Thread/Interpreter state ===")

    # Python doesn't expose PyThreadState directly, but we can
    # get useful info:
    print(f"  Current thread:  {threading.current_thread().name}")
    print(f"  Thread ID:       0x{threading.get_ident():x}")
    print(f"  Active threads:  {threading.active_count()}")

    # sys._current_frames() gives the top frame per thread
    frames = sys._current_frames()
    for thread_id, frame in frames.items():
        print(f"  Thread 0x{thread_id:x}: {frame.f_code.co_name}")
        if hasattr(frame, 'f_globals'):
            print(f"    module: {frame.f_globals.get('__name__', '?')}")
    print()

inspect_thread_state()


# ═══════════════════════════════════════════════════════════════════
# 6. Subinterpreters (PEP 554)
# ═══════════════════════════════════════════════════════════════════
#
# Subinterpreters exist within a single process but have:
#   - Separate interpreter state (PyInterpreterState)
#   - Separate GIL (so they can run in parallel!)
#   - Separate module namespace (no import lock contention)
#   - Communication via channels (queue-like)
#
# Available since Python 3.12 via the `interpreters` module.

def inspect_subinterpreters():
    """Check subinterpreter support."""
    print("=== Subinterpreters ===")

    try:
        import interpreters
        print("  interpreters module available!")
        print(f"  interpreters.list(): {interpreters.list()}")
        print()
        print("  Creating a subinterpreter:")
        interp = interpreters.create()
        print(f"    id = {interp.id}")
        print(f"    is_running = {interp.is_running}")
        print(f"  interpreters.list(): {interpreters.list()}")
        interp.close()
        print("  Subinterpreters have their OWN GIL → true parallelism!")
    except ImportError:
        print("  interpreters module NOT available")
        print("  (Requires Python 3.12+ with --enable-experimental-jit")
        print("   or built from source with subinterpreters enabled)")
    except Exception as e:
        print(f"  Error: {e}")
    print()

inspect_subinterpreters()


# ═══════════════════════════════════════════════════════════════════
# 7. Free-Threaded CPython (PEP 703, --disable-gil)
# ═══════════════════════════════════════════════════════════════════
#
# Python 3.13 introduced an optional free-threaded mode (nogil).
# Key changes:
#   - GIL is disabled at build time (--disable-gil)
#   - Thread safety via per-object locks or atomic operations
#   - Biased reference counting (most refs from owning thread)
#   - Guaranteed thread safety for pure Python code
#
# Check if we're running in free-threaded mode:

def check_free_threaded():
    """Check if running in free-threaded mode."""
    print("=== Free-threaded mode check ===")

    # In Python 3.13+, sys._is_gil_enabled() tells us
    if hasattr(sys, '_is_gil_enabled'):
        enabled = sys._is_gil_enabled()
        print(f"  GIL enabled: {enabled}")
        if not enabled:
            print("  Running in free-threaded mode (PEP 703)!")
    else:
        print(f"  sys.getswitchinterval = {sys.getswitchinterval()}s")
        print("  (pre 3.13 or standard build — GIL is always active)")

    # Check if we can check for free-threaded builds
    v = sys.version_info
    if v.major >= 3 and v.minor >= 13:
        print(f"  Python 3.13+ detected — --disable-gil support exists")
    print()

check_free_threaded()


# ═══════════════════════════════════════════════════════════════════
# 8. Weak References
# ═══════════════════════════════════════════════════════════════════
#
# A weak reference (weakref module) allows referencing an object
# without increasing its refcount. When the object dies, the weakref
# returns None (or calls a callback).
#
# Implemented in Objects/weakrefobject.c using:
#   - PyWeakReference struct (ob_ref, callback, hash)
#   - tp_weaklistoffset in PyTypeObject
#   - PyObject_ClearWeakRefs called during tp_dealloc

def inspect_weakref():
    """Demonstrate weak reference behavior."""
    print("=== Weak references ===")

    class ExpensiveObject:
        def __init__(self, name):
            self.name = name
        def __del__(self):
            print(f"    {self.name} destroyed!")

    obj = ExpensiveObject("test")
    ref = weakref.ref(obj, lambda r: print(f"    callback: {r} died"))

    print(f"  Created weakref: ref() = {ref()}")
    print(f"  ref() is obj:    {ref() is obj}")

    # Delete the strong reference
    del obj
    print(f"  After del: ref() = {ref()}")

    # WeakValueDictionary
    print()
    print("  WeakValueDictionary:")
    d = weakref.WeakValueDictionary()
    key = ExpensiveObject("key_obj")
    d["key"] = key
    print(f"    d['key'] = {d['key']}")
    del key
    try:
        d["key"]
    except KeyError:
        print(f"    After del: KeyError (value was collected)")
    print()

inspect_weakref()


# ═══════════════════════════════════════════════════════════════════
# 9. Buffer Protocol (memoryview, Py_buffer)
# ═══════════════════════════════════════════════════════════════════
#
# The buffer protocol lets objects share raw memory without copying.
# Key types: bytes, bytearray, array.array, numpy.ndarray.
#
#   Py_buffer (Include/objimpl.h):
#     buf         — pointer to raw memory
#     len         — total bytes
#     ndim        — number of dimensions
#     shape       — shape array
#     strides     — stride array
#     suboffsets  — suboffset array
#     itemsize    — size of one item
#     format      — struct format string ('B', 'h', 'i', 'd', etc.)
#     readonly    — 1 if read-only
#
# The protocol is accessed via: memoryview, or tp_as_buffer slots.

def inspect_buffer_protocol():
    """Demonstrate the buffer protocol with memoryview."""
    print("=== Buffer protocol ===")

    data = bytearray(b"hello buffer protocol!!!")  # 24 bytes, multiple of 4
    mv = memoryview(data)

    print(f"  memoryview:       {mv!r}")
    print(f"  mv.nbytes:        {mv.nbytes}")
    print(f"  mv.ndim:          {mv.ndim}")
    print(f"  mv.itemsize:      {mv.itemsize}")
    print(f"  mv.format:        {mv.format!r}")
    print(f"  mv.readonly:      {mv.readonly}")
    print(f"  mv.shape:         {mv.shape}")
    print(f"  mv.strides:       {mv.strides}")

    # Cast to different types
    print()
    print("  Cast to int32 (4 bytes):")
    int32_mv = mv.cast('I')  # unsigned 32-bit int
    print(f"    shape: {int32_mv.shape}, itemsize: {int32_mv.itemsize}")
    for i in range(len(int32_mv)):
        print(f"    [{i}] = {int32_mv[i]}")

    # Modify through buffer
    print()
    print("  Modify through memoryview:")
    mv[0:5] = b"HELLO"
    print(f"    data = {data}")
    print()

inspect_buffer_protocol()


# ═══════════════════════════════════════════════════════════════════
# 10. sys.getsizeof — Deep vs Shallow Size
# ═══════════════════════════════════════════════════════════════════
#
# sys.getsizeof only returns the OBJECT's own memory (PyObject header +
# direct fields). It does NOT include memory of referenced objects.
# For a deep (recursive) size estimate, you need to walk manually.

def inspect_deep_size():
    """Compare shallow vs deep memory use."""
    print("=== Shallow vs deep memory ===")

    def get_deep_size(obj, seen=None):
        """Recursively compute deep size of an object."""
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)
        size = sys.getsizeof(obj)

        if isinstance(obj, (list, tuple)):
            for item in obj:
                size += get_deep_size(item, seen)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                size += get_deep_size(k, seen)
                size += get_deep_size(v, seen)
        elif isinstance(obj, set):
            for item in obj:
                size += get_deep_size(item, seen)
        return size

    # Shallow vs deep for a nested structure
    nested = {"data": [1, 2, {"deep": "value" * 100}], "flag": True}
    shallow = sys.getsizeof(nested)
    deep = get_deep_size(nested)

    print(f"  Nested structure:")
    print(f"    Shallow: {shallow:>8} bytes")
    print(f"    Deep:    {deep:>8} bytes")
    print(f"    Ratio:   {deep / shallow:.1f}x")
    print()

inspect_deep_size()


# ═══════════════════════════════════════════════════════════════════
# 11. Key Insights (memories for Jafar)
# ═══════════════════════════════════════════════════════════════════
#
# 1. obmalloc uses 3 tiers: arena (256KB) → pool (4KB) → block (8B+).
#    Blocks are grouped by size class (8-byte aligned).
#    This reduces fragmentation and system call overhead.
#
# 2. Three allocator families: PyMem_Raw (no GIL, system malloc),
#    PyMem (GIL), PyObject (fast obmalloc pools). The fast path
#    is what PyType_GenericAlloc calls.
#
# 3. The GIL (Python/ceval_gil.h) is a mutex that serializes bytecode
#    execution. Released during I/O (sleep, read, write) and in C
#    extensions via Py_BEGIN_ALLOW_THREADS. Switch interval ~5ms.
#
# 4. GIL release protocol: PyEval_SaveThread() → ... → PyEval_RestoreThread().
#    This is the standard pattern for blocking C code.
#
# 5. PyThreadState per thread, PyInterpreterState per interpreter.
#    Each thread has: current frame, recursion depth, exception info.
#    sys._current_frames() shows the top frame per thread.
#
# 6. Subinterpreters (PEP 554, Python 3.12+) have separate GILs,
#    separate module namespaces, and communicate via channels.
#    True parallelism for CPU-bound work.
#
# 7. Free-threaded CPython (PEP 703, Python 3.13+) disables the GIL
#    at build time. Uses biased reference counting + atomic ops.
#    Check with sys._is_gil_enabled().
#
# 8. Weak references (Objects/weakrefobject.c) do not increment
#    refcount. PyObject_ClearWeakRefs() is called during tp_dealloc.
#    weakref module: ref, WeakValueDictionary, WeakSet.
#
# 9. Buffer protocol (Py_buffer) allows zero-copy memory sharing.
#    memoryview is the Python interface. Key fields: buf, len,
#    ndim, shape, strides, format, itemsize, readonly.
#
# 10. sys.getsizeof is shallow (object only). Deep size requires
#     recursive walking of all referenced objects.
#

if __name__ == "__main__":
    pass
