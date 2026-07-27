"""
CPython Internals --- Part 8: Memory Management Deep Dive (Phase 2)

PART 5 covered the big picture (3-tier allocator, GIL, subinterpreters).
PART 8 digs deeper into allocation internals, GC tracing, and profiling.

WHAT'S NEW:
  * obmalloc.c arena -> pool -> block (with actual arithmetic)
  * Pool header layout (pool_header struct fields)
  * Free-list management: usedp, freep, next/prev offsets
  * PyMem_SetupDebugHooks --- the debug allocator
  * tracemalloc --- allocation tracing (who allocated what)
  * GC internals: generations, thresholds, get_objects, get_referrers
  * Object free lists (int, float, tuple reuse pools)
  * Allocator replacement via PyMem_SetAllocator
  * Memory-mapped files (mmap) and copy-on-write
  * Refcount tracing and leak detection patterns

Reference:
  - Objects/obmalloc.c --- the 3-tier allocator
  - Objects/object.c --- PyObject_{Malloc,Free}
  - Python/pymem.c --- debug hooks, allocator management
  - Modules/_tracemalloc.c --- allocation tracing
  - Modules/gcmodule.c --- garbage collector
  - Include/objimpl.h --- PyObject_HEAD, PyObject_VAR_HEAD
  - https://github.com/python/cpython/blob/main/InternalDocs/allocators.md
"""

import sys
import gc
import os
import ctypes
import random
import struct
import time


# ===================================================================
# 1. obmalloc.c --- Structure Internals
# ===================================================================
#
# obmalloc.c defines the 3-tier allocator:
#
#   ARENA: 256 KB region from the OS (VirtualAlloc on Windows, mmap on Unix)
#   POOL:  4 KB page within an arena, holds blocks of ONE size class
#   BLOCK: individual allocation, 8-byte aligned, 8..512 bytes
#
# Constant key values (from Include/internal/pycore_obmalloc.h):
#   ARENA_SIZE              = 256 << 10  (262,144 bytes)
#   POOL_SIZE               = 1 << 12    (4,096 bytes)
#   POOL_SIZE_MASK          = POOL_SIZE - 1
#   BLOCK_SIZE              = 1 <<  5    (32 bytes on 64-bit)
#   MAX_BLOCKS_PER_POOL     = POOL_SIZE / BLOCK_SIZE  (128)
#
# Each pool has a header (pool_header):
#   typedef struct {
#       union { block *_padding; uint count; };
#       struct pool_header *next;     // next pool in doubly-linked list
#       struct pool_header *prev;     // previous pool
#       uint arenalist_index;         // index into arenas[] array
#       uint szidx;                   // size class index (0..63)
#       uint maxblocks;               // max blocks at this size class
#       uint freeblocks;              // number of free blocks
#       block *freeoffset;            // next free block offset
#       block *nextoffset;            // next available offset
#   } pool_header;
#
# Blocks are 8-byte aligned. Size class index formula:
#   szidx = (size - 1) >> ALIGNMENT_SHIFT
# where ALIGNMENT_SHIFT = 3 (8-byte alignment).

def simulate_pool_allocation():
    """Simulate obmalloc's pool allocation strategy."""
    print("=" * 60)
    print("1. obmalloc Arena -> Pool -> Block")
    print("=" * 60)

    ARENA_SIZE = 262144
    POOL_SIZE = 4096
    MAX_SZIDX = 63  # 0..63, max block size = 512 bytes

    blocks_per_pool = POOL_SIZE // 32  # 128 blocks at 32 bytes each

    print(f"\n  Arena size:    {ARENA_SIZE:,} bytes ({ARENA_SIZE // 1024} KB)")
    print(f"  Pool size:     {POOL_SIZE:,} bytes ({POOL_SIZE // 1024} KB)")
    print(f"  Max block size: 512 bytes (szidx 63)")
    print(f"  Min block size:   8 bytes (szidx 0)")
    print(f"  Alignment:        8 bytes")
    print(f"  Default pool blocks: ~{blocks_per_pool} (at 32-byte size class)")

    # Simulate arena breakdown
    arena_pools = ARENA_SIZE // POOL_SIZE  # 64 pools per arena
    print(f"\n  Pools per arena:     {arena_pools}")
    print(f"  Arena overhead:      ~{arena_pools * 64:,} bytes (pool headers)")
    print(f"  Usable pool space:   {POOL_SIZE - 64} bytes per pool (header removed)")

    # Simulate size class mapping
    print(f"\n  Size class examples:")
    for requested in [1, 8, 9, 16, 25, 33, 64, 100, 128, 256, 512]:
        szidx = (requested - 1) >> 3  # ALIGNMENT_SHIFT = 3
        actual_size = (szidx + 1) << 3
        waste = actual_size - requested
        print(f"    request {requested:>4}B -> szidx {szidx:>2} -> "
              f"block {actual_size:>4}B (waste {waste}B, "
              f"{waste / actual_size * 100:.0f}%)")

    print()
    print("  KEY INSIGHT: Block alignment wastes 0-7 bytes per allocation.")
    print("  This is the 'internal fragmentation' trade-off for O(1) free.")
    print()


simulate_pool_allocation()


# ===================================================================
# 2. Free-List Management
# ===================================================================
#
# Each pool tracks free blocks with two pointers:
#   freep   --- pointer to the first free block (singly-linked free list)
#   nextoffset --- offset within the pool for the next uninitialized block
#
# When a block is freed:
#   1. Its first 8 bytes are overwritten with the current freep address
#   2. freep is updated to point to the just-freed block
#   -> This is O(1) free
#
# When a block is allocated:
#   1. If freep is non-NULL, pop the first block from the free list
#   2. Otherwise, use the block at nextoffset and advance
#   -> This is O(1) allocation
#
# When a pool is empty (freep = nextoffset = end):
#   The pool is moved to the "empty" list and returned to the arena.
#
# Full pools are never freed back to the OS --- they stay in the "full" list
# and are reused when demand returns.

def simulate_free_list():
    print("=" * 60)
    print("2. Free-List Management (O(1) alloc/free)")
    print("=" * 60)

    print(r"""
  Pool state machine:

    +------------+    empty?    +------------+   has free?  +------------+
    |  USED      | ---------->  |  PARTIAL   | ---------->  |  FULL      |
    | (empty)    |              | (has free) |              | (all used) |
    |            | <----------  |            | <----------  |            |
    +------------+    freed     +------------+   alloc'd    +------------+

      USED:    pool has no allocated blocks (all slots free)
      PARTIAL: pool has a mix of free and allocated blocks
      FULL:    pool has no free blocks

  Pool linked lists for each size class:
    - usable_arenas[szidx]  -> doubly-linked list of PARTIAL pools
    - full_arenas           -> singly-linked list of FULL pools
    - empty pools are returned to the arena free list
""")

    print("  O(1) allocation sequence:")
    print("    1. Look up szidx from request size")
    print("    2. Check usable_arenas[szidx] for a PARTIAL pool")
    print("    3. If none, get a USED pool (or allocate new arena)")
    print("    4. Pop block from pool's free list (freep -> next)")
    print("    5. Return pointer to block")
    print()
    print("  O(1) free sequence:")
    print("    1. Compute pool address: pool = (block & POOL_SIZE_MASK)")
    print("    2. Push block onto pool's free list")
    print("    3. If pool was FULL, move to PARTIAL list")
    print("    4. If pool is now empty, move to USED list")
    print()


simulate_free_list()


# ===================================================================
# 3. Debug Allocator --- PyMem_SetupDebugHooks
# ===================================================================
#
# When CPython is built with Py_DEBUG or PyMem_SetupDebugHooks is called,
# the allocator is wrapped with debug checks:
#
#   Each allocation has guard bytes:
#     [8 bytes: serial number] [8 bytes: size] [data...] [4 bytes: checksum]
#
#   On free, it validates:
#     1. The serial number hasn't been freed before (double-free detection)
#     2. The checksum matches (buffer overflow detection)
#     3. The size field matches the requested size
#
#   On realloc, it checks:
#     1. The old buffer's checksum (no overflow since allocation)
#     2. Copies data and frees old with the above checks
#
# The debug pattern fills freed memory with 0xDD (dead byte) and
# newly allocated memory with 0xCD (clean byte).

def inspect_debug_allocator():
    """Check if Python is running with debug allocator."""
    print("=" * 60)
    print("3. Debug Allocator (PyMem_SetupDebugHooks)")
    print("=" * 60)

    print(f"\n  sys.version: {sys.version}")
    print(f"  Py_DEBUG build: {'d' in sys.abiflags if hasattr(sys, 'abiflags') else 'unknown'}")

    # Check if debug hooks are active
    # Python 3.13+ exposes this via sys._debugmallocstats()
    if hasattr(sys, '_debugmallocstats'):
        print("  Debug allocator API: sys._debugmallocstats() available")
    else:
        print("  Debug allocator API: NOT available")

    # Check abiflags
    if hasattr(sys, 'abiflags'):
        flags = sys.abiflags
        print(f"  ABI flags: {flags!r}")
        if 'd' in flags:
            print("    -> Py_DEBUG build (debug allocator active by default)")
        elif 't' in flags:
            print("    -> Free-threaded build (--disable-gil)")
        else:
            print("    -> Standard release build")

    print()
    print("  What debug hooks catch:")
    print("    * Buffer underflow:   fill pattern before data is checked")
    print("    * Buffer overflow:    fill pattern after data is checked")
    print("    * Use-after-free:     freed memory filled with 0xDD")
    print("    * Double-free:        serial number validation")
    print("    * API mixing:         PyMem_Malloc vs PyObject_Free mismatch")
    print()

    if hasattr(sys, '_debugmallocstats'):
        print("  sys._debugmallocstats():")
        try:
            sys._debugmallocstats()
        except Exception as e:
            print(f"    Error: {e}")
    print()


inspect_debug_allocator()


# ===================================================================
# 4. tracemalloc --- Allocation Tracing
# ===================================================================
#
# tracemalloc (Modules/_tracemalloc.c) traces every PyMem_Malloc/
# PyObject_Malloc call and records the Python traceback. It works by:
#   1. Installing a custom allocator via PyMem_SetAllocator
#   2. Intercepting each allocation to record: (size, traceback, timestamp)
#   3. Storing in a hash table keyed by (address, traceback_hash)
#
# Key API:
#   tracemalloc.start()       --- start tracing
#   tracemalloc.stop()        --- stop tracing
#   tracemalloc.get_traced_memory() --- current traced memory
#   tracemalloc.get_object_traceback(obj) --- traceback for an object
#   tracemalloc.take_snapshot() --- full memory snapshot
#
# Overhead: ~10% performance, ~1-2 bytes per tracked allocation.

def demo_tracemalloc():
    """Demonstrate tracemalloc's allocation tracing."""
    print("=" * 60)
    print("4. tracemalloc --- Allocation Tracing")
    print("=" * 60)

    try:
        import tracemalloc

        # Start tracing
        tracemalloc.start(25)  # 25-frame tracebacks
        print("\n  tracemalloc started (25-frame depth)")

        # Take a baseline snapshot
        baseline = tracemalloc.take_snapshot()
        print(f"  Baseline snapshot taken")

        # Allocate some memory in a known function
        def allocate_stuff():
            big_list = [i for i in range(100000)]
            big_dict = {str(i): i for i in range(10000)}
            big_str = "x" * 1000000
            return big_list, big_dict, big_str

        stuff = allocate_stuff()
        print(f"  Allocated list(100k), dict(10k), str(1MB)")

        # Take a diff snapshot
        current = tracemalloc.take_snapshot()
        diff = current.compare_to(baseline, 'lineno')

        print("\n  Top allocations by size (since baseline):")
        for stat in diff[:8]:
            print(f"    {stat.size_diff // 1024:>8} KB  {stat.count_diff:>6} blocks  "
                  f"{stat.traceback.format()[0] if stat.traceback else '?'}")

        # Traceback for a specific object
        print("\n  Traceback for big_list[0]:")
        tb = tracemalloc.get_object_traceback(stuff[0])
        if tb:
            for frame in tb[:5]:
                print(f"    {frame.filename}:{frame.lineno}  ({frame.function})")
        else:
            print("    (no traceback - might be integer interning)")

        # Get traced memory
        current_size, peak_size = tracemalloc.get_traced_memory()
        print(f"\n  Current traced memory: {current_size / 1024:.1f} KB")
        print(f"  Peak traced memory:    {peak_size / 1024:.1f} KB")

        tracemalloc.stop()
        print("\n  tracemalloc stopped")

    except ImportError:
        print("\n  tracemalloc not available on this platform")
    except Exception as e:
        print(f"\n  Error: {e}")


demo_tracemalloc()


# ===================================================================
# 5. GC Internals --- Generations, Thresholds, Tracing
# ===================================================================
#
# The GC (Modules/gcmodule.c) is a generational collector:
#
#   Generation 0:  700 objects (default threshold)
#   Generation 1:   10 collections of gen0
#   Generation 2:   10 collections of gen1
#
# Each collection cycle:
#   1. Find all objects in the generation
#   2. Compute reachability from roots (module globals, stack frames)
#   3. Unreachable objects are collected (tp_dealloc called)
#   4. Surviving objects are promoted to the next generation
#
# Containers (list, dict, set, tuple, custom types with Py_TPFLAGS_HAVE_GC)
# are tracked by the GC. Non-containers (int, str, float) are not tracked.

def inspect_gc():
    """Inspect the garbage collector internals."""
    print("=" * 60)
    print("5. Garbage Collector --- Generations & Tracing")
    print("=" * 60)

    gc.set_debug(gc.DEBUG_STATS)
    print("\n  GC debug: STATS enabled (shows collection summaries)")

    # Generation thresholds
    print(f"\n  GC thresholds: {gc.get_threshold()}")
    print(f"    Gen 0: {gc.get_threshold()[0]} objects -> collection")
    print(f"    Gen 1: {gc.get_threshold()[1]} gen0 collections -> gen1")
    print(f"    Gen 2: {gc.get_threshold()[2]} gen1 collections -> gen2")

    # Object counts per generation
    print(f"\n  Object counts per generation:")
    for i in range(3):
        count = gc.get_count()
        objs = gc.get_objects()  # all tracked objects
        print(f"    Gen {i}: {count[i]:>8} new objects since last {gen_name(i)} collection")

    gc.set_debug(0)

    # List tracked types
    print("\n  Top 10 tracked object types:")
    tracked = {}
    for obj in gc.get_objects():
        t = type(obj)
        tracked[t] = tracked.get(t, 0) + 1
    sorted_types = sorted(tracked.items(), key=lambda x: -x[1])
    for t, count in sorted_types[:10]:
        print(f"    {t.__name__:>20}: {count:>8}")

    print()


def gen_name(i):
    return {0: "gen0", 1: "gen1", 2: "gen2"}[i]


inspect_gc()


# ===================================================================
# 6. gc.get_referrers and gc.get_referents --- Reference Tracing
# ===================================================================
#
# These functions let you trace the reference graph at runtime:
#
#   gc.get_referrers(obj)  --- objects that REFER TO obj
#   gc.get_referents(obj)  --- objects that obj REFERS TO
#
# This is how you debug reference cycles and memory leaks.

def demo_reference_tracing():
    print("=" * 60)
    print("6. Reference Tracing --- get_referrers / get_referents")
    print("=" * 60)

    # Create a reference chain
    a = [1, 2, 3]
    b = {"key": a}
    c = (a, b)

    print(f"\n  Reference chain:")
    print(f"    a = {id(a):#x} (list)")
    print(f"    b = {id(b):#x} (dict, refers to a)")
    print(f"    c = {id(c):#x} (tuple, refers to a and b)")

    # Find referrers to 'a'
    referrers = gc.get_referrers(a)
    print(f"\n  gc.get_referrers(a) -> {len(referrers)} referrers:")
    for r in referrers[:5]:
        try:
            name = type(r).__name__
            rid = id(r)
            # Check if it's one of our known objects
            if r is b:
                print(f"    [{rid:#x}] dict  (b: our dict)")
            elif r is c:
                print(f"    [{rid:#x}] tuple (c: our tuple)")
            else:
                print(f"    [{rid:#x}] {name} (other)")
        except Exception:
            pass

    # Find referents of 'b'
    referents = gc.get_referents(b)
    print(f"\n  gc.get_referents(b) -> {len(referents)} referents:")
    for r in referents[:5]:
        try:
            name = type(r).__name__
            rid = id(r)
            if r is a:
                print(f"    [{rid:#x}] list  (a: our list)")
            else:
                short = repr(r)[:40]
                print(f"    [{rid:#x}] {name}: {short}")
        except Exception:
            pass

    # Cycle detection
    print("\n  Cycle detection demo:")
    x = []
    y = []
    x.append(y)
    y.append(x)
    print(f"    Created cycle: x -> y -> x")
    print(f"    gc.is_tracked(x): {gc.is_tracked(x)}")
    print(f"    gc.is_tracked(y): {gc.is_tracked(y)}")

    gc.collect()
    unreachable = gc.garbage
    print(f"    gc.garbage after collect: {len(unreachable)} items")
    print(f"    (gc.garbage contains objects with __del__ that form cycles)")
    print()

demo_reference_tracing()


# ===================================================================
# 7. Object Free Lists (Int, Float, Tuple)
# ===================================================================
#
# CPython maintains per-type free lists for small, commonly-allocated
# objects. These avoid the overhead of obmalloc allocation:
#
#   int:    PyLongObject free list (max 100 entries, -5..256 range cached)
#           -> small_ints[] array in Objects/longobject.c
#   float:  PyFloatObject free list (max 100 entries)
#           -> Objects/floatobject.c, NUM_FREELISTS = 100
#   tuple:  Tuple free list per size class (max 2000 per class)
#           -> Objects/tupleobject.c, MAXSAVESIZE = 20
#
# The free list is a singly-linked list using the object's first field
# (ob_type, which is never NULL for live objects) as the next pointer.
#
# When an object is freed:
#   - If the free list for its type/size is not full:
#     -> object goes onto the free list instead of being freed to obmalloc
#   - Otherwise: obmalloc free (back to pool)

def inspect_free_lists():
    """Demonstrate object free lists by allocation patterns."""
    print("=" * 60)
    print("7. Object Free Lists (Int, Float, Tuple)")
    print("=" * 60)

    # Free lists are NOT exposed via Python API, so we demonstrate
    # their effect indirectly through allocation speed.

    print("""
  Free list behavior:

    int:  -5 to 256 are cached as singletons (small_ints[])
          Any int in this range returns the SAME object
          -> is comparison works: a = 256; b = 256; a is b  (True)

    float: Freed floats go to a free list (max 100)
           Reusing free list entries is faster than obmalloc

    tuple: One free list per size (0..20)
           tuple() dealloc checks: free list full? -> obmalloc
                                   free list full? -> obmalloc
  """)

    # Demonstrate int singleton caching
    print("  Int singleton test (small ints):")
    a = 256
    b = 256
    print(f"    a = 256; b = 256; a is b -> {a is b}")
    c = 257
    d = 257
    print(f"    c = 257; d = 257; c is d -> {c is d}  (NOT cached)")
    print("    (range -5..256 are cached; outside range creates new objects)")

    # Float free list speed difference
    print("\n  Float alloc/dealloc speed:")

    def allocate_many_floats(n):
        floats = [float(i) for i in range(n)]
        for f in floats:
            pass  # let them be collected
        return floats

    # Warm up
    _ = allocate_many_floats(1000)

    # Timed: first allocation (cold free list) vs subsequent
    times = []
    for _ in range(5):
        start = time.perf_counter_ns()
        f = allocate_many_floats(100_000)
        elapsed = time.perf_counter_ns() - start
        times.append(elapsed)
        del f

    print(f"    100k float alloc cycles: {min(times) / 1e6:.1f}ms min, "
          f"{sum(times) / len(times) / 1e6:.1f}ms avg")

    # Tuple free list
    print("\n  Tuple alloc/dealloc speed (size 0..20):")
    for size in [0, 1, 5, 10, 20, 50]:
        def allocate_many_tuples(n, sz):
            tuples = [tuple(range(sz)) for _ in range(n)]
            del tuples

        start = time.perf_counter_ns()
        allocate_many_tuples(100_000, size)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        print(f"    tuple size {size:>3}: {elapsed:.1f}ms for 100k alloc/free cycles")

    print()


inspect_free_lists()


# ===================================================================
# 8. Allocator Replacement --- PyMem_SetAllocator
# ===================================================================
#
# CPython allows replacing its memory allocators at runtime via
# PyMem_SetAllocator (Include/pymem.h):
#
#   typedef struct {
#       void *(*malloc)(void *ctx, size_t size);
#       void *(*calloc)(void *ctx, size_t nelem, size_t elsize);
#       void *(*realloc)(void *ctx, void *ptr, size_t new_size);
#       void  (*free)(void *ctx, void *ptr);
#   } PyMemAllocatorEx;
#
# There are 3 domains:
#   PYMEM_DOMAIN_RAW      --- raw (no GIL)
#   PYMEM_DOMAIN_MEM      --- Python memory
#   PYMEM_DOMAIN_OBJECT   --- object memory (fast path)
#
# tracemalloc uses this to intercept all allocations.
#
# Python 3.13+ exposes this via:
#   sys._get_allocator_hooks()  --- get current allocator hooks
#   sys._set_allocator_hooks()  --- set custom allocator hooks

def inspect_allocator_api():
    """Check allocator replacement API."""
    print("=" * 60)
    print("8. Allocator Replacement API")
    print("=" * 60)

    if hasattr(sys, '_get_allocator_hooks'):
        hooks = sys._get_allocator_hooks()
        print(f"\n  sys._get_allocator_hooks() -> {hooks}")
    else:
        print("\n  sys._get_allocator_hooks() not available")
        print("  (Python 3.13+ specific)")

    if hasattr(sys, '_set_allocator_hooks'):
        print("  sys._set_allocator_hooks() available")
    else:
        print("  sys._set_allocator_hooks() not available")

    # Show the allocator domains concept
    print("""
  Allocator domains:
                     GIL?  Used by
    PYMEM_DOMAIN_RAW   [X]    PyMem_RawMalloc (C stdlib, no Python state)
    PYMEM_DOMAIN_MEM   [V]    PyMem_Malloc (Python memory)
    PYMEM_DOMAIN_OBJECT [V]    PyObject_Malloc (objects, obmalloc pools)

  Usage:
    Raw:     interpreter initialization, low-level C code
    Mem:     Python-level memory (import, modules)
    Object:  PyObject allocations (the fast path)

  You can replace each domain independently.
  tracemalloc installs hooks on PYMEM_DOMAIN_OBJECT and PYMEM_DOMAIN_MEM.
""")

    print("  Custom allocator use cases:")
    print("    * Memory usage tracking (tracemalloc style)")
    print("    * Leak detection (track all alloc/free pairs)")
    print("    * Performance profiling (measure allocation time)")
    print("    * Sandboxing (limit total memory per interpreter)")
    print()


inspect_allocator_api()


# ===================================================================
# 9. Memory-Mapped Files (mmap)
# ===================================================================
#
# mmap creates a memory-mapped file (or anonymous mapping) using
# VirtualAlloc (Windows) or mmap (Unix). The key advantage:
#
#   * Zero-copy file I/O (pages are mapped directly into address space)
#   * Copy-on-write (MAP_PRIVATE): modifications don't affect the file
#   * Shared memory between processes (MAP_SHARED)
#   * Lazy loading: only accessed pages are faulted in
#
# CPython exposes mmap via the mmap module (Modules/mmapmodule.c).
# The underlying C API is:
#   - Windows: CreateFileMapping + MapViewOfFile
#   - Unix: mmap()

def demo_mmap():
    """Demonstrate memory-mapped file behavior."""
    print("=" * 60)
    print("9. Memory-Mapped Files (mmap)")
    print("=" * 60)

    import tempfile
    import mmap

    # Create a temp file
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"A" * 4096)
        f.write(b"B" * 4096)
        f.write(b"C" * 4096)
        temp_path = f.name

    try:
        # Memory map the file
        with open(temp_path, "r+b") as f:
            with mmap.mmap(f.fileno(), 0) as m:
                print(f"\n  Mapped file size: {len(m)} bytes")
                print(f"  m[0:10]: {m[0:10]}")
                print(f"  m[4096:4100]: {m[4096:4100]}")

                # Modify through memory map
                m[0:5] = b"XXXXX"
                print(f"  After modification:")
                print(f"    m[0:10]: {m[0:10]}")

                # Copy-on-write semantics
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY) as cow:
                    print(f"\n  Copy-on-write mapping:")
                    cow[0:5] = b"YYYYY"
                    print(f"    cow[0:10]: {cow[0:10]}")
                    print(f"    m[0:10]:   {m[0:10]} (unchanged by COW write)")
                    print(f"    (MAP_PRIVATE pages are copied on write)")

        print()
        print("  KEY INSIGHT: mmap avoids the copy between kernel and")
        print("  user space. For large files, this is much faster than")
        print("  read()/write(). Python's mmap module wraps this.")
    finally:
        os.unlink(temp_path)
        print("  Temp file cleaned up")


demo_mmap()


# ===================================================================
# 10. Refcount Tracing & Leak Detection
# ===================================================================
#
# Reference counting is the primary memory management mechanism.
# Leaks happen when refcount never reaches 0.
#
#
# In CPython, you can trace refcounts with:
#   sys.getrefcount(obj) --- returns refcount + 1 (the +1 is the argument)
#   ctypes --- read ob_refcnt directly from memory
#
# For C extensions, _PyRefTotal (in debug builds) tracks total refcounts.

def demo_refcount_tracing():
    print("=" * 60)
    print("10. Refcount Tracing & Leak Detection")
    print("=" * 60)

    print("\n  sys.getrefcount basics:")
    x = [1, 2, 3]
    r1 = sys.getrefcount(x)
    print(f"    sys.getrefcount(x) = {r1} (includes the temporary)")

    # Add references
    y = x
    r2 = sys.getrefcount(x)
    print(f"    After y = x: refcount = {r2} (y holds a reference)")

    lst = [x]
    r3 = sys.getrefcount(x)
    print(f"    After lst = [x]: refcount = {r3} (lst holds a reference)")

    # Remove references
    del y
    r4 = sys.getrefcount(x)
    print(f"    After del y: refcount = {r4}")

    # Read ob_refcnt directly via ctypes (for 64-bit Python)
    print("\n  Direct ob_refcnt read via ctypes:")
    obj = "hello"
    # PyObject.ob_refcnt is at offset 0 (Py_ssize_t, 8 bytes on 64-bit)
    refcount_addr = id(obj) + 0  # offset of ob_refcnt in PyObject header
    refcnt = ctypes.c_ssize_t.from_address(refcount_addr).value
    expected = sys.getrefcount(obj) - 1  # sys.getrefcount adds 1 for the arg
    print(f"    Object: {obj!r}")
    print(f"    ob_refcnt (ctypes): {refcnt}")
    print(f"    sys.getrefcount-1:   {expected}")
    print(f"    Match: {refcnt == expected}")

    # Leak detection pattern
    print("\n  Leak detection pattern:")
    print("""
    def find_leaks():
        '''Compare object sets before/after an operation.'''
        before = {id(o) for o in gc.get_objects()}
        # ... do something ...
        gc.collect()
        after = {id(o) for o in gc.get_objects()}
        new_objs = after - before
        if new_objs:
            for obj_id in new_objs:
                obj = [o for o in gc.get_objects() if id(o) == obj_id][0]
                print(f"  NEW: {type(obj).__name__} @ {obj_id:#x}")
    """)

    print("  Memory leak categories:")
    print("    * Reference cycles (GC can collect if no __del__)")
    print("    * C extension leaks (forgotten Py_INCREF)")
    print("    * Global cache accumulation (lru_cache, interned strings)")
    print("    * Event loop callback accumulation (not unregistered)")
    print()


demo_refcount_tracing()


# ===================================================================
# 11. Key Insights (memories for Jafar)
# ===================================================================
#
# 1. obmalloc's 3-tier arena->pool->block design reduces system call
#    overhead and fragmentation. Blocks are 8-byte aligned with 63
#    size classes (8B to 512B). Larger allocations go to system malloc.
#
# 2. Free-list management is O(1): freed blocks are linked via their
#    first 8 bytes into a singly-linked list. Allocation pops from
#    the head. No coalescing needed (same-size blocks).
#
# 3. Pool state machine: USED (all free) -> PARTIAL (mixed) -> FULL.
#    Empty pools stay in the arena; they are NOT freed to the OS.
#    This avoids the thundering-herd problem of repeated mmap/munmap.
#
# 4. Debug allocator (PyMem_SetupDebugHooks) wraps every alloc/free
#    with guard bytes, serial numbers, and checksums. It catches:
#    buffer overflow/underflow, use-after-free, double-free, API mixing.
#
# 5. tracemalloc installs custom allocator hooks to record every
#    allocation with its Python traceback. It can take snapshots and
#    diff them to find allocation hotspots. Overhead ~10%.
#
# 6. GC is generational (3 generations). Gen0 threshold = 700 objects.
#    The GC only tracks container objects (list, dict, set, tuple,
#    custom with Py_TPFLAGS_HAVE_GC). Immortals (int, str) are not tracked.
#
# 7. gc.get_referrers / gc.get_referents trace the reference graph.
#    These are essential for debugging reference cycles and memory leaks.
#
# 8. Object free lists (int small_ints[], float free list, tuple free
#    lists) reuse common objects without touching obmalloc. This is
#    why small ints (-5..256) are singletons and tuple allocation is
#    extremely fast for sizes 0-20.
#
# 9. Allocator replacement (PyMem_SetAllocator) allows custom
#    allocators per domain (RAW/MEM/OBJECT). Used by tracemalloc,
#    custom profilers, and sandboxing. Python 3.13+ exposes this as
#    sys._get_allocator_hooks / sys._set_allocator_hooks.
#
# 10. mmap provides zero-copy file I/O via VirtualAlloc/mmap.
#     Copy-on-write mappings (MAP_PRIVATE) allow modification
#     without affecting the original file. Python's mmap module
#     wraps this for both Windows and Unix.
#
# 11. Refcount tracing via ctypes reads ob_refcnt directly from
#     the PyObject header. Leak detection compares object sets
#     before/after an operation. Debug builds track _PyRefTotal.
#
# SYNOPSIS: CPython's memory management is a layered system:
#
#   System (VirtualAlloc/mmap)
#   L--- Arena (256 KB)
#       L--- Pool (4 KB, one size class per pool)
#           L--- Block (8-512 bytes, 8-byte aligned)
#               L--- Type free lists (int, float, tuple)
#                   L--- Python object
#

if __name__ == "__main__":
    pass
