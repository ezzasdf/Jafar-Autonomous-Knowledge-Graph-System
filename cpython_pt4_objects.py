"""
CPython Internals — Part 4: Objects and the Type System

Covers:
  • PyObject struct (ob_refcnt, ob_type) — every object's foundation
  • PyTypeObject — metaclass machinery in C
  • Reference counting: Py_INCREF / Py_DECREF / Py_XINCREF / Py_XDECREF
  • Type implementations: int (longobject.c), str (unicodeobject.c),
    list (listobject.c), dict (dictobject.c)
  • Garbage collection: gcmodule.c, cyclic reference detection
  • Type slots: the tp_* function pointer table (tp_new, tp_init,
    tp_str, tp_iter, tp_hash, tp_richcompare, etc.)
  • Comparison protocol (tp_richcompare, Py_RETURN_NOTIMPLEMENTED)
  • The descriptor protocol (__get__, __set__, __delete__)
  • Creating custom types from C (PyType_Spec + PyType_FromSpec)

Reference:
  - CPython Internals (Anthony Shaw) — Chapter 4
  - Include/object.h, Include/typeobject.h
  - Objects/object.c, Objects/typeobject.c
  - Objects/longobject.c, Objects/unicodeobject.c
  - Objects/listobject.c, Objects/dictobject.c
  - Modules/gcmodule.c
"""

import sys
import gc
import ctypes
import struct


# ═══════════════════════════════════════════════════════════════════
# 1. PyObject — The Foundation of Everything
# ═══════════════════════════════════════════════════════════════════
#
# In CPython, EVERY Python value is a PyObject*.
#
#   // Include/object.h (simplified)
#   typedef struct _object {
#       Py_ssize_t ob_refcnt;      // Reference count
#       PyTypeObject *ob_type;     // Pointer to the type object
#   } PyObject;
#
# For variable-length objects (str, list, tuple, bytes):
#
#   typedef struct {
#       PyObject ob_base;          // PyObject header
#       Py_ssize_t ob_size;        // Number of items
#   } PyVarObject;
#
# ob_refcnt tracks how many references point to this object.
# When it reaches 0, the object is freed (Py_DECREF → _Py_Dealloc).

def inspect_refcount():
    print("=== Reference counting ===")

    a = "hello"
    print(f"  refcount of 'hello':    {sys.getrefcount(a)}")

    b = a
    print(f"  after b = a:            {sys.getrefcount(a)}")

    c = [a, a, a]
    print(f"  after c = [a, a, a]:    {sys.getrefcount(a)}")

    # Small integers are cached (interned)
    x = 42
    print(f"  refcount of 42:         {sys.getrefcount(x)}")

    # None, True, False are singletons
    print(f"  refcount of None:       {sys.getrefcount(None)}")

    # String interning
    s1 = "hello_world_interned"
    s2 = "hello_world_interned"
    print(f"  s1 is s2 (interned?):   {s1 is s2}")
    print(f"  refcount of interned:   {sys.getrefcount(s1)}")
    print()

inspect_refcount()


# ═══════════════════════════════════════════════════════════════════
# 2. PyTypeObject — Types are Objects Too
# ═══════════════════════════════════════════════════════════════════
#
# PyTypeObject is the C representation of a Python type.
# It's itself a PyObject (all types are instances of 'type').
#
# Key fields of PyTypeObject:
#   tp_name          — Name of the type ("int", "str", etc.)
#   tp_basicsize     — Size of the object in bytes
#   tp_dealloc       — Destructor (when refcount hits 0)
#   tp_repr / tp_str — repr() and str() implementations
#   tp_hash          — hash() implementation
#   tp_richcompare   — Comparison (==, <, >, etc.)
#   tp_iter / tp_iternext — Iterator protocol
#   tp_getattro      — Attribute lookup (__getattr__)
#   tp_setattro      — Attribute setting (__setattr__)
#   tp_flags         — Type flags (Py_TPFLAGS_*)
#   tp_dict          — Type's __dict__
#   tp_mro           — Method resolution order
#   tp_new           — Constructor (__new__)
#   tp_init          — Initializer (__init__)
#   tp_call          — Callable protocol (__call__)
#

def inspect_type_object():
    print("=== PyTypeObject introspection ===")

    # Every type is an instance of 'type'
    print(f"  type(int)  = {type(int)}")
    print(f"  type(str)  = {type(str)}")
    print(f"  type(type) = {type(type)}")

    # Type names
    for t in [int, str, list, dict, float, bool, type, object]:
        print(f"  {t.__name__:>10}: tp_name={t.__name__}, basic_size not exposed, flags={t.__flags__:#x}")

    # MRO (Method Resolution Order)
    print()
    print("  MRO examples:")
    for t in [int, str, list, Exception]:
        print(f"    {t.__name__:>10}: {[c.__name__ for c in t.__mro__]}")
    print()

inspect_type_object()


# ═══════════════════════════════════════════════════════════════════
# 3. int (longobject.c) — Arbitrary Precision Integers
# ═══════════════════════════════════════════════════════════════════
#
# Python's int is actually a "long" (variable-length integer).
# In CPython:
#
#   // Include/longobject.h (simplified)
#   struct _longobject {
#       PyVarObject ob_base;       // ob_refcnt + ob_type + ob_size
#       digit ob_digit[1];         // Variable-length digits (base 2^30 or 2^15)
#   };
#
# Key facts:
#   - ob_size counts digits (positive = positive number, negative = negative number)
#   - Each digit is 30 bits on 64-bit systems (PYLONG_SHIFT = 30)
#   - Small ints (-5 to 257) are cached in a static array
#   - Operations are in longobject.c: long_add, long_sub, long_mul, etc.
#

def inspect_int_internals():
    print("=== int (PyLongObject) internals ===")

    # Small ints are cached
    for i in [-5, 0, 1, 42, 256, 257, 258, -6, 1000]:
        a = i
        b = i  # may or may not be the same object
        print(f"    {i:>6}: a is b = {a is b}")

    # Check the range of cached ints
    print()
    print(f"  sys.getsizeof(0)     = {sys.getsizeof(0)} bytes")
    print(f"  sys.getsizeof(42)    = {sys.getsizeof(42)} bytes")
    print(f"  sys.getsizeof(2**30) = {sys.getsizeof(2**30)} bytes")
    print(f"  sys.getsizeof(2**60) = {sys.getsizeof(2**60)} bytes")
    print(f"  sys.getsizeof(2**90) = {sys.getsizeof(2**90)} bytes")
    # Each addition of ~30 bits adds one digit → 4 or 8 bytes
    print()

inspect_int_internals()


# ═══════════════════════════════════════════════════════════════════
# 4. str (unicodeobject.c) — Unicode String
# ═══════════════════════════════════════════════════════════════════
#
# Python 3 strings are always Unicode (PEP 393).
# CPython uses flexible string representation:
#
#   struct {
#       PyObject ob_base;
#       Py_ssize_t ob_size;      // length in characters
#       Py_hash_t ob_shash;      // cached hash (-1 = not computed)
#       char ob_sval[1];         // character data (1, 2, or 4 bytes per char)
#       struct { ... } _state;   // compact/compressed flags
#   };
#
# Three internal representations:
#   - 1 byte per char (ASCII/Latin-1)  → PyUnicode_1BYTE_KIND
#   - 2 bytes per char (UCS-2)          → PyUnicode_2BYTE_KIND
#   - 4 bytes per char (UCS-4/UTF-32)   → PyUnicode_4BYTE_KIND
#

def inspect_str_internals():
    print("=== str (PyUnicodeObject) internals ===")

    strings = [
        ("ASCII", "hello"),
        ("Latin-1", "cafe" + chr(0xE9)),
        ("BMP", "\\u4f60\\u597d"),  # repr-safe display
        ("Non-BMP", "U+10348"),
    ]

    for label, s in strings:
        byte_size = sys.getsizeof(s)
        display = s if label == "ASCII" else repr(s).strip("'")
        print(f"  {label:>10} {display:<15} getsizeof={byte_size}")

    # String interning
    a = "hello_python"
    b = "hello_python"
    print(f"  Interned check: a = b = {a!r}")
    print(f"    a is b = {a is b}")

    # Not all strings are interned
    c = "hello" + "_" + "python"
    print(f"    c (computed) is b = {c is b}")
    print()

inspect_str_internals()


# ═══════════════════════════════════════════════════════════════════
# 5. list (listobject.c) — Dynamic Array
# ═══════════════════════════════════════════════════════════════════
#
#   typedef struct {
#       PyObject ob_base;
#       Py_ssize_t ob_size;          // Number of items (len())
#       PyObject **ob_item;          // Pointer to array of PyObject*
#       Py_ssize_t allocated;        // Allocated slots (may be > ob_size)
#   } PyListObject;
#
# Growth strategy: 0, 4, 8, 16, 24, 32, 40, ... (allocated += (allocated >> 3) + 6)
# This amortizes append to O(1).

def inspect_list_internals():
    print("=== list (PyListObject) internals ===")

    lst = []
    prev_size = sys.getsizeof(lst)
    print(f"  Empty list: getsizeof = {prev_size}")

    for i in range(1, 50):
        lst.append(i)
        curr_size = sys.getsizeof(lst)
        if curr_size != prev_size:
            print(f"  After {i:>3} items: getsizeof = {curr_size} (grew by {curr_size - prev_size})")
            prev_size = curr_size

    # List resizing overhead
    print()
    print("  Key insight: list overallocates to make append O(1)")
    print("  allocated = ob_size + (ob_size >> 3) + 6")
    print()

inspect_list_internals()


# ═══════════════════════════════════════════════════════════════════
# 6. dict (dictobject.c) — Hash Table
# ═══════════════════════════════════════════════════════════════════
#
# Python 3.6+ dict uses a "compact" representation (PEP 468):
#   - Two arrays: entries[] (sparse) and indices[] (dense)
#   - Entries store (hash, key, value)
#   - Indices store the index into entries[] for O(1) lookup
#
#   struct {
#       PyObject ob_base;
#       Py_ssize_t ma_used;          // Number of entries
#       Py_ssize_t ma_version_tag;   // Changes on every mutation
#       PyDictKeysObject *ma_keys;   // Key array + hash table
#       PyObject **ma_values;        // Value array (splitted table)
#   } PyDictObject;
#
# Since Python 3.7, dict maintains insertion order.

def inspect_dict_internals():
    print("=== dict (PyDictObject) internals ===")

    d = {}
    prev_size = sys.getsizeof(d)
    print(f"  Empty dict: getsizeof = {prev_size}")

    for i in range(1, 30):
        d[f"key_{i}"] = i
        curr_size = sys.getsizeof(d)
        if curr_size != prev_size:
            print(f"  After {i:>3} keys: getsizeof = {curr_size} (grew by {curr_size - prev_size})")
            prev_size = curr_size

    # Dict key ordering (3.7+)
    print()
    print("  Insertion order (3.7+):")
    d2 = {"z": 1, "a": 2, "m": 3, "b": 4}
    for k, v in d2.items():
        print(f"    {k!r}: {v}")

    # Keys/values/items views are dynamic
    keys = d2.keys()
    d2["new"] = 5
    print(f"  Keys view after mutation: {list(keys)}")
    print()

inspect_dict_internals()


# ═══════════════════════════════════════════════════════════════════
# 7. Garbage Collection — Cyclic Reference Detection
# ═══════════════════════════════════════════════════════════════════
#
# Reference counting handles most memory management, but cannot
# handle cycles (A references B, B references A → both have refcount 1).
#
# The garbage collector (Modules/gcmodule.c) uses a generational
# approach:
#
#   Generation 0: threshold 700 — frequent collections
#   Generation 1: threshold 10  — collected when gen0 triggers 10 times
#   Generation 2: threshold 10  — collected when gen1 triggers 10 times
#
# Algorithm: mark-and-sweep with 3-color marking (white/gray/black).
# Objects that implement __del__ or have weak references need
# special handling.

def inspect_gc():
    print("=== GC (garbage collector) ===")

    # GC thresholds
    print(f"  GC thresholds: {gc.get_threshold()}")
    print(f"  GC counts:     {gc.get_count()}")

    # Creating a cycle
    class Container:
        def __init__(self):
            self.ref = None

    gc.disable()
    a = Container()
    b = Container()
    a.ref = b
    b.ref = a
    gc.enable()

    print(f"  After cycle: gc.garbage count (unreachable) = {len(gc.garbage)}")
    gc.collect()
    print(f"  After collect: gc.garbage count = {len(gc.garbage)}")
    print()

    # GC tracked objects
    print("  GC tracks:")
    for t, name in [(int, "int"), (str, "str"), (list, "list"), (dict, "dict"),
                     (Container, "Container")]:
        tracked = gc.is_tracked(t())
        print(f"    {name:>15}: gc_tracked = {tracked}")

    # Only container types (list, dict, set, tuple, custom objects) are tracked
    print()

inspect_gc()


# ═══════════════════════════════════════════════════════════════════
# 8. Type Slots — The tp_* Dispatch Table
# ═══════════════════════════════════════════════════════════════════
#
# In CPython, type behavior is defined by a table of function pointers.
# These are the "slots" in PyTypeObject.
#
# Key slots:
#   tp_dealloc       — Destructor
#   tp_repr          — repr()
#   tp_str           — str()
#   tp_hash          — hash()
#   tp_call          — calling an instance
#   tp_iter          — __iter__
#   tp_iternext      — __next__
#   tp_richcompare   — <, <=, ==, !=, >=, >
#   tp_new           — __new__ (allocator)
#   tp_init          — __init__ (initializer)
#   tp_getattro      — __getattr__
#   tp_setattro      — __setattr__
#   tp_descr_get     — __get__ (descriptor protocol)
#   tp_descr_set     — __set__ (descriptor protocol)
#   tp_alloc         — allocator
#   tp_free          — free
#   tp_getset        — property map (getset_def)
#   tp_methods       — method table (PyMethodDef[])
#   tp_members       — member definition (PyMemberDef[])
#   tp_weaklist      — weak reference list
#   tp_dictoffset    — offset for __dict__
#

def inspect_type_slots():
    print("=== Type slots ===")

    # Use type slots that are accessible from Python
    for t in [int, str, list, dict, float, bool]:
        slots = []
        if hasattr(t, "__new__"):    slots.append("tp_new")
        if hasattr(t, "__init__"):   slots.append("tp_init")
        if hasattr(t, "__repr__"):   slots.append("tp_repr")
        if hasattr(t, "__str__"):    slots.append("tp_str")
        if hasattr(t, "__hash__"):   slots.append("tp_hash")
        if hasattr(t, "__call__"):   slots.append("tp_call")
        if hasattr(t, "__iter__"):   slots.append("tp_iter")
        if hasattr(t, "__next__"):   slots.append("tp_iternext")
        if hasattr(t, "__get__"):    slots.append("tp_descr_get")
        if hasattr(t, "__set__"):    slots.append("tp_descr_set")
        print(f"  {t.__name__:>10}: {', '.join(slots)}")
    print()

inspect_type_slots()


# ═══════════════════════════════════════════════════════════════════
# 9. The Descriptor Protocol
# ═══════════════════════════════════════════════════════════════════
#
# A descriptor is any object that defines __get__, __set__, or __delete__.
#   - Data descriptor: __get__ + __set__  (e.g., property)
#   - Non-data descriptor: __get__ only   (e.g., function, classmethod)
#
# When you access obj.attr, CPython's tp_getattro:
#   1. Looks in type.__mro__ for attr
#   2. If found, checks if it's a data descriptor (has __set__)
#   3. If data descriptor: calls descriptor.__get__(obj, type)
#   4. If not data descriptor: looks in obj.__dict__
#   5. If still not found and it's a non-data descriptor: calls __get__

def inspect_descriptor():
    print("=== Descriptor protocol ===")

    class DataDescriptor:
        """Data descriptor: has both __get__ and __set__"""
        def __get__(self, obj, objtype=None):
            return 42
        def __set__(self, obj, value):
            print(f"    DataDescriptor.__set__({obj}, {value})")

    class NonDataDescriptor:
        """Non-data descriptor: only __get__"""
        def __get__(self, obj, objtype=None):
            return 99

    class MyClass:
        data_descr = DataDescriptor()
        non_data = NonDataDescriptor()
        regular_attr = "plain"

    obj = MyClass()
    print(f"  Data descriptor:     {obj.data_descr}")
    obj.data_descr = 100  # calls __set__
    print(f"  After set (still 42): {obj.__dict__.get('data_descr', 'NOT in __dict__')}")

    print(f"  Non-data descriptor: {obj.non_data}")
    obj.__dict__["non_data"] = "shadow"
    print(f"  After __dict__ shadow: {obj.non_data} (__dict__ wins)")

    # Functions are descriptors!
    print()
    print("  Functions are non-data descriptors:")
    print(f"    MyClass.non_data      = {type(MyClass.non_data)}")
    print(f"    obj.non_data          = bound method")

    # property is a data descriptor
    print()
    print("  property is a data descriptor:")

    class WithProperty:
        @property
        def val(self):
            return "from property"

    obj2 = WithProperty()
    print(f"    obj2.val = {obj2.val}")
    obj2.__dict__["val"] = "from __dict__"
    print(f"    After __dict__ set: {obj2.val} (property wins!)")
    print()

inspect_descriptor()


# ═══════════════════════════════════════════════════════════════════
# 10. Simulating PyObject Layout with ctypes
# ═══════════════════════════════════════════════════════════════════
#
# We can use ctypes to read the raw PyObject memory and see
# ob_refcnt and ob_type values.

def peek_pyobject():
    print("=== Raw PyObject memory layout ===")

    # Py_ssize_t is typically int64 on 64-bit, int32 on 32-bit
    Py_ssize_t = ctypes.c_ssize_t  # 64-bit on x64

    class PyObject(ctypes.Structure):
        _fields_ = [
            ("ob_refcnt", Py_ssize_t),
            ("ob_type", ctypes.c_void_p),  # pointer to PyTypeObject
        ]

    obj = "hello"
    buf = id(obj)  # id() returns the memory address of the PyObject
    p = ctypes.cast(buf, ctypes.POINTER(PyObject))[0]

    print(f"  Object:   {obj!r}")
    print(f"  Address:  0x{buf:x}")
    print(f"  ob_refcnt: {p.ob_refcnt}")
    print(f"  ob_type:   0x{p.ob_type:x}")

    # For variable-length objects
    class PyVarObject(ctypes.Structure):
        _fields_ = [
            ("ob_refcnt", Py_ssize_t),
            ("ob_type", ctypes.c_void_p),
            ("ob_size", Py_ssize_t),
        ]

    lst = [1, 2, 3]
    p2 = ctypes.cast(id(lst), ctypes.POINTER(PyVarObject))[0]
    print()
    print(f"  List:     {lst}")
    print(f"  ob_refcnt: {p2.ob_refcnt}")
    print(f"  ob_size:   {p2.ob_size}  (len())")
    print()

peek_pyobject()


# ═══════════════════════════════════════════════════════════════════
# 11. Key Insights (memories for Jafar)
# ═══════════════════════════════════════════════════════════════════
#
# 1. PyObject is the universal base: ob_refcnt + ob_type.
#    PyVarObject adds ob_size for variable-length types.
#    Every Python value is accessed as PyObject*.
#
# 2. Reference counting (Py_INCREF/Py_DECREF) is the primary memory
#    management. GC handles only cycles. Py_DECREF → _Py_Dealloc
#    when refcount hits 0.
#
# 3. PyTypeObject is a PyObject with a giant table of function
#    pointers (tp_* slots). Type lookup uses MRO (tp_mro).
#
# 4. int (PyLongObject) uses variable-length digits (30-bit each).
#    Small ints (-5..257) are cached. Size grows by ~4 bytes per
#    30 bits of magnitude.
#
# 5. str (PyUnicodeObject) uses 1/2/4 byte-per-character encoding
#    depending on the max code point (PEP 393). Hash is cached.
#    Short strings are interned (ob_shash set early).
#
# 6. list (PyListObject) is a dynamic array of PyObject* pointers.
#    Growth: allocated += (allocated >> 3) + 6. O(1) amortized append.
#    ob_item points to the start of the array.
#
# 7. dict (PyDictObject) uses a compact hash table (PEP 468) with
#    split key/value arrays. Insertion order preserved since 3.7.
#    ma_version_tag changes on every mutation.
#
# 8. GC uses generational mark-and-sweep (3 generations).
#    Only container types are tracked. Thresholds: (700, 10, 10).
#    gc module exposes collection control.
#
# 9. Descriptor protocol: data descriptors (__set__) win over
#    __dict__, non-data descriptors (__get__ only) lose to __dict__.
#    Functions, properties, classmethod, staticmethod are descriptors.
#
# 10. ctypes can peek at raw PyObject memory using id() as the
#     memory address. ob_refcnt, ob_type, ob_size are visible.
#

if __name__ == "__main__":
    pass
