"""Debug: test with tracemalloc and watchdog."""
import sys
import time
import traceback
import tracemalloc
import ctypes
from threading import Thread, Event
from code_sandbox import _strip_dangerous, _build_safe_globals

code = 'while True: pass'
timeout = 2.0

print(f"1. code={code!r}, timeout={timeout}", flush=True)

# Strip dangerous
code2, blocked = _strip_dangerous(code)
print(f"2. stripped: blocked={blocked}", flush=True)

# Build safe globals
safe_globals = _build_safe_globals()
print(f"3. safe_globals built", flush=True)

# Start tracemalloc
tracemalloc.start()
print(f"4. tracemalloc started", flush=True)

result_container = []
error_container = []
done_event = Event()
stop_event = Event()

def exec_code():
    local_vars = {}
    try:
        compiled = compile(code2.strip(), "<sandbox>", "exec")
        exec(compiled, safe_globals, local_vars)
        result_container.append(local_vars)
    except Exception as e:
        error_container.append((e, traceback.format_exc()))
    finally:
        done_event.set()

def target_wrapper():
    exec_code()

t = Thread(target=target_wrapper, daemon=True)
t0 = time.time()
t.start()
print(f"5. Thread started, ident={t.ident}", flush=True)

# Start watchdog
MEMORY_CHECK_INTERVAL = 0.05
def watchdog(thread_id, max_bytes, stop_ev):
    try:
        while not stop_ev.is_set():
            current, peak = tracemalloc.get_traced_memory()
            if peak > max_bytes:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_ulong(thread_id),
                    ctypes.py_object(MemoryError("OOM")))
                break
            stop_ev.wait(MEMORY_CHECK_INTERVAL)
    except Exception:
        pass

max_bytes = 256 * 1024 * 1024
w = Thread(target=watchdog, args=(t.ident, max_bytes, stop_event), daemon=True)
w.start()
print(f"6. Watchdog started, joining thread", flush=True)

t.join(timeout=timeout)
stop_event.set()
elapsed = time.time() - t0
print(f"7. join returned after {elapsed:.2f}s, done={done_event.is_set()}", flush=True)

current_mem, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"8. tracemalloc stopped, peak={peak_mem}B", flush=True)

if not done_event.is_set():
    print("9. TIMEOUT", flush=True)
else:
    print("9. COMPLETED", flush=True)
