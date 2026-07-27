"""Minimal reproduction of the sandbox timeout for debugging."""
import sys
import time
import traceback
from threading import Thread, Event

code = 'while True: pass'
timeout = 2.0

print(f"code={code!r}, timeout={timeout}", flush=True)

result_container = []
error_container = []
done_event = Event()

def exec_code():
    local_vars = {}
    try:
        compiled = compile(code.strip(), "<sandbox>", "exec")
        exec(compiled, {}, local_vars)
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
print(f"Thread started, joining with timeout={timeout}", flush=True)
t.join(timeout=timeout)
elapsed = time.time() - t0
print(f"join returned after {elapsed:.2f}s, done={done_event.is_set()}", flush=True)

if not done_event.is_set():
    print("TIMEOUT", flush=True)
else:
    print("COMPLETED", flush=True)
