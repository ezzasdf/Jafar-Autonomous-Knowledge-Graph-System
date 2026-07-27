import re
try:
    exec("print(undefined_var)")
except NameError as e:
    stderr = f"{type(e).__name__}: {e}"
    error = str(e)
    combined = f"{stderr}\n{error}"
    print("stderr:", repr(stderr))
    print("error:", repr(error))
    print("combined:", repr(combined))

    pattern = re.compile(r"NameError:\s*name\s+'(\w+)' is not defined")
    m = pattern.search(combined)
    print("Match with quotes:", m)

    # Try without quotes
    pattern2 = re.compile(r"NameError:\s*name\s+(\w+)\s+is not defined")
    m2 = pattern2.search(combined)
    print("Match without quotes:", m2)

    # Also try parsing the actual pattern used
    all_patterns = {
        "name_error": re.compile(r"NameError:\s*name\s+'(\w+)' is not defined"),
    }
    for name, p in all_patterns.items():
        m = p.search(combined)
        print(f"  {name}: match={m}", end="")
        if m:
            print(f", group={m.group(1)}", end="")
        print()
