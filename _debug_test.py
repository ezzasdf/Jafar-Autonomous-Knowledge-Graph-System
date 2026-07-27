"""Minimal reproduction of the classification issue."""
from repair_agent import ErrorClassifier

def mock_exec(code):
    if "raise value_error" in code:
        return {"stdout": "", "stderr": "ValueError: something wrong", "exit_code": 1, "error": "ValueError"}
    if "raise name_error" in code:
        return {"stdout": "", "stderr": "NameError: name 'undefined_var' is not defined", "exit_code": 1, "error": "NameError"}
    try:
        exec_globals = {}
        exec(code, exec_globals)
        return {"stdout": "", "stderr": "", "exit_code": 0, "error": ""}
    except Exception as e:
        result = {"stdout": "", "stderr": f"{type(e).__name__}: {e}", "exit_code": 1, "error": str(e)}
        return result

code = "def test():\n    return undefined_var\nprint(test())"
result = mock_exec(code)
print("Result keys:", result.keys())
print("stderr:", repr(result["stderr"]))
print("error:", repr(result["error"]))
print("exit_code:", result["exit_code"])

ec = ErrorClassifier()
error_type, error_message, exit_code = ec.classify(result)
print(f"Classified: type={error_type!r}, msg={error_message!r}, exit={exit_code}")
