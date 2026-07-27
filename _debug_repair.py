"""Debug the full repair flow."""
import repair_agent
from repair_agent import RepairAgent

def mock_exec(code):
    try:
        exec_globals = {}
        exec(code, exec_globals)
        return {"stdout": "", "stderr": "", "exit_code": 0, "error": ""}
    except Exception as e:
        return {"stdout": "", "stderr": f"{type(e).__name__}: {e}", "exit_code": 1, "error": str(e)}

code = "def test():\n    return undefined_var\nprint(test())"
result = mock_exec(code)
print("result stderr:", repr(result["stderr"]))
print("result error:", repr(result["error"]))

ec = repair_agent.ErrorClassifier()
t, m, c = ec.classify(result)
print(f"Direct classify: type={t!r}, msg={m!r}")

ra = RepairAgent(execution_runner=mock_exec)
repair = ra.repair(code, error_result=result, debug=False)
print(f"Repair result: success={repair.success}, error_type={repair.error_type!r}")
print(f"root_cause={repair.root_cause!r}")
if repair.incidents:
    inc = repair.incidents[0]
    print(f"First incident: type={inc['error_type']!r}, msg={inc['error_message']!r}")
