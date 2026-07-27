import sys; sys.path.insert(0, '.')
from test_agent import TestAgent
agent = TestAgent(None)
agent.analyze_module('memory_system.py', target_functions=['add_relationship', 'add_causal_model'])
agent.generate_tests()
for t in agent.tests:
    print(f'=== {t["target_name"]} ===')
    print(t['code'])
    print()
