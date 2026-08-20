from app.agent import run_agent

result = run_agent("Is there anything about protons in the documents?")

# Agent Response Check
print("ANSWER:")
print(result["answer"])
print()
print("QUERIES MADE:", result["queries_made"])
print("SOURCES RETRIEVED:", result["sources"])

# Basic sanity checks
assert isinstance(result, dict), "run_agent should return a dict"
assert "answer" in result and isinstance(result["answer"], str), "answer should be a string"
assert "queries_made" in result and isinstance(result["queries_made"], list), "queries_made should be a list"
assert "sources" in result and isinstance(result["sources"], list), "sources should be a list"
print("\nAll structural checks passed.")