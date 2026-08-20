from app.agent import resolve_tools, stream_final_answer

question = "Is there anything about protons in the documents?"
messages, queries_made, retrieved_sources = resolve_tools(question)
answer = "".join(stream_final_answer(messages))

# Agent Response Check
print("ANSWER:")
print(result["answer"])
print()
print("QUERIES MADE:", queries_made)
print("SOURCES RETRIEVED:", retrieved_sources)

# Basic sanity checks
assert isinstance(answer, str) and answer.strip(), "answer should be a non-empty string"
assert isinstance(queries_made, list), "queries_made should be a list"
assert isinstance(retrieved_sources, list), "retrieved_sources should be a list"
print("\nAll structural checks passed.")