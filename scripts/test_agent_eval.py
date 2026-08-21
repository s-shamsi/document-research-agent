"""
8 questions to test retrieval & generation
- 2 x document_search
- 2 x web_search
- 4 x document_search & web_search
"""
import time
from app.agent import resolve_tools, stream_final_answer


EVAL_QUESTIONS = {
    "document_only": [
        "What do the documents say about baryon decay?",
        "What is discussed about mesons in the documents?",
    ],
    "web_search_only": [
        "What are the breaking news today?",
        "What are the most significant AI developments from August 2026?",
    ],
    "mixed": [
        "How do the physics theories in the documents compare to verified experimental results?",
        "What experimental evidence beyond the documents validates the SU(3) symmetry?",
        "How does the entropy model in the documents relate to experimentally verified results?",
        "How do the theoretical predictions in the documents align with latest CERN results?",
    ],
}

for category, questions in EVAL_QUESTIONS.items():
    print(f"\n{'='*60}")
    print(f"CATEGORY: {category.replace('_', ' ').upper()}")
    print(f"{'='*60}\n")
    
    for i, q in enumerate(questions, 1):
        print(f"[{category}-{i}] Q: {q}")
        messages, queries_made, retrieved_sources = resolve_tools(q)
        answer = "".join(stream_final_answer(messages))
        
        print(f"     Queries made: {queries_made}")
        print(f"     Sources: {retrieved_sources}")
        print(f"     Answer (first 250 chars):\n     {answer[:250]}...")
        print()

        time.sleep(3)  # wait 3s between questions to avoid rate limit