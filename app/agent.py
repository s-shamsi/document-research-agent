# Placeholder for agent logic.
# This will contain:
# - search_documents()
# - web_search()
# - resolve_tools() (the agent loop)
# - stream_final_answer()

import os
import textwrap
from groq import Groq
from dotenv import load_dotenv
from app.ingest import get_collection

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# MODEL = "llama-3.3-70b-versatile"         #error: model_not_found
# MODEL = "llama-3.3-70b-specdec"           #error: model_decommissioned
# MODEL = "llama3-70b-8192"                 #error: model_decommissioned
# MODEL = "llama-3.1-8b-instant"            #error: model_not_found
# MODEL = "openai/gpt-oss-120b"             #overkill for the task
MODEL = "openai/gpt-oss-20b"
# MODEL = "qwen/qwen3.6-27b"                  #issue: internal reasoning tokens leaked into user responses


def answer_question(question: str, n_results: int = 3) -> dict:
    """
    Accepts the user's question (as a string) and returns a dictionary (containing the answer) as the output.
    """
    collection = get_collection()
    results = collection.query(query_texts = [question], n_results = n_results)

    chunks = results["documents"][0]
    sources = results["metadatas"][0]

    context = "\n \n".join(f"[Source: {s['source']}, Chunk: {s['chunk_index']}] \n {c}"
                           for c, s in zip(chunks, sources))

    prompt = textwrap.dedent(f"""
                                ### INSTRUCTIONS:
                                - You are a document research assistant. 
                                - Answer the question using **ONLY THE CONTEXT PROVIDED** below.
                                - If the context does not contain the answer, say so clearly!
                                - **DO NOT GUESS under any circumstances**.
                                - **ALWAYS CITE SOURCES** using their [Source: ..., Chunk: ...] tags.
                                
                                ### CONTEXT: {context}

                                ### QUESTION: {question}
                
                             """)

    response = client.chat.completions.create(model = MODEL,
                                              max_tokens = 1_000,
                                              messages = [{"role": "user", "content": prompt}])

    return{"answer": response.choices[0].message.content,
           "sources": [s["source"] for s in sources]}