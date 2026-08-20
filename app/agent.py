# Placeholder for agent logic.
# This will contain:
# - search_documents() ✔️
# - web_search() ❌
# - resolve_tools() (the agent loop) ✔️
# - stream_final_answer() ✔️

import os
import textwrap
import json
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
    Input: User Question (str).
    Output: LLM Response (dict) containing answer and sources.
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

def search_documents(query: str, n_results: int = 4) -> str:
    """
    The document search/retrieval function the LLM calls.
    Input: User Query → ChromaDB Semantic Search → Output: Context for LLM Response
    """
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=n_results)
    chunks = results["documents"][0]
    sources = results["metadatas"][0]
    return "\n\n".join(
        f"[Source: {s['source']}, chunk {s['chunk_index']}]\n{c}"
        for c, s in zip(chunks, sources)
    )

# JSON schema describing the search_tool to Groq.
search_tool = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": 
        """
        - Search the document collection for relevant passages. 
        - Call this whenever you need information you don't already have.
        """,
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "A focused search query"}},
            "required": ["query"]
        }
    }
}


def run_agent(question: str, max_steps: int = 5) -> dict:
    """
    Agent Orchestrator: this is the multi-step (max_steps) ReAct loop.
    Input: User question (str), max_steps of agent reasoning loop.
    Output: LLM Agent response (dict) containing answer and sources.
    Process: Input → messages = [...] records reasoning → Agent Loop OR Fallback Response → Ouput
    Agent Loop: Context + LLM Response → Check for tool_calls → (possible) tool usage(s) → Loop Answer
    """
    messages = [
        {"role": "system", 
         "content": ("You are a document research agent. You do not have context initially. "
                     "Use search_documents to find relevant information before answering. "
                     "You may search multiple times if the search result is incomplete. "
                     "Once search results are returned, answer strictly using those results.")},
        {"role": "user", "content": question}
        ]

    sources_used = []

    for step in range(max_steps):
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=1000,
            tools=[search_tool],
            messages=messages
        )
        message = response.choices[0].message

        # Safely check for tool calls regardless of finish_reason value
        if message.tool_calls:
            messages.append(message)
            for tool_call in message.tool_calls:
                # Handle unexpected tool names or malformed arguments safely
                if tool_call.function.name == "search_documents":
                    try:
                        args = json.loads(tool_call.function.arguments)
                        query_str = args.get("query", "")
                    except json.JSONDecodeError:
                        query_str = ""

                    result_text = search_documents(query_str) if query_str else "Invalid query provided."
                    if query_str and query_str not in sources_used:
                        sources_used.append(query_str)

                else:
                    result_text = "Error: Tool not found."

                # Always append a tool response to prevent API context errors
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text
                })

            continue

        # Return full response including reasoning blocks
        content = (message.content or "").strip()
        return {"answer": content, "queries_made": sources_used}

    # Fallback when max steps are reached
    messages.append({
        "role": "user",
        "content": "You reached the maximum search limit. Answer the question as best as you can using ONLY the tool output provided above."
    })

    final_resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=1000,
        messages=messages
    )

    fallback_content = (final_resp.choices[0].message.content or "").strip()

    return {"answer": "Reached max reasoning steps without a final answer. " + fallback_content,
            "queries_made": sources_used}
