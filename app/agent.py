import os
import json
from groq import Groq
from dotenv import load_dotenv
from app.ingest import get_collection
from duckduckgo_search import DDGS

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# MODEL = "llama-3.3-70b-versatile"         #error: model_not_found
# MODEL = "llama-3.3-70b-specdec"           #error: model_decommissioned
# MODEL = "llama3-70b-8192"                 #error: model_decommissioned
# MODEL = "llama-3.1-8b-instant"            #error: model_not_found
# MODEL = "openai/gpt-oss-120b"             #overkill for the task
MODEL = "openai/gpt-oss-20b"
# MODEL = "qwen/qwen3.6-27b"                  #issue: internal reasoning tokens leaked into user responses


# def answer_question(question: str, n_results: int = 3) -> dict:
#     """
#     Input: User Question (str).
#     Output: LLM Response (dict) containing answer and sources.
#     """
#     collection = get_collection()
#     results = collection.query(query_texts = [question], n_results = n_results)
#
#     chunks = results["documents"][0]
#     sources = results["metadatas"][0]
#
#     context = "\n \n".join(f"[Source: {s['source']}, Chunk: {s['chunk_index']}] \n {c}"
#                            for c, s in zip(chunks, sources))
#
#     prompt = textwrap.dedent(f"""
#                                 ### INSTRUCTIONS:
#                                 - You are a document research assistant.
#                                 - Answer the question using **ONLY THE CONTEXT PROVIDED** below.
#                                 - If the context does not contain the answer, say so clearly!
#                                 - **DO NOT GUESS under any circumstances**.
#                                 - **ALWAYS CITE SOURCES** using their [Source: ..., Chunk: ...] tags.
#
#                                 ### CONTEXT: {context}
#
#                                 ### QUESTION: {question}
#
#                              """)
#
#     response = client.chat.completions.create(model = MODEL,
#                                               max_tokens = 1_000,
#                                               messages = [{"role": "user", "content": prompt}])
#
#     return{"answer": response.choices[0].message.content,
#            "sources": [s["source"] for s in sources]}


def search_documents(query: str, n_results: int = 4) -> tuple[str, list[str]]:
    """
    The document search/retrieval function the LLM calls.
    Input: User Query
    Process: ChromaDB Semantic Search & Source Document Retrieval
    Output: (Context for LLM Response, list_of_sources)
    """
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=n_results)
    chunks = results["documents"][0]
    sources = results["metadatas"][0]
    formatted = "\n\n".join(
        f"[Source: {s['source']}, chunk {s['chunk_index']}]\n{c}"
        for c, s in zip(chunks, sources)
    )
    source_names = [s["source"] for s in sources]
    return formatted, source_names


def web_search(query: str, n_results: int = 4) -> tuple[str, list[str]]:
    """
    The public-internet search/retrieval function the LLM calls.
    Input: User Query
    Process: DuckDuckGo text search
    Output: (Context for LLM Response, list_of_source_urls)
    """
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=n_results))
    formatted = "\n\n".join(
        f"[Web: {r['title']}]({r['href']})\n{r['body']}"
        for r in results
    )
    source_urls = [r["href"] for r in results]
    return formatted, source_urls


# JSON schema describing the document_search_tool to Groq.
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

# JSON schema describing the web_search tool to Groq.
web_search_tool = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description":
        """
        - Search the public internet for information not in the user's uploaded documents.
        - Call this for current events or information outside the document collection.
        """,
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "A focused search query"}},
            "required": ["query"]
        }
    }
}


def resolve_tools(question: str, max_steps: int = 5) -> tuple[list[dict], list[str], list[str]]:
    """
    Agent Orchestrator: this is the multi-step (max_steps) ReAct loop.
    Input: User question (str), max_steps of agent reasoning loop.
    Output: (resolved conversation, queries made, sources retrieved)
            NOT a finished answer.
            The caller passes the returned messages to stream_final_answer(),
            which generates and streams the actual answer text.
    Process: Input -> messages = [...] records reasoning -> Agent Loop resolves tool calls -> Output
    Agent Loop: Context + LLM Response -> Check for tool_calls -> (possible) tool usage(s) -> Loop
    """
    messages = [
        {"role": "system",
         "content": ("You are a document research agent. You do not have context initially. "
                     "Use search_documents to find relevant information in the user's uploaded documents. "
                     "Use web_search for information not in the documents, or for external information. "
                     "You may call either tool multiple times. "
                     "Once search_documents and web_search results are returned, answer ONLY using those results. "
                     "Always cite your sources.")},
        {"role": "user", "content": question}
        ]

    queries_made = []
    retrieved_sources = []   # tracks documents/URLs retrieved

    for step in range(max_steps):
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=1000,
            tools=[search_tool, web_search_tool],
            messages=messages
        )
        message = response.choices[0].message

        # Safely check for tool calls regardless of finish_reason value
        if message.tool_calls:
            messages.append(message)
            for tool_call in message.tool_calls:
                # Handle unexpected tool names or malformed arguments safely
                try:
                    args = json.loads(tool_call.function.arguments)
                    query_str = args.get("query", "")
                except json.JSONDecodeError:
                    query_str = ""

                if not query_str:
                    result_text = "Invalid query provided."
                elif tool_call.function.name == "search_documents":
                    result_text, matched_sources = search_documents(query_str)
                    if query_str not in queries_made:
                        queries_made.append(query_str)
                    for match in matched_sources:
                        if match not in retrieved_sources:
                            retrieved_sources.append(match)
                elif tool_call.function.name == "web_search":
                    result_text, matched_sources = web_search(query_str)
                    if query_str not in queries_made:
                        queries_made.append(query_str)
                    for match in matched_sources:
                        if match not in retrieved_sources:
                            retrieved_sources.append(match)
                else:
                    result_text = "Error: Tool not found."

                # Append tool response to prevent API context errors
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text
                })

            continue  # go back to the top of the OUTER loop, ask the model again

        break  # model did NOT call a tool this round — it's ready to answer
    else:
        # Loop exhausted all max_steps without ever hitting the break above
        # (model kept calling tools on every single step)
        messages.append({
            "role": "user",
            "content": "You reached the maximum search limit. Answer the question as best "
                       "as you can using ONLY the tool output provided above."
        })

    return messages, queries_made, retrieved_sources


def stream_final_answer(messages: list[dict]):
    """
    Generator: takes the resolved conversation from resolve_tools() and streams
    the final answer text as Groq generates it, piece by piece.
    """
    stream = client.chat.completions.create(
        model=MODEL, max_tokens=1000, messages=messages, stream=True
    )
    for event in stream:
        delta = event.choices[0].delta.content
        if delta:
            yield delta