# Document Research Agent

A FastAPI backend for a *document research agent*. 

The agent:
- queries a Groq-hosted (0 cost) LLM. 
- answers questions by extracting information from: 
    1. **user-uploaded documents** (RAG over ChromaDB)
    2. **web search** (DuckDuckGo)
- orchestrates a ReAct-style tool-calling loop
- pairs with the provided React frontend that streams (markdown) responses.

> This README primarily documents the implementation in `app/`. See [Roadmap](#roadmap--future-direction) for the future development direction.

## Table of Contents

- [Objective](#objective)
- [Quick Start](#quick-start)
- [API](#api)
- [Tech Stack](#tech-stack)
- [Key Files](#key-files)
- [Logical Flow](#logical-flow)
- [FrontEnd](#frontend)
- [Evaluation](#evaluation)
- [Limitations](#limitations)
- [Future Roadmap](#future-roadmap)

## Objective 

Create the backend component of a research agent to demonstrate your technical skills and experience in machine learning integration. The service should work with the provided UI, allowing users to ask the agent research queries, and streaming back the answers. Users should be able to upload source files for embedding that the agent may use in addition to external sources.

## Quick Start

**Install:**
```bash
pip install -r requirements.txt
```

**Configure:** Create `.env`:
```env
GROQ_API_KEY=your_key_here
```

**Run:**
```bash
uvicorn app.main:app --reload --port 8787
```

## API

Health check: `GET /health`

### `POST /api/sources`
- upload (.pdf, .txt) files
- extract, chunk, and store in vectorDB (ChromaDB)

**Request:** `multipart/form-data` with `files` field  
**Response:** List of uploaded files with sizes

### `POST /api/research`
- input: ask a research question
- output: stream markdown response

**Request:**
```json
{ "request": "User question?" }
```

**Response:** markdown text stream with a sources footer listing all retrieved documents and URLs.

## Tech Stack

| Component | Tool |
|-----------|------|
| API | FastAPI |
| LLM | Groq (`gpt-oss-20b`) |
| Vector DB | ChromaDB |
| PDF parsing | pypdf |
| Web search | DuckDuckGo |

## Key Files

```
app/
  main.py      — FastAPI routes (/health, /api/sources, /api/research)
  ingest.py    — PDF/TXT parsing, chunking, ChromaDB storage
  agent.py     — ReAct loop, tool execution, streaming response
```

## Logical Flow

**Upload documents:**
- user uploads to `POST /api/sources`
- `ingest.py` extracts text
- extracted text is chunked
- text chunks are embedded and stored in ChromaDB
- documents directory is ready for semantic search

**Answer questions:**
- user sends research query to `POST /api/research`
- `agent.py` runs a ReAct loop (`max_steps=5`):
   - send query & message history to LLM available `tools=[search_tool, web_search_tool]`
   - LLM decides which tool(s) to call (if at all)
   - execute tool calls
   - append tool call history and corresponding results to message history
   - loop until final_answer OR `max_steps`
- `stream_final_answer()` re-calls LLM with `stream=True`

## FrontEnd

```bash
git clone <frontend-repo-url>
cd <frontend-folder>
npm install
npm run dev
```
- backend runs on `http://localhost:8787`
- frontend runs on `http://localhost:5173`
- Confirm the frontend points to that backend URL in `api.ts` 

## Evaluation

**To run:**
```bash
   python scripts/test_agent_eval.py
```

8 questions to test retrieval & generation:
- 2 x search_tool only calls
- 2 x web_search_tool only calls
- 4 x search_tool & web_search_tool mixed calls

 *Note: Requires GROQ_API_KEY and ChromaDB ingestion (run `scripts/test_ingest.py` first).*

**What should be tested:**
- Retrieval quality: does search output relevant chunks?
- Response Faithfulness: are the responses rooted in sources?
- Citation Usage: are cited sources used and reflected in answers?
- Hallucination Resistance: does agent decline to answer unanswerable questions instead of hallucinating?

## Limitations

- **No multi-user isolation**: all uploaded documents share one ChromaDB collection
- **No conversation memory**: each `/api/research` call is independent
- **Fixed `max_steps=5`**: stops mid-loop if exceeded, without signaling the state to the user
- **No web_search filtering**: results are ranked by DuckDuckGo only

## Future Roadmap

- **Unit tests**: Add `pytest`/`pytest-asyncio` tests.
- **Token-aware chunking**: token-aware chunking preserves document structure better.
- **Conversation memory**: support multi-turn queries so that the agent can reference and utilise searches and responses.
- **Citation validation**: verification method to confirm that cited sources actually support the claims (with judgements handed off to second-LLM).
- **Upload queuing**: upload queue for ingestion, so `/api/sources` returns immediately with a job ID.
- **OOP refactoring**: refactor ingest and agent logic into classes.
- **Error handling**: Add try-catch blocks with error messages for common failures (invalid PDFs, network timeouts, rate limits).