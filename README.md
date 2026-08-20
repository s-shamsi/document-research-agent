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
- [Evaluation](#evaluation)
- [Limitations](#limitations)
- [Future Roadmap](#future-roadmap)

## Objective 

Create the backend component of a research agent to demonstrate your technical skills and experience in machine learning integration. The service should work with the provided UI, allowing users to ask the agent research queries, and streaming back the answers. Users should be able to upload source files for embedding that the agent may use in addition to external sources.

## Quick Start

**Install:**
```bash
pip install fastapi uvicorn python-dotenv pydantic groq ddgs chromadb pypdf
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

**Response:** markdown text stream

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

## Evaluation

## Limitations

- No persistence guarantees on `chroma_db/` across environments:
    - local on-disk store, and not backed up 
    - uploaded documents share one collection
- No conversation memory:
    - each `/api/research` call is independent of previous calls
    - `max_steps=5` in `resolve_tools()` is hard-coded
- No dynamic management of agent orchestration:
    - when `n_steps = max_steps = 5`, the agent is forced to output its work, but the user is unaware
    - `max_steps = N` where $N = f(file_size, query_complexity, user_input)$ 
    - `tool_calls` are not robustly managed depending on input query and user specifications
- No retry or result-quality filtering for `web_search()` beyond DDG's default

## Future Roadmap