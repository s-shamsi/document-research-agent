from fastapi import FastAPI, File, UploadFile                               # app + file uploads.
from fastapi.middleware.cors import CORSMiddleware                          # lets React frontend call app API 
from fastapi.responses import PlainTextResponse, StreamingResponse          # error messages + api.ts compatibility
from pydantic import BaseModel                                              # pydantic class for request schemas
from dotenv import load_dotenv
# from app.agent import answer_question                                     # deprecated 1-shot "agent" response
from app.ingest import embed_uploaded_file
from app.agent import resolve_tools, stream_final_answer

load_dotenv()
app = FastAPI()

# CORS: tell browser to enable frontend-backend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                                # should restrict to trusted origins
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

# Query 1-Shot Model:
# class QuestionRequest(BaseModel):
#     question: str

# @app.post("/ask")
# def ask(req: QuestionRequest):
#     return answer_question(req.question)

@app.post("/api/sources")
def upload_sources(files: list[UploadFile] = File(...)):
    """
    Upload and embed source files.
    """
    uploaded = []
    for file in files:
        content = file.file.read()
        try:
            meta = embed_uploaded_file(
                file.filename,
                content,
                file.content_type or "application/octet-stream"
            )
            uploaded.append(meta)
        except Exception as e:
            return PlainTextResponse(f"Failed to process {file.filename}: {e}", status_code=400)
    return {"uploaded": uploaded}


class ResearchRequest(BaseModel):
    request: str


@app.post("/api/research")
def research(req: ResearchRequest):
    """
    Research endpoint: runs the agent loop, streams the final answer.
    """
    if not req.request.strip():
        return PlainTextResponse("Request cannot be empty.", status_code=400)

    try:
        messages, queries_made, retrieved_sources = resolve_tools(req.request)
    except Exception as e:
        return PlainTextResponse(f"Research failed: {e}", status_code=500)

    return StreamingResponse(stream_final_answer(messages), media_type="text/plain")