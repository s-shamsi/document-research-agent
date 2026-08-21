from pathlib import Path
from pypdf import PdfReader
from io import BytesIO
import chromadb

# create client once at module load time: repeated initialisations waste compute
_chroma_client = chromadb.PersistentClient(path="chroma_db")

def load_documents(folder: str) -> list[dict]:
    """
    Read all pdf/txt files in a folder, return list of {source, text}.
    """
    docs = []
    for path in Path(folder).glob("*"):
        if path.suffix == ".pdf":
            reader = PdfReader(str(path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif path.suffix == ".txt":
            text = path.read_text(encoding="utf-8", errors="ignore")
        else:
            continue
        docs.append({"source": str(path.name), "text": text})
    return docs

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> list[str]:
    """
    Split text into overlapping chunks of ~chunk_size characters.
    """
    assert overlap < chunk_size, "overlap must be less than chunk_size"     # prevents infinite loops.
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap   # step forward, but overlap with previous chunk
    return [c.strip() for c in chunks if c.strip()]  # remove empty chunks

def get_collection(persist_dir: str = "chroma_db"):
    """
    Connect to ChromaDB and return the documents collection.
    Note: persist_dir parameter is ignored since client is singleton at "chroma_db"
    """
    # client = chromadb.PersistentClient(path=persist_dir)
    # return client.get_or_create_collection("documents")
    return _chroma_client.get_or_create_collection("documents")

def build_vector_store(docs: list[dict], persist_dir: str = "chroma_db"):
    """
    Ingest documents, chunk them, and store them in ChromaDB with [ids, documents, metadatas] attributes.
    """
    collection = get_collection(persist_dir)

    ids, texts, metadatas = [], [], []
    for doc in docs:
        for i, chunk in enumerate(chunk_text(doc["text"])):
            ids.append(f"{doc['source']}-{i}")
            texts.append(chunk)
            metadatas.append({"source": doc["source"], "chunk_index": i})

    collection.add(ids=ids, documents=texts, metadatas=metadatas)
    return collection

def embed_uploaded_file(filename: str, content: bytes, content_type: str) -> dict:
    """
    Process an uploaded file: decode it, chunk it, embed it.
    Reuses existing chunk_text() and get_collection() functions.
    """
    # Step 1: Decode bytes to text (same logic as load_documents applied to uploaded files)
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        text = content.decode("utf-8", errors="ignore")
    
    # Step 2: Reuse existing chunking and embedding functions
    chunks = chunk_text(text)
    collection = get_collection()
    ids = [f"{filename}-{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks, metadatas=metadatas)
    
    return {"name": filename, "size": len(content), "type": content_type}