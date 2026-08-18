from pathlib import Path
from pypdf import PdfReader

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
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap   # step forward, but overlap with previous chunk
    return [c.strip() for c in chunks if c.strip()]  # remove empty chunks

