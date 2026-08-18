from app.ingest import load_documents, chunk_text

# 1. Load documents
docs = load_documents("documents")  # Adjust the folder path as needed
print(f"Successfully loaded {len(docs)} document(s).\n")

#2. Test document chunking: loop through each document, chunk it, and print the stats
for d in docs:
    chunks = chunk_text(d["text"])
    print(f"File: {d['source']} -> {len(chunks)} chunks created")