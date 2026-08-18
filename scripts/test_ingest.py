from app.ingest import load_documents, chunk_text

# 1. Load documents
docs = load_documents("documents")  # Adjust the folder path as needed
print(f"Successfully loaded {len(docs)} document(s).\n")

#2. Chunk documents: loop through each document, chunk it, and print the stats
for d in docs:
    chunks = chunk_text(d["text"])
    print(f"File: {d['source']} -> {len(chunks)} chunks created")

from app.ingest_simple import get_collection, build_vector_store

#3. Create ChromaDB client object
collection = get_collection()

# 4. Build the vector store (which handles chunking and storing)
build_vector_store(docs)
print("Ingested and created a vector DB for", len(docs), "documents")

#5. (Mono) Query the collection
results = collection.query(query_texts=["Any rooms available?"], n_results=3)

#6. Loop directly through first query in results
for i, (doc, meta, dist) in enumerate(zip(results["documents"][0], results["metadatas"][0], results["distances"][0]), start=1):
    print(f"\n[Result {i}]")
    print(f"📁 Source File : {meta.get('source')}")
    print(f"🧩 Chunk Index : {meta.get('chunk_index')}")
    print(f"📊 Distance    : {dist}")
    print("-" * 60)
    print(doc.strip())
    print("-" * 60)    

# #7. (Multi) Query the collection
# results = collection.query(query_texts=["Any rooms available?"], n_results=3)