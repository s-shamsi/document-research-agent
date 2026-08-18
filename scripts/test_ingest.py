from app.ingest import load_documents, chunk_text

# 1. Load documents
docs = load_documents("documents")  # Adjust the folder path as needed
print(f"Successfully loaded {len(docs)} document(s).\n")

#2. Chunk documents: loop through each document, chunk it, and print the stats
for d in docs:
    chunks = chunk_text(d["text"])
    print(f"File: {d['source']} -> {len(chunks)} chunks created")

from app.ingest import get_collection, build_vector_store

#3. Create ChromaDB client object
collection = get_collection()

# 4. Build the vector store (which handles chunking and storing)
build_vector_store(docs)
print("Ingested and created a vector DB for", len(docs), "documents")

#5. (Mono) Query the collection
results = collection.query(query_texts=["Any rooms available?"], n_results=3)

#6. Loop directly through single query in results
print(f"\n🔍 Results for query: 'Any rooms available?'")

for i, (doc, meta, dist) in enumerate(zip(results["documents"][0], results["metadatas"][0], results["distances"][0]), start=1):
    print(f"\n[Result {i}]")
    print(f"📁 Source File : {meta.get('source')}")
    print(f"🧩 Chunk Index : {meta.get('chunk_index')}")
    print(f"📊 Distance    : {dist}")
    print("-" * 60)
    print(doc.strip())
    print("-" * 60)    

#7. (Multi) Query the collection
queries = ["Any rooms available?",
           "What is the capital of France?",
           "Who's your Mama?"]

results = collection.query(query_texts=queries, n_results=3)

#8. Loop through the specific query index you want (e.g., index 1 for France)
target_query_idx = 1  # Change this to 0, 1, or 2 depending on what you want to inspect

print(f"\n🔍 Results for query: '{queries[target_query_idx]}'")

for i, (doc, meta, dist) in enumerate(
    zip(
        results["documents"][target_query_idx], 
        results["metadatas"][target_query_idx], 
        results["distances"][target_query_idx]
    ), 
    start=1
):
    print(f"\n[Result {i}]")
    print(f"📁 Source File : {meta.get('source')}")
    print(f"🧩 Chunk Index : {meta.get('chunk_index')}")
    print(f"📊 Distance    : {dist}")
    print("-" * 60)
    print(doc.strip())
    print("-" * 60)

# 9. Loop through all queries and their corresponding results
for q_idx, query_text in enumerate(queries):
    print(f"\n" + "=" * 60)
    print(f"🔍 QUERY {q_idx + 1}: '{query_text}'")
    print("=" * 60)
    
    docs = results["documents"][q_idx]
    metas = results["metadatas"][q_idx]
    dists = results["distances"][q_idx]
    
    if not docs:
        print("No results found.")
        continue

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        print(f"\n[Result {i}]")
        print(f"📁 Source File : {meta.get('source')}")
        print(f"🧩 Chunk Index : {meta.get('chunk_index')}")
        print(f"📊 Distance    : {dist}")
        print("-" * 60)
        print(doc.strip())
        print("-" * 60)