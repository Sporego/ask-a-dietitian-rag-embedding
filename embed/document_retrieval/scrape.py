import pymupdf4llm
from llama_index.core.schema import Document
import json
import os
from pathlib import Path

current_directory = os.getcwd()

# Append '../documents' to the current working directory
folder_path = os.path.join(current_directory, 'embed', 'documents')

# Convert the relative path to an absolute path
folder_path = os.path.abspath(folder_path)
llama_reader = pymupdf4llm.LlamaMarkdownReader()

def save_llama_docs_to_json(llama_docs: list[Document], output_path: str):
    data = []
    for doc in llama_docs:
        data.append({
            "text": doc.text,
            "metadata": doc.metadata
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved: {output_path}")

# for filename in os.listdir(folder_path):
#     if filename.endswith(".pdf"):
#         file_path = os.path.join(folder_path, filename)
        
#         try:
#             print(f"Processing {file_path}...")
#             llama_docs = llama_reader.load_data(file_path)
#             json_path = Path(file_path).with_suffix('.json')
#             save_llama_docs_to_json(llama_docs, json_path)
            
            
#         except Exception as e:
#             print(f"Error processing {file_path}: {e}")

# ------------------------ 0.  install deps -----------------------------------
# pip install "torch>=2.3" "transformers>=4.51" sentence-transformers==2.7 \
#             llama-index-core llama-index-embeddings-huggingface \
#             llama-index-vector-stores-pgvector psycopg2-binary
# -----------------------------------------------------------------------------


# -------------------- 1.  collect Document objects ---------------------------
# from llama_index.core.schema import Document
# from pathlib import Path
# import json, os

docs: list[Document] = []

for json_file in Path(folder_path).glob("*.json"):
    with open(json_file, encoding="utf-8") as f:
        payload = json.load(f)
        # payload is a list of {"text": ..., "metadata": {...}}
        for item in payload:
            docs.append(Document(text=item["text"], metadata=item["metadata"]))

print(f"Loaded {len(docs)} LlamaIndex docs")
# -----------------------------------------------------------------------------


# -------------------- 2.  build the embedding model --------------------------
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Qwen/Qwen3-Embedding-4B", device="cuda")
print("SentenceTransformer device:", model.device)
# -----------------------------------------------------------------------------


# -------------------- 3A.  (option A) Postgres / pgvector --------------------
# from llama_index.vector_stores.pgvector import PGVectorStore
# from llama_index.core import StorageContext, VectorStoreIndex

# PG_CONN = {
#     "database": "ragdb",
#     "user":     "postgres",
#     "password": "secret",
#     "host":     "localhost",
#     "port":     5432,
# }

# vector_store = PGVectorStore.from_params(
#     **PG_CONN,
#     table_name="diet_docs",
#     dimension=embed_model.output_dim      # must match output_dim above
# )

# storage_context = StorageContext.from_defaults(vector_store=vector_store)

# index = VectorStoreIndex.from_documents(
#     docs,
#     embed_model=embed_model,
#     storage_context=storage_context,
#     show_progress=True
# )
# print("🔗  All vectors are in pgvector!")
# -----------------------------------------------------------------------------


# -------------------- 3B.  (option B) local folder ---------------------------
# >>> uncomment if you just want to keep everything on disk <<<
# from llama_index.core import StorageContext, VectorStoreIndex
# storage_context = StorageContext.from_defaults()
# index = VectorStoreIndex.from_documents(
#     docs, embed_model=embed_model, storage_context=storage_context)
# index.storage_context.persist("./index")   # saves to ./index/* json + parquet
# -----------------------------------------------------------------------------


# -------------------- 4.  querying later -------------------------------------
# (Works the same for pgvector‑backed or local indexes)
# from llama_index.retrievers import VectorIndexRetriever

# retriever = VectorIndexRetriever(index, similarity_top_k=6)

# question = "What is the Recommended Dietary Allowance for iron in women 19‑50?"
# results  = retriever.retrieve(question)

# for i, node in enumerate(results, start=1):
#     print(f"[{i}] score={node.score:.3f}  page={node.node.metadata.get('page')}")
#     print(node.node.text[:200], "…\n")
# -----------------------------------------------------------------------------


# OPTIONAL: reranker, hybrid BM25‑plus‑dense, etc. can be added later

# -------------------- 5.  Vectorize and print embeddings to screen ------------
print("\n" + "="*60)
print("VECTORIZING DOCUMENTS AND PRINTING EMBEDDINGS")
print("="*60)

# Get embeddings for all documents
print(f"\nGenerating embeddings for {len(docs)} documents...")
embeddings = []

for i, doc in enumerate(docs):
    print(f"Processing document {i+1}/{len(docs)}: {doc.metadata.get('file_name', 'Unknown')}")
    
    with model.truncate_sentence_embeddings(truncate_dim=2000):
        embedding = model.encode(doc.text)
    embeddings.append({
        "document_index": i,
        "text_preview": doc.text[:100] + "..." if len(doc.text) > 100 else doc.text,
        "embedding_dimension": len(embedding),
        "embedding_preview": embedding[:5].tolist(),  # Show first 5 values
        "metadata": doc.metadata
    })
    
    print(f"  - Embedding dimension: {len(embedding)}")
    print(f"  - First 5 values: {embedding[:5].tolist()}")
    print(f"  - Page: {doc.metadata.get('page', 'N/A')}")
    print()

# Print summary
print("\n" + "="*60)
print("EMBEDDING SUMMARY")
print("="*60)
print(f"Total documents processed: {len(docs)}")
print(f"Device used: {model.device}")

# Show a few sample embeddings in detail
print(f"\nSample embeddings (showing first 3):")
for i, emb_info in enumerate(embeddings[:3]):
    print(f"\nDocument {i+1}:")
    print(f"  Text preview: {emb_info['text_preview']}")
    print(f"  Embedding (first 10 values): {emb_info['embedding_preview']}")
    print(f"  Metadata: {emb_info['metadata']}")

print("\n✅ Document vectorization complete!")
# -----------------------------------------------------------------------------
