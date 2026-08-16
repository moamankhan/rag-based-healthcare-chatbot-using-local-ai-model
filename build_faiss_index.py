import time
import chromadb
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def main():
    print("=" * 60)
    print("🚀 INSTANT FAISS VECTOR INDEX BUILDER")
    print("=" * 60)

    # 1. Load HuggingFace embedding model wrapper
    print("\n[1/3] Loading BAAI/bge-small-en-v1.5 model metadata...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # 2. Extract texts, metadatas, AND pre-computed embeddings from ChromaDB
    chroma_path = "./chroma_rag_db"
    print(f"\n[2/3] Extracting vectors directly from ChromaDB ('{chroma_path}')...")
    
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    collections = chroma_client.list_collections()
    
    if not collections:
        print("❌ Error: No collections found in ChromaDB!")
        return
        
    collection = chroma_client.get_collection(name=collections[0].name)
    total_records = collection.count()
    print(f"✅ Connected to collection: '{collections[0].name}' ({total_records:,} records)")

    # Fetch pre-computed vectors in safe batches to avoid SQLite variable limits
    batch_size = 500
    offset = 0
    
    all_texts = []
    all_metadatas = []
    all_vectors = []

    start_extract = time.time()
    while True:
        batch = collection.get(
            include=['documents', 'metadatas', 'embeddings'],
            limit=batch_size,
            offset=offset
        )
        
        if not batch['ids']:
            break
            
        all_texts.extend(batch['documents'])
        all_metadatas.extend([m or {} for m in batch['metadatas']])
        all_vectors.extend(batch['embeddings'])
        
        offset += batch_size
        print(f"   Fetched {len(all_texts):,} / {total_records:,} vectors...", end="\r")

    print(f"\n✅ Extracted {len(all_vectors):,} pre-calculated vectors in {time.time() - start_extract:.2f} seconds.")

    # 3. Build FAISS index instantly using existing embeddings
    output_dir = "./faiss_rag_db"
    print(f"\n[3/3] Building FAISS Index from existing vectors...")
    start_faiss = time.time()

    # Zip text and pre-computed vector pairs
    text_embeddings = list(zip(all_texts, all_vectors))

    # Construct FAISS index without re-calculating embeddings
    faiss_db = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=embeddings,
        metadatas=all_metadatas
    )

    # Save to disk
    faiss_db.save_local(output_dir)
    print(f"✅ FAISS index constructed and saved in {time.time() - start_faiss:.2f} seconds!")

    print("\n" + "=" * 60)
    print(f"🎉 SUCCESS! FAISS database saved in '{output_dir}'.")
    print("Created files: index.faiss and index.pkl")
    print("=" * 60)

if __name__ == "__main__":
    main()