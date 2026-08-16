from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Load the same embedding model
print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# 2. Connect to your local database
print("Connecting to local ChromaDB...")
vector_db = Chroma(
    persist_directory="./chroma_rag_db", 
    embedding_function=embedding_model
)

# 3. Interactive loop for testing
print("\n" + "="*50)
print("MEDICAL VECTOR DATABASE RETRIEVAL TESTER")
print("Type 'exit' or 'q' to quit.")
print("="*50 + "\n")

while True:
    query = input("\nEnter a medical question or symptom: ").strip()
    if query.lower() in ['exit', 'q', 'quit']:
        break
    if not query:
        continue
        
    print(f"\nSearching DB for: '{query}'...")
    # Search top 3 most relevant matches
    results = vector_db.similarity_search_with_score(query, k=3)
    
    print(f"\nFound {len(results)} relevant matches:\n" + "-"*50)
    for idx, (doc, score) in enumerate(results, 1):
        disease = doc.metadata.get('disease', 'Unknown')
        source = doc.metadata.get('source', 'Unknown')
        
        print(f"\n[MATCH {idx}] (Similarity Distance Score: {score:.4f})")
        print(f"📌 Disease/Topic : {disease}")
        print(f"📚 Source        : {source}")
        print(f"📝 Content Snippet:\n{doc.page_content[:300]}...")
        print("-" * 50)