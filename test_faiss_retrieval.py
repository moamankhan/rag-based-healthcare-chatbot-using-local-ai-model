import time
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# 1. Expanded Healthcare System Prompt
HEALTHCARE_PROMPT_TEMPLATE = """You are an expert, empathetic, and responsible Healthcare Information Assistant.
Your task is to provide a detailed, well-structured, and comprehensive explanation to answer the user's question, using ONLY the retrieved medical context below.

INSTRUCTIONS:
1. Do not give brief 1-sentence answers. Elaborate thoroughly on each concept using the context provided.
2. Structure your response into clear sections using bold titles (e.g., Overview, Symptoms, Causes, Prevention/Management).
3. Use itemized bullet points and bold key medical terms for maximum clarity and scannability.
4. Base all claims strictly on the provided medical references. Do NOT invent unsupported medical claims.
5. If the retrieved context lacks sufficient detail to answer the question, clearly state:
   "I do not have sufficient medical documentation in my database to answer this question accurately."
6. Do NOT provide personal diagnoses, emergency directives, or specific drug dosages.
7. Conclude with a brief note reminding the user to consult a qualified healthcare professional.
8. Also, ensure that your response is empathetic and supportive, acknowledging the user's concerns.
9. If you don't have the required information or if the context is insufficient, provide the link to the website containing concerned information.


Retrieved Medical Context:
--------------------------
{context}
--------------------------

User Question: {question}

Detailed & Helpful Response:"""

def load_rag_chain():
    print("=" * 60)
    print("🚀 INITIALIZING HEALTHCARE RAG TERMINAL TEST")
    print("=" * 60)

    # Load Embedding Model
    print("\n[1/3] Loading BAAI/bge-small-en-v1.5 embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # Load Local FAISS Database
    print("\n[2/3] Loading FAISS index from './faiss_rag_db'...")
    faiss_db = FAISS.load_local(
        folder_path="./faiss_rag_db",
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    print("✅ FAISS database loaded.")

    # Load Ollama LLM
    print("\n[3/3] Connecting to local Ollama LLM (qwen2.5:3b)...")
    llm = OllamaLLM(
    model="qwen2.5:3b",
    temperature=0.2,
    num_thread=6,       # Set to your CPU's physical core count (e.g., 4, 6, or 8)
    num_ctx=2048,       # Limits context window size to prevent slowdowns
    num_predict=350     # Limits maximum generated response length (~250-300 words)
    )
    print("✅ Local LLM connected.")

    prompt = PromptTemplate(
        template=HEALTHCARE_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    return faiss_db, llm, prompt

def run_rag_query(query, faiss_db, llm, prompt, k=5):
    print(f"\n🔍 Question: {query}")
    print("⏳ Searching FAISS vector index (top 5 matches)...")
    
    start_time = time.time()
    
    # Step A: Retrieve Top 5 Document Chunks
    retrieved_docs = faiss_db.similarity_search(query, k=3)
    
    # Step B: Assemble Context
    context_text = "\n\n".join([
        f"--- Reference Document {idx+1} ---\n{doc.page_content}" 
        for idx, doc in enumerate(retrieved_docs)
    ])
    
    # Step C: Format Prompt & Generate
    formatted_prompt = prompt.format(context=context_text, question=query)
    
    print("🧠 Synthesizing detailed answer using local LLM...")
    response = llm.invoke(formatted_prompt)
    elapsed = time.time() - start_time

    print(f"\n✅ Answer Generated in {elapsed:.2f} seconds:\n")
    print("-" * 60)
    print(response)
    print("-" * 60)
    
    print("\n📚 Sources Used:")
    for idx, doc in enumerate(retrieved_docs, 1):
        source = doc.metadata.get('source', 'Medical Dataset')
        disease = doc.metadata.get('disease', doc.metadata.get('topic', 'N/A'))
        print(f"  [{idx}] Topic: {disease} | Source: {source}")

def main():
    faiss_db, llm, prompt = load_rag_chain()
    
    print("\n" + "=" * 60)
    print("Medical RAG Terminal Pipeline Ready!")
    print("Enter your questions below (type 'exit' or 'q' to quit):")
    print("=" * 60)

    while True:
        query = input("\n🏥 Ask a healthcare question: ").strip()
        if query.lower() in ['exit', 'quit', 'q']:
            print("Exiting pipeline.")
            break
        if not query:
            continue
            
        run_rag_query(query, faiss_db, llm, prompt)

if __name__ == "__main__":
    main()