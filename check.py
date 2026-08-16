import chromadb

client = chromadb.PersistentClient(path="./chroma_rag_db")
collection = client.get_collection(client.list_collections()[0].name)

print("TOTAL DOCUMENTS IN DB:", collection.count())