from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import loader  # reuse your loader.py
import os

# Load and split documents
print("Loading and splitting documents...")
chunks = loader.load_and_split_documents(directory="data/")
print(f"Total chunks: {len(chunks)}")

# Create embeddings using HuggingFace
print("Creating embeddings...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create FAISS vector store
print(" Building vector store with FAISS...")
vectorstore = FAISS.from_documents(chunks, embedding_model)

# Save vectorstore locally
save_path = "vectorstore"
vectorstore.save_local(save_path)
print(f"Vector store saved to: {save_path}")

