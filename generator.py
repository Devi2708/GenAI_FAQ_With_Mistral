print("🧠 Starting generator.py...")
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
print("✅ Imported LangChain modules")
# Load embedding model (same as before)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("✅ Embedding model loaded")
# Load FAISS vectorstore
vectorstore = FAISS.load_local("vectorstore", embedding_model,allow_dangerous_deserialization=True)
print("✅ Vector DB loaded")

# Create retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Load local LLM via Ollama (make sure Ollama is running)
llm = Ollama(model="mistral")
print("✅ Local LLM loaded")

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)
print("✅ QA Chain ready")
# Ask a question
#query = input(" Ask your question: ")

#response = qa_chain.invoke(query)

# Print the result
#print("\n Answer:\n", response["result"])

#print("\n Sources:")
#for doc in response["source_documents"]:
 #   print(f"  - {doc.metadata['source']} (Page {doc.metadata.get('page', '?')})")

