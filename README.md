·       Developed a RAG system to answer enterprise FAQ’s using documents in PDF,TXT and scanned formats.

·       Integrated OCR(Tesseract + Poppler) for extracting multilingual and scanned document text, including tables and headers

·       Built document loader pipeline using LangChain, PyPDFLoader, and pdf2image with chunking via RecursiveCharacterTextSplitter.

·       Embedded and indexed documents using HuggingFaceEmbeddings and FAISS for efficient semantic search and context retrieval.

·       Integrated Mistral LLM via Ollama to locally generate answers using retrieved context chunks.

·       Deployed the solution with FastAPI backend, enabling API-based question answering over enterprise documents. 
