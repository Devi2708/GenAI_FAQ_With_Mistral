import os
import pytesseract
import re
from pdf2image import convert_from_path
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set Tesseract path for Windows users
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
folder_path = r"C:\Users\Devipriya\rag_faq_assistant\data"

def clean_text(text):
    """Clean headers, footers, multiple spaces"""
    text = re.sub(r"Page \d+ of \d+", "", text)
    text = re.sub(r"Company Confidential", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def ocr_pdf(file_path):
    """Fallback OCR loader for scanned/image-based PDFs"""
    print(f"Using OCR for scanned PDF: {file_path}")
    images = convert_from_path(file_path, dpi=300,poppler_path=r"C:\Program Files\poppler\bin")
    docs = []

    for i, img in enumerate(images):
        text = pytesseract.image_to_string(img, lang="eng")
        cleaned = clean_text(text)
        docs.append(Document(page_content=cleaned, metadata={"page": i+1, "source": file_path}))

    return docs

def extract_text_pdf(file_path):
    """Try normal PyPDFLoader first, fallback to OCR if needed"""
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.page_content = clean_text(doc.page_content)
        if sum(len(doc.page_content) for doc in docs) < 100:
            print(f"⚠️ Extracted content too short. Switching to OCR: {file_path}")
            return ocr_pdf(file_path)
        return docs
    except Exception as e:
        print(f" PyPDFLoader failed for {file_path}. Using OCR. Error: {e}")
        return ocr_pdf(file_path)

def load_and_split_documents(directory=r"C:\Users\Devipriya\Downloads\company_docs"):
    """Load and split multiple PDFs from a folder"""
    all_docs = []

    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            full_path = os.path.join(directory, filename)
            docs = extract_text_pdf(full_path)
            all_docs.extend(docs)

    print(f"Total documents loaded before splitting: {len(all_docs)}")

    # Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(all_docs)
    print(f" Split into {len(chunks)} chunks")
    return chunks

#  Run as standalone
if __name__ == "__main__":
    chunks = load_and_split_documents("data/")
    print("\n Sample Chunk:\n", chunks[0].page_content[:300])
