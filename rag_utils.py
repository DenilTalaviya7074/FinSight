import os
import uuid
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF
import pdfplumber

# Embedding model – local, small
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# ChromaDB persistence
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "data", "chroma")
os.makedirs(CHROMA_PATH, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name="report_chunks",
    embedding_function=None
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)

def extract_text_from_pdf(pdf_path):
    """Extract text using PyMuPDF."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_tables_from_pdf(pdf_path):
    """Extract tables using pdfplumber (optional)."""
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                if table:
                    tables.append(table)
    return tables

def ingest_document(file_path, company_name, ticker, report_year, kpi_list, risk_list):
    """
    Ingest a PDF: chunk, embed, store in Chroma,
    and also store KPIs & risks in SQLite (called from outside).
    """
    raw_text = extract_text_from_pdf(file_path)
    if not raw_text.strip():
        # fallback OCR could go here, but we skip for brevity
        raw_text = ""

    chunks = text_splitter.split_text(raw_text)
    if not chunks:
        return

    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"company": company_name, "year": report_year} for _ in chunks]

    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        ids=ids
    )

def retrieve_chunks(query, company_name, k=3):
    """Retrieve top-k chunks for a given company."""
    query_emb = embedder.encode([query], convert_to_numpy=True)[0]
    results = collection.query(
        query_embeddings=[query_emb.tolist()],
        n_results=k,
        where={"company": company_name}
    )
    return results["documents"][0] if results and results["documents"] else []