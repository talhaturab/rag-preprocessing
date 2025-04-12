# --- Imports ---
import torch
import asyncio
from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Local Module Imports ---
from preprocessing.utility import extract_title_summary
from doc_type_extraction import classify_document_st
from utility import process_chunks
from keyword_extraction import extract_keywords_tfidf, extract_keywords_yake
from entity_extraction import extract_entities_spacy  # using the faster version
from context_retrieval import context_retrieval_ollama


# --- Load and Split PDF ---
def load_documents(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load()

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

# --- Pipeline Execution ---
if __name__ == "__main__":
    print("🔄 Loading PDF and splitting into chunks...")

    path = "/Users/rimshatalha/Documents/toptal/2409.11974v1.pdf"
    documents = load_documents(path)
    chunks = split_documents(documents)

    print("✅ PDF loaded and split completed. \n")

    # --- One-Time Preprocessing Steps ---
    print("🔍 Extracting title and summary...")
    title, summary = extract_title_summary(documents)  # ~5-8 seconds
    print("📄 Title:", title)
    print("📝 Summary:", summary, '\n')

    print("🔍 Classifying document type...")
    doc_type = classify_document_st(documents[0].page_content)  # ~0.2 seconds
    print("🏷 Document Type:", doc_type, '\n')

    print("🔧 Enriching chunk metadata...")
    enriched_chunks = process_chunks(chunks[:4])  # fast <50ms

    # --- Per-Chunk Processing (Example: chunk[0] shown) ---
    print("💡 Extracting keywords from chunk[0]...")
    keywords = extract_keywords_yake(chunks[0].page_content)  # ~3-4ms
    print("🔑 Keywords:", keywords, '\n')

    print("👁 Extracting entities from chunk[0] using spaCy...")
    entities = extract_entities_spacy(chunks[0].page_content)  # ~0.4 seconds
    print("🔍 Entities:", entities, '\n')

    print("📚 Performing context retrieval for first 3 chunks using Ollama...")
    processed_chunks_ollama = context_retrieval_ollama(chunks[:3], documents)  # ~4-7 seconds
    print("✅ Context retrieval completed.", '\n')

    print("🚀 All processing complete.")