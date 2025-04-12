from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import asyncio
import torch
from transformers import pipeline

from utility import process_chunks
from context_retrieval import fetch_context_retrieval, context_retrieval_ollama
from summarizer import summarize_text, extract_summary_from_text
from extract_title import extract_title_combined, extract_title
from keyword_extraction import extract_keywords_ollama, extract_keywords_tfidf, extract_keywords_keybert
from questions_answered_extraction import extract_questions_from_documents
from entity_extraction import extract_entities
from doc_type_extraction import classify_document

def load_documents(pdf_path):
    # Load the PDF (each page is returned as a Document)
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents
    
def split_documents(documents):
    # Define splitter parameters
    chunk_size = 1000       # Maximum characters per chunk (tune this as needed)
    chunk_overlap = 200     # Overlap between chunks to preserve context

    # Create an instance of the RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Split each Document object (e.g., for multi-page PDFs)
    chunks = text_splitter.split_documents(documents)
    return chunks
    
# Intiate the process
print('Loading packages...')
path = "/Users/rimshatalha/Documents/toptal/2409.11974v1.pdf"
documents = load_documents(path)
chunks = split_documents(documents)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=torch.device('cpu'))
print('Loading packages completed')

# Preprocessing logic

# Add ids, and page number to metadata
enriched_chunks = process_chunks(chunks)
print("Enriched chunks:", enriched_chunks)

# Context Retrival
# Add context Retrieval to chunks using OpenAI API
processed_chunks = asyncio.run(fetch_context_retrieval(chunks[:9], documents))

# Add context Retrieval to chunks using Ollama
processed_chunks_ollama = context_retrieval_ollama(chunks[:3], documents)

# Summarizer
# Summarize using bart large
summary_text = summarize_text(chunks[0].page_content, summarizer)
# Summarize use llm
summary = extract_summary_from_text(chunks[0].page_content)

# Extract title
# Extract title using a couple of documents (better accuracy but slower)
title = extract_title_combined(chunks[:3])

# Extract title using only one document (faster)
title = extract_title(chunks[:3])

# Keywords Extractor
keywords = extract_keywords_ollama(chunks[0].page_content)
keywords = extract_keywords_tfidf(chunks[0].page_content)
keywords = extract_keywords_keybert(chunks[0].page_content)

# QuestionsAnswered Extraction
questions = extract_questions_from_documents(chunks[0].page_content)

# Entity Extraction
entities = extract_entities(chunks[0].page_content)

# Document type extraction
doc_type = classify_document(documents[0].page_content)