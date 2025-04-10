import os
import uuid

def enrich_pdf_chunk(chunk, chunk_index, total_chunks, doc_id_mapping):
    metadata = chunk.metadata
    
    # Extract PDF-related metadata
    source = metadata.get('source', "Unknown Source")
    page = metadata.get('page')
    page_label = metadata.get('page_label')
    
    # Generate or retrieve a unique document ID for the source PDF.
    if source not in doc_id_mapping:
        doc_id_mapping[source] = str(uuid.uuid4())
    doc_id = doc_id_mapping[source]
    
    # Build positional tag - use page_label if available, otherwise fall back to page number.
    if page_label:
        positional_tag = f"Page: {page_label}"
    elif page is not None:
        positional_tag = f"Page: {page}"
    else:
        positional_tag = "Page: N/A"
    
    # Create chunk position indicator (e.g., "Chunk 2 of 14")
    chunk_position = f"Chunk: {chunk_index + 1} of {total_chunks}"
    
    # Construct the overall context header
    context_header = f"DocID: {doc_id} | {chunk_position} | {positional_tag}"
    
    # Prepend the context header to the page content
    enriched_content = f"{context_header}\n{chunk.page_content}"
    
    return enriched_content

def process_chunks(chunks):
    total_chunks = len(chunks)
    enriched_chunks = []

    # Mapping from source file path to a unique document id.
    doc_id_mapping = {}
    
    for index, chunk in enumerate(chunks):
        enriched_text = enrich_pdf_chunk(chunk, index, total_chunks, doc_id_mapping)
        enriched_chunks.append(enriched_text)
    
    return enriched_chunks