import os
import uuid
from openai import OpenAI
from pydantic import BaseModel
from typing import Literal

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

def evaluate_summary(context, reference_summary, model_summary):
    class EvaluationResult(BaseModel):
        score: int  # 1 to 5
        label: Literal["Very Poor", "Poor", "Fair", "Very Good", "Excellent"]
        justification: str
        
    client = OpenAI()

    prompt = f"""You are a helpful and unbiased evaluator. Your task is to compare a generated summary with a reference summary of a given text.
        Evaluate how accurate, complete, and fluent the generated summary is compared to the reference summary.

        ## Original Text:
        {context}

        ## Reference Summary:
        {reference_summary}

        ## Model Summary:
        {model_summary}

        Now, assess the Model Summary. Give a score between 1 and 5 and labels Very Poor, Poor, Fair, Very Good, or Excellent based on the following:
        - 1: Poor – inaccurate or incomplete
        - 3: Fair – somewhat accurate, misses some important details
        - 5: Excellent – accurate, complete, fluent

        Also provide a brief justification.
    """
    
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an impartial evaluator of text quality."},
            {"role": "user", "content": prompt}  # Replace with your formatted prompt
        ],
        temperature=0,  # for consistent evaluation
        response_format=EvaluationResult
    )
    result = response.choices[0].message.parsed
    return result.score, result.label, result.justification
