import asyncio
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import time, ollama
import copy

# Define an async helper function to process one chunk.
async def process_chunk(chunk, chain, documents):
    # Create a deep copy of the chunk so we don't alter the original object.
    new_chunk = copy.deepcopy(chunk)
    
    # Retrieve the page number from the new chunk's metadata.
    page_number = new_chunk.metadata.get("page")
    # Find the source document matching the chunk's page number.
    source_doc = next((doc for doc in documents if doc.metadata.get("page") == page_number), None)
    source_text = source_doc.page_content if source_doc else new_chunk.page_content

    # Asynchronously invoke the chain to get a context summary.
    context_summary = await chain.ainvoke({"document": source_text, "chunk": new_chunk.page_content})
    # Append (or prepend) the context summary to the new_chunk's page_content.
    new_chunk.page_content = f"{context_summary.content}\n{new_chunk.page_content}"
    return new_chunk

# Define an async function to process chunks in batches concurrently.
async def process_in_batches(chunks, documents, batch_size=5):
    prompt = PromptTemplate(
        input_variables=["document", "chunk"],
        template="""
            <document>
            {document}
            </document>

            Here is a chunk from the document:
            <chunk>
            {chunk}
            </chunk>

            Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
            Answer only with the succinct context and nothing else.
        """
    )
    
    # Initialize your LLM and chain.
    llm = ChatOpenAI(model="gpt-4o")
    chain = prompt | llm  # This uses the runnable chain mechanism.

    processed_chunks = []
    # Loop over chunks in slices (batches) of batch_size.
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        # Create a list of tasks for the current batch.
        tasks = [process_chunk(chunk, chain, documents) for chunk in batch]
        # Await all tasks concurrently.
        batch_results = await asyncio.gather(*tasks)
        processed_chunks.extend(batch_results)
    
    return processed_chunks

# Example usage: process all chunks in batches.
async def fetch_context_retrieval(rag_chunks, documents):
    # Call process_in_batches to get modified copies of the chunks.
    processed_chunks = await process_in_batches(rag_chunks, documents, batch_size=10)
    processed_chunks = [chunk.page_content for chunk in processed_chunks]
    return processed_chunks
    
    
def get_summarized_context_ollama(document_content, chunk_content):
    formatted_prompt = f"""
        <document>
        {document_content}
        </document>
        
        Here is a chunk from the document:
        <chunk>
        {chunk_content}
        </chunk>
        
        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
        Answer only with the succinct context and nothing else.
    """
    start_time = time.perf_counter()
    response = ollama.chat(
        model='gemma3:1b',
        messages=[{'role': 'user', 'content': formatted_prompt}],
    )
    end_time = time.perf_counter()
    
    print(f"Execution time: {end_time - start_time:.6f} seconds")
    return response['message']['content']

def context_retrieval_ollama(chunks, documents):
    updated_chunks = []
    for chunk in chunks:
        # Create a deep copy to avoid modifying the original chunk.
        new_chunk = copy.deepcopy(chunk)
        # Retrieve the page number from the chunk's metadata.
        page_number = new_chunk.metadata.get("page")
        # Find the matching document based on the page number.
        source_doc = next((doc for doc in documents if doc.metadata.get("page") == page_number), None)
        source_text = source_doc.page_content if source_doc else new_chunk.page_content
        
        # Generate the context summary using Ollama.
        context_summary = get_summarized_context_ollama(new_chunk.page_content, source_text)
        # Update the copy of the chunk with the context summary appended.
        new_chunk.page_content = f"{context_summary}\n{new_chunk.page_content}"
        
        updated_chunks.append(new_chunk)
        
    updated_chunks = [chunk.page_content for chunk in updated_chunks]
    return updated_chunks