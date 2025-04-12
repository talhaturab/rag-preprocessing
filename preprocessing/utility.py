
import ollama
from pydantic import BaseModel
import time

# Default prompt templates
DEFAULT_TITLE_NODE_TEMPLATE = (
    """Context: {context_str}. Give a title that summarizes all of the unique 
    entities, themes, or key topics found in the above context.
    
    Also, summarize the key topics and entities of this section in a concise, informative paragraph.
    """
)

class Title(BaseModel):
    title: str
    document_summary: str

def extract_title_one_shot(documents, max_nodes=3):
    full_text = "\n\n".join(documents[:max_nodes])
    
    prompt = DEFAULT_TITLE_NODE_TEMPLATE.format(context_str=full_text)
    response = ollama.chat(
        model='qwen2.5:0.5b',
        messages=[{'role': 'user', 'content': prompt}],
        format=Title.model_json_schema(),
    )
    response = Title.model_validate_json(response.message.content)
    return response.title, response.document_summary

def extract_title_summary(documents):
    # Extract the document title using the simple functions
    if len(documents) >= 3:
        documents = documents[:3]
    documents = [doc.page_content for doc in documents]
    start_time = time.perf_counter()
    title, summary = extract_title_one_shot(documents)
    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.6f} seconds")
    return title, summary