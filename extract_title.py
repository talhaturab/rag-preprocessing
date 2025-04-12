import ollama
from pydantic import BaseModel
import time

DEFAULT_TITLE_NODE_TEMPLATE = (
        """\
    Context: {context_str}. Give a title that summarizes all of \
    the unique entities, titles or themes found in the context.
    """
)

DEFAULT_TITLE_COMBINE_TEMPLATE = (
    """{context_str}. Based on the above candidate titles and content, what is the
    one comprehensive title for this document?"""
)

class Title(BaseModel):
    title: str

def get_candidate_title(text: str) -> str:
    prompt = DEFAULT_TITLE_NODE_TEMPLATE.format(context_str=text)
    response = ollama.chat(
        model='gemma3:1b',
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 0},
        format=Title.model_json_schema(),
    )
    # Get the title candidate from the response
    candidate = Title.model_validate_json(response.message.content).title
    print(candidate)
    return candidate

def combine_candidate_titles(candidates: list) -> str:
    combined_candidates = ", ".join(candidates)
    prompt = DEFAULT_TITLE_COMBINE_TEMPLATE.format(context_str=combined_candidates)
    response = ollama.chat(
        model='gemma3:1b',
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 0},
        format=Title.model_json_schema(),
    )
    final_title = Title.model_validate_json(response.message.content).title
    return final_title

def extract_document_title(nodes: list, max_nodes: int = 3) -> str:
    candidate_titles = []
    for text in nodes[:max_nodes]:
        candidate = get_candidate_title(text)
        candidate_titles.append(candidate)
    return combine_candidate_titles(candidate_titles)

def extract_title_combined(documents):
    # Extract the document title using the simple functions
    documents = [doc.page_content for doc in documents]
    start_time = time.perf_counter()
    title = extract_document_title(documents)
    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.6f} seconds")
    return title


import ollama
from pydantic import BaseModel

# Default prompt templates
DEFAULT_TITLE_NODE_TEMPLATE = (
    "Context: {context_str}. Give a title that summarizes all of the unique "
    "entities, themes, or key topics found in the above context."
)

def extract_title_one_shot(documents, max_nodes=3):
    class Title(BaseModel):
        bestTitle:str

    full_text = "\n\n".join(documents[:max_nodes])
    
    prompt = DEFAULT_TITLE_NODE_TEMPLATE.format(context_str=full_text)
    response = ollama.chat(
        model='qwen2.5:0.5b',
        messages=[{'role': 'user', 'content': prompt}],
        format=Title.model_json_schema(),
    )
    title = Title.model_validate_json(response.message.content).bestTitle
    return title

def extract_title(documents):
    # Extract the document title using the simple functions
    if len(documents) >= 3:
        documents = documents[:3]
    documents = [doc.page_content for doc in documents]
    start_time = time.perf_counter()
    title = extract_title_one_shot(documents)
    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.6f} seconds")
    return title