import ollama
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from keybert import KeyBERT

# --- Configurable Prompt Template ---
DEFAULT_KEYWORD_EXTRACT_TEMPLATE = """\
{context_str}
Extract exactly {keywords} unique keywords that best describe this content. Return them comma-separated. 
Keywords:
"""

# --- Function: Extract keywords from a single text chunk ---
def extract_keywords_from_text(text: str, num_keywords: int = 5) -> str:
    prompt = DEFAULT_KEYWORD_EXTRACT_TEMPLATE.format(context_str=text, keywords=num_keywords)
    
    response = ollama.chat(
        model='gemma3:1b',
        messages=[{'role': 'user', 'content': prompt}],
    )
    
    return response['message']['content'].strip()

# --- Function: Extract keywords from a list of text chunks ---
def extract_keywords_ollama(doc_text, num_keywords: int = 5) -> list[dict]:
    start_time = time.perf_counter()
    keywords = extract_keywords_from_text(doc_text, num_keywords=num_keywords)
    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.6f} seconds")
    return keywords

def extract_keywords_tfidf(text: str, top_n=5) -> list[str]:
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    row = X[0].toarray()[0]
    top_indices = row.argsort()[-top_n:][::-1]
    return [feature_names[i] for i in top_indices]

def extract_keywords_keybert(text: str, top_n=5) -> list[str]:
    kw_model = KeyBERT('all-MiniLM-L6-v2')
    start_time = time.perf_counter()
    keywords = kw_model.extract_keywords(text, top_n=top_n)
    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.6f} seconds")
    return [kw for kw, _ in keywords]