import sys, time
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch

# Load the lightweight & fast embedding model (only needs to be done once)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Define candidate labels (document types)
LABELS = [
    "research paper",
    "educational lecture",
    "book",
    "magazine article",
    "news article",
    "blog post",
    "technical documentation",
    "manual",
    "newsletter",
    "press release",
    "thesis",
    "dissertation",
    "white paper",
    "novel",
    "financial statements"
]

def classify_document(text):
    # Initialize the zero-shot classification pipeline
    classifier = pipeline(
        task="zero-shot-classification",
        # model="facebook/bart-large-mnli",
        model="valhalla/distilbart-mnli-12-1",
        device = 'cpu'
    )
    
    # Perform zero-shot classification
    start_time = time.perf_counter()
    result = classifier(text, LABELS)
    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.6f} seconds")
    
    # # Print the classification results
    # print("Classification Results:")
    # for label, score in zip(result["labels"], result["scores"]):
    #     print(f"  {label}: {score:.4f}")
    
    # The top label (first in the result["labels"]) is the most likely
    most_likely_label = result["labels"][0]
    return most_likely_label

# Precompute label embeddings for faster future calls
label_embeddings = model.encode(LABELS, convert_to_tensor=True)

def classify_document_st(text: str) -> str:
    
    start_time = time.perf_counter()
    # Encode input text
    doc_embedding = model.encode(text, convert_to_tensor=True)
    
    # Compute cosine similarities
    cosine_scores = util.cos_sim(doc_embedding, label_embeddings)[0]
    
    # Get index of best matching label
    best_idx = torch.argmax(cosine_scores).item()
    result = LABELS[best_idx]
    
    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.6f} seconds")

    print(result)
    return result
