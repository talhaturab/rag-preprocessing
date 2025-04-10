import sys, time
from transformers import pipeline

def classify_document(text):
    # Initialize the zero-shot classification pipeline
    classifier = pipeline(
        task="zero-shot-classification",
        model="facebook/bart-large-mnli",
        device = 'cpu'
    )
    
    # List of candidate document labels (you can add/remove as needed)
    candidate_labels = [
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
        "white paper"
    ]
    
    # Perform zero-shot classification
    start_time = time.perf_counter()
    result = classifier(text, candidate_labels)
    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.6f} seconds")
    
    # # Print the classification results
    # print("Classification Results:")
    # for label, score in zip(result["labels"], result["scores"]):
    #     print(f"  {label}: {score:.4f}")
    
    # The top label (first in the result["labels"]) is the most likely
    most_likely_label = result["labels"][0]
    return most_likely_label