from span_marker import SpanMarkerModel
from collections import defaultdict
import time

def group_entities_by_label(entities: list) -> dict:
    """
    Converts a list of entity predictions into a dictionary grouped by label.

    Args:
        entities (list): Output from SpanMarker model, each item has 'span' and 'label'.

    Returns:
        dict: A dictionary where keys are labels (e.g., 'ORG', 'LOC') and
              values are lists of unique entity spans.
    """
    grouped = defaultdict(set)  # Use set to avoid duplicates

    for entity in entities:
        label = entity['label']
        span = entity['span'].strip()
        grouped[label].add(span)

    # Convert sets to sorted lists for clean output
    return {label: sorted(list(spans)) for label, spans in grouped.items()}

def extract_entities(text):
    # Download from the Hub
    model = SpanMarkerModel.from_pretrained("tomaarsen/span-marker-mbert-base-multinerd")
    
    # Run inference
    start_time = time.perf_counter()
    entities = model.predict(text)
    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.6f} seconds")
    
    grouped = group_entities_by_label(entities)
    return grouped