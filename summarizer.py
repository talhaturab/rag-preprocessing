import time, ollama

def summarize_text(text, summarizer):
    # Generate a summary.
    # Adjust max_length, min_length, and do_sample parameters based on your requirements.
    start_time = time.perf_counter()
    summary = summarizer(text,
                        #  max_length=200,
                        # min_length=50,
                        do_sample=False,
                        )
    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.6f} seconds")
    
    # Print out the summary
    print("Summary:")
    print(summary[0]['summary_text'])
    return summary[0]['summary_text']

# Default summary prompt template
DEFAULT_SUMMARY_EXTRACT_TEMPLATE = """\
Here is the content of the section:
{context_str}

Summarize the key topics and entities of this section in a concise, informative paragraph.

Summary:
"""

def extract_summary_from_text(text: str) -> str:
    prompt = DEFAULT_SUMMARY_EXTRACT_TEMPLATE.format(context_str=text)
    
    response = ollama.chat(
        model='gemma3:1b',  # Change to your preferred Ollama model
        messages=[{'role': 'user', 'content': prompt}]
    )
    
    summary = response['message']['content'].strip()
    return summary