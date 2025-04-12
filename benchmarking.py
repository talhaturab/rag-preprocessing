import time
import pandas as pd
from evaluate import load
import ollama
from example_texts import *
from utility import evaluate_summary

# Define the models to benchmark
models = {
    "qwen2.5:0.5b": "qwen2.5:0.5b",
    "qwen2.5:3b": "qwen2.5:3b",
    "gemma3:1b": "gemma3:1b",
    "deepseek-r1:1.5b": "deepseek-r1:1.5b",
    "tinyllama:latest": "tinyllama:latest",
    "llama3.2:latest": "llama3.2:latest",
    "smollm2:135m": "smollm2:135m",
    "ollama run smollm2:360m": "smollm2:360m",
    "phi": "phi"
}

# Define input texts of varying lengths
input_texts = {
    "100_tokens": small_context,
    
    "600_tokens": medium_context,
    
    "2000_tokens": large_context,
}

summary_texts = {
    "100_tokens": "AI is transforming industries and daily life, offering major benefits but also raising concerns around ethics, privacy, and job loss. Efforts are underway to ensure its responsible and fair use through regulations and guidelines.",
    
    "600_tokens": "Artificial intelligence is transforming industries and everyday life, offering benefits in healthcare, finance, education, and more. However, it also raises concerns around job displacement, data privacy, algorithmic bias, and ethical decision-making. As AI becomes more powerful, global efforts are underway to establish regulations and ethical frameworks that ensure its responsible use. Balancing innovation with accountability is key to building a future where AI serves humanity fairly and safely.",
    
    "2000_tokens": "Artificial Intelligence (AI) is transforming society across every domain—boosting productivity, enhancing healthcare, and revolutionizing education and finance. However, this progress comes with significant challenges, including job displacement, privacy risks, algorithmic bias, misinformation, and environmental impact. AI’s reliance on large datasets raises ethical questions around surveillance and consent, while its potential to influence public opinion and decision-making threatens democratic institutions. The technology also risks widening global inequality and demands new approaches to governance, fairness, and sustainability. To ensure AI serves humanity, society must prioritize inclusive development, transparent regulation, ethical use, and global cooperation."
}

# Load evaluation metrics
rouge = load("rouge")
bertscore = load("bertscore")

# Initialize a list to store benchmark results
results = []
summary_results = []

# Define the summary extraction prompt template
SUMMARY_PROMPT_TEMPLATE = """\
Here is the content of the section:
{context_str}

Summarize the above text in a concise manner ensuring that the main points are clearly captured.

Summary:
"""

# Benchmark each model with each input
for model_name, model_id in models.items():
    for input_size, text in input_texts.items():
        prompt = SUMMARY_PROMPT_TEMPLATE.format(context_str=text)

        # Measure inference time
        start_time = time.time()
        response = ollama.chat(
            model=model_id,
            messages=[{'role': 'user', 'content': prompt}],
            stream=False
        )
        end_time = time.time()
        inference_time = end_time - start_time

        # Extract the summary from the response
        summary = response.get('message', {}).get('content', '').strip()

        # Evaluate summary quality
        rouge_scores = rouge.compute(predictions=[summary], references=[text])
        bert_scores = bertscore.compute(predictions=[summary], references=[text], lang="en")
        llm_score, llm_label, llm_justification = evaluate_summary(text, summary_texts[input_size], summary)
        print('one done')

        results.append({
            "Model": model_name,
            "Input Size": input_size,
            "Inference Time (s)": round(inference_time, 3),
            "ROUGE-1": round(rouge_scores["rouge1"], 3),
            "ROUGE-2": round(rouge_scores["rouge2"], 3),
            "ROUGE-L": round(rouge_scores["rougeL"], 3),
            "BERTScore Precision": round(bert_scores["precision"][0], 3),
            "BERTScore Recall": round(bert_scores["recall"][0], 3),
            "BERTScore F1": round(bert_scores["f1"][0], 3),
            "LLM Judge Score": llm_score,
            "LLM Judge Label": llm_label,
            "LLM Judge Justification": llm_justification,
        })
        # Track summaries for separate CSV
        summary_results.append({
            "Model": model_name,
            "Input Size": input_size,
            "Generated Summary": summary
        })
        
        df = pd.DataFrame(results)
        df.to_csv("ollama_llm_benchmark_results.csv", index=False)
        
        summary_df = pd.DataFrame(summary_results)
        summary_df.to_csv("ollama_llm_generated_summaries.csv", index=False)


# Create a DataFrame and save results
print("Benchmarking complete. Results saved to 'ollama_llm_benchmark_results.csv'.")
