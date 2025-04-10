import ollama, time

# Template to generate questions from context
DEFAULT_QUESTION_GEN_TMPL = """\
Here is the context:
{context_str}

Generate exactly {num_questions} specific, answerable questions that this context provides direct information for. 
These should be meaningful, unique, and not generic. Only return the questions as a numbered list.
"""

def extract_questions_from_text(text: str, num_questions: int = 5) -> list[str]:
    prompt = DEFAULT_QUESTION_GEN_TMPL.format(context_str=text, num_questions=num_questions)
    
    response = ollama.chat(
        model='gemma3:1b',  # change model name as needed
        messages=[{'role': 'user', 'content': prompt}]
    )
    
    raw_output = response['message']['content'].strip()
    
    # Parse output into list of questions (numbered list handling)
    questions = [
        line.split('. ', 1)[-1].strip()
        for line in raw_output.split('\n')
        if line.strip() and '.' in line
    ]
    
    return questions[:num_questions]  # return only top N clean questions


def extract_questions_from_documents(text: str, num_questions: int = 5) -> list[dict]:
    start_time = time.perf_counter()
    questions = extract_questions_from_text(text, num_questions=num_questions)
    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.6f} seconds")
    return {
        "questions_this_excerpt_can_answer": questions
    }