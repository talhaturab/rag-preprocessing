# 📄 RAG Preprocessing Toolkit

This project provides a robust set of preprocessing tools designed to enhance **Retrieval-Augmented Generation (RAG)** pipelines. It supports chunk enrichment, context summarization, title extraction, keyword extraction, question generation, entity extraction, and document classification.

---

## 🚀 Features

- ✅ **Chunk Metadata Enrichment**
- 🧠 **Contextual Chunk Summarization** (OpenAI + Ollama options)
- 🏷️ **Title Extraction** (One-shot & Combined methods)
- 🔑 **Keyword Extraction** (TF-IDF, KeyBERT, LLM)
- ❓ **Question Generation from Context**
- 🧬 **Named Entity Recognition (NER)**
- 📚 **Document Type Classification (Zero-shot)**
- 🧪 Multiple methods benchmarked by latency and accuracy

---

## 🧱 Project Structure

```
📁 your-project/
├── main.py                         # Main orchestration script
├── utility.py                      # Chunk enrichment logic
├── context_retrieval.py           # OpenAI and Ollama-based context summarization
├── summarizer.py                  # Summarizers (HuggingFace + Ollama)
├── extract_title.py               # Title extraction logic
├── keyword_extraction.py          # 3 methods for keyword extraction
├── questions_answered_extraction.py # LLM-powered question generation
├── entity_extraction.py           # Named entity extraction using SpanMarker
├── doc_type_extraction.py         # Zero-shot doc classification
```

---

## 📦 Requirements

Install required packages:

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `langchain`, `transformers`, `torch`, `scikit-learn`
- `ollama`, `keybert`, `span-marker`, `uuid`, `pydantic`

---

## 🧪 Usage

```bash
python main.py
```

This will:
1. Load and split a PDF.
2. Enrich chunks with metadata.
3. Run summarization, title extraction, keywords, questions, entities, and classification.

---

## 📊 Benchmark Summary

| Preprocessing       | Fastest Method | Most Accurate |
|---------------------|----------------|---------------|
| Chunk Enrichment    | Local Function | —             |
| Context Retrieval   | Ollama         | OpenAI GPT-4o |
| Summarization       | Ollama         | BART          |
| Title Extraction    | One-shot       | Combined      |
| Keywords            | TF-IDF         | KeyBERT       |
| Question Extraction | Ollama         | Ollama        |
| Entity Extraction   | SpanMarker     | SpanMarker    |
| Doc Classification  | BART-MNLI      | BART-MNLI     |

---

## 🤝 Contribution

Feel free to open issues or submit pull requests. Feature suggestions and bug reports are welcome!

---

## 📄 License

MIT License

---

## 👨‍💻 Maintainer

**Talha** – [GitHub](https://github.com/) | [LinkedIn](https://linkedin.com/)

---