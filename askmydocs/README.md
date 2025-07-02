# AskMyDocs

AskMyDocs is a document-understanding app that transforms static PDFs into interactive conversations using Retrieval-Augmented Generation (RAG). Upload any document, ask a question, and receive accurate, context-grounded answers powered by OpenAI’s language models and LangChain’s modular pipelines.

Designed for students, researchers, knowledge workers, and anyone working with large or complex documents.

---

## Features

- Upload one or more PDFs (up to 200MB per file)
- Combines semantic search with generative answering (RAG architecture)
- "Smart Search Mode": hybrid keyword + embedding-based retrieval
- Automatically generates follow-up questions after each query
- Optional web fallback using SerpAPI when answer confidence is low
- Downloadable chat history as a PDF
- Supports OCR via Tesseract for scanned PDFs
- Clean, dark-themed Streamlit UI

---

## Tech Stack

| Layer            | Tool/Library         | Purpose                                                  |
|------------------|----------------------|----------------------------------------------------------|
| UI               | Streamlit            | Frontend interface                                       |
| QA Engine        | LangChain            | RAG pipeline, memory handling, and chaining              |
| Language Model   | OpenAI API           | Answer generation, question refinement                   |
| Embeddings       | OpenAIEmbeddings     | Vector representations for semantic search               |
| PDF Processing   | PyPDF2, pdf2image     | Extracting text and converting pages for OCR             |
| OCR              | Tesseract            | Optical character recognition for scanned PDFs           |
| Web Search       | SerpAPI              | Fallback search for out-of-scope or incomplete answers   |

---

## Architecture

1. **PDF Upload & Preprocessing**
   - Files are parsed using PyPDF2 and/or converted via pdf2image + Tesseract.
   - Text is split into overlapping chunks using LangChain splitters.

2. **Vector Indexing**
   - Text chunks are embedded using OpenAIEmbeddings.
   - A retriever (in-memory vector store) performs similarity-based retrieval.

3. **RAG Pipeline**
   - User questions are used to retrieve the most relevant chunks.
   - Context is passed along with the question to an OpenAI model to generate the final answer.

4. **Post-processing**
   - Generated answers are shown in a styled UI.
   - Follow-up suggestions are created via a secondary LLM prompt.
   - Full chat history can be exported as a PDF.

---

## Local Setup

1. Clone the repository:
git clone https://github.com/shruti25838/my-projects.git
cd my-projects/askmydocs

2. Install Python dependencies:
pip install -r requirements.txt

3. Install system dependencies (for OCR support):

macOS:
brew install poppler tesseract

Ubuntu:
sudo apt install poppler-utils tesseract-ocr

4. Add your API keys:

Either create a .env file or a .streamlit/secrets.toml file:

toml
OPENAI_API_KEY = "your-openai-key"
SERPAPI_API_KEY = "your-serpapi-key"

5. Run the app:
streamlit run app.py
