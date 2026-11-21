# Adaptive RAG PDF Assistant

An Adaptive Retrieval-Augmented Generation (RAG) system that answers questions using PDF documents. It dynamically selects the retrieval strategy, evaluates retrieved context, retries weak retrieval results, and falls back to web search when the documents do not provide sufficient information.

The system also supports conversational memory, allowing users to ask follow-up questions and refer to previous discussions.

## Features

- **PDF Processing:** Extracts text from PDF files and splits it into smaller chunks while preserving source and page metadata.
- **Semantic Search:** Uses vector embeddings to retrieve relevant document chunks.
- **Adaptive Routing:** Determines whether a question needs document retrieval or a direct conversational response.
- **Single-Hop & Multi-Hop Retrieval:** Selects the retrieval strategy based on the complexity of the question.
- **LLM-Based Retrieval Evaluation:** Evaluates retrieved chunks and checks whether they contain useful information.
- **Query Rewriting & Retry:** Rewrites weak queries and retries retrieval when the initial results are insufficient.
- **Web Search Fallback:** Uses web search when document retrieval fails to provide relevant information.
- **Conversational Memory:** Maintains recent conversation history and generates summaries of older messages.
- **Source Citations:** Generates answers with references to the retrieved PDF sources.
- **Evaluation Framework:** Measures retrieval quality, answer quality, context quality, prompt compliance, and latency.

## Architecture

### Adaptive RAG Workflow

1. **Question Input:** The user submits a question.
2. **Conversation Memory:** Checks whether the question refers to previous conversation details.
3. **Query Rewriting:** Converts follow-up questions into standalone questions using conversation history.
4. **Question Routing:** The Router decides whether the question requires document search or a direct response.
5. **Retrieval Classification:** If document search is needed, the system classifies the question as single-hop or multi-hop.
6. **Document Retrieval:** Retrieves relevant chunks from the ChromaDB vector database.
7. **Retrieval Evaluation:** The Judge evaluates whether the retrieved context is relevant and sufficient.
8. **Retry:** If retrieval is weak, the system uses a rewritten query and performs another search.
9. **Web Search Fallback:** If the retry is unsuccessful, the system searches the web.
10. **Answer Generation:** The Generator creates the final answer using the retrieved PDF context or web results.

### Retrieval Strategies

- **Single-Hop:** Performs one semantic search using the original question.
- **Multi-Hop:** Decomposes a complex question into smaller sub-questions, retrieves chunks for each one, and removes duplicate results.

## Technologies Used

| Component | Technology |
|---|---|
| Programming Language | Python |
| LLM Provider | OpenRouter |
| LLM Model | GPT-4o-mini |
| Embedding Model | BAAI/bge-small-en-v1.5 |
| Vector Database | ChromaDB |
| PDF Processing | PyPDF |
| Text Splitting | LangChain Text Splitters |
| Embeddings | Sentence Transformers |
| Web Search | DuckDuckGo Search |
| Environment & Dependencies | uv |
| User Interface | Streamlit |

## Project Structure

```text
AdabtiveRag/
│
├── app.py
│
├── modules/
│   ├── rag.py
│   ├── llm.py
│   ├── vectoredb.py
│   ├── embedding.py
│   ├── pdf_utils.py
│   └── web_search.py
│
├── pdfs/
│   ├── 01_AsterHome_Product_Catalog.pdf
│   ├── 02_AsterHome_Installation_Guide.pdf
│   └── 03_AsterHome_Support_Policies.pdf
│
├── evaluation/
│   └── evaluation_dataset.json
│
├── chroma_db/
│
├── .env.example
├── .gitignore
├── pyproject.toml
├── uv.lock
└── README.md
```

*Note: Evaluation scripts can also be placed inside an `evaluation/` or `tests/` directory, depending on the final project organization.*

## Installation & Setup

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd AdabtiveRag
```

### 2. Install Dependencies

This project uses `uv` for dependency management.

```bash
uv sync
```

If dependencies are not already defined in `pyproject.toml`, install the required packages:

```bash
uv add openai python-dotenv chromadb sentence-transformers torch pypdf langchain-text-splitters streamlit ddgs
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
```

Get an API key from [OpenRouter](https://openrouter.ai/).

### 4. Download the Embedding Model

The project uses:

```text
BAAI/bge-small-en-v1.5
```

The model is downloaded automatically when it is loaded for the first time. An internet connection is required for the initial download.

## Document Indexing

Place your PDF files inside the `pdfs/` directory.

The PDF processing module:

1. Extracts text from each PDF page.
2. Splits the extracted text into chunks.
3. Preserves the source filename and page number.
4. Generates a unique ID for every chunk.
5. Converts chunks into vector embeddings.
6. Stores the embeddings and metadata in ChromaDB.

**Chunking configuration:**

- Chunk size: `800`
- Chunk overlap: `150`
- Embedding dimension: `384`
- Vector similarity: Cosine similarity

The indexing process should be executed before asking questions for the first time. Re-index the documents whenever the source PDFs or embedding model change.

## Running the Application

### Streamlit Interface

From the project root:

```bash
uv run streamlit run app.py
```

### Command-Line Interface

Run the RAG module:

```bash
uv run python modules/rag.py
```

The CLI allows users to ask questions interactively and maintains conversation history during the session.

## Evaluation Framework

The project includes multiple evaluation scripts to measure different aspects of the RAG pipeline.

### 1. Retrieval Evaluation

Measures how effectively the system retrieves relevant document chunks.

| Metric | Description |
|---|---|
| Precision | Percentage of retrieved chunks that are relevant. |
| Recall | Percentage of all relevant chunks successfully retrieved. |
| MRR | Measures how highly the first relevant chunk is ranked. |
| Hop Classification Accuracy | Measures whether the Router correctly identifies single-hop and multi-hop questions. |

### 2. Answer Evaluation

Uses an LLM Judge to evaluate generated answers.

| Metric | Description |
|---|---|
| Correctness | Measures whether the generated answer matches the expected answer. |
| Faithfulness | Measures whether the answer is supported by the retrieved context. |
| Answer Relevancy | Measures how directly the answer addresses the question. |

### 3. Context Evaluation

Evaluates the quality and completeness of the context passed to the Generator.

| Metric | Description |
|---|---|
| Context Precision | Measures how many retrieved chunks are relevant. |
| Context Recall | Measures how many expected relevant chunks were retrieved. |
| Context Coverage | Checks whether all expected relevant chunks were retrieved. |

### 4. Latency Evaluation

Measures the time required to process questions and compares average latency across retrieval strategies.

### 5. Prompt Compliance Evaluation

Tests whether the Generator follows the system instructions, including:

- Using only information supported by the provided context.
- Generating factually correct answers.
- Providing valid source citations.
- Returning the required fallback response when the answer is unavailable in the context.

## Models and Their Responsibilities

The project separates the LLM responsibilities into three roles:

| Model Role | Responsibility |
|---|---|
| Router | Selects the route and classifies retrieval complexity. |
| Judge | Evaluates the relevance and sufficiency of retrieved chunks. |
| Generator | Produces the final answer using the available context. |

All three roles currently use `openai/gpt-4o-mini` through OpenRouter. Their model names can be changed in `llm.py`.

## Example Questions

Using the AsterHome sample PDF knowledge base:

```text
How much does the DoorGuard Sensor cost?

What does the DoorGuard Sensor detect?

How much does the ViewCam Indoor cost?

Which Wi-Fi frequency do AsterHome devices require?

Compare the DoorGuard Sensor and ViewCam Indoor.
```

The system can also handle follow-up questions such as:

```text
User: How much does the DoorGuard Sensor cost?

User: What does it detect?
```

The second question is rewritten using conversation history before document retrieval.

## Future Improvements

- Add a reranking stage to improve retrieval relevance.
- Improve multi-hop retrieval and query decomposition.
- Add more advanced retrieval evaluation metrics.
- Improve conversation memory management.
- Compare different embedding models and retrieval configurations.
- Optimize latency and LLM usage.

## License

This project is developed for educational and portfolio purposes.