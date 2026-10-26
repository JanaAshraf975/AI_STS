# 📄 ChatWithPDF: Advanced RAG Chatbot

Chat with your PDF files. Upload a PDF, ask questions in plain language (Arabic or English), and get answers based **only on the content of that document**, with chat memory so you can ask follow-up questions.

The project implements **Advanced RAG (Retrieval-Augmented Generation)**: instead of sending the whole PDF to an AI model, it finds the few passages that best match your question and sends only those to the model.

---

## ✨ Features

- 📤 Upload any text-based PDF
- ✂️ **Smart chunking**: splits text by paragraphs and sentences, not at fixed character positions
- 🔎 **Semantic search** with local embeddings (finds passages by meaning, not just keywords)
- 🎯 **Re-ranking**: retrieves 10 candidates, then a Cross-Encoder picks the best 3
- ✏️ **Query rewriting**: turns follow-up questions ("and what about the second one?") into standalone questions before searching
- 💬 **Chat history**: the bot remembers the recent conversation
- 🔍 Shows the retrieved context under each answer, so you can verify the source
- 🌐 LLM through **OpenRouter** (many models, including free ones), or a local model with Ollama
- 📝 Prompt template stored in a text file, so you can edit it without touching the code

---

## 🧠 How It Works

### Phase 1: Indexing (when you upload a PDF)

```
PDF file
   │
   ▼
1. Extract text          pdf_utils.py   extract_text_from_pdf()
   │
   ▼
2. Smart chunking        pdf_utils.py   chunk_text()
   │                     (800 chars, 150 overlap, split at paragraphs/sentences)
   ▼
3. Create embeddings     embedding.py   SentenceTransformer
   │                     (text → vector of numbers)
   ▼
4. Store in ChromaDB     vectoredb.py   build_vectoredb()
```

### Phase 2: Answering (each time you ask a question)

```
Question + chat history
   │
   ▼
5. Rewrite question      llm.py         rewrite_query()
   │   (makes follow-ups self-contained)
   ▼
6. Vector search         vectoredb.py   retrive_context()
   │   (top 10 closest chunks from ChromaDB)
   ▼
7. Re-rank               vectoredb.py   CrossEncoder
   │   (keeps the best 3 chunks = the "context")
   ▼
8. Build prompt          llm.py         prompt/rag_prompt.txt
   │   (context + question + chat history)
   ▼
9. Ask the LLM           llm.py         ask_gemin() → OpenRouter
   │
   ▼
Answer (shown with the retrieved context)
```

### What are "question" and "context"?

| Term | Meaning |
|------|---------|
| **question** | What the user types in the chat. |
| **context** | The 3 best passages from the PDF for that question. The LLM has never seen your PDF, so these passages are what it uses to answer. |

### Naive RAG vs this project

| Step | Naive RAG | This project (Advanced) |
|------|-----------|-------------------------|
| Chunking | Fixed 500 characters | Paragraph/sentence-aware splitting |
| Question | Used as typed | Rewritten using chat history |
| Retrieval | Top 3 from vector search | Top 10, then re-ranked down to 3 |
| Memory | None | Recent chat history |

---

## 📁 Project Structure

```
ChatWithPdf/
├── Modules/
│   ├── pdf_utils.py      # Read PDF text + smart chunking
│   ├── embedding.py      # Load the embedding model
│   ├── vectoredb.py      # ChromaDB collection + search + re-ranking
│   └── llm.py            # Prompt building, query rewriting, OpenRouter calls
├── prompt/
│   └── rag_prompt.txt    # Prompt template with {context} and {question}
├── src/
│   └── app.py            # Streamlit web app (UI + chat)
├── .env                  # Your secret API key (NOT committed to Git)
├── .env.example          # Example of what .env should look like
├── .gitignore
├── .python-version
├── pyproject.toml        # Project settings and dependencies (uv)
└── README.md
```

---

## 🧰 Requirements

- **Python 3.13** (see `.python-version`)
- **[uv](https://docs.astral.sh/uv/)**: Python package and environment manager
- A free **[OpenRouter](https://openrouter.ai)** account and API key
- Internet connection (to download the models on first run, and to call OpenRouter)

### Python dependencies

| Package | Used for |
|---------|----------|
| `streamlit` | Web interface and chat UI |
| `pypdf` | Extracting text from PDF files |
| `langchain-text-splitters` | Smart chunking (`RecursiveCharacterTextSplitter`) |
| `sentence-transformers` | Embeddings and the Cross-Encoder re-ranker |
| `chromadb` | Vector database for similarity search |
| `openai` | Client library used to talk to OpenRouter (it is OpenAI-compatible) |
| `python-dotenv` | Loading the API key from `.env` |

---

## 🚀 Setup

### 1. Install uv (if you don't have it)

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Get the project

```bash
git clone <your-repo-url>
cd ChatWithPdf
```

### 3. Install dependencies

If they are already listed in `pyproject.toml`:

```bash
uv sync
```

Otherwise, add them:

```bash
uv add streamlit pypdf sentence-transformers chromadb openai python-dotenv langchain-text-splitters
```

> `uv` creates the virtual environment automatically. You don't need to activate anything.

### 4. Get an OpenRouter API key

1. Go to **https://openrouter.ai** and sign in.
2. Open **https://openrouter.ai/keys** and click **Create Key**.
3. Copy the key (it starts with `sk-or-v1-...`). It is shown only once.

### 5. Create the `.env` file

In the project root (next to `pyproject.toml`), create a file named exactly `.env`:

```
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxx
```

- No spaces around `=`
- No quotes around the key
- Make sure `.env` is listed in `.gitignore` so it is never uploaded to GitHub

### 6. Create the prompt template

Create the folder `prompt/` in the project root and the file `prompt/rag_prompt.txt`:

```
You are a helpful assistant. Answer the question using only the context below.
If the answer is not in the context, say "I don't know".
Answer in the same language as the question.

Context:
{context}

Question:
{question}

Answer:
```

> ⚠️ Keep `{context}` and `{question}` exactly as written. If you add other curly braces (for example JSON), write them doubled: `{{ }}`.

---

## ▶️ Run the App

From the **project root** (not from inside `src/`):

```bash
uv run streamlit run src/app.py
```

Streamlit opens the app in your browser (usually http://localhost:8501).

**Usage:**
1. Upload a PDF. Wait for the "chunks loaded" message.
2. Type a question in the chat box.
3. Ask follow-up questions; the bot remembers the recent conversation.
4. Expand **Retrieved context** under any answer to see which passages were used.
5. Use **Clear chat** in the sidebar to start a new conversation. Uploading a new PDF also resets the chat.

> ⏳ On the **first run**, two models are downloaded automatically (the embedding model and the re-ranker, around 90 MB each). The page may stay blank for a minute or two. Later runs are much faster.

---

## ⚙️ Configuration

### Choose a different LLM

In `Modules/llm.py`:

```python
MODEL = "meta-llama/llama-3.3-70b-instruct:free"
```

Browse models at **https://openrouter.ai/models**. Models ending in `:free` cost nothing but have usage limits, and the list of free models changes over time. Paid models need credits (OpenRouter → Credits).

### Use a local model with Ollama

Install [Ollama](https://ollama.com), pull a model (for example `ollama pull llama3.2`), then in `llm.py`:

```python
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",   # any non-empty value works
)
MODEL = "llama3.2"
```

### Working with Arabic PDFs

The default models are English-focused. For better Arabic results:

- In `Modules/embedding.py`, use a multilingual embedding model:
  ```python
  SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
  ```
- In `Modules/vectoredb.py`, use a multilingual re-ranker:
  ```python
  CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
  ```

### Tune retrieval

| Setting | File | Default | Effect |
|---------|------|---------|--------|
| `chunk_size` | `pdf_utils.py` | 800 | Characters per chunk. Larger = more context per chunk, less precise search. |
| `overlap` | `pdf_utils.py` | 150 | Characters shared between neighbouring chunks, so sentences aren't cut off. |
| `fetch_k` | `vectoredb.py` | 10 | How many candidates the vector search returns before re-ranking. |
| `top_k` | `vectoredb.py` | 3 | How many chunks are sent to the LLM after re-ranking. |
| `history[-6:]` | `llm.py` | 6 messages | How much chat history the LLM sees (3 questions and answers). |

---

## 🛠️ Troubleshooting

| Problem | Likely cause and fix |
|---------|----------------------|
| Blank page for a long time on first run | Models are downloading. Check the terminal and wait. |
| `ModuleNotFoundError: Modules.vectoredb` | File name doesn't match the import. The file must be `Modules/vectoredb.py`. |
| `'NoneType' object has no attribute ...` on asking | `build_vectoredb` must end with `return collection`. |
| Error loading the re-ranker | Check the model name is exactly `cross-encoder/ms-marco-MiniLM-L-6-v2`. |
| `FileNotFoundError: prompt/rag_prompt.txt` | The `prompt/` folder is missing, or you ran the command from another folder. Run from the project root. |
| `KeyError` when formatting the prompt | The template has extra `{ }`. Use only `{context}` and `{question}`, or double other braces. |
| `401 Unauthorized` | API key missing or wrong. Check `.env` (name must be `OPENROUTER_API_KEY`) and restart the app. |
| `.env` not being read | File must be named `.env`, not `.env.txt` (Windows can hide extensions). |
| `429` / rate limit error | Free models have limits. Wait a bit or switch model. Note that each follow-up question uses 2 LLM calls. |
| "No text found in this PDF" | The PDF is a scanned image. `pypdf` only reads real text, so run OCR on it first. |

---

## 📝 Notes

- ChromaDB runs **in memory** (`chromadb.Client()`), so the index is rebuilt each time the app restarts or a new PDF is uploaded.
- Some names in the code are spelled in a particular way (`retrive_context`, `ask_gemin`, `PROMPT_TEMPLTE`, `vectoredb`). They are kept as they are because other files import them. If you rename one, rename it everywhere.
- Never commit your `.env` file or share your API key.

---
