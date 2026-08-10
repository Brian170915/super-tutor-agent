# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

**Conda environment: `assist`**

All Python commands must use the assist environment:
```bash
conda run -n assist python <command>
conda run -n assist pip install <package>
conda run -n assist uvicorn agent:app --host 0.0.0.0 --port 8000
```

Python executable: `D:\program\Anaconda\envs\assist\python.exe`

## Running the Application

**Important:** Always use the `assist` conda environment for this project.

```bash
# Start the FastAPI server (port 8000)
conda run -n assist python agent.py
# Or from the project directory:
D:/program/Anaconda/envs/assist/python.exe agent.py
```

The server initializes RAG infrastructure on startup (ChromaDB + BM25 index). If `data/` directory contains knowledge documents, they are automatically ingested into the vector store.

**Use `run_server.py` as a wrapper if conda run has issues with multi-line commands:**
```bash
D:/program/Anaconda/envs/assist/python.exe run_server.py
```

## API Endpoints

- `POST /chat` — Main Q&A endpoint. Form params: `session_id`, `user_input`. Returns: `session_id`, `answer`, `mindmap_mermaid`
- `GET /knowledge-structure?subject=&grade=` — Returns subject/grade/topic hierarchy
- `POST /ingest` — Upload a document to the knowledge base (multipart/form-data with file)
- `GET /session/{session_id}` — Returns chat history
- `GET /` — Serves the inline HTML frontend (chat UI + Mermaid mindmap)

## Architecture

### LangGraph Pipeline (4 nodes, linear)

```
START → query → rag → chat → thought → END
```

| Node | File | Purpose |
|------|------|---------|
| `query` | `agent/nodes.py:query_node` | Extracts `user_input` from state or last message |
| `rag` | `agent/nodes.py:rag_node` | Dual retrieval: Chroma vector search + BM25 keyword search, fused via RRF |
| `chat` | `agent/nodes.py:chat_node` | LLM answer generation with RAG context |
| `thought` | `agent/nodes.py:thought_node` | Generates Mermaid mindmap of knowledge points |

### State (`agent/state.py`)

`AgentState` extends `MessagesState` from LangGraph. Fields:
- Input: `user_input`, `session_id`
- RAG: `retrieved_docs` (List[str]), `rephrased_query` (str)
- Output: `answer` (str), `mindmap_mermaid` (str)

### RAG Pipeline (`rag/pipeline.py`)

Dual-retrieval with Reciprocal Rank Fusion (RRF):
1. LLM rephrases query to retrieval-friendly keywords
2. Chroma vector search (top 4) — uses BAAI/bge-small-zh-v1.5 embeddings (local CPU)
3. BM25 keyword search (top 4) — cached in `bm25_retriever.pkl`
4. RRF fusion (k=30) combines both result sets
5. Returns top 2 documents as context

Embedding model: `D:\knowledge\project\pythonProject\model\models-BAAI-bge-small-zh-v1.5`

### Knowledge Structure (`rag/knowledge_structure.py`)

Hardcoded curriculum map for 5 subjects (数学, 物理, 化学, 语文, 英语) with grade-level topics. Functions: `get_subjects()`, `get_grades(subject)`, `get_topics(subject, grade)`.

### Session Management (`agent/sessions.py`)

In-memory dict-based sessions with UUID. No persistence. Functions: `create_session()`, `add_message(session_id, role, content, metadata)`, `get_messages(session_id)`.

### OCR Module (`ocr/agnes_client.py`)

Uses Qwen-VL (`qwen-vl-max`) via OpenAI-compatible API for exam paper OCR. Configured via `.env`: `AGNES_API_KEY`, `AGNES_BASE_URL`, `AGNES_MODEL`. Currently not integrated into the graph pipeline (MVP is text-only Q&A).

## Running Tests

```bash
conda run -n assist python -m pytest tests/test_agent.py -v
```

3 tests are skipped: OCR client test (requires real API), and 2 end-to-end tests (require full RAG infrastructure). All other 8 tests pass.

## Key Design Decisions

- **No LangServe**: Custom `/chat` endpoint used instead of `add_routes()` to avoid route conflicts
- **Pydantic request model**: `ChatRequest` model used for `/chat` to properly parse JSON body (FastAPI requires explicit body model)
- **Inline HTML frontend**: No separate templates/static files; frontend is embedded in `agent.py` `root()` endpoint
- **MVP scope**: Current implementation is text-only Q&A. OCR and image upload paths are prepared in code but not wired into the graph
- **Session store**: In-memory dict (`session_store`); not persisted across restarts
- **Knowledge base**: Documents in `data/` directory are ingested on first startup via `KBIngestor.sync()`
- **Bug fix**: `session_store` in `agent.py` and `_sessions` in `sessions.py` were separate dicts — fixed by having `chat_endpoint` write directly to `session_store`

## Environment Variables (`.env`)

```
DASHSCOPE_API_KEY=          # Alibaba Cloud DashScope LLM key
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen3.7-max-2026-05-17
LANGCHAIN_API_KEY=          # LangSmith tracing
LANGCHAIN_TRACING_V2=true
AGNES_API_KEY=              # OCR (same as DASHSCOPE_API_KEY)
AGNES_BASE_URL=             # OCR (same as BASE_URL)
AGNES_MODEL=qwen-vl-max
```
