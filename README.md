# Multi Agent RAG workflow with Web search capabilities

A Retrieval-Augmented Generation (RAG) system featuring binary quantized embeddings, Milvus Lite vector search with reranking, an LLM router, and web search fallback via Firecrawl.

## 🚀 Features

- **Binary Quantized Embeddings**: Efficient storage and retrieval using binary vectors
- **Milvus Lite Integration**: Lightweight vector database for local development
- **Intelligent Reranking**: Enhanced retrieval quality using reciprocal rank fusion
- **Multi-Agent Workflow**: Router agent for quality evaluation and web search routing
- **Web Search Fallback**: Firecrawl integration for real-time information retrieval
- **Modular Architecture**: Clean separation of concerns for easy extensibility

## 🏗️ Architecture

```
User Query
    ↓
1. Document Retrieval (Vector Search + Reranking)
    ↓
2. RAG Response Generation
    ↓
3. Router Agent (Quality Evaluation)
    ↓
4a. SATISFACTORY → Response Synthesis
4b. UNSATISFACTORY → Web Search → Response Synthesis
    ↓
5. Final Answer
```

## 📁 Project Structure

```
paralegal-agent/
├── pyproject.toml            # Project metadata and dependencies
├── config/
│   └── settings.py           # Configuration via environment variables
├── src/
│   ├── embeddings/           # Embedding generation with binary quantization
│   ├── indexing/             # Milvus vector database integration
│   ├── retrieval/            # Vector search (no reranking in demo)
│   ├── generation/           # RAG response generation
│   └── workflows/            # Multi-agent workflow orchestration
├── examples/
│   └── test.py               # Sample usage and testing
├── data/                     # Local data (e.g., PDFs, DB files)
├── cache/                    # Model/cache directories
├── app.py                    # Streamlit app
└── README.md
```

## 🛠️ Installation

### Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

### Environment variables

Create a `.env` file in the project root or export these variables in your shell:

```bash
cp .env.example .env
# Edit .env with your API keys
```

Note: Both keys are read by `config/settings.py`. Web search is optional at runtime, but the settings loader expects both keys to be present.

## 🚀 Quick Start

### Streamlit Demo

```bash
streamlit run app.py
# or, with uv
uv run app.py
```

Upload a PDF in the sidebar, add API keys, and start chatting.

### Programmatic Usage

```python
import asyncio
import os
from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from src.embeddings.embed_data import EmbedData
from src.indexing.milvus_vdb import MilvusVDB
from src.retrieval.retriever_rerank import Retriever
from src.generation.rag import RAG
from src.workflows.agent_workflow import EnhancedRAGWorkflow

async def quick_start():
    os.environ.setdefault("OPENAI_API_KEY", "<your_key>")
    os.environ.setdefault("FIRECRAWL_API_KEY", "<your_key>")

    # 1) Load and split a PDF into text chunks
    pdf_path = "./data/your_document.pdf"
    docs = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
    text_chunks = [d.text for d in docs]

    # 2) Create embeddings (with binary quantization)
    embedder = EmbedData()
    embedder.embed(text_chunks)

    # 3) Setup Milvus Lite collection
    vdb = MilvusVDB()
    vdb.initialize_client()
    vdb.create_collection()
    vdb.ingest_data(embedder)

    # 4) Retrieval and RAG
    retriever = Retriever(vdb, embedder)
    rag = RAG(retriever)

    # 5) Enhanced workflow
    workflow = EnhancedRAGWorkflow(retriever=retriever, rag_system=rag)
    result = await workflow.run_workflow("Your question here")
    print(result["answer"])

asyncio.run(quick_start())
```

## 🔧 Configuration

Managed in `config/settings.py` (via environment variables):

```python
# Model Configuration
embedding_model = "BAAI/bge-large-en-v1.5"
llm_model = "gpt-3.5-turbo"
vector_dim = 1024

# Retrieval Configuration
top_k = 5
batch_size = 512

# Database Configuration
milvus_db_path = "./data/milvus_binary.db"
collection_name = "legal_documents"
```

## 🧩 Components

### 1) Embeddings (`src/embeddings/`)

```python
from src.embeddings.embed_data import EmbedData
embedder = EmbedData()
embedder.embed(text_chunks)
```

### 2) Vector DB (`src/indexing/`)

```python
from src.indexing.milvus_vdb import MilvusVDB
vdb = MilvusVDB()
vdb.initialize_client()
vdb.create_collection()
vdb.ingest_data(embedder)
```

### 3) Retrieval (`src/retrieval/`)

```python
from src.retrieval.retriever_rerank import Retriever
retriever = Retriever(vector_db=vdb, embed_data=embedder, top_k=5)
results = retriever.search("Your query")
```

### 4) RAG Generation (`src/generation/`)

```python
from src.generation.rag import RAG
rag = RAG(retriever=retriever)
answer = rag.query("Your question")
```

### 5) Multi-Agent Workflow (`src/workflows/`)

```python
from src.workflows.agent_workflow import EnhancedRAGWorkflow
workflow = EnhancedRAGWorkflow(retriever=retriever, rag_system=rag)
result = await workflow.run_workflow("Complex question")
```

## 🔄 Workflow Details

- **Retrieval**: Binary vector search with Hamming distance
- **RAG**: Context construction + OpenAI completion
- **Router**: LLM-based quality evaluation (SATISFACTORY/UNSATISFACTORY)
- **Web Search**: Firecrawl web search + content extraction
- **Synthesis**: Combine document and web info; refine final answer

## 🎯 Use Cases

- **Legal Research Assistant**: Index PDFs and ask targeted questions
- **Document Analysis**: Identify risks or obligations in contracts

## 🧪 Testing

Run the included example:

```bash
python examples/test.py
# or
uv run examples/test.py
```

## 🚨 Troubleshooting

- **Missing API keys**: Ensure `OPENAI_API_KEY` and `FIRECRAWL_API_KEY` are set (env or `.env`).
- **Model download issues**: Check/clear `cache/hf_cache` and ensure network access to Hugging Face.
- **Milvus Lite file locks**: Stop the app/process that holds the DB, then remove `data/*.db` files.

Enable detailed logging:

```python
from loguru import logger
logger.add("debug.log", level="DEBUG")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Submit a pull request

## 📄 License

MIT License

## 🙏 Acknowledgments

- LlamaIndex, Milvus, OpenAI, Firecrawl, HuggingFace

---
**Note**: This is an MVP implementation focused on core functionality. For production use, consider 
additional features like user authentication, rate limiting, monitoring, and robust error handling.