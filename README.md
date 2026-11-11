# ⚡ Nexus RAG — Multimodal Agent & Document Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-green.svg)](https://www.trychroma.com/)
[![Google Gemini](https://img.shields.io/badge/Gemini%20API-v1beta-orange.svg)](https://aistudio.google.com/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: Pyright](https://img.shields.io/badge/types-pyright%20100%25-blueviolet.svg)](https://github.com/microsoft/pyright)

```
[ Documents: PDF · DOCX · XLSX · CSV · JSON · TXT ] 
   │
   ▼
[ Multi-Modal Chunker & Offline MiniLM Embeddings + BM25 Sparse Index ]
   │
   ▼
[ Reciprocal Rank Fusion (RRF) + Cross-Relevance Re-Ranking ] 
   │
   ▼
[ Autonomous Agent (Google Gemini Flash Lite) ] ◄───► [ 6 Live Tools: Web · Python · Scraper · Weather ]
   │
   ▼
[ Real-Time Streaming UI + Document Inspector + Voice Input + Benchmark Dashboard ]
```

---

## 🚀 Key Features

### 1. 🔍 Advanced Retrieval & RAG Engine
- **Hybrid Search (BM25 + Chroma Vectors via RRF)**: Merges sparse BM25 keyword matching with dense cosine semantic vectors using Reciprocal Rank Fusion ($k=60$).
- **Query Transformation & HyDE-lite**: Strips conversational filler and expands vague queries into dense high-signal search terms before retrieval.
- **Two-Stage Re-Ranking**: Scores candidate passages on keyword density, exact phrase coverage, and semantic distance before passing to the LLM.
- **Multi-Format Ingestion**: Ingests `.pdf`, `.docx`, `.xlsx` (pure XML parser without Excel dependencies), `.csv`, `.json`, `.txt`, and `.md`.

### 2. 🎙️ Multimodal & Agentic Capabilities
- **Native Voice Input**: Record voice queries directly from your microphone with instant Gemini audio transcription.
- **Python REPL Tool (`execute_python`)**: Calculates complex mathematical formulas, percentages, statistical aggregations, or logic in real-time.
- **Live Web Search (`web_search`)**: Queries live internet sources via DuckDuckGo for breaking news and external facts.
- **Webpage & URL Reader (`read_url`)**: Scrapes, extracts, and summarizes live articles and URLs.
- **Weather Tool (`get_weather`)**: Live meteorological data via Open-Meteo.

### 3. 🎨 Modern Glassmorphic UI & Observability
- **Linear-Inspired Design**: Dark slate glassmorphism theme with Plus Jakarta Sans typography and glowing status indicators.
- **Interactive Document Inspector (Tab 2)**: Browse all indexed document chunks, filter by keyword, inspect page numbers, and trigger one-click summaries.
- **Multi-Chat Session Manager**: Create, switch, name, and delete conversation threads.
- **LLM-as-a-Judge Observability Dashboard (Tab 3)**: Quantitative evaluation metrics for Tool Selection Accuracy, Retrieval Hit Rate, and Faithfulness.
- **Chat Transcript Export**: One-click download of the complete chat with citations as Markdown.

---

## 🛠️ Quickstart

### Local Setup
```bash
# 1. Clone repository
git clone https://github.com/your-username/nexus-rag.git
cd nexus-rag

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API Key
echo "GEMINI_API_KEY=your_key_here" > .env

# 5. Run application
streamlit run app.py
```

### Docker Deployment
```bash
docker build -t nexus-rag .
docker run -p 8501:8501 -e GEMINI_API_KEY="your_key_here" nexus-rag
```

---

## 📂 Project Architecture

```
rag-chat/
├── app.py                  # Main Streamlit application with custom glassmorphism UI & tabs
├── yourscript.py           # Entrypoint alias
├── Dockerfile              # Production container specification
├── requirements.txt        # Pinned dependencies
├── .env                    # Environment secrets (GEMINI_API_KEY)
├── rag/
│   ├── agent.py            # Agentic tool orchestrator, streaming, and audio transcription
│   ├── store.py            # Hybrid ChromaDB + BM25 + RRF + Re-Ranking + Query Expansion
│   ├── tools.py            # 6 Agent tools (Search, Python REPL, Web Scraper, Weather, etc.)
│   └── ingest.py           # Multi-format document loader (PDF, DOCX, XLSX, CSV, JSON)
└── evaluation/
    ├── test_questions.json # Benchmark evaluation test suite
    ├── run_eval.py         # LLM-as-a-judge scoring script
    └── results.json        # Evaluation output displayed in Dashboard tab
```

---

## 🌐 Deploy Live to Streamlit Cloud (100% Free)

1. Push your project to a public or private GitHub repository.
2. Sign in to [share.streamlit.io](https://share.streamlit.io).
3. Click **"New app"**, select your repository, branch `main`, and main file `app.py`.
4. Under **Advanced Settings > Secrets**, paste:
   ```toml
   GEMINI_API_KEY = "your_actual_key"
   GEMINI_MODEL = "gemini-flash-lite-latest"
   HF_HUB_OFFLINE = "1"
   TRANSFORMERS_OFFLINE = "1"
   ```
5. Click **Deploy!** Your app will be live at `https://<your-app-name>.streamlit.app`.
