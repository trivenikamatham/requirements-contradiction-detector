# Requirements Contradiction Detector

An intelligent Django web application for automated requirements engineering, semantic verification, and contradiction detection across SRS, BRD, and system specifications.

---

## 🌟 Key Features

1. **Multi-Format Ingestion**:
   - Upload Software Requirements Specifications (**SRS**), Business Requirements Documents (**BRD**), or Architecture Memos in `.pdf`, `.docx`, and `.txt` formats.
   - Automatically segments text into individual requirement statements.
   - Preserves source provenance: tracks document name, page numbers, and section references.

2. **Vector Embeddings & Semantic Search**:
   - Generates vector representations using an offline pure-Python sublinear TF-IDF engine or OpenAI embeddings.
   - Computes cosine similarity to retrieve overlapping or related candidate requirement pairs.

3. **LLM Conflict Reasoning & Classification**:
   - Evaluates requirement pairs using LLMs (OpenAI `gpt-4o-mini`, Groq `llama-3.3-70b-versatile`, or offline Heuristic Evaluator).
   - Categorizes each pair into:
     - 🔴 **Contradictory**: Incompatible numerical thresholds, conflicting authentication mandates, opposing architectural decisions.
     - 🟡 **Overlapping / Redundant**: Duplicate functional scope or overlapping statements.
     - 🟢 **Compatible / Related**: Complementary specifications.
   - Generates confidence scores, detailed root-cause explanations, and actionable engineering resolution recommendations.

4. **Interactive Conflict Report & Auditing**:
   - KPI metrics overview (Total Reqs, Pairs Evaluated, Contradictions Found, Overlaps Detected).
   - Side-by-side comparison cards with source provenance badges.
   - Search & filtering by conflict category or keyword.
   - Export reports as **JSON** or **CSV** for auditing.
   - One-click **Demo Benchmark** pre-loaded with medical software conflicts.

---

## 🚀 Quick Start Commands

### 1. Navigate to Project Directory
```powershell
cd "c:\Users\User\Desktop\AI contradiction detector"
```

### 2. Activate Virtual Environment
```powershell
.\.venv\Scripts\activate
```

### 3. Apply Database Migrations
```powershell
python manage.py makemigrations detector
python manage.py migrate
```

### 4. Run Test Suite
```powershell
python manage.py test detector
```

### 5. Start Development Server
```powershell
python manage.py runserver
```

Now open **http://127.0.0.1:8000/** in your browser!

---

## ⚙️ Configuration & Environment Variables (`.env`)

Configure providers in `.env`:

```env
SECRET_KEY=your-django-secret-key
DEBUG=True
ALLOWED_HOSTS=*

# Provider Selection: 'mock' (safe offline heuristic), 'openai', or 'groq'
LLM_PROVIDER=mock

# Embedding Selection: 'local' (offline TF-IDF), 'openai', or 'mock'
EMBEDDING_PROVIDER=local

# OpenAI (optional)
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Groq (optional)
GROQ_API_KEY=
GROQ_MODEL=llama-3.3-70b-versatile

# Defaults
SIMILARITY_THRESHOLD=0.10
MAX_UPLOAD_SIZE_MB=15
```

---

## 📁 Project Architecture

- [`config/`](file:///c:/Users/User/Desktop/AI%20contradiction%20detector/config): Django project settings and root URL routing.
- [`detector/`](file:///c:/Users/User/Desktop/AI%20contradiction%20detector/detector): Application logic, models, views, forms, context processors.
  - [`detector/services/document_parser.py`](file:///c:/Users/User/Desktop/AI%20contradiction%20detector/detector/services/document_parser.py): Text extraction and provenance tracking for PDF, DOCX, TXT.
  - [`detector/services/embeddings.py`](file:///c:/Users/User/Desktop/AI%20contradiction%20detector/detector/services/embeddings.py): Pure-Python TF-IDF and OpenAI embedding vector generation.
  - [`detector/services/similarity.py`](file:///c:/Users/User/Desktop/AI%20contradiction%20detector/detector/services/similarity.py): Cosine similarity and candidate pair ranking.
  - [`detector/services/conflict_analyzer.py`](file:///c:/Users/User/Desktop/AI%20contradiction%20detector/detector/services/conflict_analyzer.py): LLM & heuristic reasoning engine.
- [`templates/`](file:///c:/Users/User/Desktop/AI%20contradiction%20detector/templates): Bootstrap 5 responsive UI templates (`base.html`, `upload.html`, `report.html`, `session_list.html`).
- [`static/`](file:///c:/Users/User/Desktop/AI%20contradiction%20detector/static): Custom CSS design system.
- [`sample_data/`](file:///c:/Users/User/Desktop/AI%20contradiction%20detector/sample_data): Pre-packaged benchmark requirements files (`sample_srs_hospital_v1.txt`, `sample_brd_billing_v2.txt`, `sample_srs_hospital_v1.docx`).
