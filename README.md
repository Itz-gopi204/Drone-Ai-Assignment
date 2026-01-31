# Drone Security Analyst Agent

An AI-powered security monitoring system for drone surveillance, built as part of the FlytBase AI Engineer assignment.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-1.0+-green.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-1.0+-purple.svg)
![Groq](https://img.shields.io/badge/Groq-Llama_3.3-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## Quick Links

| Resource | Link |
|----------|------|
| **Live Demo** | [https://drone-ai-assignment-kimbqjywkddhe4e5sreumg.streamlit.app/](https://drone-ai-assignment-kimbqjywkddhe4e5sreumg.streamlit.app/) |
| **Technical Report** | [docs/REPORT.md](docs/REPORT.md) |
| **Architecture Diagrams** | [docs/REPORT.md#5-solution-architecture](docs/REPORT.md#5-solution-architecture) |

---

## Highlights

| Feature | Description |
|---------|-------------|
| **6 Security Alert Rules** | Night activity, loitering, perimeter breaches, suspicious behavior |
| **Multi-Agent System** | LangGraph supervisor with specialized analyzer, alerter, and summarizer agents |
| **Dual Storage** | SQLite for structured queries + ChromaDB for semantic search |
| **VLM-Ready** | Supports BLIP-2, GPT-4 Vision, and Direct Vision Analysis |
| **Free Tier Friendly** | Uses Groq API (Llama 3.3-70B) - no credit card required |
| **142 Test Cases** | Comprehensive test coverage across all components |

---

## Table of Contents

- [System Overview](#system-overview)
- [Complete Execution Flow](#complete-execution-flow)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Component Details](#component-details)
- [Security Alert Rules](#security-alert-rules)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Design Decisions](#design-decisions)

---

## System Overview

The **Drone Security Analyst Agent** is a prototype system that processes drone telemetry and video frames to provide automated security monitoring. The system detects objects, generates real-time alerts, and maintains a queryable database of all events.

### What It Does

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   INPUT      │────▶│   ANALYZE    │────▶│    ALERT     │────▶│   OUTPUT     │
│              │     │              │     │              │     │              │
│ Video/Text   │     │ LLM extracts │     │ 6 rules      │     │ Dashboard    │
│ Drone Data   │     │ objects      │     │ evaluate     │     │ Database     │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

### Key Capabilities

| Capability | How It Works |
|------------|--------------|
| **Object Detection** | LLM analyzes frame descriptions → extracts vehicles, people, attributes |
| **Alert Generation** | 6 configurable rules check each frame → triggers HIGH/MEDIUM/LOW alerts |
| **Frame Indexing** | SQLite stores structured data + ChromaDB stores embeddings for semantic search |
| **Natural Language Queries** | Ask "Show all trucks at gate" → LangChain agent queries both databases |
| **Video Summarization** | LLM aggregates all frames → generates security report |

---

## Complete Execution Flow

### Flow 1: Live Demo Processing

```
User clicks "Run Curated Demo"
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1: LOAD SAMPLE FRAMES                                         │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ SAMPLE_FRAMES = [                                              │ │
│  │   {frame_id: 1, description: "Blue Ford F150 at main gate",   │ │
│  │    location: {name: "Main Gate", zone: "perimeter"},          │ │
│  │    timestamp: "2024-01-15T10:15:30"}                          │ │
│  │   ...5 total frames                                           │ │
│  │ ]                                                              │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2: FOR EACH FRAME → LLM ANALYSIS                              │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ analyze_frame_with_llm(description, location, timestamp)       │ │
│  │                                                                │ │
│  │ PROMPT TO LLM:                                                 │ │
│  │ "You are a security analyst. Analyze this frame:               │ │
│  │  Description: Blue Ford F150 at main gate                      │ │
│  │  Location: Main Gate (perimeter)                               │ │
│  │  Time: 10:15 (Day time)                                        │ │
│  │                                                                │ │
│  │  Check these rules:                                            │ │
│  │  - R001: Person at night (00:00-05:00) → HIGH                  │ │
│  │  - R003: Perimeter activity → MEDIUM                           │ │
│  │  - R006: Suspicious behavior → HIGH                            │ │
│  │                                                                │ │
│  │  Return JSON: {objects, alerts, threat_level}"                 │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│                              ▼                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ LLM RESPONSE:                                                  │ │
│  │ {                                                              │ │
│  │   "objects": [{"type": "vehicle", "description": "Blue Ford   │ │
│  │                F150 pickup truck"}],                          │ │
│  │   "alerts": [{"rule_id": "R003", "name": "Perimeter Activity",│ │
│  │               "priority": "MEDIUM"}],                         │ │
│  │   "threat_level": "LOW"                                       │ │
│  │ }                                                              │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 3: STORE IN DATABASE                                          │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ SQLite: db.index_frame(frame_id, timestamp, location,         │ │
│  │                        description, objects, telemetry)       │ │
│  │                                                                │ │
│  │ SQLite: db.add_alert(frame_id, rule_id, priority, description)│ │
│  │                                                                │ │
│  │ Session State: processed_frames.append(processed)             │ │
│  │ Session State: all_alerts.extend(alerts)                      │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 4: DISPLAY RESULTS                                            │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ UI shows:                                                      │ │
│  │ ┌─────────────────────────────────────────────────────────┐   │ │
│  │ │ Frame 1 | 2024-01-15T10:15:30                           │   │ │
│  │ │ 📍 Main Gate (perimeter)                                │   │ │
│  │ │ 📝 Blue Ford F150 pickup truck entering through gate    │   │ │
│  │ │ 🎯 Objects: vehicle - Blue Ford F150 pickup truck       │   │ │
│  │ └─────────────────────────────────────────────────────────┘   │ │
│  │ ┌─────────────────────────────────────────────────────────┐   │ │
│  │ │ [MEDIUM] Perimeter Activity                             │   │ │
│  │ │ Rule: R003 - Activity detected near perimeter           │   │ │
│  │ └─────────────────────────────────────────────────────────┘   │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Flow 2: Video/Image Upload Processing

```
User uploads video.mp4
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1: VIDEO PROCESSING (vlm_processor.py)                        │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ class VideoProcessor:                                          │ │
│  │     def extract_frames(video_path, interval=5):                │ │
│  │         cap = cv2.VideoCapture(video_path)                     │ │
│  │         fps = cap.get(CAP_PROP_FPS)  # e.g., 30 fps           │ │
│  │         frame_skip = fps * interval  # skip 150 frames (5s)   │ │
│  │                                                                │ │
│  │         while cap.isOpened():                                  │ │
│  │             ret, frame = cap.read()                            │ │
│  │             if frame_count % frame_skip == 0:                  │ │
│  │                 yield VideoFrame(frame_id, timestamp, frame)   │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2: VLM CAPTIONING (based on provider)                         │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Provider: "simulated" (default for demo)                       │ │
│  │ ─────────────────────────────────────────                      │ │
│  │ class SimulatedVLM:                                            │ │
│  │     def caption_frame(frame_data):                             │ │
│  │         # Returns random security scenario                     │ │
│  │         return "Person in dark clothing near warehouse"        │ │
│  │                                                                │ │
│  │ Provider: "direct" (GPT-4 Vision - RECOMMENDED)                │ │
│  │ ─────────────────────────────────────────                      │ │
│  │ class DirectVisionAnalyzer:                                    │ │
│  │     def analyze_frame(frame_data, location, timestamp):        │ │
│  │         base64_image = encode_image(frame_data)                │ │
│  │         response = openai.chat.completions.create(             │ │
│  │             model="gpt-4o",                                    │ │
│  │             messages=[{                                        │ │
│  │                 "role": "user",                                │ │
│  │                 "content": [                                   │ │
│  │                     {"type": "text", "text": security_prompt}, │ │
│  │                     {"type": "image_url", "url": base64_image} │ │
│  │                 ]                                              │ │
│  │             }]                                                 │ │
│  │         )                                                      │ │
│  │         return {objects, alerts, threat_level, analysis}       │ │
│  │                                                                │ │
│  │ Provider: "blip2" (Local GPU)                                  │ │
│  │ ─────────────────────────────────────────                      │ │
│  │ class BLIP2Captioner:                                          │ │
│  │     model = Blip2ForConditionalGeneration.from_pretrained(     │ │
│  │         "Salesforce/blip2-opt-2.7b"                            │ │
│  │     )                                                          │ │
│  │     def caption_frame(frame_data):                             │ │
│  │         inputs = processor(frame_data, return_tensors="pt")    │ │
│  │         output = model.generate(**inputs)                      │ │
│  │         return processor.decode(output)                        │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
            │
            ▼
        [Same as Flow 1: STEP 2-4]
```

### Flow 3: Natural Language Query

```
User asks: "What vehicles were detected today?"
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1: QUERY PROCESSING (bonus_features.py)                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ class SecurityQA:                                              │ │
│  │     def answer(query):                                         │ │
│  │         # 1. Get context from database                         │ │
│  │         recent_frames = db.get_recent_frames(hours=24)         │ │
│  │         recent_alerts = db.get_recent_alerts(hours=24)         │ │
│  │                                                                │ │
│  │         # 2. Build prompt with context                         │ │
│  │         prompt = f"""                                          │ │
│  │         You are a security analyst. Answer based on this data: │ │
│  │                                                                │ │
│  │         RECENT FRAMES:                                         │ │
│  │         {json.dumps(recent_frames)}                            │ │
│  │                                                                │ │
│  │         RECENT ALERTS:                                         │ │
│  │         {json.dumps(recent_alerts)}                            │ │
│  │                                                                │ │
│  │         USER QUESTION: {query}                                 │ │
│  │         """                                                    │ │
│  │                                                                │ │
│  │         # 3. Get LLM response                                  │ │
│  │         return llm.invoke(prompt)                              │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LLM RESPONSE:                                                      │
│  "Based on the surveillance data, 2 vehicles were detected today:  │
│   - Blue Ford F150 pickup truck at Main Gate (10:15)               │
│   - Red Toyota Camry at Parking Lot (14:45)                        │
│                                                                     │
│   The Ford F150 was seen 3 times, triggering a R004 Repeat Vehicle │
│   alert."                                                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                  │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                      Streamlit Web Dashboard                            │ │
│  │  ┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐  │ │
│  │  │ Live     │ Video/   │ Frame    │ Alerts   │ Query    │ Summary  │  │ │
│  │  │ Demo     │ Image    │ Process  │          │ Database │          │  │ │
│  │  └──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INTELLIGENCE LAYER                                 │
│  ┌──────────────────────────────┐  ┌──────────────────────────────────────┐ │
│  │       LLM Providers          │  │        Multi-Agent System            │ │
│  │  ┌────────────────────────┐  │  │  ┌──────────────────────────────┐   │ │
│  │  │ Groq API               │  │  │  │      Supervisor Agent        │   │ │
│  │  │ (Llama 3.3-70B)        │  │  │  │   (Routes to workers)        │   │ │
│  │  │ [DEFAULT - FREE]       │  │  │  └──────────────────────────────┘   │ │
│  │  └────────────────────────┘  │  │              │                      │ │
│  │  ┌────────────────────────┐  │  │  ┌──────────┼──────────┐            │ │
│  │  │ OpenAI API             │  │  │  ▼          ▼          ▼            │ │
│  │  │ (GPT-4o-mini)          │  │  │ Analyzer  Alerter  Searcher         │ │
│  │  │ [FALLBACK]             │  │  │ Agent     Agent    Agent            │ │
│  │  └────────────────────────┘  │  └──────────────────────────────────────┘ │
│  └──────────────────────────────┘                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PROCESSING LAYER                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐  │
│  │   VLM Processor  │  │   Alert Engine   │  │      Analyzer            │  │
│  │  ┌────────────┐  │  │  ┌────────────┐  │  │  ┌────────────────────┐  │  │
│  │  │ OpenCV     │  │  │  │ R001-R006  │  │  │  │ Object Extraction  │  │  │
│  │  │ BLIP-2     │  │  │  │ Rule Check │  │  │  │ Object Tracking    │  │  │
│  │  │ GPT-4V     │  │  │  │ Priority   │  │  │  │ Attribute Parse    │  │  │
│  │  │ Simulated  │  │  │  │ Assignment │  │  │  └────────────────────┘  │  │
│  │  └────────────┘  │  │  └────────────┘  │  └──────────────────────────┘  │
│  └──────────────────┘  └──────────────────┘                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            STORAGE LAYER                                     │
│  ┌──────────────────────────────┐  ┌──────────────────────────────────────┐ │
│  │         SQLite               │  │           ChromaDB                   │ │
│  │  ┌────────────────────────┐  │  │  ┌──────────────────────────────┐   │ │
│  │  │ frame_index            │  │  │  │ security_frames collection   │   │ │
│  │  │ alerts                 │  │  │  │ all-MiniLM-L6-v2 embeddings  │   │ │
│  │  │ detections             │  │  │  │ Semantic search              │   │ │
│  │  └────────────────────────┘  │  │  └──────────────────────────────┘   │ │
│  └──────────────────────────────┘  └──────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW                                       │
└─────────────────────────────────────────────────────────────────────────────┘

    INPUT                    PROCESSING                       OUTPUT
    ─────                    ──────────                       ──────

┌──────────┐         ┌─────────────────────┐         ┌──────────────────┐
│  Video   │────────▶│  OpenCV Extraction  │────────▶│  Frame Images    │
│  File    │         │  (1 frame/5 sec)    │         │  (numpy arrays)  │
└──────────┘         └─────────────────────┘         └────────┬─────────┘
                                                              │
                                                              ▼
┌──────────┐         ┌─────────────────────┐         ┌──────────────────┐
│  Text    │────────▶│  VLM Captioner      │────────▶│  Text Description│
│  Input   │         │  (BLIP-2/GPT-4V)    │         │  "Blue truck..." │
└──────────┘         └─────────────────────┘         └────────┬─────────┘
                                                              │
                                                              ▼
                     ┌─────────────────────┐         ┌──────────────────┐
                     │  LLM Analysis       │────────▶│  Structured JSON │
                     │  (Groq/OpenAI)      │         │  {objects, alerts│
                     │                     │         │   threat_level}  │
                     └─────────────────────┘         └────────┬─────────┘
                                                              │
                              ┌────────────────────────────────┤
                              │                                │
                              ▼                                ▼
                     ┌─────────────────┐              ┌─────────────────┐
                     │  SQLite DB      │              │  ChromaDB       │
                     │  (Structured)   │              │  (Vectors)      │
                     │                 │              │                 │
                     │  - Timestamps   │              │  - Embeddings   │
                     │  - Locations    │              │  - Similarity   │
                     │  - Alerts       │              │  - Semantic     │
                     └─────────────────┘              └─────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Git

### Quick Start

```bash
# 1. Clone repository
git clone https://github.com/Itz-gopi204/Drone-Ai-Assignment.git
cd Drone-Ai-Assignment

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API key (Groq is FREE)
# Get your key at: https://console.groq.com
echo "LLM_PROVIDER=groq" > .env
echo "GROQ_API_KEY=your-key-here" >> .env

# 5. Run the app
streamlit run streamlit_app.py
```

### Verify Installation

```bash
# Run system validation
python validate_system.py

# Run tests
pytest tests/ -v

# Quick terminal demo (no API needed)
python demo.py
```

---

## Usage

### Streamlit Dashboard (6 Tabs)

```bash
streamlit run streamlit_app.py
```

| Tab | Function | What It Does |
|-----|----------|--------------|
| **Live Demo** | Process 5 sample frames | Shows real-time AI analysis with threat levels |
| **Video/Image Upload** | Upload MP4/JPG files | Extracts frames → VLM caption → LLM analysis |
| **Frame Processing** | Analyze custom text | Enter any description → see alerts triggered |
| **Alerts** | View all alerts | Filter by HIGH/MEDIUM/LOW priority |
| **Query Database** | Ask questions | "What vehicles today?" → AI-powered answer |
| **Summary** | Generate reports | AI creates security summary of all events |

### CLI Commands

```bash
# Run curated demo (recommended)
python -m src.main --curated

# Run with 20 random events
python -m src.main --demo --events 20

# Interactive query mode
python -m src.main --interactive

# Run without API (keyword-based only)
python -m src.main --demo --no-api
```

---

## Component Details

### File Structure

```
drone-security-agent/
├── src/
│   ├── config.py              # API keys, paths, alert rules
│   ├── simulator.py           # Telemetry & frame generation
│   ├── database.py            # SQLite CRUD operations
│   ├── vector_store.py        # ChromaDB semantic search
│   ├── analyzer.py            # Object extraction & tracking
│   ├── alert_engine.py        # 6 security rules (R001-R006)
│   ├── agent.py               # LangChain agent with tools
│   ├── graph_agent.py         # LangGraph multi-agent system
│   ├── bonus_features.py      # Summarization & Q&A
│   ├── vlm_processor.py       # Video/image processing
│   ├── vision_pipeline.py     # Direct GPT-4 Vision pipeline
│   ├── batch_vision_pipeline.py  # Cost-effective BLIP + Groq pipeline
│   └── main.py                # CLI entry point
├── tests/                     # 142 test cases
├── docs/
│   ├── REPORT.md              # Technical report (IMPORTANT)
│   ├── ARCHITECTURE.md        # System architecture
│   └── FEATURE_SPEC.md        # Feature specification
├── streamlit_app.py           # Web dashboard
├── demo_batch_pipeline.py     # Batch pipeline demo
├── demo_vision_pipeline.py    # Direct vision demo
└── requirements.txt           # Dependencies
```

### Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `BatchVisionPipeline` | batch_vision_pipeline.py | **Cost-effective** video processing (BLIP + Groq) |
| `LocalVLMCaptioner` | batch_vision_pipeline.py | Local BLIP model for frame captioning |
| `DirectVisionPipeline` | vision_pipeline.py | Per-frame GPT-4 Vision analysis |
| `VLMProcessor` | vlm_processor.py | Video frame extraction + VLM captioning |
| `SecurityDatabase` | database.py | SQLite frame/alert storage |
| `FrameVectorStore` | vector_store.py | ChromaDB semantic search |
| `AlertEngine` | alert_engine.py | Evaluates 6 security rules |
| `SecurityAnalystAgent` | agent.py | LangChain agent with tools |
| `VideoSummarizer` | bonus_features.py | Generates AI summaries |
| `SecurityQA` | bonus_features.py | Natural language Q&A |

---

## Vision Processing Pipelines

The system offers **two vision processing strategies**:

### Batch Pipeline (Recommended - FREE)

```
Video → OpenCV Frames → BLIP (local GPU) → Text Descriptions → ONE Groq LLM Call → Analysis
```

| Aspect | Details |
|--------|---------|
| **Cost** | $0.00 per video |
| **VLM** | BLIP (4GB GPU) or BLIP-2 (8GB+ GPU) |
| **LLM** | Groq Llama 3.3-70B (free tier) |
| **API Calls** | 1 per video (regardless of frame count) |

```bash
# Run batch pipeline demo
python demo_batch_pipeline.py

# Process your video
python demo_batch_pipeline.py --video your_video.mp4
```

### Direct Pipeline (Per-frame API)

```
Video → OpenCV Frames → GPT-4 Vision per frame → Analysis
```

| Aspect | Details |
|--------|---------|
| **Cost** | ~$0.02 per frame |
| **VLM** | GPT-4 Vision (highest accuracy) |
| **API Calls** | 1 per frame |

```bash
# Run direct pipeline demo
python demo_vision_pipeline.py --provider direct
```

### Cost Comparison

| Pipeline | 50 Frames | 100 Frames |
|----------|-----------|------------|
| **Batch (BLIP + Groq)** | **$0.00** | **$0.00** |
| Direct (GPT-4 Vision) | $1.00 | $2.00 |

---

## Security Alert Rules

| Rule ID | Name | Priority | Trigger Condition |
|---------|------|----------|-------------------|
| **R001** | Night Activity | HIGH | Person detected between 00:00-05:00 |
| **R002** | Loitering Detection | HIGH | Same person in zone > 5 minutes |
| **R003** | Perimeter Activity | MEDIUM | Any activity in perimeter zone |
| **R004** | Repeat Vehicle | LOW | Same vehicle > 2 times in 24 hours |
| **R005** | Unknown Vehicle | MEDIUM | Unrecognized vehicle in restricted area |
| **R006** | Suspicious Behavior | HIGH | Face covering, hiding, trespassing |

### How Rules Are Checked

```python
# In streamlit_app.py → analyze_frame_with_llm()

prompt = f"""
SECURITY ALERT RULES TO CHECK:
- R001 Night Activity (HIGH): Person detected between 00:00-05:00
- R002 Loitering Detection (HIGH): Person staying in same area
- R003 Perimeter Activity (MEDIUM): Activity in perimeter zone
- R004 Repeat Vehicle (LOW): Same vehicle seen multiple times
- R005 Unknown Vehicle (MEDIUM): Unrecognized vehicle
- R006 Suspicious Behavior (HIGH): Face covering, hiding

FRAME INFO:
- Description: {description}
- Location: {location['name']} ({location['zone']})
- Time: {timestamp} ({'Night' if 0 <= hour < 5 else 'Day'})

Return JSON with triggered alerts.
"""

# LLM evaluates rules and returns:
{
    "alerts": [
        {"rule_id": "R001", "priority": "HIGH", "reason": "Person at 2:30 AM"}
    ],
    "threat_level": "HIGH"
}
```

---

## API Reference

### VLM Processing

```python
from src.vlm_processor import VLMProcessor, VLMConfig

# Configure processor
config = VLMConfig(
    provider="direct",  # "simulated", "blip2", "gpt4v", "direct"
    frame_interval_seconds=5,
    max_frames=100
)

# Process video
processor = VLMProcessor(config)
frames = processor.process_video("security_footage.mp4")

for frame in frames:
    print(f"Frame {frame.frame_id}: {frame.description}")
```

### Database Operations

```python
from src.database import SecurityDatabase

db = SecurityDatabase()

# Index a frame
db.index_frame(
    frame_id=1,
    timestamp=datetime.now(),
    location_name="Main Gate",
    location_zone="perimeter",
    description="Blue truck entering",
    objects=[{"type": "vehicle", "color": "blue"}]
)

# Query frames
results = db.query_frames(
    zone="perimeter",
    start_time=datetime.now() - timedelta(hours=1)
)

# Add alert
db.add_alert(
    frame_id=1,
    rule_id="R003",
    priority="MEDIUM",
    description="Perimeter activity detected"
)
```

### LLM Analysis

```python
from src.bonus_features import get_llm, VideoSummarizer, SecurityQA

# Get configured LLM
llm = get_llm()  # Returns Groq or OpenAI based on config

# Generate summary
summarizer = VideoSummarizer(db, use_api=True)
summary = summarizer.summarize_session()

# Ask questions
qa = SecurityQA(db, use_api=True)
answer = qa.answer("What vehicles were detected today?")
```

---

## Testing

### Run Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_alert_engine.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Summary

| Test File | Tests | What It Covers |
|-----------|-------|----------------|
| test_simulator.py | 16 | Telemetry, frames, scenarios |
| test_database.py | 22 | CRUD, queries, statistics |
| test_vector_store.py | 30 | Semantic search, embeddings |
| test_analyzer.py | 18 | Object extraction, tracking |
| test_alert_engine.py | 17 | All 6 alert rules |
| test_graph_agent.py | 26 | Multi-agent orchestration |
| test_integration.py | 13 | End-to-end pipeline |
| **Total** | **142** | **Complete coverage** |

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Simulated VLM default** | Works without GPU, demonstrates full architecture |
| **Groq over OpenAI** | Free tier available, faster inference |
| **SQLite + ChromaDB** | Structured queries + semantic search |
| **LangGraph multi-agent** | Scalable, debuggable, human-in-the-loop |
| **6 rule-based alerts** | Predictable, explainable, easy to configure |

---

## Sample Output

### Detection Log
```
[12:00:15] DETECTION: Blue Ford F150 pickup truck at Main Gate
[12:00:45] DETECTION: Same vehicle (Blue Ford F150) now at Parking Lot
[12:05:30] DETECTION: Person in safety vest near Warehouse
```

### Alert Output
```
[ALERT - HIGH] 02:30:00 | Person at Main Gate during restricted hours
[ALERT - MEDIUM] 10:15:00 | Activity detected near perimeter
[ALERT - LOW] 14:00:00 | Vehicle detected 3 times today
```

---

## Links

| Resource | URL |
|----------|-----|
| Live Demo | https://drone-ai-assignment-kimbqjywkddhe4e5sreumg.streamlit.app/ |
| GitHub | https://github.com/Itz-gopi204/Drone-Ai-Assignment |
| Technical Report | [docs/REPORT.md](docs/REPORT.md) |

---

**Author:** Gopi
