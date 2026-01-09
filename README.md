# Drone Security Analyst Agent

An AI-powered security monitoring system for drone surveillance, built as part of the FlytBase AI Engineer assignment.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-1.0+-green.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-1.0+-purple.svg)
![Groq](https://img.shields.io/badge/Groq-Llama_3.3-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Components](#components)
- [Testing](#testing)
- [AI Tools Used](#ai-tools-used)
- [Design Decisions](#design-decisions)
- [Future Improvements](#future-improvements)

---

## Overview

The **Drone Security Analyst Agent** is a prototype system that processes simulated drone telemetry data and video frames to provide automated security monitoring for property surveillance. The system detects objects (vehicles, people), generates real-time alerts based on predefined security rules, and maintains a queryable database of all detected events.

### Key Capabilities

- **Real-time Object Detection**: Identifies vehicles (with make/model/color), people, and animals from video frame descriptions
- **Intelligent Alerting**: Generates security alerts based on configurable rules (night activity, loitering, perimeter breaches)
- **Frame Indexing**: Stores all frames in a searchable SQLite database with timestamp, location, and object metadata
- **Natural Language Queries**: Ask questions like "Show all truck events at the gate"
- **Context Tracking**: Tracks recurring objects across frames (e.g., "same blue truck seen 3 times")

---

## Features

### Core Features

| Feature | Description |
|---------|-------------|
| Telemetry Simulation | Generates realistic drone telemetry (GPS, altitude, battery) |
| Video Frame Simulation | Creates frame descriptions with detected objects |
| VLM-based Analysis | Analyzes frames using simulated or API-based vision models |
| Alert Engine | Evaluates 5 predefined security rules with priority levels |
| Frame Database | SQLite-based storage with full-text and temporal queries |
| ChromaDB Vector Store | Semantic search using sentence-transformer embeddings |
| LangChain Agent | AI agent with tools for analysis, querying, and summarization |

### Bonus Features

| Feature | Description |
|---------|-------------|
| Video Summarization | Generates concise summaries of surveillance sessions |
| Follow-up Q&A | Natural language interface for querying security events |
| Daily Reports | Automated daily security reports with statistics |
| Semantic Search | Find frames by meaning, not just keywords (e.g., "suspicious activity near gate") |
| Similar Event Discovery | Find events similar to a specific incident |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES (Simulated)                          │
│  ┌─────────────────────┐              ┌─────────────────────┐           │
│  │   Drone Telemetry   │              │    Video Frames     │           │
│  │   (GPS, Altitude)   │              │   (Descriptions)    │           │
│  └──────────┬──────────┘              └──────────┬──────────┘           │
└─────────────┼────────────────────────────────────┼──────────────────────┘
              │                                    │
              ▼                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           ANALYSIS LAYER                                 │
│  ┌──────────────────────┐    ┌──────────────────────┐                   │
│  │   Frame Analyzer     │    │   Object Tracker     │                   │
│  │   (VLM/Simulated)    │    │   (Cross-frame)      │                   │
│  └──────────────────────┘    └──────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        INTELLIGENCE LAYER                                │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    LangChain Security Agent                        │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐      │  │
│  │  │Analyze  │ │ Query   │ │ Alert   │ │Summary  │ │  Q&A    │      │  │
│  │  │ Frame   │ │ Frames  │ │ Engine  │ │Generator│ │ System  │      │  │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘      │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          STORAGE LAYER                                   │
│  ┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────┐ │
│  │   Frame Index DB     │ │    Alert Log DB      │ │  ChromaDB Vector │ │
│  │      (SQLite)        │ │      (SQLite)        │ │      Store       │ │
│  └──────────────────────┘ └──────────────────────┘ └──────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Git

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/drone-security-agent.git
   cd drone-security-agent
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv

   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment** (for LLM-powered analysis)
   ```bash
   # Create .env file with your preferred LLM provider

   # Option A: Groq API (Recommended - Free tier available)
   # Get your API key at: https://console.groq.com
   echo "LLM_PROVIDER=groq" > .env
   echo "GROQ_API_KEY=your-groq-api-key-here" >> .env

   # Option B: OpenAI API (Fallback)
   echo "LLM_PROVIDER=openai" > .env
   echo "OPENAI_API_KEY=your-openai-api-key-here" >> .env
   ```

### Verify Installation

```bash
# Run system validation
python validate_system.py

# Run tests
pytest tests/ -v

# Run a quick terminal demo (no API key needed)
python demo.py

# Run full system demo
python -m src.main --curated
```

---

## Usage

### Web Interface (Streamlit)

The easiest way to explore the system is through the interactive web dashboard:

```bash
# Launch the Streamlit web app
streamlit run streamlit_app.py
```

This opens a browser with 5 interactive tabs:
- **Live Demo**: Watch real-time frame processing with alerts
- **Frame Processing**: Analyze individual frames step-by-step
- **Alerts**: View and filter security alerts by priority
- **Query Database**: Search the frame database with natural language
- **Summary**: Generate reports and ask follow-up questions

### Quick Terminal Demo

For a fast demonstration without the web interface:

```bash
# Run the terminal demo (no API key required)
python demo.py
```

### Command Line Interface

```bash
# Run curated demo (recommended for first run)
python -m src.main --curated

# Run random simulation with 20 events
python -m src.main --demo --events 20

# Start interactive mode after demo
python -m src.main --interactive

# Run without OpenAI API (simulated analysis only)
python -m src.main --demo --no-api

# Clear database before running
python -m src.main --curated --clear-db

# Disable LangGraph multi-agent system (use simple agent)
python -m src.main --demo --no-langgraph
```

### Interactive Mode Commands

Once in interactive mode, you can:

| Command | Description |
|---------|-------------|
| `stats` | Show system statistics |
| `alerts` | Show recent alerts |
| `help` | Show available commands |
| `quit` | Exit interactive mode |

**Example Queries:**
- "Show all truck events"
- "Any activity near the gate?"
- "What vehicles were seen today?"
- "Give me a summary"
- "Any recurring vehicles?"

### Programmatic Usage

```python
from src.agent import SecurityAnalystAgent
from src.database import SecurityDatabase
from datetime import datetime

# Initialize
db = SecurityDatabase()
agent = SecurityAnalystAgent(database=db, use_api=False)

# Process a frame
result = agent.process_frame(
    frame_id=1,
    timestamp=datetime.now(),
    description="Blue Ford F150 at main gate",
    location={"name": "Main Gate", "zone": "perimeter"},
    telemetry={"latitude": 37.77, "longitude": -122.41}
)

print(f"Objects detected: {result['tracked_objects']}")
print(f"Alerts: {result['alerts']}")

# Query the database
response = agent.chat("Show all vehicle events")
print(response)

# Cleanup
db.close()
```

---

## Project Structure

```
drone-security-agent/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Configuration settings (Groq/OpenAI)
│   ├── simulator.py         # Telemetry & frame simulation
│   ├── database.py          # SQLite database operations
│   ├── vector_store.py      # ChromaDB vector store for semantic search
│   ├── analyzer.py          # VLM-based frame analysis
│   ├── alert_engine.py      # Security alert rules engine
│   ├── agent.py             # LangChain security agent with tools
│   ├── graph_agent.py       # LangGraph multi-agent orchestration
│   ├── bonus_features.py    # Summarization & Q&A
│   └── main.py              # Main application entry
├── tests/
│   ├── __init__.py
│   ├── test_simulator.py    # Simulator tests (16 tests)
│   ├── test_database.py     # Database tests (22 tests)
│   ├── test_vector_store.py # Vector store tests (30 tests)
│   ├── test_analyzer.py     # Analyzer tests (18 tests)
│   ├── test_alert_engine.py # Alert engine tests (17 tests)
│   ├── test_graph_agent.py  # LangGraph agent tests (26 tests)
│   └── test_integration.py  # Integration tests (13 tests)
├── docs/
│   ├── REPORT.md            # Technical report
│   ├── FEATURE_SPEC.md      # Feature specification
│   └── ARCHITECTURE.md      # Architecture documentation
├── data/                    # Database storage (SQLite + ChromaDB)
├── output/                  # Output files
├── streamlit_app.py         # Web UI dashboard
├── demo.py                  # Quick terminal demo
├── validate_system.py       # System validation script
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variables template
└── README.md                # This file
```

---

## Components

### 1. Simulator (`simulator.py`)

Generates realistic drone telemetry and video frame data:

- **TelemetryData**: GPS coordinates, altitude, battery level
- **VideoFrame**: Frame descriptions with detected objects
- **ScenarioGenerator**: Creates contextual scenarios based on time of day
- **DroneSimulator**: Main simulation orchestrator

### 2. Database (`database.py`)

SQLite-based storage with three main tables:

- **frame_index**: Indexed frames with timestamps, locations, and objects
- **alerts**: Security alerts with priority and status
- **detections**: Individual object detections

Supports queries by:
- Time range
- Location/zone
- Object type
- Description text
- Complex multi-filter queries

### 2.5. Vector Store (`vector_store.py`)

ChromaDB-based semantic search using sentence-transformer embeddings:

- **Semantic Search**: Find frames by meaning, not keywords
- **Similarity Search**: Find frames similar to a reference frame
- **Hybrid Search**: Combine semantic search with metadata filters
- **Embedding Model**: Uses `all-MiniLM-L6-v2` for fast, accurate embeddings

Query capabilities:
- Natural language: "find suspicious activity at night"
- Object filtering: Filter by vehicle, person, or animal
- Zone filtering: Focus on specific areas (perimeter, parking, etc.)
- Alert filtering: Only show frames that triggered alerts
- Time-based: Search within specific time ranges

### 3. Analyzer (`analyzer.py`)

Frame analysis using VLM (Vision Language Models):

- **FrameAnalyzer**: Extracts objects from frame descriptions
- **ObjectTracker**: Tracks recurring objects across frames
- Supports both simulated and API-based (GPT-4 Vision) analysis

### 4. Alert Engine (`alert_engine.py`)

Rule-based security alert system:

| Rule | Trigger | Priority |
|------|---------|----------|
| R001 | Person detected 00:00-05:00 | HIGH |
| R002 | Loitering (same zone > 5 min) | HIGH |
| R003 | Activity in perimeter zone | MEDIUM |
| R004 | Same vehicle > 2 times in 24h | LOW |
| R005 | Unknown vehicle in restricted area | MEDIUM |

### 5. Agent (`agent.py`)

LangChain-powered security analyst agent with tools:

- `analyze_frame`: Process video frames
- `query_frames`: Search frame database
- `query_by_time`: Time-based queries
- `get_alerts`: Retrieve alerts
- `generate_summary`: Create activity summaries
- `get_statistics`: System statistics
- `get_recurring_objects`: Track recurring detections
- `semantic_search`: ChromaDB-powered semantic search
- `find_similar_frames`: Find similar security events

### 5.5. LangGraph Multi-Agent System (`graph_agent.py`)

Production-grade multi-agent orchestration using LangGraph:

```
            ┌─────────────────┐
            │   Supervisor    │
            │  (LLM Routing)  │
            └────────┬────────┘
                     │
    ┌────────┬───────┼───────┬────────┐
    ▼        ▼       ▼       ▼        ▼
┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐
│Analyzer││Alerter ││Searcher││Summary ││  Chat  │
└───┬────┘└───┬────┘└───┬────┘└───┬────┘└───┬────┘
    │         │         │         │         │
    └────┬────┴─────────┴─────────┴─────────┘
         │
┌────────▼────────┐
│  Human Review   │ (for critical alerts)
└────────┬────────┘
         │
┌────────▼────────┐
│Response Builder │
└─────────────────┘
```

**Key Features:**
- **Supervisor Pattern**: LLM-based intelligent routing to specialized agents
- **Human-in-the-Loop**: Critical/HIGH alerts pause for human review
- **Stateful Workflows**: Persistent state with memory checkpointing
- **Tool-Equipped Agents**: Each agent has specialized capabilities
- **Graceful Fallback**: Works offline with rule-based routing

**Agent Types:**
| Agent | Responsibility |
|-------|---------------|
| Supervisor | Routes queries to appropriate worker agents |
| Analyzer | Extracts objects and activities from frames |
| Alerter | Evaluates security rules and generates alerts |
| Searcher | Semantic and keyword search across database |
| Summarizer | Generates reports and activity summaries |
| Human Review | Flags critical alerts for human approval |

### 6. Bonus Features (`bonus_features.py`)

- **VideoSummarizer**: Generates session summaries and daily reports
- **SecurityQA**: Natural language Q&A for security queries

---

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test Files

```bash
pytest tests/test_simulator.py -v
pytest tests/test_database.py -v
pytest tests/test_alert_engine.py -v
pytest tests/test_integration.py -v
```

### Test Coverage

```bash
pytest tests/ --cov=src --cov-report=html
```

### Test Cases Summary

| Test File | Test Count | Coverage |
|-----------|------------|----------|
| test_simulator.py | 16 | Telemetry, frames, scenarios, streaming |
| test_database.py | 22 | CRUD operations, queries, statistics |
| test_analyzer.py | 18 | Object extraction, tracking, security relevance |
| test_alert_engine.py | 17 | All 5 alert rules, cooldown, formatting |
| test_vector_store.py | 30 | ChromaDB semantic search, filters |
| test_graph_agent.py | 26 | LangGraph multi-agent orchestration |
| test_integration.py | 13 | End-to-end pipeline, persistence, performance |
| **Total** | **142** | **Complete test coverage** |

---

## AI Tools Used

### Claude Code (Primary)

- **Architecture Design**: Generated system architecture and component structure
- **Code Generation**: Created all Python modules with proper abstractions
- **Documentation**: Generated README, feature spec, and architecture docs
- **Test Cases**: Created comprehensive test suite
- **Debugging**: Helped identify and fix issues during development

### Impact on Workflow

| Task | Without AI | With AI | Time Saved |
|------|-----------|---------|------------|
| Architecture design | 4 hours | 1 hour | 75% |
| Core implementation | 12 hours | 4 hours | 67% |
| Test writing | 4 hours | 1.5 hours | 62% |
| Documentation | 3 hours | 1 hour | 67% |

---

## Design Decisions

### 1. Simulated VLM Instead of Real Vision Model

**Decision**: Use text-based frame descriptions instead of actual image processing.

**Rationale**:
- Faster prototyping without GPU requirements
- Easier to test and demonstrate
- Same architecture supports real VLM integration
- Assignment focus is on system design, not computer vision

### 2. SQLite + ChromaDB for Storage

**Decision**: Use SQLite for structured data and ChromaDB for semantic search.

**Rationale**:
- SQLite: Zero configuration, portable, handles structured queries
- ChromaDB: Provides semantic search capabilities with vector embeddings
- Sentence-transformers: Fast local embeddings without API costs
- Dual storage enables both keyword and semantic queries
- Easy to migrate to production databases later

### 3. LangChain + LangGraph Multi-Agent System

**Decision**: Use LangChain with LangGraph for multi-agent orchestration.

**Rationale**:
- Industry-standard framework
- Built-in tool management
- Supports multiple LLM backends
- Easy to extend with new capabilities
- **LangGraph provides:**
  - Supervisor pattern for intelligent query routing
  - Specialized worker agents (Analyzer, Alerter, Searcher, Summarizer)
  - Human-in-the-loop for critical/high priority alerts
  - Stateful workflows with memory checkpointing
  - Graceful fallback to rule-based routing when offline

### 4. Rule-Based Alerting

**Decision**: Implement deterministic alert rules instead of ML-based anomaly detection.

**Rationale**:
- Predictable and explainable
- Easy to configure and test
- No training data required
- Appropriate for prototype scope

### 5. Offline-First Design

**Decision**: System works without API access.

**Rationale**:
- Enables testing without API costs
- Demonstrates core logic independently
- API enhances but doesn't replace functionality

---

## Future Improvements

### If Time Permitted

1. **Real VLM Integration**
   - Integrate BLIP-2 or LLaVA for actual image analysis
   - Process real video frames from sample footage

2. **Enhanced Vector Search**
   - Cross-modal search (image + text)
   - Temporal pattern detection using embeddings

3. **Real-time Streaming**
   - WebSocket-based live updates
   - Dashboard UI for monitoring

4. **Advanced Anomaly Detection**
   - ML-based unusual pattern detection
   - Behavioral analysis over time

5. **Multi-Drone Support**
   - Coordinate multiple drone feeds
   - Cross-drone object tracking

6. **Enhanced Visualization**
   - Web dashboard with maps
   - Timeline view of events
   - Alert notification system

---

## Sample Output

### Detection Log
```
[12:00:15] DETECTION: Blue Ford F150 pickup truck spotted at Main Gate
[12:00:45] DETECTION: Same vehicle (Blue Ford F150) now at Parking Lot
[12:05:30] DETECTION: Person in safety vest near Warehouse
```

### Alert Output
```
[ALERT - HIGH] 02:30:00 | Person detected at Main Gate during restricted hours | Location: Main Gate
[ALERT - MEDIUM] 10:15:00 | Activity detected near perimeter at Back Fence | Location: Back Fence
[ALERT - LOW] 14:00:00 | Vehicle (Blue Ford F150) detected 3 times today | Location: Main Gate
```

### Query Results
```
Query: "Show all truck events today"
Result: Found 4 frames with vehicles:
  • [10:15] Blue Ford F150 entering main gate
  • [10:45] Blue Ford F150 at parking lot
  • [14:00] Red delivery truck at loading dock
  • [16:30] Blue Ford F150 leaving via main gate
```

---

## License

This project was created as part of the FlytBase AI Engineer assignment.

---

## Contact

For questions about this implementation, please contact the repository owner.

---

*Built with Claude Code assistance*
