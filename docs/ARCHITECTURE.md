# System Architecture: Drone Security Analyst Agent

## Overview

This document describes the technical architecture for the Drone Security Analyst Agent, a real-time security monitoring system that processes drone telemetry and video data.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA SOURCES                                        │
│  ┌─────────────────────┐              ┌─────────────────────┐                   │
│  │   Drone Telemetry   │              │    Video Frames     │                   │
│  │   (Simulated)       │              │    (Simulated)      │                   │
│  │  - Timestamp        │              │  - Frame ID         │                   │
│  │  - GPS Location     │              │  - Image/Description│                   │
│  │  - Altitude         │              │  - Timestamp        │                   │
│  │  - Battery          │              │                     │                   │
│  └──────────┬──────────┘              └──────────┬──────────┘                   │
└─────────────┼────────────────────────────────────┼──────────────────────────────┘
              │                                    │
              ▼                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DATA INGESTION LAYER                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                        Data Stream Processor                             │    │
│  │   - Synchronizes telemetry with video frames                            │    │
│  │   - Creates unified event objects                                        │    │
│  │   - Handles data validation                                              │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           ANALYSIS LAYER                                         │
│                                                                                  │
│  ┌──────────────────────┐    ┌──────────────────────┐    ┌─────────────────┐   │
│  │   Vision Language    │    │    Object Tracker    │    │  Context        │   │
│  │   Model (VLM)        │    │                      │    │  Manager        │   │
│  │                      │    │  - Track objects     │    │                 │   │
│  │  - Frame analysis    │───▶│    across frames     │───▶│  - Maintain     │   │
│  │  - Object detection  │    │  - Assign IDs        │    │    session      │   │
│  │  - Scene description │    │  - Count entries     │    │    context      │   │
│  │                      │    │                      │    │  - Historical   │   │
│  │  [BLIP-2 / GPT-4V]   │    │                      │    │    awareness    │   │
│  └──────────────────────┘    └──────────────────────┘    └─────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           INTELLIGENCE LAYER                                     │
│                                                                                  │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                        LangChain Security Agent                            │  │
│  │                                                                            │  │
│  │   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   │  │
│  │   │   Alert     │   │   Query     │   │  Summary    │   │   Q&A       │   │  │
│  │   │   Tool      │   │   Tool      │   │   Tool      │   │   Tool      │   │  │
│  │   │             │   │             │   │             │   │             │   │  │
│  │   │ Evaluates   │   │ Searches    │   │ Generates   │   │ Answers     │   │  │
│  │   │ alert rules │   │ frame DB    │   │ summaries   │   │ questions   │   │  │
│  │   └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘   │  │
│  │                                                                            │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           STORAGE LAYER                                          │
│                                                                                  │
│  ┌──────────────────────┐    ┌──────────────────────┐    ┌─────────────────┐   │
│  │   Frame Index DB     │    │    Alert Log         │    │  Event Log      │   │
│  │   (SQLite)           │    │    (SQLite)          │    │  (JSON/SQLite)  │   │
│  │                      │    │                      │    │                 │   │
│  │  - frame_id          │    │  - alert_id          │    │  - event_id     │   │
│  │  - timestamp         │    │  - timestamp         │    │  - timestamp    │   │
│  │  - location          │    │  - priority          │    │  - type         │   │
│  │  - objects[]         │    │  - description       │    │  - description  │   │
│  │  - description       │    │  - location          │    │  - metadata     │   │
│  │  - embeddings        │    │  - status            │    │                 │   │
│  └──────────────────────┘    └──────────────────────┘    └─────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           OUTPUT LAYER                                           │
│                                                                                  │
│  ┌──────────────────────┐    ┌──────────────────────┐    ┌─────────────────┐   │
│  │   Console Output     │    │    Alert Display     │    │  Query Results  │   │
│  │   (Real-time logs)   │    │    (Priority-based)  │    │  (Formatted)    │   │
│  └──────────────────────┘    └──────────────────────┘    └─────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Data Sources (Simulated)

#### Telemetry Simulator
Generates realistic drone telemetry data:
```python
{
    "timestamp": "2024-01-15T12:00:00Z",
    "drone_id": "DRONE-001",
    "location": {
        "name": "Main Gate",
        "lat": 37.7749,
        "lon": -122.4194,
        "zone": "perimeter"
    },
    "altitude": 50.0,  # meters
    "battery": 85,     # percentage
    "status": "patrolling"
}
```

#### Video Frame Simulator
Generates simulated video frames with text descriptions:
```python
{
    "frame_id": 1,
    "timestamp": "2024-01-15T12:00:00Z",
    "description": "Blue Ford F150 pickup truck entering through main gate",
    "simulated_objects": ["vehicle:truck:blue", "gate:main"]
}
```

---

### 2. Analysis Layer

#### Vision Language Model (VLM)
- **Purpose**: Analyze video frames and generate detailed descriptions
- **Implementation Options**:
  - **BLIP-2** (Local): Open-source, runs locally, good for prototyping
  - **GPT-4 Vision** (API): Higher accuracy, requires API key
  - **LLaVA** (Local): Open-source alternative
- **For Prototype**: Using simulated text descriptions with LLM analysis

#### Object Tracker
- Maintains object identity across frames
- Tracks entry/exit counts
- Identifies recurring objects (e.g., "same blue truck seen 3 times")

#### Context Manager
- Maintains sliding window of recent events
- Provides historical context to agent
- Enables pattern detection

---

### 3. Intelligence Layer (LangChain Agent)

```
┌─────────────────────────────────────────────────────────────────┐
│                    LANGCHAIN AGENT ARCHITECTURE                  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                     LLM (GPT-4 / Claude)                  │   │
│  │                                                           │   │
│  │   System Prompt:                                          │   │
│  │   "You are a security analyst agent monitoring a          │   │
│  │    property via drone surveillance. Analyze events,       │   │
│  │    generate alerts, and answer security queries."         │   │
│  │                                                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                        TOOLS                              │   │
│  │                                                           │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐            │   │
│  │  │ analyze_   │ │ check_     │ │ query_     │            │   │
│  │  │ frame      │ │ alerts     │ │ history    │            │   │
│  │  └────────────┘ └────────────┘ └────────────┘            │   │
│  │                                                           │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐            │   │
│  │  │ log_       │ │ generate_  │ │ get_       │            │   │
│  │  │ detection  │ │ summary    │ │ context    │            │   │
│  │  └────────────┘ └────────────┘ └────────────┘            │   │
│  │                                                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Agent Tools

| Tool | Purpose | Input | Output |
|------|---------|-------|--------|
| `analyze_frame` | Process a video frame | Frame data + telemetry | Detected objects, description |
| `check_alerts` | Evaluate alert rules | Detection event | Alert (if triggered) or None |
| `query_history` | Search frame database | Query string | Matching frames |
| `log_detection` | Store detection | Detection data | Confirmation |
| `generate_summary` | Create summary | Time range | Summary text |
| `get_context` | Get recent events | None | Context window |

---

### 4. Alert Rules Engine

```python
ALERT_RULES = [
    {
        "id": "R001",
        "name": "Night Activity",
        "condition": "person detected AND time between 00:00-05:00",
        "priority": "HIGH",
        "message": "Person detected during restricted hours"
    },
    {
        "id": "R002",
        "name": "Loitering Detection",
        "condition": "same person in same zone for > 5 minutes",
        "priority": "HIGH",
        "message": "Person loitering detected"
    },
    {
        "id": "R003",
        "name": "Perimeter Activity",
        "condition": "any object near fence/perimeter",
        "priority": "MEDIUM",
        "message": "Activity detected near perimeter"
    },
    {
        "id": "R004",
        "name": "Repeat Vehicle",
        "condition": "same vehicle detected > 2 times in 24h",
        "priority": "LOW",
        "message": "Recurring vehicle pattern detected"
    }
]
```

---

### 5. Storage Schema

#### Frame Index Table
```sql
CREATE TABLE frame_index (
    frame_id INTEGER PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    location_name TEXT,
    location_zone TEXT,
    latitude REAL,
    longitude REAL,
    description TEXT,
    objects JSON,  -- ["vehicle:truck:blue", "person"]
    alert_triggered BOOLEAN DEFAULT FALSE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_timestamp ON frame_index(timestamp);
CREATE INDEX idx_location ON frame_index(location_zone);
```

#### Alerts Table
```sql
CREATE TABLE alerts (
    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    frame_id INTEGER,
    rule_id TEXT,
    priority TEXT,  -- HIGH, MEDIUM, LOW
    description TEXT,
    location TEXT,
    status TEXT DEFAULT 'active',  -- active, acknowledged, resolved
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (frame_id) REFERENCES frame_index(frame_id)
);
```

#### Detection Log Table
```sql
CREATE TABLE detections (
    detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    frame_id INTEGER,
    object_type TEXT,  -- person, vehicle, animal
    object_subtype TEXT,  -- truck, car, dog
    object_attributes JSON,  -- {"color": "blue", "make": "Ford"}
    location_zone TEXT,
    confidence REAL,
    FOREIGN KEY (frame_id) REFERENCES frame_index(frame_id)
);
```

---

## Data Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW DIAGRAM                                │
│                                                                               │
│   1. INGEST          2. ANALYZE         3. PROCESS         4. STORE/OUTPUT   │
│                                                                               │
│  ┌─────────┐       ┌─────────────┐     ┌─────────────┐     ┌─────────────┐   │
│  │Telemetry│──┐    │             │     │             │     │             │   │
│  │ Stream  │  │    │    VLM      │     │   Agent     │     │  Database   │   │
│  └─────────┘  ├───▶│  Analysis   │────▶│  Processing │────▶│  Storage    │   │
│  ┌─────────┐  │    │             │     │             │     │             │   │
│  │  Video  │──┘    │             │     │             │     │             │   │
│  │ Frames  │       └─────────────┘     └──────┬──────┘     └─────────────┘   │
│  └─────────┘                                  │                               │
│                                               ▼                               │
│                                        ┌─────────────┐                        │
│                                        │   Alerts    │                        │
│                                        │   Output    │                        │
│                                        └─────────────┘                        │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Processing Steps

1. **Frame Arrival**: Video frame + telemetry received
2. **VLM Analysis**: Frame analyzed for objects/scene description
3. **Object Tracking**: Objects matched with previous detections
4. **Context Update**: Context manager updated with new event
5. **Alert Check**: Alert rules evaluated against detection
6. **Storage**: Frame indexed in database
7. **Output**: Logs displayed, alerts triggered if needed

---

## Technology Stack

| Component | Technology | Justification |
|-----------|------------|---------------|
| Language | Python 3.10+ | Rich AI/ML ecosystem |
| LLM | OpenAI GPT-4 / Claude | High reasoning capability |
| Agent Framework | LangChain | Robust tool integration |
| VLM | BLIP-2 / Simulated | Prototype flexibility |
| Database | SQLite | Simple, no server needed |
| Vector Store | ChromaDB (optional) | Semantic search capability |

---

## Scalability Considerations

### Current (Prototype)
- Single drone
- Simulated data
- SQLite storage
- Local processing

### Future (Production)
- Multiple drones → Message queue (Kafka/RabbitMQ)
- Real video → GPU cluster for VLM
- SQLite → PostgreSQL/TimescaleDB
- Local → Kubernetes deployment

---

## Security Considerations

1. **Data Privacy**: Video frames contain sensitive information
2. **Access Control**: Alert system should have role-based access
3. **Audit Trail**: All actions logged for compliance
4. **Encryption**: Data at rest and in transit should be encrypted

---

*Architecture Version: 1.0*
*Last Updated: January 2025*
