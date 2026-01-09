# Drone Security Analyst Agent
## Technical Report

**Assignment**: FlytBase AI Engineer
**Author**: AI Engineer Candidate
**Date**: January 2025

---

## Executive Summary

This report documents the design, implementation, and evaluation of the Drone Security Analyst Agent prototype. The system processes simulated drone telemetry and video data to provide automated security monitoring, object detection, alert generation, and intelligent querying capabilities.

**Key Achievements:**
- Functional prototype with all required components
- 5 predefined security alert rules
- Frame-by-frame indexing with SQLite + ChromaDB semantic search
- LangChain v1 with **create_react_agent** (latest API)
- **LangGraph multi-agent orchestration** with supervisor pattern
- **Groq API integration** (Llama 3.3-70B, free tier available)
- Human-in-the-loop for critical alerts
- Comprehensive test suite with **142 test cases**
- **Streamlit Web Dashboard** for interactive demonstration
- Bonus features: Video summarization and Q&A system

---

## 1. Problem Approach

### 1.1 Understanding the Requirements

The assignment required building a prototype that:
1. Processes simulated drone telemetry and video frames
2. Analyzes video content to identify objects/events
3. Generates real-time security alerts
4. Implements frame-by-frame indexing for querying

### 1.2 Solution Strategy

I adopted a **modular architecture** with clear separation of concerns:

```
Simulator → Analyzer → Alert Engine → Database
                ↓
         LangGraph Multi-Agent System
         (Supervisor → Worker Agents → Human Review)
```

This design allows:
- Independent testing of each component
- Easy replacement of simulated components with real ones
- Scalability for production deployment

---

## 2. Design Decisions

### 2.1 Simulated Video Frames vs Real VLM

**Decision**: Use text-based frame descriptions instead of actual image processing.

**Justification**:
- The assignment explicitly mentions "Simulate video frames with text descriptions"
- Faster development and testing cycle
- No GPU requirements for prototype
- Same architecture supports real VLM integration later
- Demonstrates the system design without computer vision complexity

**Trade-offs**:
- Cannot demonstrate actual image recognition accuracy
- Limited to predefined scenarios

### 2.2 SQLite vs Vector Database

**Decision**: Use SQLite for frame indexing instead of ChromaDB or Pinecone.

**Justification**:
- Zero configuration - just works
- Portable single-file database
- Sufficient for prototype scale (thousands of frames)
- Full-text search capabilities
- Easy to migrate to PostgreSQL for production

**Trade-offs**:
- No semantic/vector search
- Single-machine limitation

### 2.3 LangChain + LangGraph vs Custom Agent

**Decision**: Use LangChain framework with **LangGraph for multi-agent orchestration**.

**Justification**:
- Industry-standard framework (as mentioned in assignment)
- Built-in tool management and function calling
- Supports multiple LLM providers (OpenAI, Anthropic, etc.)
- Well-documented and maintained
- Easy to extend with new tools
- **LangGraph provides:**
  - Supervisor pattern for intelligent routing
  - Stateful workflows with memory checkpointing
  - Human-in-the-loop capabilities for critical alerts
  - Parallel agent execution for performance

**Trade-offs**:
- Additional dependency
- Some overhead for simple use cases

### 2.4 Rule-Based vs ML Alerting

**Decision**: Implement deterministic rule-based alerting.

**Justification**:
- Predictable and explainable behavior
- Easy to configure and test
- No training data required
- Appropriate for security-critical applications where false negatives are costly
- Rules can be customized per deployment

**Trade-offs**:
- Cannot detect novel anomalies
- Requires manual rule creation

---

## 3. Implementation Details

### 3.1 Core Components

| Component | Lines of Code | Purpose |
|-----------|--------------|---------|
| simulator.py | ~350 | Generates telemetry and frames |
| database.py | ~400 | SQLite storage and queries |
| vector_store.py | ~750 | ChromaDB semantic search |
| analyzer.py | ~380 | VLM-based frame analysis |
| alert_engine.py | ~350 | Security rule evaluation |
| agent.py | ~500 | LangChain agent orchestration |
| **graph_agent.py** | **~1100** | **LangGraph multi-agent system** |
| bonus_features.py | ~300 | Summarization and Q&A |
| main.py | ~450 | Application entry point |

### 3.2 Alert Rules Implementation

```python
ALERT_RULES = [
    {
        "id": "R001",
        "name": "Night Activity",
        "condition": "person detected AND time between 00:00-05:00",
        "priority": "HIGH"
    },
    {
        "id": "R002",
        "name": "Loitering Detection",
        "condition": "same person in same zone for > 5 minutes",
        "priority": "HIGH"
    },
    # ... 3 more rules
]
```

### 3.3 Database Schema

```sql
-- Frame indexing table
CREATE TABLE frame_index (
    frame_id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    location_name TEXT,
    location_zone TEXT,
    description TEXT,
    objects JSON,
    alert_triggered INTEGER
);

-- Alerts table
CREATE TABLE alerts (
    alert_id INTEGER PRIMARY KEY,
    timestamp TEXT NOT NULL,
    frame_id INTEGER,
    rule_id TEXT,
    priority TEXT,
    description TEXT
);
```

---

## 4. Testing Strategy

### 4.1 Test Coverage

| Test File | Test Cases | Focus Area |
|-----------|-----------|------------|
| test_simulator.py | 16 | Data generation |
| test_database.py | 22 | Storage operations |
| test_analyzer.py | 18 | Frame analysis |
| test_alert_engine.py | 17 | Alert rules |
| test_vector_store.py | 30 | Semantic search |
| test_graph_agent.py | 26 | LangGraph multi-agent |
| test_integration.py | 13 | End-to-end |
| **Total** | **142** | **All components** |

### 4.2 Key Test Scenarios

1. **Truck Detection**: Verify "Blue Ford F150" is correctly identified
2. **Night Alert**: Confirm person at 2 AM triggers HIGH alert
3. **Recurring Vehicle**: Track same vehicle across multiple frames
4. **Query Accuracy**: "Show all trucks" returns correct results
5. **Cooldown**: Prevent duplicate alerts within 5 minutes

### 4.3 Sample Test Output

```
tests/test_alert_engine.py::TestAlertEngine::test_night_activity_alert PASSED
tests/test_alert_engine.py::TestAlertEngine::test_loitering_alert PASSED
tests/test_database.py::TestSecurityDatabase::test_query_frames_by_description PASSED
tests/test_integration.py::TestEndToEndPipeline::test_simulate_and_process_events PASSED
```

---

## 5. Results and Examples

### 5.1 Sample Detection Output

```
[12:00:15] Main Gate
  Blue Ford F150 pickup truck entering through main gate
  Detected: vehicle (blue pickup truck)

[12:05:30] Warehouse
  Person in dark clothing walking near warehouse
  Detected: person (unknown)
  [ALERT - MEDIUM] Activity detected near storage zone
```

### 5.2 Sample Query Results

**Query**: "Show all truck events today"

**Result**:
```json
{
  "query": "truck",
  "result_count": 4,
  "frames": [
    {"frame_id": 1, "timestamp": "2024-01-15T10:15:00", "description": "Blue Ford F150 at gate"},
    {"frame_id": 7, "timestamp": "2024-01-15T14:00:00", "description": "Same blue Ford F150 returning"},
    ...
  ]
}
```

### 5.3 Sample Alert

```
[ALERT - HIGH] 02:30:00
Rule: R001 (Night Activity)
Description: Person detected at Main Gate during restricted hours (02:30)
Location: Main Gate
Status: active
```

---

## 6. AI Tools Usage

### 6.1 Claude Code Impact

| Task | Manual Estimate | With Claude | Saved |
|------|----------------|-------------|-------|
| Architecture design | 4 hours | 1 hour | 75% |
| Core modules | 12 hours | 4 hours | 67% |
| Test cases | 4 hours | 1.5 hours | 62% |
| Documentation | 3 hours | 1 hour | 67% |
| **Total** | **23 hours** | **7.5 hours** | **67%** |

### 6.2 Specific Contributions

1. **Architecture**: Claude suggested the modular design with clear interfaces
2. **Code Structure**: Generated well-organized Python modules with dataclasses
3. **Alert Rules**: Helped design the 5 security rules and their conditions
4. **Test Cases**: Created comprehensive test scenarios including edge cases
5. **Documentation**: Generated README, feature spec, and this report

### 6.3 Manual Customization

While Claude Code generated the foundation, I customized:
- Alert rule thresholds for realistic behavior
- Scenario templates for meaningful demo data
- Query parsing logic for natural language
- Output formatting for readability

---

## 7. Limitations and Future Work

### 7.1 Current Limitations

1. **No Real Image Processing**: Uses text descriptions, not actual computer vision
2. **Single Drone**: No multi-drone coordination
3. **Local Storage**: SQLite doesn't scale horizontally

### 7.2 Implemented Improvements

The following were implemented beyond the basic requirements:
- **ChromaDB Vector Store**: Semantic search with sentence-transformers
- **Streamlit Web Dashboard**: Interactive 5-tab UI for demonstration
- **Groq API Support**: Free-tier LLM with Llama 3.3-70B
- **LangChain v1 Compatibility**: Updated to latest APIs (create_react_agent)

### 7.3 Future Improvements

**Short-term (with more time):**
- Integrate BLIP-2 or LLaVA for actual frame analysis
- Add real-time WebSocket streaming for live updates

**Long-term (production):**
- Kubernetes deployment with horizontal scaling
- ML-based anomaly detection
- Multi-drone support with cross-correlation
- Mobile push notifications

---

## 8. Conclusion

The Drone Security Analyst Agent prototype successfully demonstrates:

✅ **Core Requirements**
- Telemetry and video frame processing
- Object detection and logging
- Real-time alert generation
- Frame-by-frame indexing and querying

✅ **Bonus Features**
- Video summarization
- Follow-up Q&A capability

✅ **Quality**
- Comprehensive test coverage
- Well-documented codebase
- Modular architecture for extensibility

The system is ready for demonstration and provides a solid foundation for production development.

---

## Appendix A: Running the Demo

```bash
# Install dependencies
pip install -r requirements.txt

# Quick terminal demo (no API key needed)
python demo.py

# Web dashboard (recommended)
streamlit run streamlit_app.py

# Full system demo
python -m src.main --curated

# Interactive mode
python -m src.main --interactive

# Run all 142 tests
pytest tests/ -v
```

## Appendix B: Repository Structure

```
drone-security-agent/
├── src/               # Source code (10 modules)
├── tests/             # Test suite (142 tests)
├── docs/              # Documentation
├── data/              # Database storage (SQLite + ChromaDB)
├── streamlit_app.py   # Web UI dashboard
├── demo.py            # Quick terminal demo
├── validate_system.py # System validation script
└── README.md          # Setup instructions
```

## Appendix C: LLM Configuration

```bash
# .env file configuration

# Option A: Groq API (Recommended - Free tier)
LLM_PROVIDER=groq
GROQ_API_KEY=your-key-here

# Option B: OpenAI API
LLM_PROVIDER=openai
OPENAI_API_KEY=your-key-here
```

---

*Report generated with Claude Code assistance*
