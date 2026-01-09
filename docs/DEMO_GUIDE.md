# Quick Demo Guide

## 5-Minute Demo for Evaluators

This guide provides a quick walkthrough to demonstrate all features of the Drone Security Analyst Agent.

---

## Prerequisites

```bash
# Ensure you're in the project directory
cd drone-security-agent

# Activate virtual environment (if using)
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

---

## Demo Option 1: Streamlit Web Dashboard (Recommended)

```bash
streamlit run streamlit_app.py
```

### Tab 1: Live Demo
1. Click **"Run Curated Demo"** button
2. Watch the AI process 5 sample frames in real-time
3. See detected objects and triggered alerts

### Tab 2: Frame Processing
1. Enter a description like:
   ```
   a FEMALE LADY WITH BAG AND COVERING HER FACE IS DETECTED which is very suspicious
   ```
2. Select location: **Main Gate**, Zone: **perimeter**
3. Click **"Analyze Frame with AI"**
4. See:
   - AI detects: Person with bag, face covered
   - Alert triggered: R006 Suspicious Behavior (HIGH)
   - Threat Level: MEDIUM/HIGH

### Tab 3: Alerts
- View all triggered alerts filtered by priority (HIGH/MEDIUM/LOW)

### Tab 4: Query Database
1. Run demo first (Tab 1)
2. Ask: **"What vehicles were detected?"**
3. AI responds with vehicles from the database

### Tab 5: Summary
1. Click **"Generate AI Summary"**
2. Get a comprehensive security report

---

## Demo Option 2: Terminal Demo (No UI)

```bash
# Quick demo (no API key needed)
python demo.py

# Full curated demo
python -m src.main --curated
```

---

## Demo Option 3: Run Tests

```bash
# Run all 142 tests
pytest tests/ -v

# Run specific test
pytest tests/test_alert_engine.py -v
```

---

## Key Features to Highlight

| Feature | How to Demo |
|---------|-------------|
| LLM-Powered Analysis | Frame Processing tab - AI analyzes descriptions |
| Alert Rules (6 rules) | Enter suspicious descriptions, see alerts |
| Database Queries | Query tab - ask natural language questions |
| AI Summaries | Summary tab - generate reports |
| Object Detection | Live Demo - see extracted objects |

---

## Sample Test Inputs

### 1. Suspicious Person
```
Person in dark hoodie covering face near fence at 2 AM
```
**Expected:** R001 (Night Activity) + R006 (Suspicious Behavior)

### 2. Vehicle Detection
```
Blue Ford F150 pickup truck entering main gate
```
**Expected:** Vehicle detected with color and type

### 3. Perimeter Alert
```
Unknown person walking near back fence area
```
**Expected:** R003 (Perimeter Activity) if zone is perimeter

---

## Architecture Highlights

```
User Input → LLM Analysis (Groq/Llama 3.3-70B) → Object Extraction → Alert Rules → Database → Response
```

- **LangChain v1** with create_react_agent
- **LangGraph** multi-agent orchestration
- **SQLite** + **ChromaDB** for storage
- **Streamlit** for web UI

---

## Contact

Repository: https://github.com/Itz-gopi204/Drone-Ai-Assignment
