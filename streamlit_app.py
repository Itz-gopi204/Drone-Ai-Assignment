"""
Drone Security Analyst Agent - Demo Frontend

A Streamlit-based web interface to demonstrate the agent's capabilities:
- Frame processing and object detection
- Real-time alert generation
- Database querying
- Video summarization

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import json
import time
from datetime import datetime, timedelta
import random
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import actual agents and database
try:
    from src.database import SecurityDatabase
    from src.bonus_features import VideoSummarizer, SecurityQA, get_llm, has_api_key
    from src.config import LLM_PROVIDER, GROQ_API_KEY, OPENAI_API_KEY
    AGENTS_AVAILABLE = True
except ImportError as e:
    print(f"Agent import error: {e}")
    AGENTS_AVAILABLE = False

# Import VLM processor for video/image upload
try:
    from src.vlm_processor import (
        VLMProcessor, VLMConfig, get_vlm_status,
        process_uploaded_video, process_uploaded_image
    )
    VLM_AVAILABLE = True
except ImportError as e:
    print(f"VLM import error: {e}")
    VLM_AVAILABLE = False

# Import Direct Vision Pipeline (per-frame GPT-4 Vision - EXPENSIVE)
try:
    from src.vision_pipeline import (
        DirectVisionPipeline, PipelineConfig, FrameAnalysisResult,
        get_pipeline_status, process_video_with_vision, process_image_with_vision
    )
    VISION_PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"Vision Pipeline import error: {e}")
    VISION_PIPELINE_AVAILABLE = False

# Import Batch Vision Pipeline (RECOMMENDED - Cost Effective)
try:
    from src.batch_vision_pipeline import (
        BatchVisionPipeline, BatchPipelineConfig, BatchAnalysisResult,
        get_batch_pipeline_status, process_video_batch
    )
    BATCH_PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"Batch Pipeline import error: {e}")
    BATCH_PIPELINE_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Drone Security Analyst Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling - with explicit text colors for visibility
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5 !important;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555555 !important;
        text-align: center;
        margin-bottom: 2rem;
    }
    .frame-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #1a1a1a !important;
    }
    .frame-box strong {
        color: #1a1a1a !important;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #b71c1c !important;
    }
    .alert-high strong {
        color: #c62828 !important;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #e65100 !important;
    }
    .alert-medium strong {
        color: #ef6c00 !important;
    }
    .alert-low {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #0d47a1 !important;
    }
    .alert-low strong {
        color: #1565c0 !important;
    }
    .detection-box {
        background-color: #e8f5e9;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.25rem 0;
        color: #1b5e20 !important;
    }
    .detection-box strong {
        color: #2e7d32 !important;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #1a1a1a !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================== Simulated Data ====================

SAMPLE_FRAMES = [
    {
        "frame_id": 1,
        "timestamp": "2024-01-15T10:15:30",
        "description": "Blue Ford F150 pickup truck entering through main gate",
        "location": {"name": "Main Gate", "zone": "perimeter"},
        "telemetry": {"altitude": 50, "battery": 95, "drone_id": "DRONE-001"}
    },
    {
        "frame_id": 2,
        "timestamp": "2024-01-15T10:20:45",
        "description": "Person in dark clothing walking near warehouse loading dock",
        "location": {"name": "Warehouse", "zone": "storage"},
        "telemetry": {"altitude": 45, "battery": 93, "drone_id": "DRONE-001"}
    },
    {
        "frame_id": 3,
        "timestamp": "2024-01-15T00:30:15",
        "description": "Unknown person loitering near back fence for 10 minutes",
        "location": {"name": "Back Fence", "zone": "perimeter"},
        "telemetry": {"altitude": 55, "battery": 88, "drone_id": "DRONE-001"}
    },
    {
        "frame_id": 4,
        "timestamp": "2024-01-15T14:45:00",
        "description": "Red Toyota Camry parked in visitor parking area",
        "location": {"name": "Parking Lot", "zone": "parking"},
        "telemetry": {"altitude": 40, "battery": 85, "drone_id": "DRONE-001"}
    },
    {
        "frame_id": 5,
        "timestamp": "2024-01-15T02:15:00",
        "description": "Person detected near main office building entrance",
        "location": {"name": "Office Building", "zone": "main"},
        "telemetry": {"altitude": 50, "battery": 80, "drone_id": "DRONE-001"}
    },
]

ALERT_RULES = [
    {"id": "R001", "name": "Night Activity", "condition": "Person detected between 00:00-05:00", "priority": "HIGH"},
    {"id": "R002", "name": "Loitering Detection", "condition": "Same person in zone > 5 minutes", "priority": "HIGH"},
    {"id": "R003", "name": "Perimeter Activity", "condition": "Activity in perimeter zone", "priority": "MEDIUM"},
    {"id": "R004", "name": "Repeat Vehicle", "condition": "Same vehicle > 2 times in 24h", "priority": "LOW"},
    {"id": "R005", "name": "Unknown Vehicle", "condition": "Unrecognized vehicle in restricted area", "priority": "MEDIUM"},
    {"id": "R006", "name": "Suspicious Behavior", "condition": "Suspicious activity detected (covering face, hiding, etc.)", "priority": "HIGH"},
]

# ==================== Helper Functions ====================

def analyze_frame_with_llm(description: str, location: dict, timestamp: datetime) -> dict:
    """Use LLM to analyze frame description and extract objects + alerts."""
    if not (AGENTS_AVAILABLE and has_api_key()):
        return {"objects": [], "alerts": [], "analysis": "LLM not available"}

    try:
        llm = get_llm()
        if not llm:
            return {"objects": [], "alerts": [], "analysis": "LLM not available"}

        # Create prompt for LLM analysis
        prompt = f"""You are a security analyst for a drone surveillance system. Analyze this frame description and provide a security analysis.

FRAME INFORMATION:
- Description: {description}
- Location: {location['name']} (Zone: {location['zone']})
- Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}
- Time of day: {'Night time (restricted hours)' if 0 <= timestamp.hour < 5 else 'Day time'}

SECURITY ALERT RULES:
- R001 Night Activity (HIGH): Person detected between 00:00-05:00
- R002 Loitering Detection (HIGH): Person staying in same area for extended period
- R003 Perimeter Activity (MEDIUM): Any activity in perimeter zone
- R004 Repeat Vehicle (LOW): Same vehicle seen multiple times
- R005 Unknown Vehicle (MEDIUM): Unrecognized vehicle in restricted area
- R006 Suspicious Behavior (HIGH): Face covering, hiding, suspicious actions, trespassing

INSTRUCTIONS:
1. Identify ALL objects in the description (people, vehicles, animals, items)
2. For each object, extract attributes (color, type, behavior, clothing, etc.)
3. Check if ANY security alert rules should be triggered
4. Provide your analysis

Respond in this EXACT JSON format:
{{
    "objects": [
        {{"type": "person/vehicle/animal/object", "description": "detailed description with attributes"}}
    ],
    "alerts": [
        {{"rule_id": "R00X", "name": "Rule Name", "priority": "HIGH/MEDIUM/LOW", "reason": "why this alert was triggered"}}
    ],
    "analysis": "Brief security analysis of the scene",
    "threat_level": "NONE/LOW/MEDIUM/HIGH/CRITICAL"
}}

If no objects are detected, return empty arrays. Be thorough - identify ALL security concerns."""

        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)

        # Parse JSON from response
        import re
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            result = json.loads(json_match.group())
            return result
        else:
            return {"objects": [], "alerts": [], "analysis": response_text, "threat_level": "UNKNOWN"}

    except Exception as e:
        return {"objects": [], "alerts": [], "analysis": f"LLM error: {str(e)}", "threat_level": "UNKNOWN"}

def extract_objects(description: str) -> list:
    """Fallback: Extract objects from frame description using keywords."""
    objects = []
    description_lower = description.lower()

    # Vehicle detection
    vehicle_keywords = ["truck", "car", "van", "motorcycle", "pickup", "suv", "sedan", "camry", "f150", "vehicle", "bike", "bicycle"]
    colors = ["blue", "red", "black", "white", "silver", "gray", "green", "yellow", "orange", "brown"]

    for keyword in vehicle_keywords:
        if keyword in description_lower:
            color = next((c for c in colors if c in description_lower), "unknown")
            objects.append({"type": "vehicle", "subtype": keyword, "color": color})
            break

    # Person detection - expanded keywords
    person_keywords = ["person", "man", "woman", "lady", "female", "male", "individual", "someone",
                       "figure", "intruder", "worker", "visitor", "stranger", "people", "human",
                       "guy", "girl", "boy", "child", "adult", "suspect", "trespasser"]

    person_detected = any(kw in description_lower for kw in person_keywords)
    if person_detected:
        attributes = []
        if "bag" in description_lower or "backpack" in description_lower:
            attributes.append("carrying bag")
        if "cover" in description_lower or "mask" in description_lower or "hiding" in description_lower:
            attributes.append("face covered")
        if "suspicious" in description_lower:
            attributes.append("suspicious")
        if "dark" in description_lower:
            attributes.append("dark clothing")

        objects.append({
            "type": "person",
            "attributes": ", ".join(attributes) if attributes else "detected"
        })

    if "bag" in description_lower and not person_detected:
        objects.append({"type": "object", "subtype": "bag"})

    return objects

def check_alerts(frame: dict) -> list:
    """Fallback: Check alert rules against frame data using keywords."""
    alerts = []
    timestamp = datetime.fromisoformat(frame["timestamp"])
    description = frame["description"].lower()
    zone = frame["location"]["zone"]

    person_keywords = ["person", "man", "woman", "lady", "female", "male", "individual", "someone",
                       "figure", "intruder", "worker", "visitor", "stranger", "people", "human",
                       "guy", "girl", "boy", "child", "adult", "suspect", "trespasser"]
    person_detected = any(kw in description for kw in person_keywords)

    if person_detected and (timestamp.hour >= 0 and timestamp.hour < 5):
        alerts.append({
            "rule_id": "R001",
            "name": "Night Activity",
            "priority": "HIGH",
            "description": f"Person detected at {frame['location']['name']} during restricted hours ({timestamp.strftime('%H:%M')})"
        })

    suspicious_keywords = ["suspicious", "cover", "hiding", "mask", "hooded", "running", "fleeing",
                          "trespassing", "breaking", "climbing", "sneaking"]
    if person_detected and any(kw in description for kw in suspicious_keywords):
        alerts.append({
            "rule_id": "R006",
            "name": "Suspicious Behavior",
            "priority": "HIGH",
            "description": f"Suspicious activity detected at {frame['location']['name']}: {frame['description'][:100]}"
        })

    if "loitering" in description or "10 minutes" in description:
        alerts.append({
            "rule_id": "R002",
            "name": "Loitering Detection",
            "priority": "HIGH",
            "description": f"Person loitering detected at {frame['location']['name']}"
        })

    if zone == "perimeter" and ("person" in description or "vehicle" in description or "truck" in description):
        alerts.append({
            "rule_id": "R003",
            "name": "Perimeter Activity",
            "priority": "MEDIUM",
            "description": f"Activity detected near perimeter at {frame['location']['name']}"
        })

    return alerts

def search_frames(query: str, frames: list) -> list:
    """Search frames by query with smart matching."""
    query_lower = query.lower()

    # Handle "show all" / "every frame" / "all frames" type queries
    all_keywords = ["all frame", "every frame", "all event", "show all", "list all", "each frame", "all data"]
    if any(kw in query_lower for kw in all_keywords):
        return frames

    # Extract meaningful search words (filter out common words)
    stop_words = {"give", "me", "show", "find", "get", "the", "a", "an", "to", "for", "of",
                  "information", "related", "about", "what", "where", "when", "is", "are",
                  "was", "were", "been", "being", "have", "has", "had", "do", "does", "did",
                  "and", "or", "but", "in", "on", "at", "by", "with", "from"}

    query_words = [w for w in query_lower.split() if w not in stop_words and len(w) > 2]

    results = []
    for frame in frames:
        desc_lower = frame["description"].lower()
        location_name = frame["location"]["name"].lower()
        zone = frame["location"]["zone"].lower()

        # Check if any query word matches description, location, or zone
        for word in query_words:
            if word in desc_lower or word in location_name or word in zone:
                results.append(frame)
                break

    # If no results with word matching, try original substring match
    if not results and query_words:
        for frame in frames:
            if query_lower in frame["description"].lower():
                results.append(frame)

    return results

# ==================== Session State ====================

if "processed_frames" not in st.session_state:
    st.session_state.processed_frames = []
if "all_alerts" not in st.session_state:
    st.session_state.all_alerts = []
if "frame_database" not in st.session_state:
    st.session_state.frame_database = []

# Initialize database and agents
if "db" not in st.session_state:
    if AGENTS_AVAILABLE:
        try:
            st.session_state.db = SecurityDatabase()
            st.session_state.summarizer = VideoSummarizer(st.session_state.db, use_api=True)
            st.session_state.qa_system = SecurityQA(st.session_state.db, use_api=True)
            st.session_state.llm_available = has_api_key()
        except Exception as e:
            st.session_state.db = None
            st.session_state.summarizer = None
            st.session_state.qa_system = None
            st.session_state.llm_available = False
    else:
        st.session_state.db = None
        st.session_state.summarizer = None
        st.session_state.qa_system = None
        st.session_state.llm_available = False

# ==================== Main UI ====================

# Header
st.markdown('<p class="main-header">🛡️ Drone Security Analyst Agent</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Property Security Monitoring System</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("LLM Provider")
    if AGENTS_AVAILABLE and st.session_state.llm_available:
        provider_name = LLM_PROVIDER.upper() if AGENTS_AVAILABLE else "None"
        if provider_name == "GROQ":
            st.success("✅ Groq API Connected")
            st.info("Using: llama-3.3-70b-versatile")
        else:
            st.success("✅ OpenAI API Connected")
            st.info("Using: gpt-4o-mini")
    else:
        st.warning("⚠️ No LLM API configured")
        st.info("Add GROQ_API_KEY or OPENAI_API_KEY to .env")

    st.divider()

    st.subheader("📋 Alert Rules")
    for rule in ALERT_RULES:
        with st.expander(f"{rule['id']}: {rule['name']}"):
            st.write(f"**Condition:** {rule['condition']}")
            st.write(f"**Priority:** {rule['priority']}")

    st.divider()

    st.subheader("📊 Statistics")
    col1, col2 = st.columns(2)
    col1.metric("Frames Processed", len(st.session_state.processed_frames))
    col2.metric("Alerts Generated", len(st.session_state.all_alerts))

# Main content tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎬 Live Demo",
    "📹 Video/Image Upload",
    "🔍 Frame Processing",
    "⚠️ Alerts",
    "🔎 Query Database",
    "📝 Summary"
])

# ==================== Tab 1: Live Demo ====================
with tab1:
    st.header("Live Processing Demo")
    st.write("Watch the agent process simulated drone footage in real-time.")

    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("▶️ Run Curated Demo", type="primary", use_container_width=True):
            st.session_state.processed_frames = []
            st.session_state.all_alerts = []
            st.session_state.frame_database = []

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, frame in enumerate(SAMPLE_FRAMES):
                status_text.text(f"🤖 AI Agent processing Frame {frame['frame_id']}...")
                progress_bar.progress((i + 1) / len(SAMPLE_FRAMES))

                # Use LLM for processing if available
                timestamp = datetime.fromisoformat(frame["timestamp"])
                if AGENTS_AVAILABLE and has_api_key():
                    llm_result = analyze_frame_with_llm(
                        frame["description"],
                        frame["location"],
                        timestamp
                    )
                    objects = llm_result.get("objects", [])
                    # Convert LLM alerts to our format
                    llm_alerts = llm_result.get("alerts", [])
                    alerts = []
                    for a in llm_alerts:
                        alerts.append({
                            "rule_id": a.get("rule_id", "R000"),
                            "name": a.get("name", "Alert"),
                            "priority": a.get("priority", "MEDIUM"),
                            "description": a.get("reason", "Security concern detected")
                        })
                else:
                    # Fallback to keyword matching
                    objects = extract_objects(frame["description"])
                    alerts = check_alerts(frame)

                processed = {
                    **frame,
                    "objects": objects,
                    "alerts": alerts,
                    "processed_at": datetime.now().isoformat()
                }

                st.session_state.processed_frames.append(processed)
                st.session_state.frame_database.append(processed)
                st.session_state.all_alerts.extend(alerts)

                # Save to actual database for AI queries
                if AGENTS_AVAILABLE and st.session_state.db:
                    try:
                        # Save frame to database
                        st.session_state.db.index_frame(
                            frame_id=frame["frame_id"],
                            timestamp=datetime.fromisoformat(frame["timestamp"]),
                            location_name=frame["location"]["name"],
                            location_zone=frame["location"]["zone"],
                            description=frame["description"],
                            objects=objects,
                            telemetry=frame.get("telemetry", {})
                        )
                        # Save alerts to database
                        for alert in alerts:
                            st.session_state.db.add_alert(
                                frame_id=frame["frame_id"],
                                rule_id=alert["rule_id"],
                                priority=alert["priority"],
                                description=alert["description"]
                            )
                    except Exception as e:
                        pass  # Continue even if DB save fails

            status_text.text("✅ Demo complete!")
            st.success(f"Processed {len(SAMPLE_FRAMES)} frames, generated {len(st.session_state.all_alerts)} alerts")
            if AGENTS_AVAILABLE and st.session_state.db:
                st.info("💾 Data saved to database - AI queries now available!")
            st.rerun()

    with col2:
        if st.session_state.processed_frames:
            st.subheader("Recent Detections")
            for frame in st.session_state.processed_frames[-3:]:
                with st.container():
                    # Format objects for display (handle both LLM and fallback format)
                    obj_list = []
                    for o in frame.get('objects', []):
                        if 'description' in o:
                            obj_list.append(f"{o['type']}: {o['description'][:30]}")
                        else:
                            obj_list.append(o.get('type', 'unknown'))

                    st.markdown(f"""
                    <div class="frame-box">
                        <strong>Frame {frame['frame_id']}</strong> | {frame['timestamp']}<br>
                        📍 {frame['location']['name']} ({frame['location']['zone']})<br>
                        📝 {frame['description']}<br>
                        🎯 Objects: {', '.join(obj_list) or 'None detected'}
                    </div>
                    """, unsafe_allow_html=True)

    # Show recent alerts
    if st.session_state.all_alerts:
        st.subheader("🚨 Recent Alerts")
        for alert in st.session_state.all_alerts[-5:]:
            priority_class = f"alert-{alert['priority'].lower()}"
            st.markdown(f"""
            <div class="{priority_class}">
                <strong>[{alert['priority']}] {alert['name']}</strong><br>
                {alert['description']}
            </div>
            """, unsafe_allow_html=True)

# ==================== Tab 2: Video/Image Upload ====================
with tab2:
    st.header("📹 Video & Image Upload")
    st.write("Upload a video file or image to process with the Vision Language Model (VLM).")

    # Show Pipeline status
    if BATCH_PIPELINE_AVAILABLE:
        batch_status = get_batch_pipeline_status()
        col_status1, col_status2, col_status3, col_status4 = st.columns(4)
        with col_status1:
            if batch_status["groq_available"]:
                st.success("✅ Groq LLM Ready")
            else:
                st.warning("⚠️ No Groq API Key")
        with col_status2:
            if batch_status.get("cuda_available"):
                st.success(f"✅ GPU: {batch_status.get('gpu_name', 'Unknown')[:20]}")
            else:
                st.info("ℹ️ No GPU (CPU mode)")
        with col_status3:
            if batch_status["blip2_available"]:
                st.success("✅ BLIP Ready")
            else:
                st.info("ℹ️ Simulated VLM")
        with col_status4:
            st.success(f"💰 Cost: FREE")

        # Strategy selection with clear explanation
        st.markdown("""
        ### Vision Processing Strategy

        | Strategy | How it Works | Cost |
        |----------|--------------|------|
        | **batch** (Recommended) | Frames → Local VLM/Simulated → Text → ONE LLM call | **FREE** |
        | **direct** | Each frame → GPT-4 Vision API | ~$0.02/frame |

        **Batch Pipeline** (Recommended):
        1. Extract frames from video
        2. Generate text descriptions (BLIP-2 local or simulated)
        3. Send ALL descriptions to Groq LLM in ONE call
        4. Get comprehensive analysis with alerts

        **Cost Comparison:**
        - 50 frames with GPT-4 Vision: ~$1.00
        - 50 frames with Batch Pipeline: **$0.00** (Groq is free!)
        """)

        processing_strategy = st.selectbox(
            "Select Processing Strategy",
            options=["batch", "direct"],
            index=0,
            help="'batch' is recommended - uses free Groq API for analysis"
        )

        st.divider()

        # Two columns for Video and Image upload
        upload_col1, upload_col2 = st.columns(2)

        with upload_col1:
            st.subheader("🎬 Video Upload")
            video_file = st.file_uploader(
                "Upload a video file",
                type=["mp4", "avi", "mov", "mkv"],
                key="video_upload"
            )

            if video_file:
                st.video(video_file)

                frame_interval = st.slider("Frame extraction interval (seconds)", 1, 30, 5)
                max_frames = st.slider("Maximum frames to extract", 5, 50, 20)

                if st.button("🚀 Process Video", type="primary", key="process_video"):
                    # Reset video file and save to temp
                    video_file.seek(0)
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                        tmp.write(video_file.read())
                        tmp_path = tmp.name

                    try:
                        if processing_strategy == "batch":
                            # BATCH PIPELINE (RECOMMENDED - FREE)
                            st.info("🔄 Using Batch Pipeline (FREE with Groq)")

                            batch_config = BatchPipelineConfig(
                                frame_interval_seconds=frame_interval,
                                max_frames=max_frames,
                                vlm_provider="auto"  # Auto-detect BLIP if GPU available
                            )

                            pipeline = BatchVisionPipeline(config=batch_config)

                            # Progress tracking
                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            # Step 1: Extract frames and generate captions
                            status_text.text("📹 Extracting frames and generating descriptions...")
                            frames = pipeline.extract_frames(tmp_path)
                            progress_bar.progress(0.5)

                            # Step 2: Analyze ALL frames in ONE LLM call
                            status_text.text("🤖 Analyzing all frames with LLM (ONE API call)...")
                            result = pipeline.analyze_batch(frames)
                            progress_bar.progress(1.0)

                            # Process results
                            for frame_result in result.frames:
                                # Convert alerts to UI format
                                alerts = []
                                for a in frame_result.get("alerts", []):
                                    alerts.append({
                                        "rule_id": a.get("rule_id", "R000"),
                                        "name": a.get("name", "Alert"),
                                        "priority": a.get("priority", "MEDIUM"),
                                        "description": a.get("reason", a.get("name", "Security concern"))
                                    })

                                processed = {
                                    "frame_id": frame_result.get("frame_id"),
                                    "timestamp": frame_result.get("timestamp", datetime.now().isoformat()),
                                    "description": frame_result.get("description", ""),
                                    "location": frame_result.get("location", {}),
                                    "objects": frame_result.get("objects", []),
                                    "alerts": alerts,
                                    "threat_level": frame_result.get("threat_level", "UNKNOWN")
                                }

                                st.session_state.processed_frames.append(processed)
                                st.session_state.frame_database.append(processed)
                                st.session_state.all_alerts.extend(alerts)

                            # Show batch analysis results
                            status_text.text("✅ Batch analysis complete!")
                            st.success(f"Processed {len(result.frames)} frames in ONE API call!")

                            # Show summary
                            st.subheader("📊 Analysis Summary")
                            col_sum1, col_sum2, col_sum3 = st.columns(3)
                            col_sum1.metric("Frames Analyzed", len(result.frames))
                            col_sum2.metric("Alerts Generated", len(result.alerts))
                            col_sum3.metric("Threat Level", result.threat_assessment)

                            st.write(f"**Summary:** {result.summary}")

                            if result.statistics.get("patterns"):
                                st.subheader("🔍 Patterns Detected")
                                for pattern in result.statistics["patterns"]:
                                    st.write(f"- {pattern}")

                        else:
                            # DIRECT PIPELINE (EXPENSIVE - per frame API call)
                            st.warning("⚠️ Using Direct Pipeline (costs ~$0.02 per frame)")

                            pipeline_config = PipelineConfig(
                                provider="direct" if get_pipeline_status().get("direct_vision_available") else "simulated",
                                frame_interval_seconds=frame_interval,
                                max_frames=max_frames,
                                store_to_database=False
                            )

                            db = st.session_state.db if AGENTS_AVAILABLE else None
                            pipeline = DirectVisionPipeline(config=pipeline_config, database=db)

                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            frames_extracted = list(pipeline.extract_frames(tmp_path))
                            total_frames = len(frames_extracted)
                            st.info(f"📹 Extracted {total_frames} frames from video")

                            for i, frame_info in enumerate(frames_extracted):
                                status_text.text(f"🤖 Analyzing frame {i+1}/{total_frames} with Vision AI...")
                                progress_bar.progress((i + 1) / total_frames)

                                result = pipeline.analyze_frame(
                                    frame_data=frame_info["frame_data"],
                                    location=frame_info["location"],
                                    timestamp=frame_info["timestamp"],
                                    telemetry=frame_info["telemetry"],
                                    frame_id=frame_info["frame_id"]
                                )

                                alerts = []
                                for a in result.alerts:
                                    alerts.append({
                                        "rule_id": a.get("rule_id", "R000"),
                                        "name": a.get("name", "Alert"),
                                        "priority": a.get("priority", "MEDIUM"),
                                        "description": a.get("reason", a.get("name", "Security concern"))
                                    })

                                processed = {
                                    "frame_id": result.frame_id,
                                    "timestamp": result.timestamp.isoformat(),
                                    "description": result.description,
                                    "location": result.location,
                                    "telemetry": result.telemetry,
                                    "objects": result.objects,
                                    "alerts": alerts,
                                    "threat_level": result.threat_level,
                                    "analysis": result.analysis
                                }

                                st.session_state.processed_frames.append(processed)
                                st.session_state.frame_database.append(processed)
                                st.session_state.all_alerts.extend(alerts)

                            status_text.text("✅ Video processing complete!")
                            st.success(f"Processed {total_frames} frames, generated {len(st.session_state.all_alerts)} alerts")

                        st.rerun()

                    except Exception as e:
                        st.error(f"Error processing video: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
                    finally:
                        # Clean up temp file
                        import os
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)

        with upload_col2:
            st.subheader("🖼️ Image Upload")
            image_file = st.file_uploader(
                "Upload an image",
                type=["jpg", "jpeg", "png", "bmp"],
                key="image_upload"
            )

            # Location selection for image
            image_location = st.selectbox(
                "Select Image Location",
                options=["Main Gate", "Parking Lot", "Warehouse", "Loading Dock", "Back Fence", "Office Building"],
                key="image_location"
            )
            image_zone = st.selectbox(
                "Select Zone",
                options=["perimeter", "parking", "storage", "operations", "main"],
                key="image_zone"
            )

            if image_file:
                st.image(image_file, caption="Uploaded Image", use_container_width=True)

                if st.button("🔍 Analyze Image with Vision AI", type="primary", key="analyze_image"):
                    with st.spinner("🤖 Analyzing image with Direct Vision Pipeline..."):
                        try:
                            image_file.seek(0)
                            from PIL import Image
                            image = Image.open(image_file)

                            # Create pipeline - use direct if available, else simulated
                            provider = "direct" if get_pipeline_status().get("direct_vision_available") else "simulated"
                            pipeline_config = PipelineConfig(provider=provider)
                            db = st.session_state.db if AGENTS_AVAILABLE else None
                            pipeline = DirectVisionPipeline(config=pipeline_config, database=db)

                            # Analyze image with full context
                            location = {"name": image_location, "zone": image_zone}
                            timestamp = datetime.now()

                            result = pipeline.analyze_frame(
                                frame_data=image,
                                location=location,
                                timestamp=timestamp,
                                frame_id=len(st.session_state.processed_frames) + 1
                            )

                            st.success("✅ Image analyzed with Vision AI!")

                            # Display results in organized sections
                            col_result1, col_result2 = st.columns(2)

                            with col_result1:
                                st.subheader("📝 Scene Description")
                                st.info(result.description)

                                st.subheader("🎯 Detected Objects")
                                if result.objects:
                                    for obj in result.objects:
                                        st.markdown(f"""
                                        <div class="detection-box">
                                            <strong>{obj.get('type', 'object').title()}</strong>: {obj.get('description', 'N/A')}
                                        </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.write("No objects detected")

                            with col_result2:
                                st.subheader("⚠️ Security Alerts")
                                if result.alerts:
                                    for alert in result.alerts:
                                        priority = alert.get('priority', 'MEDIUM')
                                        priority_class = f"alert-{priority.lower()}"
                                        st.markdown(f"""
                                        <div class="{priority_class}">
                                            <strong>[{priority}] {alert.get('name', 'Alert')}</strong><br>
                                            Rule: {alert.get('rule_id', 'N/A')}<br>
                                            {alert.get('reason', '')}
                                        </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.success("No security alerts triggered")

                                st.subheader("🎚️ Threat Assessment")
                                threat_colors = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢", "NONE": "⚪"}
                                st.markdown(f"**Threat Level:** {threat_colors.get(result.threat_level, '❓')} **{result.threat_level}**")

                            # Show full analysis
                            with st.expander("📊 Full Security Analysis"):
                                st.write(result.analysis)
                                st.json(result.to_dict())

                            # Store results
                            alerts = []
                            for a in result.alerts:
                                alerts.append({
                                    "rule_id": a.get("rule_id", "R000"),
                                    "name": a.get("name", "Alert"),
                                    "priority": a.get("priority", "MEDIUM"),
                                    "description": a.get("reason", a.get("name", "Security concern"))
                                })

                            processed = {
                                "frame_id": result.frame_id,
                                "timestamp": timestamp.isoformat(),
                                "description": result.description,
                                "location": location,
                                "objects": result.objects,
                                "alerts": alerts,
                                "threat_level": result.threat_level,
                                "analysis": result.analysis
                            }
                            st.session_state.processed_frames.append(processed)
                            st.session_state.frame_database.append(processed)
                            st.session_state.all_alerts.extend(alerts)

                            # Save to database
                            if AGENTS_AVAILABLE and st.session_state.db:
                                try:
                                    st.session_state.db.index_frame(
                                        frame_id=result.frame_id,
                                        timestamp=timestamp,
                                        location_name=location["name"],
                                        location_zone=location["zone"],
                                        description=result.description,
                                        objects=result.objects,
                                        telemetry={}
                                    )
                                    for alert in alerts:
                                        st.session_state.db.add_alert(
                                            frame_id=result.frame_id,
                                            rule_id=alert["rule_id"],
                                            priority=alert["priority"],
                                            description=alert["description"]
                                        )
                                    st.info("💾 Saved to database")
                                except Exception as db_err:
                                    pass

                        except Exception as e:
                            st.error(f"Error analyzing image: {str(e)}")

        # Show processed frames from video/image
        if st.session_state.processed_frames:
            st.divider()
            st.subheader("📊 Processed Frames")
            for frame in st.session_state.processed_frames[-5:]:
                with st.expander(f"Frame {frame['frame_id']} - {frame.get('location', {}).get('name', 'Unknown')}"):
                    st.write(f"**Description:** {frame['description']}")
                    st.write(f"**Timestamp:** {frame['timestamp']}")
                    if frame.get('objects'):
                        st.write(f"**Objects:** {frame['objects']}")
                    if frame.get('alerts'):
                        st.write(f"**Alerts:** {frame['alerts']}")
    elif VLM_AVAILABLE:
        # Fallback to old VLM processor if new pipeline not available
        st.warning("⚠️ New Vision Pipeline not available. Using legacy VLM processor.")
        vlm_status = get_vlm_status()
        vlm_provider = st.selectbox(
            "Select VLM Provider (Legacy)",
            options=["simulated", "blip2", "gpt4v"],
            index=0
        )
        st.info("For best results, install openai package and set OPENAI_API_KEY")
    else:
        st.error("⚠️ Vision modules not available. Please check installation.")
        st.code("pip install opencv-python openai Pillow")

# ==================== Tab 3: Frame Processing ====================
with tab3:
    st.header("🔍 Frame-by-Frame Processing")
    st.write("See how the agent analyzes individual frames.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input: Simulated Frame")

        # Custom frame input
        custom_desc = st.text_area(
            "Frame Description",
            value="Blue Ford F150 pickup truck entering through main gate",
            height=100,
            help="Describe what's in the frame. Use keywords like: person, man, woman, lady, female, truck, car, suspicious, covering face, bag, etc."
        )

        col_a, col_b = st.columns(2)
        with col_a:
            location_name = st.selectbox("Location", ["Main Gate", "Warehouse", "Parking Lot", "Back Fence", "Office Building"])
        with col_b:
            zone = st.selectbox("Zone", ["perimeter", "storage", "parking", "main"])

        # Full date and time input
        st.write("**📅 Timestamp:**")
        col_date, col_time = st.columns(2)
        with col_date:
            frame_date = st.date_input("Date", value=datetime.now().date())
        with col_time:
            frame_time = st.time_input("Time", value=datetime.now().time())

        if st.button("🔬 Analyze Frame with AI", type="primary"):
            # Create frame with full datetime
            full_timestamp = datetime.combine(frame_date, frame_time)
            test_frame = {
                "frame_id": random.randint(100, 999),
                "timestamp": full_timestamp.isoformat(),
                "description": custom_desc,
                "location": {"name": location_name, "zone": zone},
                "telemetry": {"altitude": 50, "battery": 90, "drone_id": "DRONE-001"}
            }

            # Use LLM for analysis if available
            if AGENTS_AVAILABLE and has_api_key():
                with st.spinner("🤖 AI Agent analyzing frame..."):
                    llm_result = analyze_frame_with_llm(
                        custom_desc,
                        {"name": location_name, "zone": zone},
                        full_timestamp
                    )
                    st.session_state.current_analysis = {
                        "frame": test_frame,
                        "llm_analysis": llm_result,
                        "objects": llm_result.get("objects", []),
                        "alerts": llm_result.get("alerts", []),
                        "analysis_text": llm_result.get("analysis", ""),
                        "threat_level": llm_result.get("threat_level", "UNKNOWN"),
                        "used_llm": True
                    }
            else:
                # Fallback to keyword matching
                st.session_state.current_analysis = {
                    "frame": test_frame,
                    "objects": extract_objects(custom_desc),
                    "alerts": check_alerts(test_frame),
                    "used_llm": False
                }

    with col2:
        st.subheader("Output: Analysis Results")

        if "current_analysis" in st.session_state:
            analysis = st.session_state.current_analysis

            # Show if LLM was used
            if analysis.get("used_llm"):
                st.success("🤖 **AI Agent Analysis** (Powered by LLM)")

                # Show threat level
                threat_level = analysis.get("threat_level", "UNKNOWN")
                threat_colors = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢", "NONE": "⚪"}
                st.markdown(f"**Threat Level:** {threat_colors.get(threat_level, '❓')} **{threat_level}**")

                # Show AI analysis text
                if analysis.get("analysis_text"):
                    st.markdown(f"""
                    <div class="frame-box">
                        <strong>🧠 AI Analysis:</strong><br>
                        {analysis['analysis_text']}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("⚡ Fallback Analysis (keyword-based)")

            # Show detected objects
            st.write("**🎯 Detected Objects:**")
            if analysis["objects"]:
                for obj in analysis["objects"]:
                    # Handle both LLM format and fallback format
                    obj_type = obj.get('type', 'unknown')
                    if 'description' in obj:
                        # LLM format
                        detail = f" - {obj['description']}"
                    elif obj_type == 'vehicle':
                        detail = f" - {obj.get('subtype', '')} ({obj.get('color', '')})"
                    elif obj_type == 'person':
                        attrs = obj.get('attributes', 'detected')
                        detail = f" - {attrs}" if attrs else ""
                    else:
                        detail = f" - {obj.get('subtype', '')}" if obj.get('subtype') else ""

                    st.markdown(f"""
                    <div class="detection-box">
                        <strong>{obj_type.title()}</strong>{detail}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No objects detected")

            # Show alerts
            st.write("**⚠️ Triggered Alerts:**")
            if analysis["alerts"]:
                for alert in analysis["alerts"]:
                    priority = alert.get('priority', 'MEDIUM')
                    priority_class = f"alert-{priority.lower()}"
                    rule_id = alert.get('rule_id', 'N/A')
                    name = alert.get('name', 'Alert')
                    # Handle both LLM format (reason) and fallback format (description)
                    desc = alert.get('reason', alert.get('description', ''))
                    st.markdown(f"""
                    <div class="{priority_class}">
                        <strong>[{priority}] {name}</strong><br>
                        Rule: {rule_id}<br>
                        {desc}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("No alerts triggered")

            # Show JSON output
            with st.expander("📄 Raw JSON Output"):
                st.json(analysis)

# ==================== Tab 4: Alerts ====================
with tab4:
    st.header("⚠️ Security Alerts")

    col1, col2, col3 = st.columns(3)

    high_alerts = [a for a in st.session_state.all_alerts if a["priority"] == "HIGH"]
    medium_alerts = [a for a in st.session_state.all_alerts if a["priority"] == "MEDIUM"]
    low_alerts = [a for a in st.session_state.all_alerts if a["priority"] == "LOW"]

    with col1:
        st.metric("🔴 HIGH", len(high_alerts))
    with col2:
        st.metric("🟠 MEDIUM", len(medium_alerts))
    with col3:
        st.metric("🔵 LOW", len(low_alerts))

    st.divider()

    # Filter
    priority_filter = st.multiselect("Filter by Priority", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM", "LOW"])

    filtered_alerts = [a for a in st.session_state.all_alerts if a["priority"] in priority_filter]

    if filtered_alerts:
        for alert in filtered_alerts:
            priority_class = f"alert-{alert['priority'].lower()}"
            st.markdown(f"""
            <div class="{priority_class}">
                <strong>[{alert['priority']}] {alert['name']}</strong> | Rule: {alert['rule_id']}<br>
                {alert['description']}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No alerts to display. Run the demo first!")

# ==================== Tab 5: Query Database ====================
with tab5:
    st.header("🔎 Query Frame Database")
    st.write("Search through indexed frames using natural language queries.")

    # LLM-powered query section
    if st.session_state.llm_available and st.session_state.qa_system:
        st.subheader("🤖 AI-Powered Query")
        ai_query = st.text_input("Ask the AI about surveillance data", placeholder="e.g., 'What vehicles were detected?' or 'Any security alerts?'", key="ai_query")

        if st.button("🧠 Ask AI", type="primary"):
            if ai_query:
                with st.spinner("AI is analyzing..."):
                    try:
                        answer = st.session_state.qa_system.answer(ai_query)
                        st.markdown(f"""
                        <div class="frame-box">
                            <strong>🤖 AI Response:</strong><br>
                            {answer.replace(chr(10), '<br>')}
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"AI query failed: {e}")
            else:
                st.warning("Please enter a question")

        st.divider()

    # Simple text search
    st.subheader("🔍 Text Search")
    query = st.text_input("Enter your query", placeholder="e.g., 'show all truck events' or 'person near warehouse'", key="text_query")

    col1, col2 = st.columns(2)
    with col1:
        search_button = st.button("🔍 Search", type="primary", key="search_btn")
    with col2:
        show_all = st.button("📋 Show All Frames")

    if search_button and query:
        results = search_frames(query, st.session_state.frame_database)

        st.subheader(f"Results for: '{query}'")
        st.write(f"Found {len(results)} matching frames")

        if results:
            for frame in results:
                st.markdown(f"""
                <div class="frame-box">
                    <strong>Frame {frame['frame_id']}</strong> | {frame['timestamp']}<br>
                    📍 {frame['location']['name']}<br>
                    📝 {frame['description']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No matching frames found")

    if show_all:
        st.subheader("All Indexed Frames")
        if st.session_state.frame_database:
            for frame in st.session_state.frame_database:
                st.markdown(f"""
                <div class="frame-box">
                    <strong>Frame {frame['frame_id']}</strong> | {frame['timestamp']}<br>
                    📍 {frame['location']['name']} ({frame['location']['zone']})<br>
                    📝 {frame['description']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No frames in database. Run the demo first!")

    # Example queries
    st.divider()
    st.subheader("💡 Example Queries")
    example_queries = [
        "truck",
        "person",
        "warehouse",
        "gate",
        "loitering"
    ]

    cols = st.columns(len(example_queries))
    for i, eq in enumerate(example_queries):
        if cols[i].button(eq, key=f"eq_{i}"):
            results = search_frames(eq, st.session_state.frame_database)
            st.write(f"**Results for '{eq}':** {len(results)} frames found")
            for frame in results:
                st.write(f"- Frame {frame['frame_id']}: {frame['description'][:60]}...")

# ==================== Tab 6: Summary ====================
with tab6:
    st.header("📝 Video Summary")
    st.write("AI-generated summary of surveillance activity.")

    # LLM-powered summary
    if st.session_state.llm_available and st.session_state.summarizer:
        if st.button("🤖 Generate AI Summary", type="primary"):
            with st.spinner("AI is generating summary..."):
                try:
                    ai_summary = st.session_state.summarizer.summarize_session()
                    st.subheader("🤖 AI-Generated Summary")
                    st.markdown(f"""
                    <div class="frame-box">
                        {ai_summary.replace(chr(10), '<br>')}
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"AI summary failed: {e}")
                    st.info("Falling back to basic summary...")

        st.divider()

    if st.button("📊 Generate Basic Summary", type="secondary" if st.session_state.llm_available else "primary"):
        if st.session_state.frame_database:
            # Generate summary
            total_frames = len(st.session_state.frame_database)
            total_alerts = len(st.session_state.all_alerts)
            high_priority = len([a for a in st.session_state.all_alerts if a["priority"] == "HIGH"])

            vehicles = sum(1 for f in st.session_state.frame_database if any(o["type"] == "vehicle" for o in f.get("objects", [])))
            persons = sum(1 for f in st.session_state.frame_database if any(o["type"] == "person" for o in f.get("objects", [])))

            st.subheader("📈 Activity Summary")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Frames", total_frames)
            col2.metric("Vehicles Detected", vehicles)
            col3.metric("Persons Detected", persons)
            col4.metric("Alerts Generated", total_alerts)

            st.divider()

            # Text summary
            st.subheader("📄 Narrative Summary")

            summary_text = f"""
            **Surveillance Period Summary**

            During the monitoring period, the drone security system processed **{total_frames} video frames**
            across multiple locations on the property.

            **Key Findings:**
            - **{vehicles} vehicle(s)** were detected, including entries through the main gate
            - **{persons} person(s)** were identified in various zones
            - **{total_alerts} security alert(s)** were generated, with **{high_priority} high-priority** incidents

            **Notable Events:**
            """

            st.markdown(summary_text)

            for alert in st.session_state.all_alerts:
                st.write(f"- [{alert['priority']}] {alert['description']}")

            st.divider()

            # Recommendations
            st.subheader("💡 Recommendations")
            if high_priority > 0:
                st.warning("⚠️ High-priority alerts detected. Recommend immediate review of flagged incidents.")
            else:
                st.success("✅ No critical security concerns identified during this period.")
        else:
            st.info("Run the demo first to generate a summary!")

    # Q&A Section
    st.divider()
    st.subheader("❓ Follow-up Questions")

    question = st.text_input("Ask a question about the surveillance data", placeholder="e.g., 'What objects were detected?'", key="summary_question")

    if st.button("Ask", key="summary_ask") and question:
        # Use LLM if available
        if st.session_state.llm_available and st.session_state.qa_system:
            with st.spinner("AI is thinking..."):
                try:
                    answer = st.session_state.qa_system.answer(question)
                    st.markdown(f"""
                    <div class="frame-box">
                        <strong>🤖 AI Answer:</strong><br>
                        {answer.replace(chr(10), '<br>')}
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"AI query failed: {e}")
        else:
            # Fallback to basic keyword matching
            question_lower = question.lower()

            if "object" in question_lower or "detect" in question_lower:
                st.write("**Answer:** The following objects were detected during surveillance:")
                objects_found = set()
                for frame in st.session_state.frame_database:
                    for obj in frame.get("objects", []):
                        if obj["type"] == "vehicle":
                            objects_found.add(f"Vehicle ({obj.get('subtype', 'unknown')} - {obj.get('color', 'unknown')})")
                        else:
                            objects_found.add(obj["type"].title())

                for obj in objects_found:
                    st.write(f"- {obj}")

            elif "alert" in question_lower:
                st.write(f"**Answer:** {len(st.session_state.all_alerts)} alerts were generated.")
                for alert in st.session_state.all_alerts:
                    st.write(f"- [{alert['priority']}] {alert['name']}: {alert['description']}")

            elif "truck" in question_lower:
                truck_frames = [f for f in st.session_state.frame_database if "truck" in f["description"].lower()]
                st.write(f"**Answer:** {len(truck_frames)} frame(s) contain truck detections.")

            else:
                st.write("**Answer:** I can answer questions about detected objects, alerts, vehicles, and persons. Try asking about specific items!")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #555555 !important; font-size: 0.9rem; background-color: rgba(240,242,246,0.8); padding: 1rem; border-radius: 8px;">
    <span style="color: #1E88E5;">🛡️ Drone Security Analyst Agent</span> | Built with LangChain + LangGraph + Groq<br>
    <span style="color: #333333;">FlytBase AI Engineer Assignment</span>
</div>
""", unsafe_allow_html=True)
