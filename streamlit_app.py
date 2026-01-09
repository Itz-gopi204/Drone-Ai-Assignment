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

# Page configuration
st.set_page_config(
    page_title="Drone Security Analyst Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .frame-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .alert-low {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .detection-box {
        background-color: #e8f5e9;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.25rem 0;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
]

# ==================== Helper Functions ====================

def extract_objects(description: str) -> list:
    """Extract objects from frame description."""
    objects = []
    description_lower = description.lower()

    # Vehicle detection
    vehicle_keywords = ["truck", "car", "van", "motorcycle", "pickup", "suv", "sedan", "camry", "f150"]
    colors = ["blue", "red", "black", "white", "silver", "gray", "green"]

    for keyword in vehicle_keywords:
        if keyword in description_lower:
            color = next((c for c in colors if c in description_lower), "unknown")
            objects.append({"type": "vehicle", "subtype": keyword, "color": color})
            break

    # Person detection
    if "person" in description_lower or "man" in description_lower or "woman" in description_lower:
        objects.append({"type": "person", "attributes": "detected"})

    return objects

def check_alerts(frame: dict) -> list:
    """Check alert rules against frame data."""
    alerts = []
    timestamp = datetime.fromisoformat(frame["timestamp"])
    description = frame["description"].lower()
    zone = frame["location"]["zone"]

    # R001: Night Activity
    if "person" in description and (timestamp.hour >= 0 and timestamp.hour < 5):
        alerts.append({
            "rule_id": "R001",
            "name": "Night Activity",
            "priority": "HIGH",
            "description": f"Person detected at {frame['location']['name']} during restricted hours ({timestamp.strftime('%H:%M')})"
        })

    # R002: Loitering
    if "loitering" in description or "10 minutes" in description:
        alerts.append({
            "rule_id": "R002",
            "name": "Loitering Detection",
            "priority": "HIGH",
            "description": f"Person loitering detected at {frame['location']['name']}"
        })

    # R003: Perimeter Activity
    if zone == "perimeter" and ("person" in description or "vehicle" in description or "truck" in description):
        alerts.append({
            "rule_id": "R003",
            "name": "Perimeter Activity",
            "priority": "MEDIUM",
            "description": f"Activity detected near perimeter at {frame['location']['name']}"
        })

    return alerts

def search_frames(query: str, frames: list) -> list:
    """Search frames by query."""
    query_lower = query.lower()
    results = []

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

# ==================== Main UI ====================

# Header
st.markdown('<p class="main-header">🛡️ Drone Security Analyst Agent</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Property Security Monitoring System</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("LLM Provider")
    llm_provider = st.selectbox("Select Provider", ["Groq (Free)", "OpenAI"])

    if llm_provider == "Groq (Free)":
        st.info("Using: llama-3.3-70b-versatile")
    else:
        st.info("Using: gpt-4o-mini")

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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎬 Live Demo",
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
                status_text.text(f"Processing Frame {frame['frame_id']}...")
                progress_bar.progress((i + 1) / len(SAMPLE_FRAMES))

                # Simulate processing delay
                time.sleep(0.8)

                # Process frame
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

            status_text.text("✅ Demo complete!")
            st.success(f"Processed {len(SAMPLE_FRAMES)} frames, generated {len(st.session_state.all_alerts)} alerts")
            st.rerun()

    with col2:
        if st.session_state.processed_frames:
            st.subheader("Recent Detections")
            for frame in st.session_state.processed_frames[-3:]:
                with st.container():
                    st.markdown(f"""
                    <div class="frame-box">
                        <strong>Frame {frame['frame_id']}</strong> | {frame['timestamp']}<br>
                        📍 {frame['location']['name']} ({frame['location']['zone']})<br>
                        📝 {frame['description']}<br>
                        🎯 Objects: {', '.join([f"{o['type']}" for o in frame['objects']]) or 'None detected'}
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

# ==================== Tab 2: Frame Processing ====================
with tab2:
    st.header("🔍 Frame-by-Frame Processing")
    st.write("See how the agent analyzes individual frames.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input: Simulated Frame")

        # Custom frame input
        custom_desc = st.text_area(
            "Frame Description",
            value="Blue Ford F150 pickup truck entering through main gate",
            height=100
        )

        col_a, col_b = st.columns(2)
        with col_a:
            location_name = st.selectbox("Location", ["Main Gate", "Warehouse", "Parking Lot", "Back Fence", "Office Building"])
        with col_b:
            zone = st.selectbox("Zone", ["perimeter", "storage", "parking", "main"])

        hour = st.slider("Hour of Day", 0, 23, 12)

        if st.button("🔬 Analyze Frame", type="primary"):
            # Create frame
            test_frame = {
                "frame_id": random.randint(100, 999),
                "timestamp": f"2024-01-15T{hour:02d}:30:00",
                "description": custom_desc,
                "location": {"name": location_name, "zone": zone},
                "telemetry": {"altitude": 50, "battery": 90, "drone_id": "DRONE-001"}
            }

            st.session_state.current_analysis = {
                "frame": test_frame,
                "objects": extract_objects(custom_desc),
                "alerts": check_alerts(test_frame)
            }

    with col2:
        st.subheader("Output: Analysis Results")

        if "current_analysis" in st.session_state:
            analysis = st.session_state.current_analysis

            # Show detected objects
            st.write("**🎯 Detected Objects:**")
            if analysis["objects"]:
                for obj in analysis["objects"]:
                    st.markdown(f"""
                    <div class="detection-box">
                        <strong>{obj['type'].title()}</strong>
                        {f" - {obj.get('subtype', '')} ({obj.get('color', '')})" if obj['type'] == 'vehicle' else ''}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No objects detected")

            # Show alerts
            st.write("**⚠️ Triggered Alerts:**")
            if analysis["alerts"]:
                for alert in analysis["alerts"]:
                    priority_class = f"alert-{alert['priority'].lower()}"
                    st.markdown(f"""
                    <div class="{priority_class}">
                        <strong>[{alert['priority']}] {alert['name']}</strong><br>
                        Rule: {alert['rule_id']}<br>
                        {alert['description']}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("No alerts triggered")

            # Show JSON output
            with st.expander("📄 Raw JSON Output"):
                st.json(analysis)

# ==================== Tab 3: Alerts ====================
with tab3:
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

# ==================== Tab 4: Query Database ====================
with tab4:
    st.header("🔎 Query Frame Database")
    st.write("Search through indexed frames using natural language queries.")

    query = st.text_input("Enter your query", placeholder="e.g., 'show all truck events' or 'person near warehouse'")

    col1, col2 = st.columns(2)
    with col1:
        search_button = st.button("🔍 Search", type="primary")
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

# ==================== Tab 5: Summary ====================
with tab5:
    st.header("📝 Video Summary")
    st.write("AI-generated summary of surveillance activity.")

    if st.button("📊 Generate Summary", type="primary"):
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

    question = st.text_input("Ask a question about the surveillance data", placeholder="e.g., 'What objects were detected?'")

    if st.button("Ask") and question:
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
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    🛡️ Drone Security Analyst Agent | Built with LangChain + LangGraph + Groq<br>
    FlytBase AI Engineer Assignment
</div>
""", unsafe_allow_html=True)
