"""
Configuration settings for the Drone Security Analyst Agent.
Supports both local (.env) and Streamlit Cloud (st.secrets) deployment.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_config_value(key: str, default: str = "") -> str:
    """
    Get configuration value from Streamlit secrets or environment variables.
    Priority: Streamlit secrets > Environment variables > Default
    """
    # Try Streamlit secrets first (for cloud deployment)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    # Fall back to environment variables (for local development)
    return os.getenv(key, default)


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
DB_PATH = DATA_DIR / "security_agent.db"
CHROMA_DB_PATH = DATA_DIR / "chromadb"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
CHROMA_DB_PATH.mkdir(exist_ok=True)

# LLM Provider Configuration
# Supported providers: "groq", "openai"
LLM_PROVIDER = get_config_value("LLM_PROVIDER", "groq")  # Default to Groq (free tier available)

# Groq API Configuration (Free tier available - https://console.groq.com)
GROQ_API_KEY = get_config_value("GROQ_API_KEY", "")
GROQ_MODEL_NAME = get_config_value("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")  # Fast & powerful

# OpenAI API Configuration (fallback)
OPENAI_API_KEY = get_config_value("OPENAI_API_KEY", "")
OPENAI_MODEL_NAME = get_config_value("OPENAI_MODEL_NAME", "gpt-4o-mini")

# Active model based on provider
MODEL_NAME = GROQ_MODEL_NAME if LLM_PROVIDER == "groq" else OPENAI_MODEL_NAME

# Available Groq Models (for reference):
# - llama-3.3-70b-versatile (recommended - fast, powerful)
# - llama-3.1-70b-versatile (good reasoning)
# - llama-3.1-8b-instant (fastest, lighter tasks)
# - mixtral-8x7b-32768 (good for complex analysis)
# - gemma2-9b-it (Google's model)

# ChromaDB / Vector Store Configuration
VECTOR_STORE_CONFIG = {
    "embedding_model": "all-MiniLM-L6-v2",  # Fast, good quality embeddings
    "collection_name": "security_frames",
    "similarity_threshold": 0.7,  # Minimum similarity score for matches
}

# Simulation Configuration
SIMULATION_CONFIG = {
    "drone_id": "DRONE-001",
    "property_name": "SecureProperty-Alpha",
    "patrol_locations": [
        {"name": "Main Gate", "zone": "perimeter", "lat": 37.7749, "lon": -122.4194},
        {"name": "Parking Lot", "zone": "parking", "lat": 37.7750, "lon": -122.4190},
        {"name": "Warehouse", "zone": "storage", "lat": 37.7748, "lon": -122.4185},
        {"name": "Loading Dock", "zone": "operations", "lat": 37.7747, "lon": -122.4180},
        {"name": "Back Fence", "zone": "perimeter", "lat": 37.7745, "lon": -122.4175},
        {"name": "Office Building", "zone": "main", "lat": 37.7751, "lon": -122.4195},
        {"name": "Garage", "zone": "vehicles", "lat": 37.7752, "lon": -122.4188},
    ],
    "frame_interval_seconds": 5,  # Time between simulated frames
}

# Alert Rules Configuration
ALERT_RULES = [
    {
        "id": "R001",
        "name": "Night Activity",
        "description": "Person detected during night hours (00:00-05:00)",
        "conditions": {
            "object_type": "person",
            "time_range": {"start": "00:00", "end": "05:00"}
        },
        "priority": "HIGH",
        "message_template": "Person detected at {location} during restricted hours ({time})"
    },
    {
        "id": "R002",
        "name": "Loitering Detection",
        "description": "Same person detected in same zone for extended period",
        "conditions": {
            "object_type": "person",
            "same_zone_duration_minutes": 5
        },
        "priority": "HIGH",
        "message_template": "Person loitering at {location} for extended period"
    },
    {
        "id": "R003",
        "name": "Perimeter Activity",
        "description": "Activity detected in perimeter zone",
        "conditions": {
            "zone": "perimeter",
            "object_types": ["person", "vehicle"]
        },
        "priority": "MEDIUM",
        "message_template": "Activity detected near perimeter at {location}"
    },
    {
        "id": "R004",
        "name": "Repeat Vehicle Entry",
        "description": "Same vehicle detected multiple times in 24 hours",
        "conditions": {
            "object_type": "vehicle",
            "repeat_count_24h": 2
        },
        "priority": "LOW",
        "message_template": "Vehicle ({details}) detected {count} times today"
    },
    {
        "id": "R005",
        "name": "Unknown Vehicle",
        "description": "Unrecognized vehicle in restricted area",
        "conditions": {
            "object_type": "vehicle",
            "zone": ["storage", "operations"],
            "unknown": True
        },
        "priority": "MEDIUM",
        "message_template": "Unknown vehicle detected at {location}"
    },
]

# Object Detection Categories
OBJECT_CATEGORIES = {
    "person": ["person", "human", "individual", "man", "woman", "worker", "intruder"],
    "vehicle": ["car", "truck", "van", "motorcycle", "bicycle", "pickup", "suv", "sedan"],
    "animal": ["dog", "cat", "bird", "wildlife"],
}

# Vehicle attributes for detection
VEHICLE_COLORS = ["red", "blue", "black", "white", "silver", "gray", "green", "yellow"]
VEHICLE_MAKES = ["Ford", "Toyota", "Honda", "Chevrolet", "BMW", "Mercedes", "Tesla"]
VEHICLE_TYPES = ["sedan", "SUV", "pickup truck", "van", "motorcycle"]

# Logging Configuration
LOG_CONFIG = {
    "level": "INFO",
    "format": "[%(asctime)s] %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S"
}

# Agent Configuration
AGENT_CONFIG = {
    "temperature": 0.3,  # Lower for more consistent analysis
    "max_tokens": 1000,
    "context_window_size": 10,  # Number of recent events to keep in context
}
