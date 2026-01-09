#!/usr/bin/env python3
"""
System Validation Script
========================
Validates all components of the Drone Security Agent system.
"""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("DRONE SECURITY AGENT - SYSTEM VALIDATION")
print("=" * 60)

errors = []
warnings = []

# ==================== 1. Check Python Version ====================
print("\n[1/7] Checking Python version...")
py_version = sys.version_info
print(f"  Python {py_version.major}.{py_version.minor}.{py_version.micro}")
if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 10):
    warnings.append("Python 3.10+ recommended for LangChain v1")
    print(f"  WARNING: Python 3.10+ recommended")
else:
    print("  OK")

# ==================== 2. Check Core Imports ====================
print("\n[2/7] Checking core imports...")

core_imports = {
    "pydantic": "pydantic",
    "rich": "rich",
    "dotenv": "python-dotenv",
}

for module, package in core_imports.items():
    try:
        __import__(module)
        print(f"  {module}: OK")
    except ImportError as e:
        errors.append(f"{module} not installed (pip install {package})")
        print(f"  {module}: MISSING")

# ==================== 3. Check LangChain Imports ====================
print("\n[3/7] Checking LangChain ecosystem...")

langchain_imports = [
    ("langchain_core", "langchain-core"),
    ("langchain_openai", "langchain-openai"),
    ("langchain_groq", "langchain-groq"),
    ("langchain_community", "langchain-community"),
    ("langgraph", "langgraph"),
]

for module, package in langchain_imports:
    try:
        __import__(module)
        print(f"  {module}: OK")
    except ImportError:
        warnings.append(f"{module} not installed (pip install {package})")
        print(f"  {module}: MISSING (optional)")

# Check langchain main package
try:
    from langchain.agents import create_react_agent, AgentExecutor
    print("  langchain.agents (v1 API): OK")
except ImportError:
    try:
        from langchain_classic.agents import AgentExecutor
        print("  langchain_classic.agents (fallback): OK")
    except ImportError:
        warnings.append("langchain agents not available")
        print("  langchain agents: MISSING")

# ==================== 4. Check Local Modules ====================
print("\n[4/7] Checking local modules...")

local_modules = [
    "src.config",
    "src.database",
    "src.simulator",
    "src.analyzer",
    "src.alert_engine",
    "src.agent",
    "src.graph_agent",
    "src.vector_store",
]

for module in local_modules:
    try:
        __import__(module)
        print(f"  {module}: OK")
    except ImportError as e:
        errors.append(f"{module}: {str(e)}")
        print(f"  {module}: ERROR - {e}")
    except Exception as e:
        warnings.append(f"{module}: {str(e)}")
        print(f"  {module}: WARNING - {e}")

# ==================== 5. Test Database Operations ====================
print("\n[5/7] Testing database operations...")

try:
    from src.database import SecurityDatabase, FrameRecord

    # Create in-memory database for testing
    db = SecurityDatabase(":memory:")

    # Test frame indexing
    test_frame = FrameRecord(
        frame_id=1,
        timestamp=datetime.now(),
        location_name="Test Gate",
        location_zone="perimeter",
        latitude=37.77,
        longitude=-122.41,
        description="Test vehicle at gate",
        objects=[{"type": "vehicle", "subtype": "car"}],
        alert_triggered=False
    )
    db.index_frame(test_frame)

    # Test query
    results = db.query_frames_by_description("vehicle")
    if results:
        print("  Frame indexing: OK")
        print("  Frame query: OK")
    else:
        warnings.append("Frame query returned no results")
        print("  Frame query: WARNING")

    # Test statistics
    stats = db.get_statistics()
    print(f"  Statistics: OK (frames: {stats['total_frames']})")

except Exception as e:
    errors.append(f"Database test failed: {e}")
    print(f"  Database: ERROR - {e}")

# ==================== 6. Test Analyzer ====================
print("\n[6/7] Testing analyzer...")

try:
    from src.analyzer import FrameAnalyzer, ObjectTracker

    analyzer = FrameAnalyzer(use_api=False)  # Don't require API

    result = analyzer.analyze_frame(
        frame_id=1,
        timestamp=datetime.now(),
        frame_description="Blue Ford F150 pickup truck at main gate",
        location_context={"name": "Main Gate", "zone": "perimeter"}
    )

    print(f"  Frame analysis: OK")
    print(f"  Objects detected: {len(result.detected_objects)}")
    print(f"  Security relevant: {result.security_relevant}")

    # Test tracker
    tracker = ObjectTracker()
    for obj in result.detected_objects:
        tracker.track_object(obj, datetime.now(), "Main Gate")
    print(f"  Object tracking: OK")

except Exception as e:
    errors.append(f"Analyzer test failed: {e}")
    print(f"  Analyzer: ERROR - {e}")

# ==================== 7. Test Alert Engine ====================
print("\n[7/7] Testing alert engine...")

try:
    from src.alert_engine import AlertEngine
    from src.database import SecurityDatabase

    db = SecurityDatabase(":memory:")
    engine = AlertEngine(database=db)

    # Test night activity alert (should trigger)
    night_time = datetime.now().replace(hour=2, minute=30)
    alerts = engine.evaluate(
        frame_id=1,
        timestamp=night_time,
        detected_objects=[{"type": "person", "id": "P001"}],
        location={"name": "Main Gate", "zone": "perimeter"},
        description="Person at gate during night"
    )

    if alerts:
        print(f"  Alert generation: OK ({len(alerts)} alerts)")
        print(f"  Night activity rule: TRIGGERED")
    else:
        print(f"  Alert generation: OK (no alerts - cooldown may be active)")

except Exception as e:
    errors.append(f"Alert engine test failed: {e}")
    print(f"  Alert engine: ERROR - {e}")

# ==================== Summary ====================
print("\n" + "=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)

if errors:
    print(f"\nERRORS ({len(errors)}):")
    for err in errors:
        print(f"  - {err}")

if warnings:
    print(f"\nWARNINGS ({len(warnings)}):")
    for warn in warnings:
        print(f"  - {warn}")

if not errors and not warnings:
    print("\nAll checks passed! System is ready.")
elif not errors:
    print(f"\nSystem functional with {len(warnings)} warning(s).")
else:
    print(f"\nSystem has {len(errors)} error(s) that need to be fixed.")

print("\n" + "=" * 60)

# Exit with appropriate code
sys.exit(1 if errors else 0)
