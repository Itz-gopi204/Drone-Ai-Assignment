#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: End-to-End Vision Pipeline for Drone Security Agent

This script demonstrates the BEST strategy for video processing:
1. Video -> Frame extraction with OpenCV
2. Frame + Context -> GPT-4 Vision -> Complete security analysis
3. Results stored in database

Usage:
    python demo_vision_pipeline.py                    # Run with simulated mode
    python demo_vision_pipeline.py --video path.mp4  # Process actual video
    python demo_vision_pipeline.py --image path.jpg  # Process single image
    python demo_vision_pipeline.py --provider direct # Use GPT-4 Vision (requires OPENAI_API_KEY)
"""

import os
import sys
import argparse
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.vision_pipeline import (
    DirectVisionPipeline, PipelineConfig, get_pipeline_status,
    process_video_with_vision, process_image_with_vision
)
from src.database import SecurityDatabase


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_result(result: dict):
    """Print analysis result in a formatted way."""
    print(f"\n📷 Frame {result.get('frame_id', 'N/A')}")
    print(f"   📍 Location: {result.get('location', {}).get('name', 'Unknown')}")
    print(f"   🕐 Timestamp: {result.get('timestamp', 'N/A')}")
    print(f"   📝 Description: {result.get('description', 'N/A')[:80]}...")
    print(f"   🎯 Objects: {len(result.get('objects', []))}")
    for obj in result.get('objects', [])[:3]:
        print(f"      - {obj.get('type', 'unknown')}: {obj.get('description', 'N/A')[:50]}")
    print(f"   ⚠️  Alerts: {len(result.get('alerts', []))}")
    for alert in result.get('alerts', []):
        print(f"      - [{alert.get('priority', 'MEDIUM')}] {alert.get('name', 'Alert')}: {alert.get('reason', '')[:40]}")
    print(f"   🎚️  Threat Level: {result.get('threat_level', 'UNKNOWN')}")


def demo_simulated():
    """Run demo with simulated VLM (no API needed)."""
    print_header("DEMO: Simulated Vision Pipeline")

    # Check status
    status = get_pipeline_status()
    print(f"\nPipeline Status:")
    print(f"  - OpenAI Available: {status['openai_available']}")
    print(f"  - OpenAI Key Configured: {status['openai_key_configured']}")
    print(f"  - Recommended Provider: {status['recommended_provider']}")

    # Create database
    db = SecurityDatabase(":memory:")

    # Create pipeline in simulated mode
    config = PipelineConfig(
        provider="simulated",
        frame_interval_seconds=5,
        max_frames=5,
        store_to_database=True
    )

    pipeline = DirectVisionPipeline(config=config, database=db)

    print("\n🎬 Processing simulated frames...")

    # Create simulated frames using numpy
    import numpy as np

    for i in range(5):
        # Create a blank test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Vary location
        locations = [
            {"name": "Main Gate", "zone": "perimeter"},
            {"name": "Parking Lot", "zone": "parking"},
            {"name": "Warehouse", "zone": "storage"},
            {"name": "Loading Dock", "zone": "operations"},
            {"name": "Back Fence", "zone": "perimeter"},
        ]

        # Analyze frame
        result = pipeline.analyze_frame(
            frame_data=test_image,
            location=locations[i % len(locations)],
            timestamp=datetime.now(),
            frame_id=i + 1
        )

        print_result(result.to_dict())

    # Print summary
    print_header("SUMMARY")
    stats = db.get_statistics()
    print(f"📊 Total Frames: {stats['total_frames']}")
    print(f"⚠️  Total Alerts: {stats['total_alerts']}")
    print(f"🔴 High Priority: {stats['high_priority_alerts']}")

    # Get all alerts
    alerts = db.get_alerts(limit=10)
    if alerts:
        print("\n📋 All Alerts:")
        for alert in alerts:
            print(f"   [{alert.priority}] Frame {alert.frame_id}: {alert.description[:50]}...")


def demo_video(video_path: str, provider: str = "simulated"):
    """Process an actual video file."""
    print_header(f"DEMO: Video Processing ({provider})")

    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return

    # Check status
    status = get_pipeline_status()
    if provider == "direct" and not status["direct_vision_available"]:
        print("⚠️  Direct Vision not available (no OpenAI API key)")
        print("   Falling back to simulated mode")
        provider = "simulated"

    # Create database
    db = SecurityDatabase()

    print(f"\n📹 Processing video: {video_path}")
    print(f"🤖 Provider: {provider}")

    def progress_callback(current, total, result):
        print(f"\r   Processing frame {current}/{total}: {result.description[:40]}...", end="")

    # Process video
    results = process_video_with_vision(
        video_path=video_path,
        provider=provider,
        frame_interval=5,
        max_frames=20,
        progress_callback=progress_callback,
        database=db
    )

    print("\n")

    # Print results
    for result in results:
        print_result(result)

    # Print summary
    print_header("SUMMARY")
    print(f"📊 Total Frames Processed: {len(results)}")

    total_alerts = sum(len(r.get('alerts', [])) for r in results)
    high_alerts = sum(1 for r in results for a in r.get('alerts', []) if a.get('priority') == 'HIGH')

    print(f"⚠️  Total Alerts: {total_alerts}")
    print(f"🔴 High Priority: {high_alerts}")

    threat_levels = [r.get('threat_level', 'NONE') for r in results]
    if 'CRITICAL' in threat_levels:
        print("🚨 CRITICAL THREATS DETECTED!")
    elif 'HIGH' in threat_levels:
        print("⚠️  HIGH threat level detected")


def demo_image(image_path: str, provider: str = "simulated"):
    """Process a single image."""
    print_header(f"DEMO: Image Analysis ({provider})")

    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return

    # Check status
    status = get_pipeline_status()
    if provider == "direct" and not status["direct_vision_available"]:
        print("⚠️  Direct Vision not available (no OpenAI API key)")
        print("   Falling back to simulated mode")
        provider = "simulated"

    # Create database
    db = SecurityDatabase()

    print(f"\n🖼️  Processing image: {image_path}")
    print(f"🤖 Provider: {provider}")

    # Process image
    result = process_image_with_vision(
        image_path_or_data=image_path,
        provider=provider,
        location={"name": "Main Gate", "zone": "perimeter"},
        timestamp=datetime.now(),
        database=db
    )

    # Print result
    print_result(result)

    print_header("COMPLETE ANALYSIS")
    print(f"📝 Full Analysis:\n{result.get('analysis', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(
        description="Demo: End-to-End Vision Pipeline for Drone Security"
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Path to video file to process"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to image file to process"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="simulated",
        choices=["direct", "simulated"],
        help="VLM provider (direct uses GPT-4 Vision, simulated for demo)"
    )

    args = parser.parse_args()

    print_header("DRONE SECURITY VISION PIPELINE DEMO")
    print("""
This demo shows the BEST strategy for video/image processing:

+-------------+     +------------------+     +-------------------------+
| Video/Image | --> | OpenCV Extract   | --> | Frame + Context         |
| Input       |     | Frames           |     | - image_data            |
+-------------+     +------------------+     | - location zone         |
                                             | - timestamp             |
                                             +-----------+-------------+
                                                         |
                                                         v
                                             +-------------------------+
                                             | GPT-4 Vision (Direct)   |
                                             | Single API call returns:|
                                             | - objects detected      |
                                             | - security alerts       |
                                             | - threat level          |
                                             +-----------+-------------+
                                                         |
                                                         v
                                             +-------------------------+
                                             | Database Storage        |
                                             | - SQLite (structured)   |
                                             | - ChromaDB (vectors)    |
                                             +-------------------------+
    """)

    if args.video:
        demo_video(args.video, args.provider)
    elif args.image:
        demo_image(args.image, args.provider)
    else:
        demo_simulated()

    print("\n✅ Demo complete!")
    print("\nTo run with actual video:")
    print("  python demo_vision_pipeline.py --video your_video.mp4 --provider direct")
    print("\nTo run with image:")
    print("  python demo_vision_pipeline.py --image your_image.jpg --provider direct")


if __name__ == "__main__":
    main()
