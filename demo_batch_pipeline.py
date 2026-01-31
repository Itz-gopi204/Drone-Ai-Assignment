#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: Batch Vision Pipeline - Cost Effective Approach

This is the RECOMMENDED strategy for video processing:
1. Video -> Extract frames with OpenCV
2. Each frame -> Local VLM (BLIP-2) or simulated -> Text description (FREE)
3. ALL descriptions -> ONE LLM call (Groq - FREE) -> Complete analysis

Cost: $0.00 (vs $1.00+ for GPT-4 Vision per frame)

Usage:
    python demo_batch_pipeline.py                    # Run demo
    python demo_batch_pipeline.py --video path.mp4  # Process actual video
"""

import os
import sys
import argparse
from datetime import datetime, timedelta

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.batch_vision_pipeline import (
    BatchVisionPipeline, BatchPipelineConfig, FrameData,
    get_batch_pipeline_status
)


def print_header(text: str):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def demo_batch():
    """Run demo with batch pipeline."""
    print_header("BATCH VISION PIPELINE DEMO")

    # Show status
    status = get_batch_pipeline_status()
    print(f"\nPipeline Status:")
    print(f"  - BLIP-2 Available: {status['blip2_available']}")
    print(f"  - Groq LLM Available: {status['groq_available']}")
    print(f"  - Cost: {status['cost']}")
    print(f"  - API calls per video: {status['api_calls_per_video']}")

    print("\n" + "-" * 60)
    print("COST COMPARISON:")
    print("-" * 60)
    print("  GPT-4 Vision (per frame):  50 frames x $0.02 = $1.00")
    print("  Batch Pipeline:            BLIP-2 (free) + 1 Groq call = $0.00")
    print("-" * 60)

    # Create pipeline
    config = BatchPipelineConfig(vlm_provider="simulated")
    pipeline = BatchVisionPipeline(config=config)

    # Simulate frames
    print("\n[Step 1] Generating frame descriptions with local VLM...")
    frames = []
    for i in range(10):
        description = pipeline.captioner.caption_frame(None)
        frame = FrameData(
            frame_id=i + 1,
            timestamp=datetime.now() + timedelta(seconds=i * 5),
            location=config.location_zones[i % len(config.location_zones)],
            telemetry={"drone_id": "DRONE-001", "altitude": 50},
            description=description
        )
        frames.append(frame)
        print(f"  Frame {i+1}: {description[:50]}...")

    # Analyze batch
    print(f"\n[Step 2] Sending ALL {len(frames)} frames to LLM in ONE call...")
    result = pipeline.analyze_batch(frames)

    # Show results
    print_header("ANALYSIS RESULTS")

    print(f"\nFrames Analyzed: {len(result.frames)}")
    print(f"Total Alerts: {len(result.alerts)}")
    print(f"Threat Assessment: {result.threat_assessment}")
    print(f"Processing Time: {result.processing_time_ms:.0f}ms")

    print(f"\nSummary: {result.summary}")

    if result.alerts:
        print("\nSecurity Alerts:")
        for alert in result.alerts[:5]:
            print(f"  [{alert['priority']}] Frame {alert['frame_id']}: {alert['name']} - {alert['reason'][:40]}...")

    if result.statistics.get("patterns"):
        print("\nPatterns Detected:")
        for pattern in result.statistics["patterns"]:
            print(f"  - {pattern}")

    print("\n" + "=" * 60)
    print("  KEY BENEFIT: Only ONE LLM API call for the entire video!")
    print("=" * 60)


def demo_video(video_path: str):
    """Process actual video with batch pipeline."""
    print_header(f"PROCESSING VIDEO: {video_path}")

    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return

    status = get_batch_pipeline_status()
    print(f"\nUsing: {status['recommended_vlm']} VLM + Groq LLM")
    print(f"Cost: {status['cost']}")

    config = BatchPipelineConfig(
        frame_interval_seconds=5,
        max_frames=30,
        vlm_provider=status['recommended_vlm']
    )

    pipeline = BatchVisionPipeline(config=config)
    result = pipeline.process_video(video_path)

    print_header("RESULTS")
    print(f"Frames: {len(result.frames)}")
    print(f"Alerts: {len(result.alerts)}")
    print(f"Threat: {result.threat_assessment}")
    print(f"\nSummary: {result.summary}")


def main():
    parser = argparse.ArgumentParser(
        description="Demo: Batch Vision Pipeline - Cost Effective Video Analysis"
    )
    parser.add_argument("--video", type=str, help="Path to video file")

    args = parser.parse_args()

    print("""
    +------------------------------------------------------------------+
    |          BATCH VISION PIPELINE - RECOMMENDED APPROACH            |
    +------------------------------------------------------------------+
    |                                                                  |
    |  The SMART way to process surveillance video:                    |
    |                                                                  |
    |  1. Extract frames from video                                    |
    |  2. Generate text descriptions (BLIP-2 local - FREE)             |
    |  3. Send ALL descriptions to LLM in ONE call (Groq - FREE)       |
    |  4. Get complete analysis with alerts                            |
    |                                                                  |
    |  COST: $0.00 per video (vs $1.00+ with GPT-4 Vision per frame)   |
    |                                                                  |
    +------------------------------------------------------------------+
    """)

    if args.video:
        demo_video(args.video)
    else:
        demo_batch()

    print("\nTo process your own video:")
    print("  python demo_batch_pipeline.py --video your_video.mp4")


if __name__ == "__main__":
    main()
