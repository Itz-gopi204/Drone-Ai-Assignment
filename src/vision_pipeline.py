"""
Unified Vision Pipeline for Drone Security Agent.

This is the BEST STRATEGY implementation that:
1. Takes video input → Extracts frames with OpenCV
2. For each frame: Sends image + context → GPT-4 Vision → Complete security analysis
3. Stores results in database (SQLite + ChromaDB)
4. Generates alerts and tracks objects

Pipeline Flow:
┌─────────────┐     ┌──────────────────┐     ┌─────────────────────────┐
│ Video File  │ ──► │ OpenCV Extract   │ ──► │ Frame + Metadata        │
│ (.mp4/.avi) │     │ Frames           │     │ - image_data            │
└─────────────┘     └──────────────────┘     │ - timestamp             │
                                             │ - location zone         │
                                             │ - telemetry             │
                                             └───────────┬─────────────┘
                                                         │
                                                         ▼
                                             ┌─────────────────────────┐
                                             │ GPT-4 Vision (Direct)   │
                                             │ Single API call returns:│
                                             │ - description           │
                                             │ - objects detected      │
                                             │ - security alerts       │
                                             │ - threat_level          │
                                             └───────────┬─────────────┘
                                                         │
                                                         ▼
                                             ┌─────────────────────────┐
                                             │ Store & Alert           │
                                             │ - SQLite (structured)   │
                                             │ - ChromaDB (vector)     │
                                             │ - Alert notifications   │
                                             └─────────────────────────┘
"""

import os
import cv2
import base64
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Generator, Callable
from dataclasses import dataclass, field
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for required packages
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Import config
try:
    from .config import OPENAI_API_KEY
except ImportError:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Import database
try:
    from .database import SecurityDatabase
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    SecurityDatabase = None


@dataclass
class FrameAnalysisResult:
    """Complete analysis result for a single frame."""
    frame_id: int
    timestamp: datetime
    image_data: any  # PIL Image or numpy array
    location: Dict
    telemetry: Dict

    # VLM Analysis Results
    description: str = ""
    objects: List[Dict] = field(default_factory=list)
    alerts: List[Dict] = field(default_factory=list)
    analysis: str = ""
    threat_level: str = "NONE"

    # Processing metadata
    provider: str = "direct"
    processing_time_ms: float = 0
    raw_response: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage/serialization."""
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp.isoformat(),
            "location": self.location,
            "telemetry": self.telemetry,
            "description": self.description,
            "objects": self.objects,
            "alerts": self.alerts,
            "analysis": self.analysis,
            "threat_level": self.threat_level,
            "provider": self.provider,
            "processing_time_ms": self.processing_time_ms
        }


@dataclass
class PipelineConfig:
    """Configuration for the vision pipeline."""
    # Video processing
    frame_interval_seconds: int = 5
    max_frames: int = 50

    # VLM settings
    provider: str = "direct"  # "direct", "gpt4v", "simulated"
    model: str = "gpt-4o"
    max_tokens: int = 500

    # Location zones (cycle through for extracted frames)
    location_zones: List[Dict] = field(default_factory=lambda: [
        {"name": "Main Gate", "zone": "perimeter"},
        {"name": "Parking Lot", "zone": "parking"},
        {"name": "Warehouse", "zone": "storage"},
        {"name": "Loading Dock", "zone": "operations"},
        {"name": "Back Fence", "zone": "perimeter"},
    ])

    # Database settings
    store_to_database: bool = True

    # Callbacks
    on_frame_processed: Optional[Callable] = None
    on_alert_generated: Optional[Callable] = None


class DirectVisionPipeline:
    """
    The BEST approach: Direct Vision Analysis Pipeline.

    Sends actual images directly to GPT-4 Vision for complete security analysis
    in a single API call. No information loss from intermediate captioning.

    Features:
    - Extracts frames from video using OpenCV
    - Sends each frame + context to GPT-4 Vision
    - Gets complete analysis (objects, alerts, threat level) in ONE call
    - Stores results in SQLite + ChromaDB
    - Supports real-time callback notifications
    """

    SECURITY_ANALYSIS_PROMPT = """You are an expert security analyst for a drone surveillance system.
Analyze this security camera image and provide a complete security assessment.

CONTEXT:
- Location: {location_name} (Zone: {location_zone})
- Timestamp: {timestamp}
- Time Context: {time_context}

SECURITY ALERT RULES TO EVALUATE:
- R001 Night Activity (HIGH): Person detected between 00:00-05:00
- R002 Loitering Detection (HIGH): Person staying in same area for extended period
- R003 Perimeter Activity (MEDIUM): Any activity in perimeter zone
- R004 Repeat Vehicle (LOW): Same vehicle seen multiple times
- R005 Unknown Vehicle (MEDIUM): Unrecognized vehicle in restricted area
- R006 Suspicious Behavior (HIGH): Face covering, hiding, carrying suspicious items

INSTRUCTIONS:
1. Describe what you see in the image in detail
2. Identify ALL objects: people (count, clothing, behavior), vehicles (type, color, make), other items
3. Evaluate EACH security alert rule - determine if it should be triggered
4. Assess the overall threat level based on your analysis

Respond in this EXACT JSON format:
{{
    "description": "Detailed description of the scene",
    "objects": [
        {{"type": "person", "description": "detailed description with clothing, behavior, position"}},
        {{"type": "vehicle", "description": "type, color, make/model if visible, position"}}
    ],
    "alerts": [
        {{"rule_id": "R00X", "name": "Rule Name", "priority": "HIGH/MEDIUM/LOW", "reason": "specific reason why triggered"}}
    ],
    "analysis": "Security analysis explaining your assessment and any concerns",
    "threat_level": "NONE/LOW/MEDIUM/HIGH/CRITICAL"
}}

IMPORTANT:
- If no objects are detected, return empty arrays
- If no alerts are triggered, return empty alerts array with threat_level "NONE"
- Be thorough - security depends on accurate detection
- Consider the time of day and location zone in your assessment"""

    def __init__(self, config: PipelineConfig = None, database: SecurityDatabase = None):
        """
        Initialize the Direct Vision Pipeline.

        Args:
            config: Pipeline configuration
            database: Security database for storing results
        """
        self.config = config or PipelineConfig()
        self.database = database

        # Initialize OpenAI client
        self.client = None
        if OPENAI_AVAILABLE:
            api_key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = openai.OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized for Direct Vision Pipeline")
            else:
                logger.warning("No OpenAI API key found - will use simulated mode")
        else:
            logger.warning("OpenAI package not installed - will use simulated mode")

    def _frame_to_base64(self, frame_data) -> str:
        """Convert frame to base64 for API calls."""
        if PIL_AVAILABLE and isinstance(frame_data, Image.Image):
            import io
            buffer = io.BytesIO()
            frame_data.save(buffer, format="JPEG", quality=85)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Assume numpy array (from OpenCV)
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buffer).decode("utf-8")

    def _get_time_context(self, timestamp: datetime) -> str:
        """Get human-readable time context for the prompt."""
        hour = timestamp.hour
        if 0 <= hour < 5:
            return "Night time (RESTRICTED HOURS 00:00-05:00)"
        elif 5 <= hour < 7:
            return "Early morning"
        elif 7 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 17:
            return "Afternoon"
        elif 17 <= hour < 20:
            return "Evening"
        else:
            return "Night time"

    def analyze_frame(self, frame_data, location: Dict, timestamp: datetime,
                      telemetry: Dict = None, frame_id: int = 1) -> FrameAnalysisResult:
        """
        Analyze a single frame with GPT-4 Vision.

        This is the core method - sends image + context to GPT-4 Vision
        and gets complete security analysis in ONE API call.

        Args:
            frame_data: Image data (PIL Image or numpy array)
            location: Location info {"name": "...", "zone": "..."}
            timestamp: Frame timestamp
            telemetry: Optional telemetry data
            frame_id: Frame identifier

        Returns:
            FrameAnalysisResult with complete analysis
        """
        import time
        start_time = time.time()

        result = FrameAnalysisResult(
            frame_id=frame_id,
            timestamp=timestamp,
            image_data=frame_data,
            location=location,
            telemetry=telemetry or {},
            provider=self.config.provider
        )

        # If no client available, use simulated mode
        if not self.client or self.config.provider == "simulated":
            return self._simulated_analysis(result)

        try:
            # Convert image to base64
            base64_image = self._frame_to_base64(frame_data)

            # Build prompt with context
            prompt = self.SECURITY_ANALYSIS_PROMPT.format(
                location_name=location.get("name", "Unknown"),
                location_zone=location.get("zone", "unknown"),
                timestamp=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                time_context=self._get_time_context(timestamp)
            )

            # Call GPT-4 Vision
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.config.max_tokens
            )

            response_text = response.choices[0].message.content
            result.raw_response = response_text

            # Parse JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                parsed = json.loads(json_match.group())
                result.description = parsed.get("description", "")
                result.objects = parsed.get("objects", [])
                result.alerts = parsed.get("alerts", [])
                result.analysis = parsed.get("analysis", "")
                result.threat_level = parsed.get("threat_level", "UNKNOWN")
            else:
                result.description = response_text
                result.analysis = response_text

        except Exception as e:
            logger.error(f"Error analyzing frame {frame_id}: {e}")
            result.description = f"Error analyzing frame: {str(e)}"
            result.threat_level = "UNKNOWN"

        result.processing_time_ms = (time.time() - start_time) * 1000
        return result

    def _simulated_analysis(self, result: FrameAnalysisResult) -> FrameAnalysisResult:
        """Generate simulated analysis for testing/demo."""
        import random

        scenarios = [
            {
                "description": "Blue Ford F150 pickup truck entering through main gate",
                "objects": [{"type": "vehicle", "description": "Blue Ford F150 pickup truck, entering facility"}],
                "alerts": [{"rule_id": "R003", "name": "Perimeter Activity", "priority": "MEDIUM", "reason": "Vehicle activity detected in perimeter zone"}],
                "threat_level": "LOW"
            },
            {
                "description": "Person in dark hoodie walking near perimeter fence",
                "objects": [{"type": "person", "description": "Individual wearing dark hoodie, walking along fence line"}],
                "alerts": [{"rule_id": "R003", "name": "Perimeter Activity", "priority": "MEDIUM", "reason": "Person detected in perimeter zone"}],
                "threat_level": "MEDIUM"
            },
            {
                "description": "Empty parking lot, no activity detected",
                "objects": [],
                "alerts": [],
                "threat_level": "NONE"
            },
            {
                "description": "Two workers in safety vests near warehouse entrance",
                "objects": [
                    {"type": "person", "description": "Worker 1 in orange safety vest"},
                    {"type": "person", "description": "Worker 2 in orange safety vest"}
                ],
                "alerts": [],
                "threat_level": "NONE"
            },
            {
                "description": "Unknown person loitering near restricted area, face partially covered",
                "objects": [{"type": "person", "description": "Individual with face partially covered, loitering near restricted area"}],
                "alerts": [
                    {"rule_id": "R002", "name": "Loitering Detection", "priority": "HIGH", "reason": "Person loitering in area for extended period"},
                    {"rule_id": "R006", "name": "Suspicious Behavior", "priority": "HIGH", "reason": "Face partially covered, suspicious behavior"}
                ],
                "threat_level": "HIGH"
            },
        ]

        # Check if night hours for R001
        if 0 <= result.timestamp.hour < 5:
            scenario = scenarios[4].copy()  # Use suspicious scenario
            scenario["alerts"].append({
                "rule_id": "R001",
                "name": "Night Activity",
                "priority": "HIGH",
                "reason": f"Activity detected during restricted hours ({result.timestamp.strftime('%H:%M')})"
            })
            scenario["threat_level"] = "CRITICAL"
        else:
            scenario = random.choice(scenarios)

        result.description = scenario["description"]
        result.objects = scenario["objects"]
        result.alerts = scenario.get("alerts", [])
        result.analysis = f"Simulated analysis for demo. Location: {result.location.get('name', 'Unknown')}"
        result.threat_level = scenario.get("threat_level", "NONE")
        result.provider = "simulated"
        result.processing_time_ms = 50  # Simulated

        return result

    def extract_frames(self, video_path: str) -> Generator[Dict, None, None]:
        """
        Extract frames from video file.

        Args:
            video_path: Path to video file

        Yields:
            Dict with frame_data, timestamp, location, telemetry
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        fps = max(cap.get(cv2.CAP_PROP_FPS), 1)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        frame_skip = int(fps * self.config.frame_interval_seconds)

        logger.info(f"Video: {duration:.1f}s, {fps:.1f} fps, extracting every {self.config.frame_interval_seconds}s")

        frame_count = 0
        extracted_count = 0
        start_time = datetime.now()

        while cap.isOpened() and extracted_count < self.config.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % max(frame_skip, 1) == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Calculate timestamp
                timestamp = start_time + timedelta(seconds=frame_count / fps)

                # Assign location (cycle through zones)
                location = self.config.location_zones[extracted_count % len(self.config.location_zones)]

                # Generate telemetry
                telemetry = {
                    "drone_id": "DRONE-001",
                    "timestamp": timestamp.isoformat(),
                    "latitude": 37.7749 + (extracted_count * 0.0001),
                    "longitude": -122.4194 + (extracted_count * 0.0001),
                    "altitude": 50,
                    "battery": max(100 - extracted_count, 20),
                    "frame_in_video": frame_count
                }

                yield {
                    "frame_id": extracted_count + 1,
                    "frame_data": frame_rgb,
                    "timestamp": timestamp,
                    "location": location,
                    "telemetry": telemetry
                }

                extracted_count += 1

            frame_count += 1

        cap.release()
        logger.info(f"Extracted {extracted_count} frames from video")

    def process_video(self, video_path: str, progress_callback: Callable = None) -> List[FrameAnalysisResult]:
        """
        Process entire video through the vision pipeline.

        This is the main entry point for video processing:
        1. Extract frames
        2. Analyze each frame with GPT-4 Vision
        3. Store results in database
        4. Call callbacks for alerts

        Args:
            video_path: Path to video file
            progress_callback: Optional callback(current, total, result) for progress updates

        Returns:
            List of FrameAnalysisResult objects
        """
        results = []
        frames_list = list(self.extract_frames(video_path))
        total_frames = len(frames_list)

        logger.info(f"Processing {total_frames} frames through vision pipeline")

        for i, frame_info in enumerate(frames_list):
            # Analyze frame
            result = self.analyze_frame(
                frame_data=frame_info["frame_data"],
                location=frame_info["location"],
                timestamp=frame_info["timestamp"],
                telemetry=frame_info["telemetry"],
                frame_id=frame_info["frame_id"]
            )

            results.append(result)

            # Store in database if configured
            if self.config.store_to_database and self.database:
                self._store_result(result)

            # Call frame processed callback
            if self.config.on_frame_processed:
                self.config.on_frame_processed(result)

            # Call alert callback if alerts generated
            if result.alerts and self.config.on_alert_generated:
                for alert in result.alerts:
                    self.config.on_alert_generated(result, alert)

            # Progress callback
            if progress_callback:
                progress_callback(i + 1, total_frames, result)

            logger.info(f"Frame {result.frame_id}/{total_frames}: {result.description[:50]}... [{result.threat_level}]")

        return results

    def process_image(self, image_path_or_data, location: Dict = None,
                      timestamp: datetime = None) -> FrameAnalysisResult:
        """
        Process a single image through the vision pipeline.

        Args:
            image_path_or_data: Path to image file, PIL Image, or numpy array
            location: Location info
            timestamp: Timestamp (defaults to now)

        Returns:
            FrameAnalysisResult with complete analysis
        """
        # Load image if path
        if isinstance(image_path_or_data, str):
            if PIL_AVAILABLE:
                image = Image.open(image_path_or_data)
            else:
                image = cv2.imread(image_path_or_data)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path_or_data

        # Default values
        location = location or {"name": "Unknown", "zone": "unknown"}
        timestamp = timestamp or datetime.now()

        result = self.analyze_frame(
            frame_data=image,
            location=location,
            timestamp=timestamp,
            frame_id=1
        )

        # Store in database if configured
        if self.config.store_to_database and self.database:
            self._store_result(result)

        return result

    def _store_result(self, result: FrameAnalysisResult):
        """Store analysis result in database."""
        if not self.database:
            return

        try:
            # Store frame
            self.database.index_frame(
                frame_id=result.frame_id,
                timestamp=result.timestamp,
                location_name=result.location.get("name", "Unknown"),
                location_zone=result.location.get("zone", "unknown"),
                description=result.description,
                objects=result.objects,
                telemetry=result.telemetry
            )

            # Store alerts
            for alert in result.alerts:
                self.database.add_alert(
                    frame_id=result.frame_id,
                    rule_id=alert.get("rule_id", "R000"),
                    priority=alert.get("priority", "MEDIUM"),
                    description=alert.get("reason", alert.get("name", "Alert"))
                )

        except Exception as e:
            logger.error(f"Error storing result in database: {e}")


def process_video_with_vision(
    video_path: str,
    provider: str = "direct",
    frame_interval: int = 5,
    max_frames: int = 50,
    progress_callback: Callable = None,
    database: SecurityDatabase = None
) -> List[Dict]:
    """
    Convenience function to process a video with the vision pipeline.

    Args:
        video_path: Path to video file
        provider: VLM provider ("direct", "gpt4v", "simulated")
        frame_interval: Seconds between frame extraction
        max_frames: Maximum frames to extract
        progress_callback: Optional callback(current, total, result)
        database: Optional database for storage

    Returns:
        List of analysis results as dictionaries
    """
    config = PipelineConfig(
        provider=provider,
        frame_interval_seconds=frame_interval,
        max_frames=max_frames,
        store_to_database=database is not None
    )

    pipeline = DirectVisionPipeline(config=config, database=database)
    results = pipeline.process_video(video_path, progress_callback)

    return [r.to_dict() for r in results]


def process_image_with_vision(
    image_path_or_data,
    provider: str = "direct",
    location: Dict = None,
    timestamp: datetime = None,
    database: SecurityDatabase = None
) -> Dict:
    """
    Convenience function to process an image with the vision pipeline.

    Args:
        image_path_or_data: Image path, PIL Image, or numpy array
        provider: VLM provider
        location: Location info
        timestamp: Timestamp
        database: Optional database for storage

    Returns:
        Analysis result as dictionary
    """
    config = PipelineConfig(
        provider=provider,
        store_to_database=database is not None
    )

    pipeline = DirectVisionPipeline(config=config, database=database)
    result = pipeline.process_image(image_path_or_data, location, timestamp)

    return result.to_dict()


# Quick status check
def get_pipeline_status() -> Dict:
    """Get status of the vision pipeline components."""
    has_openai_key = OPENAI_AVAILABLE and bool(os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY)

    return {
        "openai_available": OPENAI_AVAILABLE,
        "openai_key_configured": has_openai_key,
        "pil_available": PIL_AVAILABLE,
        "database_available": DATABASE_AVAILABLE,
        "direct_vision_available": has_openai_key,
        "recommended_provider": "direct" if has_openai_key else "simulated"
    }


if __name__ == "__main__":
    print("=" * 60)
    print("DIRECT VISION PIPELINE TEST")
    print("=" * 60)

    status = get_pipeline_status()
    print(f"\nPipeline Status: {json.dumps(status, indent=2)}")

    # Test with simulated mode
    config = PipelineConfig(provider="simulated")
    pipeline = DirectVisionPipeline(config=config)

    # Create a test image (blank)
    import numpy as np
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)

    result = pipeline.analyze_frame(
        frame_data=test_image,
        location={"name": "Main Gate", "zone": "perimeter"},
        timestamp=datetime.now(),
        frame_id=1
    )

    print(f"\nTest Result:")
    print(f"  Description: {result.description}")
    print(f"  Objects: {len(result.objects)}")
    print(f"  Alerts: {len(result.alerts)}")
    print(f"  Threat Level: {result.threat_level}")
    print(f"  Processing Time: {result.processing_time_ms:.0f}ms")
