"""
Video Frame Analyzer using Vision Language Models (VLM)

This module provides frame analysis capabilities using either:
1. Simulated analysis (for prototype)
2. OpenAI GPT-4 Vision API
3. Local VLM models (BLIP-2, LLaVA)
"""

import re
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .config import OPENAI_API_KEY, OBJECT_CATEGORIES, VEHICLE_COLORS, VEHICLE_MAKES


@dataclass
class AnalysisResult:
    """Result of frame analysis."""
    frame_id: int
    timestamp: datetime
    description: str
    detected_objects: List[Dict]
    security_relevant: bool
    confidence: float
    raw_analysis: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            "description": self.description,
            "detected_objects": self.detected_objects,
            "security_relevant": self.security_relevant,
            "confidence": self.confidence,
            "raw_analysis": self.raw_analysis
        }


class FrameAnalyzer:
    """
    Analyzes video frames to detect objects and generate descriptions.

    For the prototype, this uses simulated analysis based on text descriptions.
    In production, this would integrate with actual VLM APIs.
    """

    def __init__(self, use_api: bool = False):
        """
        Initialize the frame analyzer.

        Args:
            use_api: Whether to use actual VLM API (requires API key)
        """
        self.use_api = use_api and bool(OPENAI_API_KEY)
        self.analysis_history: List[AnalysisResult] = []

        if self.use_api:
            self._init_openai_client()

    def _init_openai_client(self):
        """Initialize OpenAI client for GPT-4 Vision."""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=OPENAI_API_KEY)
        except ImportError:
            print("OpenAI package not installed. Using simulated analysis.")
            self.use_api = False

    def analyze_frame(
        self,
        frame_id: int,
        timestamp: datetime,
        frame_description: str,
        location_context: Dict
    ) -> AnalysisResult:
        """
        Analyze a video frame (simulated or via API).

        Args:
            frame_id: Unique frame identifier
            timestamp: Frame timestamp
            frame_description: Text description of frame content
            location_context: Location metadata from telemetry

        Returns:
            AnalysisResult with detected objects and analysis
        """
        if self.use_api:
            return self._analyze_with_api(frame_id, timestamp, frame_description, location_context)
        else:
            return self._analyze_simulated(frame_id, timestamp, frame_description, location_context)

    def _analyze_simulated(
        self,
        frame_id: int,
        timestamp: datetime,
        frame_description: str,
        location_context: Dict
    ) -> AnalysisResult:
        """
        Perform simulated analysis based on text description.

        This parses the description to extract objects and determine
        security relevance without requiring actual image processing.
        """
        detected_objects = self._extract_objects_from_description(frame_description)
        security_relevant = self._check_security_relevance(
            detected_objects, timestamp, location_context
        )

        # Generate enhanced description
        enhanced_description = self._enhance_description(
            frame_description, detected_objects, location_context
        )

        result = AnalysisResult(
            frame_id=frame_id,
            timestamp=timestamp,
            description=enhanced_description,
            detected_objects=detected_objects,
            security_relevant=security_relevant,
            confidence=0.85,  # Simulated confidence
            raw_analysis=frame_description
        )

        self.analysis_history.append(result)
        return result

    def _analyze_with_api(
        self,
        frame_id: int,
        timestamp: datetime,
        frame_description: str,
        location_context: Dict
    ) -> AnalysisResult:
        """
        Analyze using OpenAI GPT-4 Vision API.

        Note: For prototype, we send text description instead of actual image.
        In production, this would send base64-encoded images.
        """
        prompt = f"""Analyze this security camera frame description and extract relevant information.

Frame Description: {frame_description}
Location: {location_context.get('name', 'Unknown')} ({location_context.get('zone', 'Unknown zone')})
Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}

Please provide:
1. A list of detected objects with their attributes (type, color, make/model if vehicle)
2. Whether this frame is security-relevant (contains people, vehicles, or suspicious activity)
3. Any potential security concerns

Respond in JSON format:
{{
    "objects": [
        {{"type": "vehicle|person|animal", "subtype": "...", "color": "...", "attributes": {{}}}}
    ],
    "security_relevant": true/false,
    "concerns": ["..."],
    "enhanced_description": "..."
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a security analyst AI analyzing drone surveillance footage."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            analysis_text = response.choices[0].message.content
            analysis_data = json.loads(analysis_text)

            result = AnalysisResult(
                frame_id=frame_id,
                timestamp=timestamp,
                description=analysis_data.get("enhanced_description", frame_description),
                detected_objects=analysis_data.get("objects", []),
                security_relevant=analysis_data.get("security_relevant", False),
                confidence=0.90,
                raw_analysis=analysis_text
            )

            self.analysis_history.append(result)
            return result

        except Exception as e:
            print(f"API analysis failed: {e}. Falling back to simulated analysis.")
            return self._analyze_simulated(frame_id, timestamp, frame_description, location_context)

    def _extract_objects_from_description(self, description: str) -> List[Dict]:
        """
        Extract objects from a text description.

        Uses keyword matching and pattern recognition to identify
        objects mentioned in the description.
        """
        objects = []
        desc_lower = description.lower()

        # Vehicle detection
        vehicle_keywords = ["truck", "car", "van", "sedan", "suv", "pickup", "motorcycle", "vehicle", "forklift"]
        for keyword in vehicle_keywords:
            if keyword in desc_lower:
                obj = {
                    "type": "vehicle",
                    "subtype": keyword,
                    "attributes": {}
                }

                # Extract color
                for color in VEHICLE_COLORS:
                    if color in desc_lower:
                        obj["color"] = color
                        break

                # Extract make
                for make in VEHICLE_MAKES:
                    if make.lower() in desc_lower:
                        obj["make"] = make
                        break

                # Extract model (e.g., F150, Camry)
                model_patterns = [r'f-?150', r'f-?250', r'camry', r'civic', r'accord', r'mustang', r'model\s*[3sxy]']
                for pattern in model_patterns:
                    match = re.search(pattern, desc_lower)
                    if match:
                        obj["model"] = match.group().upper().replace('-', '')
                        break

                # Check for recurring/same vehicle
                if any(word in desc_lower for word in ["same", "again", "returning", "second", "repeat"]):
                    obj["attributes"]["recurring"] = True

                objects.append(obj)
                break  # Only detect one vehicle per description

        # Person detection
        person_keywords = ["person", "people", "worker", "man", "woman", "individual", "someone", "guard", "intruder"]
        for keyword in person_keywords:
            if keyword in desc_lower:
                obj = {
                    "type": "person",
                    "subtype": keyword if keyword not in ["person", "someone"] else "unknown",
                    "attributes": {}
                }

                # Check for suspicious indicators
                suspicious_keywords = ["loitering", "suspicious", "dark clothing", "unknown", "unauthorized", "intruder"]
                if any(word in desc_lower for word in suspicious_keywords):
                    obj["attributes"]["suspicious"] = True

                # Check for worker/authorized
                if any(word in desc_lower for word in ["worker", "safety vest", "uniform", "guard", "security"]):
                    obj["attributes"]["authorized"] = True

                objects.append(obj)
                break

        # Animal detection
        animal_keywords = ["dog", "cat", "bird", "deer", "animal", "wildlife"]
        for keyword in animal_keywords:
            if keyword in desc_lower:
                objects.append({
                    "type": "animal",
                    "subtype": keyword,
                    "attributes": {}
                })
                break

        return objects

    def _check_security_relevance(
        self,
        objects: List[Dict],
        timestamp: datetime,
        location_context: Dict
    ) -> bool:
        """
        Determine if frame content is security-relevant.

        Security-relevant frames contain:
        - People (especially at night or in restricted areas)
        - Vehicles (especially unknown or in restricted areas)
        - Any suspicious activity indicators
        """
        if not objects:
            return False

        # Any person detection is security-relevant
        if any(obj.get("type") == "person" for obj in objects):
            return True

        # Vehicles in certain zones
        restricted_zones = ["perimeter", "storage", "operations"]
        if location_context.get("zone") in restricted_zones:
            if any(obj.get("type") == "vehicle" for obj in objects):
                return True

        # Night time activity (00:00 - 05:00)
        hour = timestamp.hour
        if 0 <= hour < 5:
            if objects:  # Any activity at night
                return True

        # Suspicious attributes
        for obj in objects:
            if obj.get("attributes", {}).get("suspicious"):
                return True
            if obj.get("attributes", {}).get("recurring"):
                return True

        # Default: vehicles are mildly relevant
        if any(obj.get("type") == "vehicle" for obj in objects):
            return True

        return False

    def _enhance_description(
        self,
        original: str,
        objects: List[Dict],
        location_context: Dict
    ) -> str:
        """
        Generate an enhanced description with structured information.
        """
        parts = [original]

        # Add location context
        location_name = location_context.get("name", "Unknown location")
        zone = location_context.get("zone", "")

        if objects:
            obj_summary = []
            for obj in objects:
                obj_desc = obj.get("type", "object")
                if obj.get("color"):
                    obj_desc = f"{obj['color']} {obj_desc}"
                if obj.get("make"):
                    obj_desc = f"{obj['make']} {obj_desc}"
                if obj.get("subtype") and obj["subtype"] != obj["type"]:
                    obj_desc = f"{obj_desc} ({obj['subtype']})"
                obj_summary.append(obj_desc)

            if obj_summary:
                parts.append(f"Detected: {', '.join(obj_summary)}")

        return " | ".join(parts)

    def get_analysis_summary(self, last_n: int = 10) -> Dict:
        """Get summary of recent analyses."""
        recent = self.analysis_history[-last_n:] if self.analysis_history else []

        total_objects = sum(len(a.detected_objects) for a in recent)
        security_relevant_count = sum(1 for a in recent if a.security_relevant)

        object_types = {}
        for analysis in recent:
            for obj in analysis.detected_objects:
                obj_type = obj.get("type", "unknown")
                object_types[obj_type] = object_types.get(obj_type, 0) + 1

        return {
            "total_frames_analyzed": len(recent),
            "security_relevant_frames": security_relevant_count,
            "total_objects_detected": total_objects,
            "object_type_breakdown": object_types,
            "average_confidence": sum(a.confidence for a in recent) / len(recent) if recent else 0
        }


class ObjectTracker:
    """
    Tracks objects across multiple frames to identify patterns.

    Maintains state about seen objects and detects recurring appearances.
    """

    def __init__(self, memory_hours: int = 24):
        """
        Initialize the object tracker.

        Args:
            memory_hours: How long to remember objects
        """
        self.memory_hours = memory_hours
        self.tracked_objects: Dict[str, List[Dict]] = {
            "vehicles": [],
            "persons": [],
            "animals": []
        }

    def track_object(self, obj: Dict, timestamp: datetime, location: str) -> Dict:
        """
        Track an object and check for recurring appearances.

        Args:
            obj: Object data from analysis
            timestamp: Detection timestamp
            location: Location name

        Returns:
            Updated object data with tracking info
        """
        obj_type = obj.get("type", "unknown")
        tracking_key = self._generate_tracking_key(obj)

        # Determine category
        category = "vehicles" if obj_type == "vehicle" else \
                   "persons" if obj_type == "person" else "animals"

        # Check for existing tracking
        existing = self._find_matching_object(obj, category)

        tracking_entry = {
            "key": tracking_key,
            "timestamp": timestamp,
            "location": location,
            "object_data": obj
        }

        if existing:
            existing["sightings"].append(tracking_entry)
            obj["tracking"] = {
                "id": existing["id"],
                "total_sightings": len(existing["sightings"]),
                "first_seen": existing["sightings"][0]["timestamp"].isoformat(),
                "recurring": len(existing["sightings"]) > 1
            }
        else:
            new_id = f"{category[:-1]}_{len(self.tracked_objects[category]) + 1}"
            new_tracked = {
                "id": new_id,
                "key": tracking_key,
                "sightings": [tracking_entry]
            }
            self.tracked_objects[category].append(new_tracked)
            obj["tracking"] = {
                "id": new_id,
                "total_sightings": 1,
                "first_seen": timestamp.isoformat(),
                "recurring": False
            }

        return obj

    def _generate_tracking_key(self, obj: Dict) -> str:
        """Generate a unique key for object matching."""
        parts = [obj.get("type", "unknown")]

        if obj.get("color"):
            parts.append(obj["color"])
        if obj.get("make"):
            parts.append(obj["make"])
        if obj.get("model"):
            parts.append(obj["model"])
        if obj.get("subtype"):
            parts.append(obj["subtype"])

        return "_".join(parts).lower()

    def _find_matching_object(self, obj: Dict, category: str) -> Optional[Dict]:
        """Find a matching tracked object."""
        tracking_key = self._generate_tracking_key(obj)

        for tracked in self.tracked_objects[category]:
            if tracked["key"] == tracking_key:
                return tracked

        return None

    def get_recurring_objects(self, min_sightings: int = 2) -> List[Dict]:
        """Get all objects seen multiple times."""
        recurring = []

        for category, objects in self.tracked_objects.items():
            for tracked in objects:
                if len(tracked["sightings"]) >= min_sightings:
                    recurring.append({
                        "id": tracked["id"],
                        "category": category,
                        "key": tracked["key"],
                        "sighting_count": len(tracked["sightings"]),
                        "locations": list(set(s["location"] for s in tracked["sightings"])),
                        "first_seen": tracked["sightings"][0]["timestamp"],
                        "last_seen": tracked["sightings"][-1]["timestamp"]
                    })

        return recurring

    def get_object_history(self, object_id: str) -> Optional[Dict]:
        """Get full history for a tracked object."""
        for category, objects in self.tracked_objects.items():
            for tracked in objects:
                if tracked["id"] == object_id:
                    return {
                        "id": tracked["id"],
                        "category": category,
                        "sightings": [
                            {
                                "timestamp": s["timestamp"].isoformat(),
                                "location": s["location"],
                                "details": s["object_data"]
                            }
                            for s in tracked["sightings"]
                        ]
                    }
        return None

    def clear_old_entries(self):
        """Remove entries older than memory window."""
        cutoff = datetime.now()
        from datetime import timedelta
        cutoff = cutoff - timedelta(hours=self.memory_hours)

        for category in self.tracked_objects:
            for tracked in self.tracked_objects[category]:
                tracked["sightings"] = [
                    s for s in tracked["sightings"]
                    if s["timestamp"] > cutoff
                ]

            # Remove empty tracked objects
            self.tracked_objects[category] = [
                t for t in self.tracked_objects[category]
                if t["sightings"]
            ]


if __name__ == "__main__":
    # Test the analyzer
    analyzer = FrameAnalyzer(use_api=False)

    test_descriptions = [
        "Blue Ford F150 pickup truck entering through main gate",
        "Person in dark clothing walking near warehouse at night",
        "Empty parking lot, no activity detected",
        "Security guard on patrol near fence",
        "Same blue Ford F150 now at parking lot - second visit today"
    ]

    print("=" * 60)
    print("FRAME ANALYZER TEST")
    print("=" * 60)

    for i, desc in enumerate(test_descriptions):
        result = analyzer.analyze_frame(
            frame_id=i + 1,
            timestamp=datetime.now(),
            frame_description=desc,
            location_context={"name": "Main Gate", "zone": "perimeter"}
        )

        print(f"\nFrame {result.frame_id}:")
        print(f"  Description: {result.description}")
        print(f"  Objects: {result.detected_objects}")
        print(f"  Security Relevant: {result.security_relevant}")

    print(f"\n{'=' * 60}")
    summary = analyzer.get_analysis_summary()
    print(f"Analysis Summary: {json.dumps(summary, indent=2)}")
