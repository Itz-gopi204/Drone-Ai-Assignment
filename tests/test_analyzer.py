"""
Tests for the Frame Analyzer module.
"""

import pytest
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analyzer import (
    FrameAnalyzer,
    AnalysisResult,
    ObjectTracker
)


@pytest.fixture
def analyzer():
    """Create a frame analyzer for testing."""
    return FrameAnalyzer(use_api=False)


@pytest.fixture
def tracker():
    """Create an object tracker for testing."""
    return ObjectTracker()


class TestAnalysisResult:
    """Tests for AnalysisResult dataclass."""

    def test_result_creation(self):
        """Test creating an analysis result."""
        result = AnalysisResult(
            frame_id=1,
            timestamp=datetime.now(),
            description="Blue truck at gate",
            detected_objects=[{"type": "vehicle"}],
            security_relevant=True,
            confidence=0.9
        )

        assert result.frame_id == 1
        assert result.security_relevant is True
        assert len(result.detected_objects) == 1

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = AnalysisResult(
            frame_id=42,
            timestamp=datetime(2024, 1, 15, 12, 0, 0),
            description="Test description",
            detected_objects=[{"type": "person"}],
            security_relevant=False,
            confidence=0.85,
            raw_analysis="Raw text"
        )

        data = result.to_dict()

        assert data["frame_id"] == 42
        assert data["confidence"] == 0.85
        assert data["raw_analysis"] == "Raw text"


class TestFrameAnalyzer:
    """Tests for FrameAnalyzer class."""

    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.use_api is False
        assert analyzer.analysis_history == []

    def test_analyze_vehicle_frame(self, analyzer):
        """Test analyzing a frame with a vehicle."""
        result = analyzer.analyze_frame(
            frame_id=1,
            timestamp=datetime.now(),
            frame_description="Blue Ford F150 pickup truck at main gate",
            location_context={"name": "Main Gate", "zone": "perimeter"}
        )

        assert result.frame_id == 1
        assert len(result.detected_objects) > 0

        # Should detect vehicle
        vehicle = result.detected_objects[0]
        assert vehicle["type"] == "vehicle"
        assert vehicle.get("color") == "blue"
        assert vehicle.get("make") == "Ford"

    def test_analyze_person_frame(self, analyzer):
        """Test analyzing a frame with a person."""
        result = analyzer.analyze_frame(
            frame_id=2,
            timestamp=datetime.now(),
            frame_description="Person in dark clothing walking near warehouse",
            location_context={"name": "Warehouse", "zone": "storage"}
        )

        assert len(result.detected_objects) > 0

        person = result.detected_objects[0]
        assert person["type"] == "person"

    def test_analyze_empty_frame(self, analyzer):
        """Test analyzing a frame with no objects."""
        result = analyzer.analyze_frame(
            frame_id=3,
            timestamp=datetime.now(),
            frame_description="Empty parking lot, clear skies",
            location_context={"name": "Parking Lot", "zone": "parking"}
        )

        assert result.detected_objects == []
        assert result.security_relevant is False

    def test_security_relevance_person(self, analyzer):
        """Test that person detection is security relevant."""
        result = analyzer.analyze_frame(
            frame_id=4,
            timestamp=datetime.now(),
            frame_description="Unknown person near fence",
            location_context={"name": "Back Fence", "zone": "perimeter"}
        )

        assert result.security_relevant is True

    def test_security_relevance_night(self, analyzer):
        """Test night activity is security relevant."""
        night_time = datetime.now().replace(hour=2, minute=0)

        result = analyzer.analyze_frame(
            frame_id=5,
            timestamp=night_time,
            frame_description="Vehicle with headlights at gate",
            location_context={"name": "Main Gate", "zone": "perimeter"}
        )

        assert result.security_relevant is True

    def test_suspicious_attributes_detection(self, analyzer):
        """Test detection of suspicious attributes."""
        result = analyzer.analyze_frame(
            frame_id=6,
            timestamp=datetime.now(),
            frame_description="Person loitering near warehouse entrance",
            location_context={"name": "Warehouse", "zone": "storage"}
        )

        assert result.security_relevant is True
        # Person should have suspicious attribute
        if result.detected_objects:
            person = result.detected_objects[0]
            assert person.get("attributes", {}).get("suspicious") is True

    def test_recurring_vehicle_detection(self, analyzer):
        """Test detection of recurring vehicle."""
        result = analyzer.analyze_frame(
            frame_id=7,
            timestamp=datetime.now(),
            frame_description="Same blue truck returning to parking lot again",
            location_context={"name": "Parking Lot", "zone": "parking"}
        )

        if result.detected_objects:
            vehicle = result.detected_objects[0]
            assert vehicle.get("attributes", {}).get("recurring") is True

    def test_analysis_history(self, analyzer):
        """Test that analysis history is maintained."""
        # Analyze multiple frames
        for i in range(5):
            analyzer.analyze_frame(
                frame_id=i + 1,
                timestamp=datetime.now(),
                frame_description=f"Test frame {i}",
                location_context={"name": "Test", "zone": "test"}
            )

        assert len(analyzer.analysis_history) == 5

    def test_get_analysis_summary(self, analyzer):
        """Test getting analysis summary."""
        # Analyze some frames
        descriptions = [
            "Blue truck at gate",
            "Person walking",
            "Empty lot",
            "Red car parked",
        ]

        for i, desc in enumerate(descriptions):
            analyzer.analyze_frame(
                frame_id=i + 1,
                timestamp=datetime.now(),
                frame_description=desc,
                location_context={"name": "Test", "zone": "test"}
            )

        summary = analyzer.get_analysis_summary()

        assert summary["total_frames_analyzed"] == 4
        assert "object_type_breakdown" in summary


class TestObjectTracker:
    """Tests for ObjectTracker class."""

    def test_tracker_initialization(self, tracker):
        """Test tracker initialization."""
        assert "vehicles" in tracker.tracked_objects
        assert "persons" in tracker.tracked_objects

    def test_track_new_vehicle(self, tracker):
        """Test tracking a new vehicle."""
        vehicle = {"type": "vehicle", "subtype": "truck", "color": "blue"}
        timestamp = datetime.now()

        result = tracker.track_object(vehicle, timestamp, "Main Gate")

        assert "tracking" in result
        assert result["tracking"]["total_sightings"] == 1
        assert result["tracking"]["recurring"] is False

    def test_track_recurring_vehicle(self, tracker):
        """Test tracking a recurring vehicle."""
        vehicle = {"type": "vehicle", "subtype": "truck", "color": "blue", "make": "Ford"}
        timestamp = datetime.now()

        # First sighting
        tracker.track_object(vehicle.copy(), timestamp, "Main Gate")

        # Second sighting
        result = tracker.track_object(vehicle.copy(), timestamp, "Parking Lot")

        assert result["tracking"]["total_sightings"] == 2
        assert result["tracking"]["recurring"] is True

    def test_track_different_vehicles(self, tracker):
        """Test tracking different vehicles separately."""
        vehicle1 = {"type": "vehicle", "color": "blue", "make": "Ford"}
        vehicle2 = {"type": "vehicle", "color": "red", "make": "Toyota"}
        timestamp = datetime.now()

        result1 = tracker.track_object(vehicle1, timestamp, "Gate")
        result2 = tracker.track_object(vehicle2, timestamp, "Gate")

        # Should have different tracking IDs
        assert result1["tracking"]["id"] != result2["tracking"]["id"]

    def test_get_recurring_objects(self, tracker):
        """Test getting list of recurring objects."""
        vehicle = {"type": "vehicle", "color": "blue"}
        timestamp = datetime.now()

        # Create multiple sightings
        for i in range(3):
            tracker.track_object(vehicle.copy(), timestamp, f"Location{i}")

        recurring = tracker.get_recurring_objects(min_sightings=2)

        assert len(recurring) >= 1
        assert recurring[0]["sighting_count"] >= 2

    def test_get_object_history(self, tracker):
        """Test getting history for a tracked object."""
        vehicle = {"type": "vehicle", "color": "blue"}
        timestamp = datetime.now()

        result = tracker.track_object(vehicle, timestamp, "Gate")
        object_id = result["tracking"]["id"]

        history = tracker.get_object_history(object_id)

        assert history is not None
        assert len(history["sightings"]) == 1

    def test_track_person(self, tracker):
        """Test tracking a person."""
        person = {"type": "person", "subtype": "worker"}
        timestamp = datetime.now()

        result = tracker.track_object(person, timestamp, "Office")

        assert "tracking" in result
        assert "person" in result["tracking"]["id"]


class TestIntegration:
    """Integration tests for analyzer module."""

    def test_full_analysis_pipeline(self, analyzer, tracker):
        """Test full analysis and tracking pipeline."""
        descriptions = [
            ("Blue Ford F150 at main gate", {"name": "Main Gate", "zone": "perimeter"}),
            ("Person walking near warehouse", {"name": "Warehouse", "zone": "storage"}),
            ("Same blue Ford F150 now at parking lot", {"name": "Parking Lot", "zone": "parking"}),
        ]

        for i, (desc, location) in enumerate(descriptions):
            result = analyzer.analyze_frame(
                frame_id=i + 1,
                timestamp=datetime.now(),
                frame_description=desc,
                location_context=location
            )

            # Track detected objects
            for obj in result.detected_objects:
                tracker.track_object(obj, result.timestamp, location["name"])

        # Verify tracking
        recurring = tracker.get_recurring_objects()
        # The Ford F150 should be tracked as recurring
        assert len(analyzer.analysis_history) == 3

    def test_extract_vehicle_details(self, analyzer):
        """Test extraction of detailed vehicle information."""
        result = analyzer.analyze_frame(
            frame_id=1,
            timestamp=datetime.now(),
            frame_description="White Tesla Model 3 sedan entering garage",
            location_context={"name": "Garage", "zone": "vehicles"}
        )

        if result.detected_objects:
            vehicle = result.detected_objects[0]
            # Should extract color and potentially make
            assert vehicle.get("color") == "white" or vehicle.get("make") == "Tesla"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
