"""
Tests for the Database module.
"""

import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import (
    SecurityDatabase,
    FrameRecord,
    AlertRecord,
    DetectionRecord,
    parse_natural_query
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    db = SecurityDatabase(db_path=db_path)
    yield db

    db.close()
    db_path.unlink(missing_ok=True)


@pytest.fixture
def sample_frame():
    """Create a sample frame record."""
    return FrameRecord(
        frame_id=1,
        timestamp=datetime.now(),
        location_name="Main Gate",
        location_zone="perimeter",
        latitude=37.7749,
        longitude=-122.4194,
        description="Blue Ford F150 entering main gate",
        objects=[{"type": "vehicle", "subtype": "truck", "color": "blue", "make": "Ford"}]
    )


class TestFrameRecord:
    """Tests for FrameRecord class."""

    def test_frame_record_creation(self, sample_frame):
        """Test creating a frame record."""
        assert sample_frame.frame_id == 1
        assert sample_frame.location_name == "Main Gate"
        assert len(sample_frame.objects) == 1

    def test_frame_record_to_dict(self, sample_frame):
        """Test converting frame record to dictionary."""
        data = sample_frame.to_dict()

        assert data["frame_id"] == 1
        assert data["location_name"] == "Main Gate"
        assert isinstance(data["objects"], list)


class TestSecurityDatabase:
    """Tests for SecurityDatabase class."""

    def test_database_initialization(self, temp_db):
        """Test database initializes correctly."""
        assert temp_db.connection is not None

    def test_index_frame(self, temp_db, sample_frame):
        """Test indexing a frame."""
        frame_id = temp_db.index_frame(sample_frame)

        assert frame_id == sample_frame.frame_id

    def test_get_frame(self, temp_db, sample_frame):
        """Test retrieving a frame."""
        temp_db.index_frame(sample_frame)
        retrieved = temp_db.get_frame(sample_frame.frame_id)

        assert retrieved is not None
        assert retrieved.frame_id == sample_frame.frame_id
        assert retrieved.description == sample_frame.description

    def test_get_nonexistent_frame(self, temp_db):
        """Test retrieving a frame that doesn't exist."""
        retrieved = temp_db.get_frame(9999)
        assert retrieved is None

    def test_query_frames_by_time(self, temp_db):
        """Test querying frames by time range."""
        now = datetime.now()

        # Create frames at different times
        for i in range(5):
            frame = FrameRecord(
                frame_id=i + 1,
                timestamp=now - timedelta(hours=i),
                location_name="Test Location",
                location_zone="test",
                latitude=0.0,
                longitude=0.0,
                description=f"Test frame {i}",
                objects=[]
            )
            temp_db.index_frame(frame)

        # Query last 2 hours
        start = now - timedelta(hours=2)
        results = temp_db.query_frames_by_time(start, now)

        assert len(results) >= 2

    def test_query_frames_by_location(self, temp_db):
        """Test querying frames by location zone."""
        # Create frames in different zones
        zones = ["perimeter", "perimeter", "parking", "storage"]
        for i, zone in enumerate(zones):
            frame = FrameRecord(
                frame_id=i + 1,
                timestamp=datetime.now(),
                location_name=f"Location {i}",
                location_zone=zone,
                latitude=0.0,
                longitude=0.0,
                description=f"Test frame {i}",
                objects=[]
            )
            temp_db.index_frame(frame)

        results = temp_db.query_frames_by_location("perimeter")
        assert len(results) == 2

    def test_query_frames_by_description(self, temp_db):
        """Test querying frames by description text."""
        frames = [
            ("Blue truck at gate", [{"type": "vehicle"}]),
            ("Person walking", [{"type": "person"}]),
            ("Red truck in parking", [{"type": "vehicle"}]),
        ]

        for i, (desc, objs) in enumerate(frames):
            frame = FrameRecord(
                frame_id=i + 1,
                timestamp=datetime.now(),
                location_name="Test",
                location_zone="test",
                latitude=0.0,
                longitude=0.0,
                description=desc,
                objects=objs
            )
            temp_db.index_frame(frame)

        results = temp_db.query_frames_by_description("truck")
        assert len(results) == 2

    def test_query_frames_complex(self, temp_db):
        """Test complex query with multiple filters."""
        now = datetime.now()

        # Create diverse frames
        frames_data = [
            ("perimeter", "Truck at gate", [{"type": "vehicle"}]),
            ("perimeter", "Person at fence", [{"type": "person"}]),
            ("parking", "Car parked", [{"type": "vehicle"}]),
        ]

        for i, (zone, desc, objs) in enumerate(frames_data):
            frame = FrameRecord(
                frame_id=i + 1,
                timestamp=now - timedelta(hours=i),
                location_name=f"Location {i}",
                location_zone=zone,
                latitude=0.0,
                longitude=0.0,
                description=desc,
                objects=objs
            )
            temp_db.index_frame(frame)

        # Query vehicles in perimeter
        results = temp_db.query_frames_complex(
            object_type="vehicle",
            location_zone="perimeter"
        )

        assert len(results) == 1
        assert "Truck" in results[0].description


class TestAlertOperations:
    """Tests for alert-related database operations."""

    def test_log_alert(self, temp_db):
        """Test logging an alert."""
        alert = AlertRecord(
            alert_id=None,
            timestamp=datetime.now(),
            frame_id=1,
            rule_id="R001",
            priority="HIGH",
            description="Test alert",
            location="Main Gate",
            status="active"
        )

        alert_id = temp_db.log_alert(alert)
        assert alert_id is not None

    def test_get_alerts(self, temp_db):
        """Test retrieving alerts."""
        # Create multiple alerts
        for i, priority in enumerate(["HIGH", "MEDIUM", "LOW"]):
            alert = AlertRecord(
                alert_id=None,
                timestamp=datetime.now(),
                frame_id=i + 1,
                rule_id=f"R00{i+1}",
                priority=priority,
                description=f"Test alert {i}",
                location="Test Location",
                status="active"
            )
            temp_db.log_alert(alert)

        # Get all alerts
        all_alerts = temp_db.get_alerts()
        assert len(all_alerts) == 3

        # Get high priority only
        high_alerts = temp_db.get_alerts(priority="HIGH")
        assert len(high_alerts) == 1

    def test_update_alert_status(self, temp_db):
        """Test updating alert status."""
        alert = AlertRecord(
            alert_id=None,
            timestamp=datetime.now(),
            frame_id=1,
            rule_id="R001",
            priority="HIGH",
            description="Test alert",
            location="Test",
            status="active"
        )

        alert_id = temp_db.log_alert(alert)
        temp_db.update_alert_status(alert_id, "resolved")

        alerts = temp_db.get_alerts(status="resolved")
        assert len(alerts) == 1


class TestDetectionOperations:
    """Tests for detection-related database operations."""

    def test_log_detection(self, temp_db):
        """Test logging a detection."""
        detection = DetectionRecord(
            detection_id=None,
            timestamp=datetime.now(),
            frame_id=1,
            object_type="vehicle",
            object_subtype="truck",
            object_attributes={"color": "blue"},
            location_zone="perimeter",
            confidence=0.95
        )

        detection_id = temp_db.log_detection(detection)
        assert detection_id is not None

    def test_get_detections_by_type(self, temp_db):
        """Test getting detections by type."""
        # Create detections
        for obj_type in ["vehicle", "vehicle", "person"]:
            detection = DetectionRecord(
                detection_id=None,
                timestamp=datetime.now(),
                frame_id=1,
                object_type=obj_type,
                object_subtype="unknown",
                object_attributes={},
                location_zone="test",
                confidence=0.9
            )
            temp_db.log_detection(detection)

        vehicles = temp_db.get_detections_by_type("vehicle")
        assert len(vehicles) == 2

    def test_count_object_occurrences(self, temp_db):
        """Test counting object occurrences."""
        # Create multiple vehicle detections
        for i in range(5):
            detection = DetectionRecord(
                detection_id=None,
                timestamp=datetime.now(),
                frame_id=i + 1,
                object_type="vehicle",
                object_subtype="truck",
                object_attributes={},
                location_zone="test",
                confidence=0.9
            )
            temp_db.log_detection(detection)

        count = temp_db.count_object_occurrences("vehicle")
        assert count == 5

        count_trucks = temp_db.count_object_occurrences("vehicle", "truck")
        assert count_trucks == 5


class TestStatistics:
    """Tests for statistics operations."""

    def test_get_statistics(self, temp_db, sample_frame):
        """Test getting database statistics."""
        # Add some data
        temp_db.index_frame(sample_frame)

        alert = AlertRecord(
            alert_id=None,
            timestamp=datetime.now(),
            frame_id=1,
            rule_id="R001",
            priority="HIGH",
            description="Test",
            location="Test",
            status="active"
        )
        temp_db.log_alert(alert)

        stats = temp_db.get_statistics()

        assert stats["total_frames"] >= 1
        assert stats["total_alerts"] >= 1

    def test_clear_all_data(self, temp_db, sample_frame):
        """Test clearing all data."""
        temp_db.index_frame(sample_frame)

        temp_db.clear_all_data()

        stats = temp_db.get_statistics()
        assert stats["total_frames"] == 0


class TestNaturalQueryParser:
    """Tests for natural language query parsing."""

    def test_parse_vehicle_query(self):
        """Test parsing vehicle-related queries."""
        params = parse_natural_query("show me all trucks")

        assert params.get("object_type") == "vehicle"

    def test_parse_person_query(self):
        """Test parsing person-related queries."""
        params = parse_natural_query("any people detected?")

        assert params.get("object_type") == "person"

    def test_parse_location_query(self):
        """Test parsing location queries."""
        params = parse_natural_query("activity at the gate")

        assert params.get("description_contains") == "gate"

    def test_parse_time_query(self):
        """Test parsing time-based queries."""
        params = parse_natural_query("events from today")

        assert "start_time" in params
        assert "end_time" in params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
