"""
Tests for the Alert Engine module.
"""

import pytest
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.alert_engine import (
    AlertEngine,
    Alert,
    AlertPriority,
    AlertFormatter
)


@pytest.fixture
def alert_engine():
    """Create an alert engine for testing."""
    return AlertEngine(database=None)


class TestAlertPriority:
    """Tests for AlertPriority enum."""

    def test_priority_values(self):
        """Test priority enum values."""
        assert AlertPriority.LOW.value == "LOW"
        assert AlertPriority.MEDIUM.value == "MEDIUM"
        assert AlertPriority.HIGH.value == "HIGH"
        assert AlertPriority.CRITICAL.value == "CRITICAL"


class TestAlert:
    """Tests for Alert dataclass."""

    def test_alert_creation(self):
        """Test creating an alert."""
        alert = Alert(
            rule_id="R001",
            rule_name="Night Activity",
            priority=AlertPriority.HIGH,
            timestamp=datetime.now(),
            frame_id=1,
            location="Main Gate",
            description="Person detected at night",
            details={"test": True}
        )

        assert alert.rule_id == "R001"
        assert alert.priority == AlertPriority.HIGH

    def test_alert_to_dict(self):
        """Test converting alert to dictionary."""
        alert = Alert(
            rule_id="R001",
            rule_name="Test Rule",
            priority=AlertPriority.MEDIUM,
            timestamp=datetime(2024, 1, 15, 12, 0, 0),
            frame_id=42,
            location="Test Location",
            description="Test description",
            details={}
        )

        data = alert.to_dict()

        assert data["rule_id"] == "R001"
        assert data["priority"] == "MEDIUM"
        assert data["frame_id"] == 42

    def test_alert_str(self):
        """Test alert string representation."""
        alert = Alert(
            rule_id="R001",
            rule_name="Test",
            priority=AlertPriority.HIGH,
            timestamp=datetime(2024, 1, 15, 12, 0, 0),
            frame_id=1,
            location="Gate",
            description="Test alert",
            details={}
        )

        alert_str = str(alert)
        assert "HIGH" in alert_str
        assert "Gate" in alert_str


class TestAlertEngine:
    """Tests for AlertEngine class."""

    def test_engine_initialization(self, alert_engine):
        """Test engine initializes with rules."""
        assert len(alert_engine.rules) > 0

    def test_night_activity_alert(self, alert_engine):
        """Test R001: Night activity detection."""
        # Test at 2 AM with person detected
        timestamp = datetime.now().replace(hour=2, minute=0)
        objects = [{"type": "person", "subtype": "unknown"}]
        location = {"name": "Main Gate", "zone": "perimeter"}

        alerts = alert_engine.evaluate(
            frame_id=1,
            timestamp=timestamp,
            detected_objects=objects,
            location=location,
            description="Person at gate"
        )

        # Should trigger night activity alert
        assert len(alerts) > 0
        assert any(a.rule_id == "R001" for a in alerts)

    def test_no_night_alert_during_day(self, alert_engine):
        """Test no night alert during daytime."""
        # Test at noon with person detected
        timestamp = datetime.now().replace(hour=12, minute=0)
        objects = [{"type": "person", "subtype": "worker"}]
        location = {"name": "Office", "zone": "main"}

        alerts = alert_engine.evaluate(
            frame_id=1,
            timestamp=timestamp,
            detected_objects=objects,
            location=location,
            description="Worker at office"
        )

        # Should not trigger R001 (night activity)
        night_alerts = [a for a in alerts if a.rule_id == "R001"]
        assert len(night_alerts) == 0

    def test_loitering_alert(self, alert_engine):
        """Test R002: Loitering detection."""
        timestamp = datetime.now().replace(hour=14, minute=0)
        objects = [{
            "type": "person",
            "subtype": "unknown",
            "attributes": {"loitering": True}
        }]
        location = {"name": "Warehouse", "zone": "storage"}

        alerts = alert_engine.evaluate(
            frame_id=1,
            timestamp=timestamp,
            detected_objects=objects,
            location=location,
            description="Person loitering"
        )

        assert any(a.rule_id == "R002" for a in alerts)

    def test_perimeter_activity_alert(self, alert_engine):
        """Test R003: Perimeter activity detection."""
        timestamp = datetime.now().replace(hour=10, minute=0)
        objects = [{"type": "vehicle", "subtype": "truck"}]
        location = {"name": "Back Fence", "zone": "perimeter"}

        alerts = alert_engine.evaluate(
            frame_id=1,
            timestamp=timestamp,
            detected_objects=objects,
            location=location,
            description="Vehicle near fence"
        )

        assert any(a.rule_id == "R003" for a in alerts)

    def test_repeat_vehicle_alert(self, alert_engine):
        """Test R004: Repeat vehicle detection."""
        timestamp = datetime.now().replace(hour=12, minute=0)
        objects = [{
            "type": "vehicle",
            "subtype": "truck",
            "color": "blue",
            "make": "Ford",
            "tracking": {
                "recurring": True,
                "total_sightings": 3
            }
        }]
        location = {"name": "Main Gate", "zone": "perimeter"}

        alerts = alert_engine.evaluate(
            frame_id=1,
            timestamp=timestamp,
            detected_objects=objects,
            location=location,
            description="Blue truck again"
        )

        assert any(a.rule_id == "R004" for a in alerts)

    def test_no_alert_for_empty_frame(self, alert_engine):
        """Test no alert for frame with no objects."""
        timestamp = datetime.now().replace(hour=10, minute=0)
        objects = []
        location = {"name": "Parking Lot", "zone": "parking"}

        alerts = alert_engine.evaluate(
            frame_id=1,
            timestamp=timestamp,
            detected_objects=objects,
            location=location,
            description="Empty parking lot"
        )

        assert len(alerts) == 0

    def test_cooldown_prevents_duplicate_alerts(self, alert_engine):
        """Test that cooldown prevents duplicate alerts."""
        timestamp = datetime.now().replace(hour=2, minute=0)
        objects = [{"type": "person", "subtype": "unknown"}]
        location = {"name": "Main Gate", "zone": "perimeter"}

        # First evaluation
        alerts1 = alert_engine.evaluate(
            frame_id=1,
            timestamp=timestamp,
            detected_objects=objects,
            location=location,
            description="Person at gate"
        )

        # Second evaluation (should be in cooldown)
        alerts2 = alert_engine.evaluate(
            frame_id=2,
            timestamp=timestamp + timedelta(seconds=10),
            detected_objects=objects,
            location=location,
            description="Still person at gate"
        )

        # First should trigger, second should be blocked by cooldown
        assert len(alerts1) > 0
        # R001 should be in cooldown for same location
        r001_alerts = [a for a in alerts2 if a.rule_id == "R001"]
        assert len(r001_alerts) == 0


class TestAlertStatistics:
    """Tests for alert statistics."""

    def test_get_alert_statistics(self, alert_engine):
        """Test getting alert statistics."""
        # Generate some alerts
        for i in range(3):
            timestamp = datetime.now().replace(hour=2, minute=i)
            objects = [{"type": "person"}]
            location = {"name": f"Location{i}", "zone": "perimeter"}

            alert_engine.evaluate(
                frame_id=i,
                timestamp=timestamp,
                detected_objects=objects,
                location=location,
                description=f"Test {i}"
            )

        stats = alert_engine.get_alert_statistics()

        assert "total_alerts" in stats
        assert "by_priority" in stats
        assert "by_rule" in stats


class TestAlertFormatter:
    """Tests for AlertFormatter class."""

    @pytest.fixture
    def sample_alert(self):
        """Create a sample alert."""
        return Alert(
            rule_id="R001",
            rule_name="Night Activity",
            priority=AlertPriority.HIGH,
            timestamp=datetime(2024, 1, 15, 2, 30, 0),
            frame_id=42,
            location="Main Gate",
            description="Person detected at night",
            details={}
        )

    def test_format_console(self, sample_alert):
        """Test console formatting."""
        formatted = AlertFormatter.format_console(sample_alert)

        assert "ALERT" in formatted
        assert "HIGH" in formatted
        assert "Main Gate" in formatted

    def test_format_log(self, sample_alert):
        """Test log formatting."""
        formatted = AlertFormatter.format_log(sample_alert)

        assert "R001" in formatted
        assert "HIGH" in formatted
        assert "2024" in formatted

    def test_format_json(self, sample_alert):
        """Test JSON formatting."""
        import json

        formatted = AlertFormatter.format_json(sample_alert)
        data = json.loads(formatted)

        assert data["rule_id"] == "R001"
        assert data["priority"] == "HIGH"


class TestIntegration:
    """Integration tests for alert engine."""

    def test_multiple_alerts_single_frame(self, alert_engine):
        """Test that multiple alerts can be triggered from one frame."""
        # Night + Perimeter should trigger multiple rules
        timestamp = datetime.now().replace(hour=2, minute=0)
        objects = [
            {"type": "person", "subtype": "unknown"},
            {"type": "vehicle", "subtype": "truck"}
        ]
        location = {"name": "Back Fence", "zone": "perimeter"}

        alerts = alert_engine.evaluate(
            frame_id=1,
            timestamp=timestamp,
            detected_objects=objects,
            location=location,
            description="Person and vehicle at fence at night"
        )

        # Should trigger both R001 (night) and R003 (perimeter)
        rule_ids = {a.rule_id for a in alerts}
        assert "R001" in rule_ids  # Night activity
        assert "R003" in rule_ids  # Perimeter activity


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
