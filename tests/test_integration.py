"""
Integration Tests for Drone Security Analyst Agent

These tests verify the complete system works together correctly.
"""

import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulator import DroneSimulator
from src.database import SecurityDatabase, FrameRecord
from src.analyzer import FrameAnalyzer, ObjectTracker
from src.alert_engine import AlertEngine
from src.agent import SecurityAnalystAgent


@pytest.fixture
def temp_db():
    """Create a temporary database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    db = SecurityDatabase(db_path=db_path)
    yield db

    db.close()
    db_path.unlink(missing_ok=True)


@pytest.fixture
def full_system(temp_db):
    """Create a complete system for testing."""
    return {
        "database": temp_db,
        "simulator": DroneSimulator(),
        "analyzer": FrameAnalyzer(use_api=False),
        "tracker": ObjectTracker(),
        "alert_engine": AlertEngine(database=temp_db)
    }


class TestEndToEndPipeline:
    """End-to-end pipeline tests."""

    def test_simulate_and_process_events(self, full_system):
        """Test simulating events and processing them through the pipeline."""
        simulator = full_system["simulator"]
        analyzer = full_system["analyzer"]
        database = full_system["database"]
        alert_engine = full_system["alert_engine"]
        tracker = full_system["tracker"]

        # Generate events
        events = list(simulator.generate_stream(num_events=10))

        processed_count = 0
        alerts_generated = []

        for event in events:
            # Analyze frame
            result = analyzer.analyze_frame(
                frame_id=event.frame.frame_id,
                timestamp=event.frame.timestamp,
                frame_description=event.frame.description,
                location_context={
                    "name": event.frame.location_name,
                    "zone": event.frame.location_zone
                }
            )

            # Track objects
            for obj in result.detected_objects:
                tracker.track_object(obj, result.timestamp, event.frame.location_name)

            # Check alerts
            alerts = alert_engine.evaluate(
                frame_id=event.frame.frame_id,
                timestamp=event.frame.timestamp,
                detected_objects=result.detected_objects,
                location={
                    "name": event.frame.location_name,
                    "zone": event.frame.location_zone
                },
                description=result.description
            )

            # Index frame
            frame_record = FrameRecord(
                frame_id=event.frame.frame_id,
                timestamp=event.frame.timestamp,
                location_name=event.frame.location_name,
                location_zone=event.frame.location_zone,
                latitude=event.telemetry.latitude,
                longitude=event.telemetry.longitude,
                description=result.description,
                objects=result.detected_objects,
                alert_triggered=bool(alerts)
            )
            database.index_frame(frame_record)

            processed_count += 1
            alerts_generated.extend(alerts)

        # Verify results
        assert processed_count == 10

        stats = database.get_statistics()
        assert stats["total_frames"] == 10

    def test_query_after_processing(self, full_system):
        """Test querying data after processing events."""
        simulator = full_system["simulator"]
        analyzer = full_system["analyzer"]
        database = full_system["database"]

        # Process some events with known content
        test_descriptions = [
            "Blue Ford truck at main gate",
            "Person walking near warehouse",
            "Red car in parking lot",
            "Security guard at fence",
            "Blue Ford truck leaving through gate"
        ]

        for i, desc in enumerate(test_descriptions):
            location = {"name": "Test Location", "zone": "test"}

            result = analyzer.analyze_frame(
                frame_id=i + 1,
                timestamp=datetime.now(),
                frame_description=desc,
                location_context=location
            )

            frame_record = FrameRecord(
                frame_id=i + 1,
                timestamp=datetime.now(),
                location_name="Test Location",
                location_zone="test",
                latitude=0.0,
                longitude=0.0,
                description=result.description,
                objects=result.detected_objects,
                alert_triggered=False
            )
            database.index_frame(frame_record)

        # Query for trucks
        truck_results = database.query_frames_by_description("truck")
        assert len(truck_results) >= 2  # Should find Ford truck entries

        # Query for person
        person_results = database.query_frames_by_description("Person")
        assert len(person_results) >= 1

    def test_alert_generation_pipeline(self, full_system):
        """Test that alerts are properly generated and stored."""
        database = full_system["database"]
        alert_engine = full_system["alert_engine"]

        # Create a night-time event that should trigger alert
        night_time = datetime.now().replace(hour=2, minute=30)
        objects = [{"type": "person", "subtype": "unknown"}]
        location = {"name": "Main Gate", "zone": "perimeter"}

        alerts = alert_engine.evaluate(
            frame_id=1,
            timestamp=night_time,
            detected_objects=objects,
            location=location,
            description="Person at gate at night"
        )

        # Should generate at least one alert
        assert len(alerts) > 0

        # Alerts should be in database
        db_alerts = database.get_alerts()
        assert len(db_alerts) > 0

    def test_recurring_object_tracking(self, full_system):
        """Test that recurring objects are properly tracked."""
        tracker = full_system["tracker"]
        analyzer = full_system["analyzer"]

        # Simulate same vehicle appearing multiple times
        vehicle_descriptions = [
            "Blue Ford F150 truck at main gate",
            "Blue Ford F150 truck at parking lot",
            "Blue Ford F150 truck leaving via back gate"
        ]

        for i, desc in enumerate(vehicle_descriptions):
            result = analyzer.analyze_frame(
                frame_id=i + 1,
                timestamp=datetime.now(),
                frame_description=desc,
                location_context={"name": f"Location{i}", "zone": "perimeter"}
            )

            for obj in result.detected_objects:
                tracker.track_object(obj, result.timestamp, f"Location{i}")

        # Check for recurring vehicle
        recurring = tracker.get_recurring_objects(min_sightings=2)
        assert len(recurring) >= 1


class TestSecurityAgentIntegration:
    """Tests for SecurityAnalystAgent integration."""

    def test_agent_process_frame(self, temp_db):
        """Test agent frame processing."""
        agent = SecurityAnalystAgent(database=temp_db, use_api=False)

        result = agent.process_frame(
            frame_id=1,
            timestamp=datetime.now(),
            description="Blue truck entering main gate",
            location={"name": "Main Gate", "zone": "perimeter"},
            telemetry={"latitude": 37.77, "longitude": -122.41}
        )

        assert result["frame_id"] == 1
        assert "analysis" in result
        assert "tracked_objects" in result

    def test_agent_chat_offline(self, temp_db):
        """Test agent chat functionality (offline mode)."""
        agent = SecurityAnalystAgent(database=temp_db, use_api=False)

        # Process some frames first
        for i in range(3):
            agent.process_frame(
                frame_id=i + 1,
                timestamp=datetime.now(),
                description=f"Test frame {i} with truck",
                location={"name": "Gate", "zone": "perimeter"},
                telemetry={"latitude": 37.77, "longitude": -122.41}
            )

        # Test queries
        response = agent.chat("Show me all truck events")
        assert len(response) > 0

        response = agent.chat("Give me a summary")
        assert len(response) > 0

    def test_agent_context_summary(self, temp_db):
        """Test agent context summary."""
        agent = SecurityAnalystAgent(database=temp_db, use_api=False)

        # Process some events
        agent.process_frame(
            frame_id=1,
            timestamp=datetime.now(),
            description="Test frame",
            location={"name": "Gate", "zone": "perimeter"},
            telemetry={"latitude": 37.77, "longitude": -122.41}
        )

        summary = agent.get_context_summary()

        assert "Frames indexed" in summary
        assert "alerts" in summary.lower()


class TestDemoScenarios:
    """Tests for demo scenario functionality."""

    def test_curated_demo_scenario(self, temp_db):
        """Test the curated demo scenario."""
        simulator = DroneSimulator()
        analyzer = FrameAnalyzer(use_api=False)
        alert_engine = AlertEngine(database=temp_db)

        events = simulator.generate_demo_scenario()

        # Should generate meaningful events
        assert len(events) > 5

        # Process all events
        all_alerts = []
        for event in events:
            result = analyzer.analyze_frame(
                frame_id=event.frame.frame_id,
                timestamp=event.frame.timestamp,
                frame_description=event.frame.description,
                location_context={
                    "name": event.frame.location_name,
                    "zone": event.frame.location_zone
                }
            )

            alerts = alert_engine.evaluate(
                frame_id=event.frame.frame_id,
                timestamp=event.frame.timestamp,
                detected_objects=result.detected_objects,
                location={
                    "name": event.frame.location_name,
                    "zone": event.frame.location_zone
                },
                description=result.description
            )

            all_alerts.extend(alerts)

        # Demo should generate some detections
        assert len(analyzer.analysis_history) == len(events)


class TestDataPersistence:
    """Tests for data persistence."""

    def test_data_survives_session(self):
        """Test that data persists across database sessions."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = Path(f.name)

        try:
            # Session 1: Write data
            db1 = SecurityDatabase(db_path=db_path)
            frame = FrameRecord(
                frame_id=1,
                timestamp=datetime.now(),
                location_name="Test",
                location_zone="test",
                latitude=0.0,
                longitude=0.0,
                description="Persistent test frame",
                objects=[{"type": "vehicle"}]
            )
            db1.index_frame(frame)
            db1.close()

            # Session 2: Read data
            db2 = SecurityDatabase(db_path=db_path)
            retrieved = db2.get_frame(1)

            assert retrieved is not None
            assert retrieved.description == "Persistent test frame"

            db2.close()

        finally:
            db_path.unlink(missing_ok=True)


class TestErrorHandling:
    """Tests for error handling."""

    def test_handle_empty_description(self, full_system):
        """Test handling of empty frame descriptions."""
        analyzer = full_system["analyzer"]

        result = analyzer.analyze_frame(
            frame_id=1,
            timestamp=datetime.now(),
            frame_description="",
            location_context={"name": "Test", "zone": "test"}
        )

        # Should not crash, should return empty objects
        assert result.detected_objects == []

    def test_handle_invalid_timestamp(self, full_system):
        """Test handling of edge case timestamps."""
        alert_engine = full_system["alert_engine"]

        # Boundary time (exactly midnight)
        midnight = datetime.now().replace(hour=0, minute=0, second=0)
        objects = [{"type": "person"}]
        location = {"name": "Gate", "zone": "perimeter"}

        # Should not crash
        alerts = alert_engine.evaluate(
            frame_id=1,
            timestamp=midnight,
            detected_objects=objects,
            location=location,
            description="Test at midnight"
        )

        # Midnight is within night hours, should trigger alert
        assert any(a.rule_id == "R001" for a in alerts)


class TestPerformance:
    """Basic performance tests."""

    def test_process_many_events(self, temp_db):
        """Test processing a large number of events."""
        agent = SecurityAnalystAgent(database=temp_db, use_api=False)

        import time
        start = time.time()

        for i in range(100):
            agent.process_frame(
                frame_id=i + 1,
                timestamp=datetime.now(),
                description=f"Test frame {i}",
                location={"name": "Gate", "zone": "perimeter"},
                telemetry={"latitude": 37.77, "longitude": -122.41}
            )

        elapsed = time.time() - start

        # Should process 100 frames in reasonable time (< 10 seconds without API)
        assert elapsed < 10

        stats = temp_db.get_statistics()
        assert stats["total_frames"] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
