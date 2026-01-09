"""
Tests for the Drone Simulator module.
"""

import pytest
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulator import (
    DroneSimulator,
    TelemetryData,
    VideoFrame,
    SimulatedEvent,
    ScenarioGenerator
)


class TestTelemetryData:
    """Tests for TelemetryData class."""

    def test_telemetry_creation(self):
        """Test creating telemetry data."""
        telemetry = TelemetryData(
            timestamp=datetime.now(),
            drone_id="DRONE-001",
            location_name="Main Gate",
            location_zone="perimeter",
            latitude=37.7749,
            longitude=-122.4194,
            altitude=50.0,
            battery_percent=85,
            status="patrolling"
        )

        assert telemetry.drone_id == "DRONE-001"
        assert telemetry.location_name == "Main Gate"
        assert telemetry.battery_percent == 85

    def test_telemetry_to_dict(self):
        """Test converting telemetry to dictionary."""
        telemetry = TelemetryData(
            timestamp=datetime(2024, 1, 15, 12, 0, 0),
            drone_id="DRONE-001",
            location_name="Main Gate",
            location_zone="perimeter",
            latitude=37.7749,
            longitude=-122.4194,
            altitude=50.0,
            battery_percent=85,
            status="patrolling"
        )

        data = telemetry.to_dict()

        assert data["drone_id"] == "DRONE-001"
        assert data["location"]["name"] == "Main Gate"
        assert data["battery"] == 85


class TestVideoFrame:
    """Tests for VideoFrame class."""

    def test_frame_creation(self):
        """Test creating video frame."""
        frame = VideoFrame(
            frame_id=1,
            timestamp=datetime.now(),
            description="Blue truck at gate",
            detected_objects=[{"type": "vehicle", "color": "blue"}],
            location_name="Main Gate",
            location_zone="perimeter"
        )

        assert frame.frame_id == 1
        assert "truck" in frame.description
        assert len(frame.detected_objects) == 1

    def test_frame_str(self):
        """Test frame string representation."""
        frame = VideoFrame(
            frame_id=42,
            timestamp=datetime.now(),
            description="Security event detected",
            detected_objects=[],
            location_name="Warehouse",
            location_zone="storage"
        )

        assert "42" in str(frame)
        assert "Security event" in str(frame)


class TestScenarioGenerator:
    """Tests for scenario generation."""

    def test_time_period_detection(self):
        """Test correct time period detection."""
        assert ScenarioGenerator.get_time_period(8) == "morning"
        assert ScenarioGenerator.get_time_period(14) == "afternoon"
        assert ScenarioGenerator.get_time_period(20) == "evening"
        assert ScenarioGenerator.get_time_period(2) == "night"

    def test_scenario_generation(self):
        """Test scenario generation."""
        location = {"name": "Main Gate", "zone": "perimeter"}
        scenario = ScenarioGenerator.generate_scenario(10, location)

        assert "description" in scenario
        assert "objects" in scenario
        assert isinstance(scenario["objects"], list)


class TestDroneSimulator:
    """Tests for the main DroneSimulator class."""

    def test_simulator_initialization(self):
        """Test simulator initialization."""
        simulator = DroneSimulator()

        assert simulator.frame_counter == 0
        assert simulator.battery_level == 100
        assert simulator.config is not None

    def test_simulator_with_start_time(self):
        """Test simulator with custom start time."""
        start = datetime(2024, 1, 15, 12, 0, 0)
        simulator = DroneSimulator(start_time=start)

        assert simulator.current_time == start

    def test_generate_telemetry(self):
        """Test telemetry generation."""
        simulator = DroneSimulator()
        location = {"name": "Test Location", "zone": "test", "lat": 37.0, "lon": -122.0}

        telemetry = simulator.generate_telemetry(location)

        assert telemetry.location_name == "Test Location"
        assert telemetry.drone_id == "DRONE-001"
        assert 0 <= telemetry.battery_percent <= 100

    def test_generate_frame(self):
        """Test frame generation."""
        simulator = DroneSimulator()
        location = {"name": "Main Gate", "zone": "perimeter", "lat": 37.0, "lon": -122.0}

        frame = simulator.generate_frame(location)

        assert frame.frame_id == 1
        assert frame.location_name == "Main Gate"
        assert isinstance(frame.description, str)

    def test_generate_event(self):
        """Test event generation."""
        simulator = DroneSimulator()
        event = simulator.generate_event()

        assert isinstance(event, SimulatedEvent)
        assert isinstance(event.telemetry, TelemetryData)
        assert isinstance(event.frame, VideoFrame)

    def test_generate_stream(self):
        """Test event stream generation."""
        simulator = DroneSimulator()
        events = list(simulator.generate_stream(num_events=10))

        assert len(events) == 10
        # Frame IDs should be sequential
        assert events[0].frame.frame_id == 1
        assert events[9].frame.frame_id == 10

    def test_battery_drain(self):
        """Test that battery drains over time."""
        simulator = DroneSimulator()
        initial_battery = simulator.battery_level

        # Generate multiple events
        for _ in range(10):
            simulator.generate_event()

        assert simulator.battery_level < initial_battery

    def test_time_progression(self):
        """Test that time progresses between events."""
        start_time = datetime(2024, 1, 15, 12, 0, 0)
        simulator = DroneSimulator(start_time=start_time)

        event1 = simulator.generate_event()
        event2 = simulator.generate_event()

        assert event2.frame.timestamp > event1.frame.timestamp

    def test_demo_scenario_generation(self):
        """Test curated demo scenario generation."""
        simulator = DroneSimulator()
        events = simulator.generate_demo_scenario()

        assert len(events) > 0
        # Should include the blue Ford F150 scenario
        descriptions = [e.frame.description for e in events]
        assert any("Ford" in d or "truck" in d.lower() for d in descriptions)

    def test_location_cycling(self):
        """Test that simulator cycles through locations."""
        simulator = DroneSimulator()
        locations_seen = set()

        for _ in range(20):
            event = simulator.generate_event()
            locations_seen.add(event.frame.location_name)

        # Should have seen multiple locations
        assert len(locations_seen) > 1


class TestIntegration:
    """Integration tests for simulator."""

    def test_full_simulation_cycle(self):
        """Test a full simulation cycle."""
        simulator = DroneSimulator()

        # Generate events
        events = list(simulator.generate_stream(num_events=5))

        # Verify data integrity
        for event in events:
            assert event.telemetry.timestamp == event.frame.timestamp
            assert isinstance(event.to_dict(), dict)

    def test_special_events_inclusion(self):
        """Test that special events are included when requested."""
        simulator = DroneSimulator()

        # Generate many events with special scenarios
        events = list(simulator.generate_stream(num_events=50, include_special=True))

        # At least some should have non-empty object lists
        events_with_objects = [e for e in events if e.frame.detected_objects]
        assert len(events_with_objects) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
