"""
Drone Telemetry and Video Frame Simulator

Generates realistic simulated data for testing the security analyst agent.
"""

import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Generator
from dataclasses import dataclass, field, asdict
import json

from .config import (
    SIMULATION_CONFIG,
    VEHICLE_COLORS,
    VEHICLE_MAKES,
    VEHICLE_TYPES,
)


@dataclass
class TelemetryData:
    """Drone telemetry data structure."""
    timestamp: datetime
    drone_id: str
    location_name: str
    location_zone: str
    latitude: float
    longitude: float
    altitude: float
    battery_percent: int
    status: str

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "drone_id": self.drone_id,
            "location": {
                "name": self.location_name,
                "zone": self.location_zone,
                "lat": self.latitude,
                "lon": self.longitude
            },
            "altitude": self.altitude,
            "battery": self.battery_percent,
            "status": self.status
        }

    def __str__(self) -> str:
        return f"[{self.timestamp.strftime('%H:%M:%S')}] Drone at {self.location_name} ({self.location_zone})"


@dataclass
class VideoFrame:
    """Simulated video frame with description."""
    frame_id: int
    timestamp: datetime
    description: str
    detected_objects: List[Dict]
    location_name: str
    location_zone: str

    def to_dict(self) -> Dict:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp.isoformat(),
            "description": self.description,
            "detected_objects": self.detected_objects,
            "location": {
                "name": self.location_name,
                "zone": self.location_zone
            }
        }

    def __str__(self) -> str:
        return f"Frame {self.frame_id}: {self.description}"


@dataclass
class SimulatedEvent:
    """Combined telemetry and video frame event."""
    telemetry: TelemetryData
    frame: VideoFrame

    def to_dict(self) -> Dict:
        return {
            "telemetry": self.telemetry.to_dict(),
            "frame": self.frame.to_dict()
        }


class ScenarioGenerator:
    """Generates realistic security scenarios for simulation."""

    # Scenario templates for different times of day
    SCENARIOS = {
        "morning": [
            {
                "description": "{color} {make} {vehicle_type} arriving at {location}",
                "objects": [{"type": "vehicle", "subtype": "{vehicle_type}", "color": "{color}", "make": "{make}"}],
                "weight": 30
            },
            {
                "description": "Worker in safety vest walking near {location}",
                "objects": [{"type": "person", "subtype": "worker", "attributes": {"clothing": "safety vest"}}],
                "weight": 25
            },
            {
                "description": "Delivery truck backing into loading dock",
                "objects": [{"type": "vehicle", "subtype": "truck", "color": "white", "attributes": {"purpose": "delivery"}}],
                "weight": 20
            },
            {
                "description": "Empty area, no activity detected at {location}",
                "objects": [],
                "weight": 25
            },
        ],
        "afternoon": [
            {
                "description": "{color} {make} sedan parked at {location}",
                "objects": [{"type": "vehicle", "subtype": "sedan", "color": "{color}", "make": "{make}"}],
                "weight": 25
            },
            {
                "description": "Two people conversing near {location}",
                "objects": [
                    {"type": "person", "subtype": "individual", "id": "person_1"},
                    {"type": "person", "subtype": "individual", "id": "person_2"}
                ],
                "weight": 20
            },
            {
                "description": "Forklift operating near warehouse",
                "objects": [{"type": "vehicle", "subtype": "forklift", "attributes": {"industrial": True}}],
                "weight": 15
            },
            {
                "description": "Security guard on patrol at {location}",
                "objects": [{"type": "person", "subtype": "security", "attributes": {"uniform": True}}],
                "weight": 15
            },
            {
                "description": "Clear view of {location}, no movement",
                "objects": [],
                "weight": 25
            },
        ],
        "evening": [
            {
                "description": "{color} {vehicle_type} exiting through main gate",
                "objects": [{"type": "vehicle", "subtype": "{vehicle_type}", "color": "{color}"}],
                "weight": 30
            },
            {
                "description": "Last worker leaving for the day at {location}",
                "objects": [{"type": "person", "subtype": "worker"}],
                "weight": 25
            },
            {
                "description": "Automated lights turning on at {location}",
                "objects": [],
                "weight": 25
            },
            {
                "description": "Stray cat near {location}",
                "objects": [{"type": "animal", "subtype": "cat"}],
                "weight": 20
            },
        ],
        "night": [
            {
                "description": "Dark area at {location}, motion sensors active",
                "objects": [],
                "weight": 35
            },
            {
                "description": "Person walking near {location}",
                "objects": [{"type": "person", "subtype": "unknown", "attributes": {"suspicious": True}}],
                "weight": 15
            },
            {
                "description": "Security patrol vehicle at {location}",
                "objects": [{"type": "vehicle", "subtype": "SUV", "attributes": {"security": True}}],
                "weight": 20
            },
            {
                "description": "Person loitering near fence at {location}",
                "objects": [{"type": "person", "subtype": "unknown", "attributes": {"loitering": True}}],
                "weight": 10
            },
            {
                "description": "Wildlife (deer) crossing near {location}",
                "objects": [{"type": "animal", "subtype": "deer"}],
                "weight": 10
            },
            {
                "description": "Unknown vehicle with headlights off near {location}",
                "objects": [{"type": "vehicle", "subtype": "unknown", "attributes": {"suspicious": True, "lights_off": True}}],
                "weight": 10
            },
        ],
        "special": [
            {
                "description": "Blue Ford F150 pickup truck entering through main gate",
                "objects": [{"type": "vehicle", "subtype": "pickup truck", "color": "blue", "make": "Ford", "model": "F150"}],
                "weight": 1,
                "force_location": "Main Gate"
            },
            {
                "description": "Person in dark clothing loitering at main gate",
                "objects": [{"type": "person", "subtype": "unknown", "attributes": {"clothing": "dark", "loitering": True}}],
                "weight": 1,
                "force_location": "Main Gate"
            },
            {
                "description": "Same blue Ford F150 now at parking lot",
                "objects": [{"type": "vehicle", "subtype": "pickup truck", "color": "blue", "make": "Ford", "model": "F150", "recurring": True}],
                "weight": 1,
                "force_location": "Parking Lot"
            },
            {
                "description": "Red Toyota sedan at garage entrance",
                "objects": [{"type": "vehicle", "subtype": "sedan", "color": "red", "make": "Toyota"}],
                "weight": 1,
                "force_location": "Garage"
            },
        ]
    }

    @classmethod
    def get_time_period(cls, hour: int) -> str:
        """Determine time period based on hour."""
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 22:
            return "evening"
        else:
            return "night"

    @classmethod
    def generate_scenario(cls, hour: int, location: Dict, force_special: bool = False) -> Dict:
        """Generate a scenario based on time and location."""
        if force_special and cls.SCENARIOS["special"]:
            scenario = random.choice(cls.SCENARIOS["special"])
        else:
            time_period = cls.get_time_period(hour)
            scenarios = cls.SCENARIOS[time_period]
            weights = [s["weight"] for s in scenarios]
            scenario = random.choices(scenarios, weights=weights, k=1)[0]

        # Fill in template variables
        color = random.choice(VEHICLE_COLORS)
        make = random.choice(VEHICLE_MAKES)
        vehicle_type = random.choice(VEHICLE_TYPES)
        location_name = scenario.get("force_location", location["name"])

        description = scenario["description"].format(
            color=color,
            make=make,
            vehicle_type=vehicle_type,
            location=location_name
        )

        # Process objects
        objects = []
        for obj in scenario["objects"]:
            processed_obj = {}
            for key, value in obj.items():
                if isinstance(value, str):
                    processed_obj[key] = value.format(
                        color=color,
                        make=make,
                        vehicle_type=vehicle_type
                    )
                else:
                    processed_obj[key] = value
            objects.append(processed_obj)

        return {
            "description": description,
            "objects": objects,
            "location_name": location_name
        }


class DroneSimulator:
    """
    Simulates drone telemetry and video frame data.

    This simulator generates realistic data streams for testing
    the security analyst agent without requiring actual drone hardware.
    """

    def __init__(self, start_time: Optional[datetime] = None):
        """
        Initialize the simulator.

        Args:
            start_time: Starting timestamp for simulation. Defaults to current time.
        """
        self.config = SIMULATION_CONFIG
        self.current_time = start_time or datetime.now()
        self.frame_counter = 0
        self.battery_level = 100
        self.current_location_idx = 0
        self.tracked_objects: Dict[str, List] = {}  # Track recurring objects

    def _get_next_location(self) -> Dict:
        """Get the next patrol location in sequence."""
        location = self.config["patrol_locations"][self.current_location_idx]
        self.current_location_idx = (self.current_location_idx + 1) % len(self.config["patrol_locations"])
        return location

    def _update_battery(self) -> int:
        """Simulate battery drain."""
        drain = random.uniform(0.1, 0.3)
        self.battery_level = max(0, self.battery_level - drain)
        return int(self.battery_level)

    def generate_telemetry(self, location: Dict) -> TelemetryData:
        """Generate telemetry data for current position."""
        return TelemetryData(
            timestamp=self.current_time,
            drone_id=self.config["drone_id"],
            location_name=location["name"],
            location_zone=location["zone"],
            latitude=location["lat"] + random.uniform(-0.0001, 0.0001),
            longitude=location["lon"] + random.uniform(-0.0001, 0.0001),
            altitude=random.uniform(30, 60),
            battery_percent=self._update_battery(),
            status="patrolling"
        )

    def generate_frame(self, location: Dict, force_special: bool = False) -> VideoFrame:
        """Generate a simulated video frame with detected objects."""
        self.frame_counter += 1

        scenario = ScenarioGenerator.generate_scenario(
            hour=self.current_time.hour,
            location=location,
            force_special=force_special
        )

        return VideoFrame(
            frame_id=self.frame_counter,
            timestamp=self.current_time,
            description=scenario["description"],
            detected_objects=scenario["objects"],
            location_name=scenario.get("location_name", location["name"]),
            location_zone=location["zone"]
        )

    def generate_event(self, force_special: bool = False) -> SimulatedEvent:
        """Generate a complete event with telemetry and frame."""
        location = self._get_next_location()

        telemetry = self.generate_telemetry(location)
        frame = self.generate_frame(location, force_special)

        # Advance time
        self.current_time += timedelta(seconds=self.config["frame_interval_seconds"])

        return SimulatedEvent(telemetry=telemetry, frame=frame)

    def generate_stream(self, num_events: int = 50, include_special: bool = True) -> Generator[SimulatedEvent, None, None]:
        """
        Generate a stream of simulated events.

        Args:
            num_events: Number of events to generate
            include_special: Whether to include special security scenarios

        Yields:
            SimulatedEvent objects
        """
        special_indices = set()
        if include_special:
            # Insert special events at random positions
            num_special = min(5, num_events // 10)
            special_indices = set(random.sample(range(num_events), num_special))

        for i in range(num_events):
            force_special = i in special_indices
            yield self.generate_event(force_special=force_special)

    def generate_demo_scenario(self) -> List[SimulatedEvent]:
        """
        Generate a curated demo scenario showcasing key features.

        Returns:
            List of events demonstrating various security scenarios
        """
        events = []

        # Reset to controlled start time (noon)
        self.current_time = datetime.now().replace(hour=12, minute=0, second=0)

        demo_scenarios = [
            # Normal morning activity
            ("Main Gate", "Blue Ford F150 pickup truck entering through main gate"),
            ("Parking Lot", "Blue Ford F150 parking in visitor area"),
            ("Office Building", "Worker in safety vest entering office building"),

            # Afternoon activity
            ("Warehouse", "Forklift moving pallets near warehouse"),
            ("Loading Dock", "Delivery truck at loading dock"),
            ("Garage", "Red Toyota sedan at garage entrance"),

            # Same vehicle returning (pattern detection)
            ("Main Gate", "Blue Ford F150 pickup truck entering through main gate again"),
            ("Parking Lot", "Same blue Ford F150 now at parking lot - second visit today"),

            # Evening transition
            ("Office Building", "Workers leaving office building for the day"),
            ("Main Gate", "Multiple vehicles exiting through main gate"),

            # Night scenarios (shift time to night)
            ("Back Fence", "Dark area at back fence, motion sensors active"),
            ("Main Gate", "Person in dark clothing near main gate at night"),  # Should trigger alert
            ("Warehouse", "Security patrol vehicle checking warehouse"),
        ]

        for location_name, description in demo_scenarios:
            # Find location
            location = next(
                (loc for loc in self.config["patrol_locations"] if loc["name"] == location_name),
                self.config["patrol_locations"][0]
            )

            telemetry = self.generate_telemetry(location)

            # Parse objects from description
            objects = self._parse_description_to_objects(description)

            self.frame_counter += 1
            frame = VideoFrame(
                frame_id=self.frame_counter,
                timestamp=self.current_time,
                description=description,
                detected_objects=objects,
                location_name=location_name,
                location_zone=location["zone"]
            )

            events.append(SimulatedEvent(telemetry=telemetry, frame=frame))

            # Advance time (more for transitions)
            if "night" in description.lower() or "evening" in description.lower():
                self.current_time += timedelta(hours=2)
            else:
                self.current_time += timedelta(minutes=random.randint(5, 15))

        return events

    def _parse_description_to_objects(self, description: str) -> List[Dict]:
        """Parse a description string to extract object data."""
        objects = []
        desc_lower = description.lower()

        # Vehicle detection
        for vehicle_type in ["truck", "sedan", "suv", "van", "pickup", "forklift"]:
            if vehicle_type in desc_lower:
                obj = {"type": "vehicle", "subtype": vehicle_type}

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

                # Check for recurring
                if "same" in desc_lower or "again" in desc_lower or "second" in desc_lower:
                    obj["recurring"] = True

                objects.append(obj)
                break

        # Person detection
        if any(word in desc_lower for word in ["person", "worker", "workers", "people"]):
            obj = {"type": "person"}

            if "worker" in desc_lower:
                obj["subtype"] = "worker"
            elif "security" in desc_lower:
                obj["subtype"] = "security"
            else:
                obj["subtype"] = "unknown"

            if "dark clothing" in desc_lower or "night" in desc_lower:
                obj["attributes"] = {"suspicious": True}

            objects.append(obj)

        return objects


def run_simulation_demo():
    """Run a demonstration of the simulator."""
    print("=" * 60)
    print("DRONE SECURITY SIMULATOR - DEMO")
    print("=" * 60)

    simulator = DroneSimulator()
    events = simulator.generate_demo_scenario()

    for event in events:
        print(f"\n{event.telemetry}")
        print(f"  {event.frame}")
        print(f"  Objects: {event.frame.detected_objects}")

    print(f"\n{'=' * 60}")
    print(f"Generated {len(events)} demo events")
    print("=" * 60)


if __name__ == "__main__":
    run_simulation_demo()
