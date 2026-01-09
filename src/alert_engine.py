"""
Security Alert Engine

Evaluates security rules against detected events and generates alerts.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from .config import ALERT_RULES
from .database import SecurityDatabase, AlertRecord


class AlertPriority(Enum):
    """Alert priority levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    """Security alert structure."""
    rule_id: str
    rule_name: str
    priority: AlertPriority
    timestamp: datetime
    frame_id: int
    location: str
    description: str
    details: Dict

    def to_dict(self) -> Dict:
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "frame_id": self.frame_id,
            "location": self.location,
            "description": self.description,
            "details": self.details
        }

    def __str__(self) -> str:
        return f"[ALERT - {self.priority.value}] {self.timestamp.strftime('%H:%M:%S')} | {self.description} | Location: {self.location}"


class AlertEngine:
    """
    Evaluates security rules and generates alerts.

    Processes detected objects and events against predefined rules
    to determine if alerts should be triggered.
    """

    def __init__(self, database: Optional[SecurityDatabase] = None):
        """
        Initialize the alert engine.

        Args:
            database: Optional database for persisting alerts
        """
        self.database = database
        self.rules = self._load_rules()
        self.recent_alerts: List[Alert] = []
        self.alert_cooldowns: Dict[str, datetime] = {}  # Prevent duplicate alerts
        self.context_window: List[Dict] = []  # Recent events for pattern detection
        self.cooldown_seconds = 300  # 5 minute cooldown per rule per location

    def _load_rules(self) -> List[Dict]:
        """Load alert rules from configuration."""
        return ALERT_RULES

    def evaluate(
        self,
        frame_id: int,
        timestamp: datetime,
        detected_objects: List[Dict],
        location: Dict,
        description: str
    ) -> List[Alert]:
        """
        Evaluate all rules against a detection event.

        Args:
            frame_id: Frame identifier
            timestamp: Event timestamp
            detected_objects: List of detected objects
            location: Location data
            description: Frame description

        Returns:
            List of triggered alerts
        """
        triggered_alerts = []

        # Add to context window
        self._update_context({
            "frame_id": frame_id,
            "timestamp": timestamp,
            "objects": detected_objects,
            "location": location,
            "description": description
        })

        for rule in self.rules:
            alert = self._evaluate_rule(
                rule, frame_id, timestamp, detected_objects, location, description
            )
            if alert:
                # Check cooldown
                cooldown_key = f"{rule['id']}_{location.get('name', 'unknown')}"
                if self._check_cooldown(cooldown_key):
                    triggered_alerts.append(alert)
                    self._set_cooldown(cooldown_key)

                    # Persist to database if available
                    if self.database:
                        self._persist_alert(alert)

        return triggered_alerts

    def _evaluate_rule(
        self,
        rule: Dict,
        frame_id: int,
        timestamp: datetime,
        detected_objects: List[Dict],
        location: Dict,
        description: str
    ) -> Optional[Alert]:
        """Evaluate a single rule."""
        conditions = rule.get("conditions", {})
        rule_id = rule["id"]

        # R001: Night Activity (person detected at night)
        if rule_id == "R001":
            return self._check_night_activity(rule, frame_id, timestamp, detected_objects, location)

        # R002: Loitering Detection
        elif rule_id == "R002":
            return self._check_loitering(rule, frame_id, timestamp, detected_objects, location)

        # R003: Perimeter Activity
        elif rule_id == "R003":
            return self._check_perimeter_activity(rule, frame_id, timestamp, detected_objects, location)

        # R004: Repeat Vehicle Entry
        elif rule_id == "R004":
            return self._check_repeat_vehicle(rule, frame_id, timestamp, detected_objects, location)

        # R005: Unknown Vehicle in Restricted Area
        elif rule_id == "R005":
            return self._check_unknown_vehicle(rule, frame_id, timestamp, detected_objects, location)

        return None

    def _check_night_activity(
        self,
        rule: Dict,
        frame_id: int,
        timestamp: datetime,
        objects: List[Dict],
        location: Dict
    ) -> Optional[Alert]:
        """Check for person detected during night hours (00:00-05:00)."""
        hour = timestamp.hour

        # Check if it's night time
        if not (0 <= hour < 5):
            return None

        # Check for person detection
        persons = [obj for obj in objects if obj.get("type") == "person"]
        if not persons:
            return None

        # Generate alert
        message = rule["message_template"].format(
            location=location.get("name", "Unknown"),
            time=timestamp.strftime("%H:%M")
        )

        return Alert(
            rule_id=rule["id"],
            rule_name=rule["name"],
            priority=AlertPriority[rule["priority"]],
            timestamp=timestamp,
            frame_id=frame_id,
            location=location.get("name", "Unknown"),
            description=message,
            details={"persons_detected": len(persons), "hour": hour}
        )

    def _check_loitering(
        self,
        rule: Dict,
        frame_id: int,
        timestamp: datetime,
        objects: List[Dict],
        location: Dict
    ) -> Optional[Alert]:
        """Check for loitering (same person in same zone for extended period)."""
        # Check for person with loitering attribute
        for obj in objects:
            if obj.get("type") == "person":
                attributes = obj.get("attributes", {})
                if attributes.get("loitering") or attributes.get("suspicious"):
                    message = rule["message_template"].format(
                        location=location.get("name", "Unknown")
                    )

                    return Alert(
                        rule_id=rule["id"],
                        rule_name=rule["name"],
                        priority=AlertPriority[rule["priority"]],
                        timestamp=timestamp,
                        frame_id=frame_id,
                        location=location.get("name", "Unknown"),
                        description=message,
                        details={"object_attributes": attributes}
                    )

        # Also check context window for repeated person sightings in same location
        person_sightings = self._get_context_sightings("person", location.get("name"), minutes=5)
        if len(person_sightings) >= 3:
            message = rule["message_template"].format(
                location=location.get("name", "Unknown")
            )

            return Alert(
                rule_id=rule["id"],
                rule_name=rule["name"],
                priority=AlertPriority[rule["priority"]],
                timestamp=timestamp,
                frame_id=frame_id,
                location=location.get("name", "Unknown"),
                description=message,
                details={"sighting_count": len(person_sightings)}
            )

        return None

    def _check_perimeter_activity(
        self,
        rule: Dict,
        frame_id: int,
        timestamp: datetime,
        objects: List[Dict],
        location: Dict
    ) -> Optional[Alert]:
        """Check for activity in perimeter zone."""
        zone = location.get("zone", "")

        if zone != "perimeter":
            return None

        # Check for person or vehicle in perimeter
        relevant_objects = [
            obj for obj in objects
            if obj.get("type") in ["person", "vehicle"]
        ]

        if not relevant_objects:
            return None

        message = rule["message_template"].format(
            location=location.get("name", "Unknown")
        )

        return Alert(
            rule_id=rule["id"],
            rule_name=rule["name"],
            priority=AlertPriority[rule["priority"]],
            timestamp=timestamp,
            frame_id=frame_id,
            location=location.get("name", "Unknown"),
            description=message,
            details={
                "objects_detected": [
                    {"type": obj.get("type"), "subtype": obj.get("subtype")}
                    for obj in relevant_objects
                ]
            }
        )

    def _check_repeat_vehicle(
        self,
        rule: Dict,
        frame_id: int,
        timestamp: datetime,
        objects: List[Dict],
        location: Dict
    ) -> Optional[Alert]:
        """Check for vehicles with repeated entries."""
        vehicles = [obj for obj in objects if obj.get("type") == "vehicle"]

        for vehicle in vehicles:
            # Check for recurring attribute from tracker
            tracking = vehicle.get("tracking", {})
            attributes = vehicle.get("attributes", {})

            if tracking.get("recurring") or attributes.get("recurring"):
                sightings = tracking.get("total_sightings", 2)

                # Build vehicle description
                vehicle_desc = self._build_vehicle_description(vehicle)

                message = rule["message_template"].format(
                    details=vehicle_desc,
                    count=sightings
                )

                return Alert(
                    rule_id=rule["id"],
                    rule_name=rule["name"],
                    priority=AlertPriority[rule["priority"]],
                    timestamp=timestamp,
                    frame_id=frame_id,
                    location=location.get("name", "Unknown"),
                    description=message,
                    details={
                        "vehicle": vehicle_desc,
                        "total_sightings": sightings,
                        "tracking_id": tracking.get("id")
                    }
                )

        return None

    def _check_unknown_vehicle(
        self,
        rule: Dict,
        frame_id: int,
        timestamp: datetime,
        objects: List[Dict],
        location: Dict
    ) -> Optional[Alert]:
        """Check for unknown vehicles in restricted areas."""
        zone = location.get("zone", "")
        restricted_zones = ["storage", "operations"]

        if zone not in restricted_zones:
            return None

        vehicles = [obj for obj in objects if obj.get("type") == "vehicle"]

        for vehicle in vehicles:
            attributes = vehicle.get("attributes", {})

            # Check if vehicle appears unknown or suspicious
            if attributes.get("suspicious") or attributes.get("unknown"):
                message = rule["message_template"].format(
                    location=location.get("name", "Unknown")
                )

                return Alert(
                    rule_id=rule["id"],
                    rule_name=rule["name"],
                    priority=AlertPriority[rule["priority"]],
                    timestamp=timestamp,
                    frame_id=frame_id,
                    location=location.get("name", "Unknown"),
                    description=message,
                    details={"vehicle": vehicle, "zone": zone}
                )

        return None

    def _build_vehicle_description(self, vehicle: Dict) -> str:
        """Build a human-readable vehicle description."""
        parts = []

        if vehicle.get("color"):
            parts.append(vehicle["color"])
        if vehicle.get("make"):
            parts.append(vehicle["make"])
        if vehicle.get("model"):
            parts.append(vehicle["model"])
        if vehicle.get("subtype"):
            parts.append(vehicle["subtype"])

        return " ".join(parts) if parts else "Unknown vehicle"

    def _update_context(self, event: Dict):
        """Update the context window with a new event."""
        self.context_window.append(event)

        # Keep only last 50 events or last 30 minutes
        cutoff = datetime.now() - timedelta(minutes=30)
        self.context_window = [
            e for e in self.context_window
            if e["timestamp"] > cutoff
        ][-50:]

    def _get_context_sightings(
        self,
        object_type: str,
        location: str,
        minutes: int = 5
    ) -> List[Dict]:
        """Get recent sightings of an object type at a location."""
        cutoff = datetime.now() - timedelta(minutes=minutes)

        sightings = []
        for event in self.context_window:
            if event["timestamp"] < cutoff:
                continue
            if event["location"].get("name") != location:
                continue

            for obj in event["objects"]:
                if obj.get("type") == object_type:
                    sightings.append(event)
                    break

        return sightings

    def _check_cooldown(self, key: str) -> bool:
        """Check if an alert can be triggered (not in cooldown)."""
        if key not in self.alert_cooldowns:
            return True

        cooldown_until = self.alert_cooldowns[key]
        return datetime.now() > cooldown_until

    def _set_cooldown(self, key: str):
        """Set cooldown for an alert key."""
        self.alert_cooldowns[key] = datetime.now() + timedelta(seconds=self.cooldown_seconds)

    def _persist_alert(self, alert: Alert):
        """Persist an alert to the database."""
        if not self.database:
            return

        alert_record = AlertRecord(
            alert_id=None,
            timestamp=alert.timestamp,
            frame_id=alert.frame_id,
            rule_id=alert.rule_id,
            priority=alert.priority.value,
            description=alert.description,
            location=alert.location,
            status="active"
        )

        self.database.log_alert(alert_record)

    def get_recent_alerts(self, limit: int = 10) -> List[Alert]:
        """Get recent alerts."""
        return self.recent_alerts[-limit:]

    def get_alerts_by_priority(self, priority: AlertPriority) -> List[Alert]:
        """Get alerts filtered by priority."""
        return [a for a in self.recent_alerts if a.priority == priority]

    def get_alert_statistics(self) -> Dict:
        """Get statistics about generated alerts."""
        total = len(self.recent_alerts)

        by_priority = {p.value: 0 for p in AlertPriority}
        by_rule = {}
        by_location = {}

        for alert in self.recent_alerts:
            by_priority[alert.priority.value] += 1
            by_rule[alert.rule_id] = by_rule.get(alert.rule_id, 0) + 1
            by_location[alert.location] = by_location.get(alert.location, 0) + 1

        return {
            "total_alerts": total,
            "by_priority": by_priority,
            "by_rule": by_rule,
            "by_location": by_location
        }


class AlertFormatter:
    """Formats alerts for different output types."""

    @staticmethod
    def format_console(alert: Alert) -> str:
        """Format alert for console output."""
        priority_colors = {
            AlertPriority.LOW: "\033[94m",      # Blue
            AlertPriority.MEDIUM: "\033[93m",   # Yellow
            AlertPriority.HIGH: "\033[91m",     # Red
            AlertPriority.CRITICAL: "\033[95m"  # Magenta
        }
        reset = "\033[0m"

        color = priority_colors.get(alert.priority, "")

        return (
            f"{color}[ALERT - {alert.priority.value}]{reset} "
            f"{alert.timestamp.strftime('%H:%M:%S')} | "
            f"{alert.description} | "
            f"Location: {alert.location}"
        )

    @staticmethod
    def format_log(alert: Alert) -> str:
        """Format alert for log file."""
        return (
            f"[{alert.timestamp.isoformat()}] "
            f"[{alert.priority.value}] "
            f"[{alert.rule_id}] "
            f"{alert.description} "
            f"at {alert.location}"
        )

    @staticmethod
    def format_json(alert: Alert) -> str:
        """Format alert as JSON string."""
        import json
        return json.dumps(alert.to_dict(), indent=2)


if __name__ == "__main__":
    # Test the alert engine
    from datetime import datetime

    print("=" * 60)
    print("ALERT ENGINE TEST")
    print("=" * 60)

    engine = AlertEngine()

    # Test scenarios
    test_cases = [
        {
            "name": "Night activity",
            "timestamp": datetime.now().replace(hour=2, minute=30),
            "objects": [{"type": "person", "subtype": "unknown"}],
            "location": {"name": "Main Gate", "zone": "perimeter"}
        },
        {
            "name": "Loitering person",
            "timestamp": datetime.now().replace(hour=14, minute=0),
            "objects": [{"type": "person", "subtype": "unknown", "attributes": {"loitering": True}}],
            "location": {"name": "Warehouse", "zone": "storage"}
        },
        {
            "name": "Perimeter vehicle",
            "timestamp": datetime.now().replace(hour=10, minute=0),
            "objects": [{"type": "vehicle", "subtype": "truck", "color": "blue"}],
            "location": {"name": "Back Fence", "zone": "perimeter"}
        },
        {
            "name": "Recurring vehicle",
            "timestamp": datetime.now().replace(hour=12, minute=0),
            "objects": [{"type": "vehicle", "subtype": "truck", "color": "blue", "make": "Ford", "tracking": {"recurring": True, "total_sightings": 3}}],
            "location": {"name": "Main Gate", "zone": "perimeter"}
        },
    ]

    for i, test in enumerate(test_cases):
        print(f"\nTest {i + 1}: {test['name']}")
        alerts = engine.evaluate(
            frame_id=i + 1,
            timestamp=test["timestamp"],
            detected_objects=test["objects"],
            location=test["location"],
            description=f"Test frame for {test['name']}"
        )

        if alerts:
            for alert in alerts:
                print(f"  {AlertFormatter.format_console(alert)}")
        else:
            print("  No alerts triggered")

    print(f"\n{'=' * 60}")
    stats = engine.get_alert_statistics()
    print(f"Alert Statistics: {stats}")
