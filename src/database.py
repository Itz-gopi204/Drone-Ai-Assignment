"""
Frame Indexing Database System

Provides storage and querying capabilities for video frames,
detections, and alerts.
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass

from .config import DB_PATH


@dataclass
class FrameRecord:
    """Record structure for indexed frames."""
    frame_id: int
    timestamp: datetime
    location_name: str
    location_zone: str
    latitude: float
    longitude: float
    description: str
    objects: List[Dict]
    alert_triggered: bool = False

    def to_dict(self) -> Dict:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            "location_name": self.location_name,
            "location_zone": self.location_zone,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "description": self.description,
            "objects": self.objects,
            "alert_triggered": self.alert_triggered
        }


@dataclass
class AlertRecord:
    """Record structure for alerts."""
    alert_id: Optional[int]
    timestamp: datetime
    frame_id: int
    rule_id: str
    priority: str
    description: str
    location: str
    status: str = "active"

    def to_dict(self) -> Dict:
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            "frame_id": self.frame_id,
            "rule_id": self.rule_id,
            "priority": self.priority,
            "description": self.description,
            "location": self.location,
            "status": self.status
        }


@dataclass
class DetectionRecord:
    """Record structure for object detections."""
    detection_id: Optional[int]
    timestamp: datetime
    frame_id: int
    object_type: str
    object_subtype: str
    object_attributes: Dict
    location_zone: str
    confidence: float = 1.0


class SecurityDatabase:
    """
    SQLite database for storing and querying security data.

    Provides frame indexing, alert storage, and detection logging
    with support for complex queries.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the database connection.

        Args:
            db_path: Path to SQLite database file. Uses default if not specified.
        """
        self.db_path = db_path or DB_PATH
        self.connection: Optional[sqlite3.Connection] = None
        self._initialize_database()

    def _initialize_database(self):
        """Create database tables if they don't exist."""
        self.connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.connection.row_factory = sqlite3.Row

        cursor = self.connection.cursor()

        # Frame index table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS frame_index (
                frame_id INTEGER PRIMARY KEY,
                timestamp TEXT NOT NULL,
                location_name TEXT,
                location_zone TEXT,
                latitude REAL,
                longitude REAL,
                description TEXT,
                objects TEXT,
                alert_triggered INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                frame_id INTEGER,
                rule_id TEXT,
                priority TEXT,
                description TEXT,
                location TEXT,
                status TEXT DEFAULT 'active',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (frame_id) REFERENCES frame_index(frame_id)
            )
        """)

        # Detections table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                frame_id INTEGER,
                object_type TEXT,
                object_subtype TEXT,
                object_attributes TEXT,
                location_zone TEXT,
                confidence REAL DEFAULT 1.0,
                FOREIGN KEY (frame_id) REFERENCES frame_index(frame_id)
            )
        """)

        # Create indexes for efficient querying
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_frame_timestamp ON frame_index(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_frame_location ON frame_index(location_zone)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alert_timestamp ON alerts(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alert_priority ON alerts(priority)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_detection_type ON detections(object_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_detection_frame ON detections(frame_id)")

        self.connection.commit()

    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None

    # ==================== Frame Operations ====================

    def index_frame(self, frame: FrameRecord) -> int:
        """
        Index a video frame in the database.

        Args:
            frame: FrameRecord to store

        Returns:
            frame_id of the indexed frame
        """
        cursor = self.connection.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO frame_index
            (frame_id, timestamp, location_name, location_zone, latitude, longitude, description, objects, alert_triggered)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            frame.frame_id,
            frame.timestamp.isoformat() if isinstance(frame.timestamp, datetime) else frame.timestamp,
            frame.location_name,
            frame.location_zone,
            frame.latitude,
            frame.longitude,
            frame.description,
            json.dumps(frame.objects),
            1 if frame.alert_triggered else 0
        ))

        self.connection.commit()
        return frame.frame_id

    def get_frame(self, frame_id: int) -> Optional[FrameRecord]:
        """Get a specific frame by ID."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM frame_index WHERE frame_id = ?", (frame_id,))
        row = cursor.fetchone()

        if row:
            return self._row_to_frame_record(row)
        return None

    def _row_to_frame_record(self, row: sqlite3.Row) -> FrameRecord:
        """Convert database row to FrameRecord."""
        return FrameRecord(
            frame_id=row["frame_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            location_name=row["location_name"],
            location_zone=row["location_zone"],
            latitude=row["latitude"],
            longitude=row["longitude"],
            description=row["description"],
            objects=json.loads(row["objects"]) if row["objects"] else [],
            alert_triggered=bool(row["alert_triggered"])
        )

    # ==================== Query Operations ====================

    def query_frames_by_time(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[FrameRecord]:
        """Query frames within a time range."""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT * FROM frame_index
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp ASC
        """, (start_time.isoformat(), end_time.isoformat()))

        return [self._row_to_frame_record(row) for row in cursor.fetchall()]

    def query_frames_by_location(self, location_zone: str) -> List[FrameRecord]:
        """Query frames by location zone."""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT * FROM frame_index
            WHERE location_zone = ?
            ORDER BY timestamp DESC
        """, (location_zone,))

        return [self._row_to_frame_record(row) for row in cursor.fetchall()]

    def query_frames_by_object_type(self, object_type: str) -> List[FrameRecord]:
        """Query frames containing a specific object type."""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT * FROM frame_index
            WHERE objects LIKE ?
            ORDER BY timestamp DESC
        """, (f'%"type": "{object_type}"%',))

        # Also try alternate JSON format
        results = [self._row_to_frame_record(row) for row in cursor.fetchall()]

        if not results:
            cursor.execute("""
                SELECT * FROM frame_index
                WHERE objects LIKE ?
                ORDER BY timestamp DESC
            """, (f'%"type":"{object_type}"%',))
            results = [self._row_to_frame_record(row) for row in cursor.fetchall()]

        return results

    def query_frames_by_description(self, search_term: str) -> List[FrameRecord]:
        """Search frames by description text."""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT * FROM frame_index
            WHERE description LIKE ?
            ORDER BY timestamp DESC
        """, (f'%{search_term}%',))

        return [self._row_to_frame_record(row) for row in cursor.fetchall()]

    def query_frames_complex(
        self,
        object_type: Optional[str] = None,
        location_zone: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        description_contains: Optional[str] = None,
        limit: int = 100
    ) -> List[FrameRecord]:
        """
        Complex query with multiple filters.

        Args:
            object_type: Filter by object type (e.g., "vehicle", "person")
            location_zone: Filter by zone (e.g., "perimeter", "parking")
            start_time: Filter by start time
            end_time: Filter by end time
            description_contains: Search term in description
            limit: Maximum results to return

        Returns:
            List of matching FrameRecords
        """
        conditions = []
        params = []

        if object_type:
            conditions.append("objects LIKE ?")
            params.append(f'%"{object_type}"%')

        if location_zone:
            conditions.append("location_zone = ?")
            params.append(location_zone)

        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time.isoformat())

        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time.isoformat())

        if description_contains:
            conditions.append("description LIKE ?")
            params.append(f'%{description_contains}%')

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        cursor = self.connection.cursor()
        cursor.execute(f"""
            SELECT * FROM frame_index
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT ?
        """, params)

        return [self._row_to_frame_record(row) for row in cursor.fetchall()]

    def get_all_frames(self, limit: int = 1000) -> List[FrameRecord]:
        """Get all indexed frames."""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT * FROM frame_index
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

        return [self._row_to_frame_record(row) for row in cursor.fetchall()]

    # ==================== Alert Operations ====================

    def log_alert(self, alert: AlertRecord) -> int:
        """
        Log a security alert.

        Args:
            alert: AlertRecord to store

        Returns:
            alert_id of the logged alert
        """
        cursor = self.connection.cursor()

        cursor.execute("""
            INSERT INTO alerts (timestamp, frame_id, rule_id, priority, description, location, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            alert.timestamp.isoformat() if isinstance(alert.timestamp, datetime) else alert.timestamp,
            alert.frame_id,
            alert.rule_id,
            alert.priority,
            alert.description,
            alert.location,
            alert.status
        ))

        self.connection.commit()
        return cursor.lastrowid

    def get_alerts(
        self,
        priority: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[AlertRecord]:
        """Get alerts with optional filtering."""
        conditions = []
        params = []

        if priority:
            conditions.append("priority = ?")
            params.append(priority)

        if status:
            conditions.append("status = ?")
            params.append(status)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        cursor = self.connection.cursor()
        cursor.execute(f"""
            SELECT * FROM alerts
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT ?
        """, params)

        return [self._row_to_alert_record(row) for row in cursor.fetchall()]

    def _row_to_alert_record(self, row: sqlite3.Row) -> AlertRecord:
        """Convert database row to AlertRecord."""
        return AlertRecord(
            alert_id=row["alert_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            frame_id=row["frame_id"],
            rule_id=row["rule_id"],
            priority=row["priority"],
            description=row["description"],
            location=row["location"],
            status=row["status"]
        )

    def update_alert_status(self, alert_id: int, status: str):
        """Update an alert's status."""
        cursor = self.connection.cursor()
        cursor.execute("""
            UPDATE alerts SET status = ? WHERE alert_id = ?
        """, (status, alert_id))
        self.connection.commit()

    # ==================== Detection Operations ====================

    def log_detection(self, detection: DetectionRecord) -> int:
        """Log an object detection."""
        cursor = self.connection.cursor()

        cursor.execute("""
            INSERT INTO detections
            (timestamp, frame_id, object_type, object_subtype, object_attributes, location_zone, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            detection.timestamp.isoformat() if isinstance(detection.timestamp, datetime) else detection.timestamp,
            detection.frame_id,
            detection.object_type,
            detection.object_subtype,
            json.dumps(detection.object_attributes),
            detection.location_zone,
            detection.confidence
        ))

        self.connection.commit()
        return cursor.lastrowid

    def get_detections_by_type(self, object_type: str, hours: int = 24) -> List[DetectionRecord]:
        """Get detections of a specific type within time window."""
        since = datetime.now() - timedelta(hours=hours)

        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT * FROM detections
            WHERE object_type = ? AND timestamp >= ?
            ORDER BY timestamp DESC
        """, (object_type, since.isoformat()))

        return [self._row_to_detection_record(row) for row in cursor.fetchall()]

    def count_object_occurrences(
        self,
        object_type: str,
        object_subtype: Optional[str] = None,
        hours: int = 24
    ) -> int:
        """Count occurrences of an object type within time window."""
        since = datetime.now() - timedelta(hours=hours)

        cursor = self.connection.cursor()

        if object_subtype:
            cursor.execute("""
                SELECT COUNT(*) FROM detections
                WHERE object_type = ? AND object_subtype = ? AND timestamp >= ?
            """, (object_type, object_subtype, since.isoformat()))
        else:
            cursor.execute("""
                SELECT COUNT(*) FROM detections
                WHERE object_type = ? AND timestamp >= ?
            """, (object_type, since.isoformat()))

        return cursor.fetchone()[0]

    def _row_to_detection_record(self, row: sqlite3.Row) -> DetectionRecord:
        """Convert database row to DetectionRecord."""
        return DetectionRecord(
            detection_id=row["detection_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            frame_id=row["frame_id"],
            object_type=row["object_type"],
            object_subtype=row["object_subtype"],
            object_attributes=json.loads(row["object_attributes"]) if row["object_attributes"] else {},
            location_zone=row["location_zone"],
            confidence=row["confidence"]
        )

    # ==================== Statistics ====================

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        cursor = self.connection.cursor()

        cursor.execute("SELECT COUNT(*) FROM frame_index")
        total_frames = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM alerts")
        total_alerts = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM alerts WHERE priority = 'HIGH'")
        high_priority_alerts = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM detections")
        total_detections = cursor.fetchone()[0]

        cursor.execute("""
            SELECT object_type, COUNT(*) as count
            FROM detections
            GROUP BY object_type
        """)
        detections_by_type = {row[0]: row[1] for row in cursor.fetchall()}

        return {
            "total_frames": total_frames,
            "total_alerts": total_alerts,
            "high_priority_alerts": high_priority_alerts,
            "total_detections": total_detections,
            "detections_by_type": detections_by_type
        }

    def clear_all_data(self):
        """Clear all data from the database (for testing)."""
        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM detections")
        cursor.execute("DELETE FROM alerts")
        cursor.execute("DELETE FROM frame_index")
        self.connection.commit()


# Convenience function for natural language query parsing
def parse_natural_query(query: str) -> Dict[str, Any]:
    """
    Parse a natural language query into database query parameters.

    Args:
        query: Natural language query string

    Returns:
        Dictionary of query parameters
    """
    query_lower = query.lower()
    params = {}

    # Object type detection
    if any(word in query_lower for word in ["truck", "car", "vehicle", "sedan", "pickup"]):
        params["object_type"] = "vehicle"
    elif any(word in query_lower for word in ["person", "people", "someone", "worker"]):
        params["object_type"] = "person"
    elif any(word in query_lower for word in ["animal", "dog", "cat"]):
        params["object_type"] = "animal"

    # Location detection
    locations = ["gate", "parking", "warehouse", "garage", "fence", "perimeter"]
    for loc in locations:
        if loc in query_lower:
            params["description_contains"] = loc
            break

    # Time detection
    if "today" in query_lower:
        params["start_time"] = datetime.now().replace(hour=0, minute=0, second=0)
        params["end_time"] = datetime.now()
    elif "yesterday" in query_lower:
        yesterday = datetime.now() - timedelta(days=1)
        params["start_time"] = yesterday.replace(hour=0, minute=0, second=0)
        params["end_time"] = yesterday.replace(hour=23, minute=59, second=59)
    elif "night" in query_lower or "midnight" in query_lower:
        params["start_time"] = datetime.now().replace(hour=0, minute=0, second=0)
        params["end_time"] = datetime.now().replace(hour=5, minute=0, second=0)

    return params


if __name__ == "__main__":
    # Test database operations
    db = SecurityDatabase()

    # Test frame indexing
    test_frame = FrameRecord(
        frame_id=1,
        timestamp=datetime.now(),
        location_name="Main Gate",
        location_zone="perimeter",
        latitude=37.7749,
        longitude=-122.4194,
        description="Blue Ford F150 entering main gate",
        objects=[{"type": "vehicle", "subtype": "truck", "color": "blue"}]
    )

    db.index_frame(test_frame)
    print(f"Indexed frame: {test_frame.frame_id}")

    # Test query
    results = db.query_frames_by_description("truck")
    print(f"Found {len(results)} frames with 'truck'")

    # Print statistics
    stats = db.get_statistics()
    print(f"Database stats: {stats}")

    db.close()
