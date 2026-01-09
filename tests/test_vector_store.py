"""
Tests for the ChromaDB Vector Store Module

This module tests the FrameVectorStore class and its semantic search capabilities.
"""

import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

# Skip all tests if ChromaDB is not available
pytest.importorskip("chromadb")
pytest.importorskip("sentence_transformers")

from src.vector_store import FrameVectorStore, SemanticQueryParser


class TestFrameVectorStore:
    """Tests for the FrameVectorStore class."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary directory for the vector store."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def vector_store(self, temp_db_path):
        """Create a vector store with temporary storage."""
        store = FrameVectorStore(persist_directory=temp_db_path)
        return store

    def test_initialization(self, temp_db_path):
        """Test that the vector store initializes correctly."""
        store = FrameVectorStore(persist_directory=temp_db_path)
        assert store is not None
        assert store.collection is not None

    def test_add_frame(self, vector_store):
        """Test adding a frame to the vector store."""
        frame_id = 1
        timestamp = datetime.now()
        description = "Blue Ford F150 pickup truck at main gate"
        objects = [{"type": "vehicle", "subtype": "truck", "color": "blue"}]
        location_name = "Main Gate"
        location_zone = "perimeter"

        # Add frame
        vector_store.add_frame(
            frame_id=frame_id,
            timestamp=timestamp,
            description=description,
            objects=objects,
            location_name=location_name,
            location_zone=location_zone,
            alert_triggered=False
        )

        # Verify frame was added
        stats = vector_store.get_statistics()
        assert stats["total_frames"] == 1

    def test_add_multiple_frames(self, vector_store):
        """Test adding multiple frames."""
        frames = [
            {
                "frame_id": 1,
                "description": "Blue Ford F150 at main gate",
                "objects": [{"type": "vehicle", "color": "blue"}],
                "location_name": "Main Gate",
                "location_zone": "perimeter"
            },
            {
                "frame_id": 2,
                "description": "Person walking near warehouse",
                "objects": [{"type": "person"}],
                "location_name": "Warehouse",
                "location_zone": "storage"
            },
            {
                "frame_id": 3,
                "description": "Red delivery truck at loading dock",
                "objects": [{"type": "vehicle", "color": "red"}],
                "location_name": "Loading Dock",
                "location_zone": "service"
            }
        ]

        for frame in frames:
            vector_store.add_frame(
                frame_id=frame["frame_id"],
                timestamp=datetime.now(),
                description=frame["description"],
                objects=frame["objects"],
                location_name=frame["location_name"],
                location_zone=frame["location_zone"]
            )

        stats = vector_store.get_statistics()
        assert stats["total_frames"] == 3

    def test_semantic_search(self, vector_store):
        """Test semantic similarity search."""
        # Add test frames
        frames = [
            {
                "frame_id": 1,
                "description": "Blue Ford F150 pickup truck entering through main gate",
                "objects": [{"type": "vehicle", "subtype": "truck", "color": "blue"}],
                "location_name": "Main Gate",
                "location_zone": "perimeter"
            },
            {
                "frame_id": 2,
                "description": "Person in dark clothing walking near warehouse",
                "objects": [{"type": "person"}],
                "location_name": "Warehouse",
                "location_zone": "storage"
            },
            {
                "frame_id": 3,
                "description": "Red delivery van at loading dock",
                "objects": [{"type": "vehicle", "subtype": "van", "color": "red"}],
                "location_name": "Loading Dock",
                "location_zone": "service"
            }
        ]

        for frame in frames:
            vector_store.add_frame(
                frame_id=frame["frame_id"],
                timestamp=datetime.now(),
                description=frame["description"],
                objects=frame["objects"],
                location_name=frame["location_name"],
                location_zone=frame["location_zone"]
            )

        # Search for trucks - should find the Ford F150
        results = vector_store.semantic_search("trucks at the gate", n_results=5)
        assert len(results) > 0
        # The truck frame should be in the results (results are VectorSearchResult objects)
        frame_ids = [r.frame_id for r in results]
        assert 1 in frame_ids

    def test_hybrid_search_with_object_filter(self, vector_store):
        """Test hybrid search with object type filter."""
        # Add test frames
        frames = [
            {
                "frame_id": 1,
                "description": "Vehicle at gate",
                "objects": [{"type": "vehicle"}],
                "location_name": "Main Gate",
                "location_zone": "perimeter"
            },
            {
                "frame_id": 2,
                "description": "Person at gate",
                "objects": [{"type": "person"}],
                "location_name": "Main Gate",
                "location_zone": "perimeter"
            }
        ]

        for frame in frames:
            vector_store.add_frame(
                frame_id=frame["frame_id"],
                timestamp=datetime.now(),
                description=frame["description"],
                objects=frame["objects"],
                location_name=frame["location_name"],
                location_zone=frame["location_zone"]
            )

        # Search with vehicle filter
        results = vector_store.hybrid_search(
            query="activity at gate",
            object_type="vehicle",
            n_results=5
        )

        # Should only return the vehicle frame
        assert len(results) == 1
        assert results[0].frame_id == 1

    def test_hybrid_search_with_zone_filter(self, vector_store):
        """Test hybrid search with location zone filter."""
        # Add test frames
        frames = [
            {
                "frame_id": 1,
                "description": "Vehicle at gate",
                "objects": [{"type": "vehicle"}],
                "location_name": "Main Gate",
                "location_zone": "perimeter"
            },
            {
                "frame_id": 2,
                "description": "Vehicle at warehouse",
                "objects": [{"type": "vehicle"}],
                "location_name": "Warehouse",
                "location_zone": "storage"
            }
        ]

        for frame in frames:
            vector_store.add_frame(
                frame_id=frame["frame_id"],
                timestamp=datetime.now(),
                description=frame["description"],
                objects=frame["objects"],
                location_name=frame["location_name"],
                location_zone=frame["location_zone"]
            )

        # Search with perimeter zone filter
        results = vector_store.hybrid_search(
            query="vehicles",
            location_zone="perimeter",
            n_results=5
        )

        # Should only return the perimeter frame
        assert len(results) == 1
        assert results[0].frame_id == 1

    def test_hybrid_search_alerts_only(self, vector_store):
        """Test hybrid search with alerts_only filter."""
        # Add test frames
        vector_store.add_frame(
            frame_id=1,
            timestamp=datetime.now(),
            description="Normal activity at gate",
            objects=[{"type": "vehicle"}],
            location_name="Main Gate",
            location_zone="perimeter",
            alert_triggered=False
        )
        vector_store.add_frame(
            frame_id=2,
            timestamp=datetime.now(),
            description="Suspicious activity at gate",
            objects=[{"type": "person"}],
            location_name="Main Gate",
            location_zone="perimeter",
            alert_triggered=True
        )

        # Search with alerts only
        results = vector_store.hybrid_search(
            query="activity at gate",
            alerts_only=True,
            n_results=5
        )

        # Should only return the alert frame
        assert len(results) == 1
        assert results[0].frame_id == 2
        assert results[0].metadata.get("alert_triggered") == True

    def test_find_similar_frames(self, vector_store):
        """Test finding similar frames."""
        # Add test frames with varying similarity
        frames = [
            {
                "frame_id": 1,
                "description": "Blue Ford F150 pickup truck at main gate",
                "objects": [{"type": "vehicle", "color": "blue"}],
                "location_name": "Main Gate"
            },
            {
                "frame_id": 2,
                "description": "Blue pickup truck exiting main gate",
                "objects": [{"type": "vehicle", "color": "blue"}],
                "location_name": "Main Gate"
            },
            {
                "frame_id": 3,
                "description": "Person walking near warehouse",
                "objects": [{"type": "person"}],
                "location_name": "Warehouse"
            }
        ]

        for frame in frames:
            vector_store.add_frame(
                frame_id=frame["frame_id"],
                timestamp=datetime.now(),
                description=frame["description"],
                objects=frame["objects"],
                location_name=frame["location_name"],
                location_zone="general"
            )

        # Find frames similar to frame 1 (blue truck)
        similar = vector_store.find_similar_frames(frame_id=1, n_results=2)

        # Frame 2 (also blue truck) should be most similar
        assert len(similar) > 0
        # The most similar frame should be frame 2
        if len(similar) >= 1:
            assert similar[0].frame_id == 2

    def test_get_frame_by_id(self, vector_store):
        """Test retrieving a frame by ID."""
        frame_id = 42
        description = "Test frame description"

        vector_store.add_frame(
            frame_id=frame_id,
            timestamp=datetime.now(),
            description=description,
            objects=[{"type": "vehicle"}],
            location_name="Test Location",
            location_zone="test"
        )

        frame = vector_store.get_frame_by_id(frame_id)
        assert frame is not None
        assert frame["frame_id"] == frame_id
        # Description may be enriched with metadata for better semantic search
        assert description in frame["description"]

    def test_get_frame_by_id_not_found(self, vector_store):
        """Test retrieving a non-existent frame."""
        frame = vector_store.get_frame_by_id(999)
        assert frame is None

    def test_delete_frame(self, vector_store):
        """Test deleting a frame."""
        # Add a frame
        vector_store.add_frame(
            frame_id=1,
            timestamp=datetime.now(),
            description="Test frame",
            objects=[],
            location_name="Test",
            location_zone="test"
        )

        # Verify it exists
        assert vector_store.get_frame_by_id(1) is not None

        # Delete it
        result = vector_store.delete_frame(1)
        assert result == True

        # Verify it's gone
        assert vector_store.get_frame_by_id(1) is None

    def test_clear_collection(self, vector_store):
        """Test clearing all frames."""
        # Add multiple frames
        for i in range(5):
            vector_store.add_frame(
                frame_id=i,
                timestamp=datetime.now(),
                description=f"Frame {i}",
                objects=[],
                location_name="Test",
                location_zone="test"
            )

        # Verify frames exist
        stats = vector_store.get_statistics()
        assert stats["total_frames"] == 5

        # Clear collection
        vector_store.clear_collection()

        # Verify all frames are gone
        stats = vector_store.get_statistics()
        assert stats["total_frames"] == 0

    def test_get_statistics(self, vector_store):
        """Test getting statistics."""
        # Add frames with different attributes
        vector_store.add_frame(
            frame_id=1,
            timestamp=datetime.now(),
            description="Vehicle frame",
            objects=[{"type": "vehicle"}],
            location_name="Gate",
            location_zone="perimeter"
        )
        vector_store.add_frame(
            frame_id=2,
            timestamp=datetime.now(),
            description="Person frame with alert",
            objects=[{"type": "person"}],
            location_name="Warehouse",
            location_zone="storage",
            alert_triggered=True
        )

        stats = vector_store.get_statistics()

        assert stats["total_frames"] == 2
        assert stats["alert_frames"] == 1
        assert "vehicle" in stats["object_types"]
        assert "person" in stats["object_types"]
        assert "perimeter" in stats["zones"]
        assert "storage" in stats["zones"]


class TestSemanticQueryParser:
    """Tests for the SemanticQueryParser class."""

    def test_parse_vehicle_query(self):
        """Test parsing a vehicle-related query."""
        result = SemanticQueryParser.parse_query("Show me all trucks at the gate")

        assert result["object_type"] == "vehicle"
        assert result["normalized_query"] == "show me all trucks at the gate"

    def test_parse_person_query(self):
        """Test parsing a person-related query."""
        result = SemanticQueryParser.parse_query("Find people near the warehouse")

        assert result["object_type"] == "person"

    def test_parse_alert_query(self):
        """Test parsing an alert-related query."""
        result = SemanticQueryParser.parse_query("Show security alerts from today")

        assert result["alerts_only"] == True

    def test_parse_location_query(self):
        """Test parsing a query with location."""
        result = SemanticQueryParser.parse_query("Activity at main gate")

        # Should extract location from query
        assert "main gate" in result["normalized_query"]

    def test_parse_zone_query_perimeter(self):
        """Test parsing a query mentioning perimeter."""
        result = SemanticQueryParser.parse_query("Check perimeter for activity")

        assert result["location_zone"] == "perimeter"

    def test_parse_zone_query_parking(self):
        """Test parsing a query mentioning parking."""
        result = SemanticQueryParser.parse_query("Vehicles in parking lot")

        assert result["location_zone"] == "parking"

    def test_parse_empty_query(self):
        """Test parsing an empty query."""
        result = SemanticQueryParser.parse_query("")

        assert result["normalized_query"] == ""
        assert result.get("object_type") is None

    def test_parse_complex_query(self):
        """Test parsing a complex query with multiple keywords."""
        result = SemanticQueryParser.parse_query(
            "Show suspicious vehicles near the perimeter that triggered alerts"
        )

        assert result["object_type"] == "vehicle"
        assert result["location_zone"] == "perimeter"
        assert result["alerts_only"] == True


class TestVectorStoreIntegration:
    """Integration tests for the vector store with realistic scenarios."""

    @pytest.fixture
    def populated_store(self, tmp_path):
        """Create a vector store with sample security data."""
        store = FrameVectorStore(persist_directory=str(tmp_path / "chromadb"))

        # Simulate a day of security footage
        base_time = datetime.now().replace(hour=0, minute=0, second=0)

        scenarios = [
            # Morning delivery
            {"frame_id": 1, "hours": 8, "desc": "White delivery van arriving at loading dock",
             "objects": [{"type": "vehicle", "subtype": "van", "color": "white"}],
             "location": "Loading Dock", "zone": "service"},

            # Employee arriving
            {"frame_id": 2, "hours": 9, "desc": "Person in business attire entering through main gate",
             "objects": [{"type": "person"}],
             "location": "Main Gate", "zone": "perimeter"},

            # Suspicious activity
            {"frame_id": 3, "hours": 2, "desc": "Unknown person near warehouse at night",
             "objects": [{"type": "person"}],
             "location": "Warehouse", "zone": "storage", "alert": True},

            # Regular patrol
            {"frame_id": 4, "hours": 14, "desc": "Security vehicle patrolling perimeter",
             "objects": [{"type": "vehicle", "subtype": "suv"}],
             "location": "Perimeter Road", "zone": "perimeter"},

            # Recurring vehicle
            {"frame_id": 5, "hours": 10, "desc": "Blue Ford F150 pickup at main gate",
             "objects": [{"type": "vehicle", "subtype": "truck", "color": "blue"}],
             "location": "Main Gate", "zone": "perimeter"},

            {"frame_id": 6, "hours": 15, "desc": "Blue Ford F150 pickup leaving via main gate",
             "objects": [{"type": "vehicle", "subtype": "truck", "color": "blue"}],
             "location": "Main Gate", "zone": "perimeter"},
        ]

        for scenario in scenarios:
            timestamp = base_time + timedelta(hours=scenario["hours"])
            store.add_frame(
                frame_id=scenario["frame_id"],
                timestamp=timestamp,
                description=scenario["desc"],
                objects=scenario["objects"],
                location_name=scenario["location"],
                location_zone=scenario["zone"],
                alert_triggered=scenario.get("alert", False)
            )

        return store

    def test_find_deliveries(self, populated_store):
        """Test finding delivery-related events."""
        results = populated_store.semantic_search("deliveries or vans", n_results=5)

        # Should find the delivery van
        frame_ids = [r.frame_id for r in results]
        assert 1 in frame_ids  # The delivery van frame

    def test_find_night_activity(self, populated_store):
        """Test finding night-time activity."""
        results = populated_store.hybrid_search(
            query="night activity",
            alerts_only=True,
            n_results=5
        )

        # Should find the suspicious night activity
        assert len(results) >= 1
        assert any(r.frame_id == 3 for r in results)

    def test_find_recurring_vehicle(self, populated_store):
        """Test finding a recurring vehicle."""
        results = populated_store.semantic_search(
            "blue Ford F150 truck",
            n_results=5
        )

        # Should find both blue truck frames
        frame_ids = [r.frame_id for r in results]
        assert 5 in frame_ids
        assert 6 in frame_ids

    def test_perimeter_monitoring(self, populated_store):
        """Test querying perimeter zone events."""
        results = populated_store.hybrid_search(
            query="perimeter activity",
            location_zone="perimeter",
            n_results=10
        )

        # Should find all perimeter events
        assert len(results) >= 3  # Gate entries and patrol

    def test_semantic_similarity_ranking(self, populated_store):
        """Test that semantic search ranks results by relevance."""
        # Search for trucks
        results = populated_store.semantic_search("pickup trucks", n_results=5)

        # The Ford F150 frames should rank higher than other vehicles
        if len(results) >= 2:
            top_ids = [r.frame_id for r in results[:2]]
            # At least one of the blue truck frames should be in top 2
            assert 5 in top_ids or 6 in top_ids
