"""
Vector Store Module using ChromaDB

Provides semantic search capabilities for video frame indexing.
Enables natural language queries like "find all suspicious activity"
or "show events similar to the blue truck incident".
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
import hashlib

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("ChromaDB not installed. Run: pip install chromadb sentence-transformers")

from .config import DATA_DIR


@dataclass
class VectorSearchResult:
    """Result from vector similarity search."""
    frame_id: int
    timestamp: str
    location: str
    description: str
    objects: List[Dict]
    similarity_score: float
    metadata: Dict

    def to_dict(self) -> Dict:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "location": self.location,
            "description": self.description,
            "objects": self.objects,
            "similarity_score": self.similarity_score,
            "metadata": self.metadata
        }


class FrameVectorStore:
    """
    ChromaDB-based vector store for semantic frame indexing.

    Enables:
    - Semantic search: "find suspicious activity near the gate"
    - Similarity search: "find events similar to this alert"
    - Hybrid queries: combine semantic + metadata filters
    """

    COLLECTION_NAME = "security_frames"

    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the vector store.

        Args:
            persist_directory: Directory for persistent storage
            embedding_model: Sentence transformer model for embeddings
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB is required. Install with: pip install chromadb sentence-transformers")

        self.persist_directory = Path(persist_directory) if persist_directory else (DATA_DIR / "chromadb")
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Use sentence-transformers for embeddings
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embedding_function,
            metadata={"description": "Security video frame embeddings"}
        )

        print(f"Vector store initialized with {self.collection.count()} frames")

    def _generate_document_id(self, frame_id: int, timestamp: datetime) -> str:
        """Generate unique document ID."""
        content = f"{frame_id}_{timestamp.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _create_searchable_text(
        self,
        description: str,
        objects: List[Dict],
        location: str,
        timestamp: datetime
    ) -> str:
        """
        Create rich searchable text from frame data.

        Combines description, objects, location, and time context
        for better semantic matching.
        """
        parts = [description]

        # Add object descriptions
        for obj in objects:
            obj_desc = obj.get("type", "")
            if obj.get("color"):
                obj_desc = f"{obj['color']} {obj_desc}"
            if obj.get("make"):
                obj_desc = f"{obj['make']} {obj_desc}"
            if obj.get("subtype"):
                obj_desc = f"{obj_desc} {obj['subtype']}"
            if obj_desc:
                parts.append(f"Detected: {obj_desc}")

            # Add attributes
            attrs = obj.get("attributes", {})
            if attrs.get("suspicious"):
                parts.append("suspicious activity")
            if attrs.get("loitering"):
                parts.append("loitering behavior")
            if attrs.get("recurring"):
                parts.append("recurring appearance")

        # Add location context
        parts.append(f"Location: {location}")

        # Add time context
        hour = timestamp.hour
        if 0 <= hour < 5:
            parts.append("night time activity")
        elif 5 <= hour < 12:
            parts.append("morning activity")
        elif 12 <= hour < 17:
            parts.append("afternoon activity")
        else:
            parts.append("evening activity")

        return " | ".join(parts)

    def add_frame(
        self,
        frame_id: int,
        timestamp: datetime,
        description: str,
        objects: List[Dict],
        location_name: str,
        location_zone: str,
        alert_triggered: bool = False,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Add a frame to the vector store.

        Args:
            frame_id: Unique frame identifier
            timestamp: Frame timestamp
            description: Frame description
            objects: List of detected objects
            location_name: Location name
            location_zone: Location zone
            alert_triggered: Whether alert was triggered
            metadata: Additional metadata

        Returns:
            Document ID
        """
        doc_id = self._generate_document_id(frame_id, timestamp)

        # Create searchable text
        searchable_text = self._create_searchable_text(
            description, objects, location_name, timestamp
        )

        # Prepare metadata
        doc_metadata = {
            "frame_id": frame_id,
            "timestamp": timestamp.isoformat(),
            "location_name": location_name,
            "location_zone": location_zone,
            "alert_triggered": alert_triggered,
            "object_count": len(objects),
            "has_person": any(o.get("type") == "person" for o in objects),
            "has_vehicle": any(o.get("type") == "vehicle" for o in objects),
            "hour": timestamp.hour,
            "objects_json": json.dumps(objects)
        }

        if metadata:
            doc_metadata.update(metadata)

        # Add to collection (upsert to handle duplicates)
        self.collection.upsert(
            ids=[doc_id],
            documents=[searchable_text],
            metadatas=[doc_metadata]
        )

        return doc_id

    def semantic_search(
        self,
        query: str,
        n_results: int = 10,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None
    ) -> List[VectorSearchResult]:
        """
        Perform semantic similarity search.

        Args:
            query: Natural language query
            n_results: Maximum results to return
            where: Metadata filter (e.g., {"location_zone": "perimeter"})
            where_document: Document content filter

        Returns:
            List of VectorSearchResult ordered by similarity
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=["documents", "metadatas", "distances"]
        )

        search_results = []

        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i] if results["distances"] else 0

                # Convert distance to similarity score (ChromaDB uses L2 distance)
                similarity = 1 / (1 + distance)

                # Parse objects from JSON
                objects = []
                if metadata.get("objects_json"):
                    try:
                        objects = json.loads(metadata["objects_json"])
                    except:
                        pass

                search_results.append(VectorSearchResult(
                    frame_id=metadata.get("frame_id", 0),
                    timestamp=metadata.get("timestamp", ""),
                    location=metadata.get("location_name", ""),
                    description=results["documents"][0][i],
                    objects=objects,
                    similarity_score=similarity,
                    metadata=metadata
                ))

        return search_results

    def find_similar_frames(
        self,
        frame_id: int,
        n_results: int = 5
    ) -> List[VectorSearchResult]:
        """
        Find frames similar to a given frame.

        Args:
            frame_id: Reference frame ID
            n_results: Number of similar frames to return

        Returns:
            List of similar frames
        """
        # Get the frame's document
        results = self.collection.get(
            where={"frame_id": frame_id},
            include=["documents", "embeddings"]
        )

        if not results["documents"]:
            return []

        # Search using the frame's text
        return self.semantic_search(
            query=results["documents"][0],
            n_results=n_results + 1  # +1 because it will include itself
        )[1:]  # Exclude the frame itself

    def search_by_object_type(
        self,
        object_type: str,
        query: Optional[str] = None,
        n_results: int = 20
    ) -> List[VectorSearchResult]:
        """
        Search frames containing specific object type.

        Args:
            object_type: "person", "vehicle", or "animal"
            query: Optional semantic query to refine results
            n_results: Maximum results

        Returns:
            Matching frames
        """
        # Build metadata filter
        if object_type == "person":
            where = {"has_person": True}
        elif object_type == "vehicle":
            where = {"has_vehicle": True}
        else:
            where = None

        search_query = query or f"frames with {object_type}"

        return self.semantic_search(
            query=search_query,
            n_results=n_results,
            where=where
        )

    def search_by_time_range(
        self,
        start_hour: int,
        end_hour: int,
        query: Optional[str] = None,
        n_results: int = 20
    ) -> List[VectorSearchResult]:
        """
        Search frames within a time range.

        Args:
            start_hour: Start hour (0-23)
            end_hour: End hour (0-23)
            query: Optional semantic query
            n_results: Maximum results

        Returns:
            Matching frames
        """
        # ChromaDB where clause for hour range
        if start_hour <= end_hour:
            where = {
                "$and": [
                    {"hour": {"$gte": start_hour}},
                    {"hour": {"$lte": end_hour}}
                ]
            }
        else:
            # Handle overnight range (e.g., 22:00 to 05:00)
            where = {
                "$or": [
                    {"hour": {"$gte": start_hour}},
                    {"hour": {"$lte": end_hour}}
                ]
            }

        search_query = query or "activity during this time period"

        return self.semantic_search(
            query=search_query,
            n_results=n_results,
            where=where
        )

    def search_by_location(
        self,
        location_zone: str,
        query: Optional[str] = None,
        n_results: int = 20
    ) -> List[VectorSearchResult]:
        """
        Search frames at a specific location zone.

        Args:
            location_zone: Zone name (e.g., "perimeter", "parking")
            query: Optional semantic query
            n_results: Maximum results

        Returns:
            Matching frames
        """
        where = {"location_zone": location_zone}
        search_query = query or f"activity at {location_zone}"

        return self.semantic_search(
            query=search_query,
            n_results=n_results,
            where=where
        )

    def search_alerts_only(
        self,
        query: Optional[str] = None,
        n_results: int = 20
    ) -> List[VectorSearchResult]:
        """
        Search only frames that triggered alerts.

        Args:
            query: Optional semantic query
            n_results: Maximum results

        Returns:
            Frames with alerts
        """
        where = {"alert_triggered": True}
        search_query = query or "security alert incident"

        return self.semantic_search(
            query=search_query,
            n_results=n_results,
            where=where
        )

    def hybrid_search(
        self,
        query: str,
        object_type: Optional[str] = None,
        location_zone: Optional[str] = None,
        time_range: Optional[Tuple[int, int]] = None,
        alerts_only: bool = False,
        n_results: int = 10
    ) -> List[VectorSearchResult]:
        """
        Perform hybrid search with semantic query and filters.

        Args:
            query: Natural language query
            object_type: Filter by object type
            location_zone: Filter by location
            time_range: Filter by hour range (start, end)
            alerts_only: Only include frames with alerts
            n_results: Maximum results

        Returns:
            Matching frames
        """
        # Build where clause
        conditions = []

        if object_type == "person":
            conditions.append({"has_person": True})
        elif object_type == "vehicle":
            conditions.append({"has_vehicle": True})

        if location_zone:
            conditions.append({"location_zone": location_zone})

        if alerts_only:
            conditions.append({"alert_triggered": True})

        if time_range:
            start_hour, end_hour = time_range
            if start_hour <= end_hour:
                conditions.append({"hour": {"$gte": start_hour}})
                conditions.append({"hour": {"$lte": end_hour}})

        # Combine conditions
        where = None
        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        return self.semantic_search(
            query=query,
            n_results=n_results,
            where=where
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        count = self.collection.count()

        # Sample some data for stats
        sample = self.collection.get(
            limit=min(100, count),
            include=["metadatas"]
        )

        stats = {
            "total_frames": count,
            "with_persons": 0,
            "with_vehicles": 0,
            "with_alerts": 0,
            "alert_frames": 0,
            "by_zone": {},
            "zones": set(),
            "object_types": set()
        }

        if sample["metadatas"]:
            for meta in sample["metadatas"]:
                if meta.get("has_person"):
                    stats["with_persons"] += 1
                    stats["object_types"].add("person")
                if meta.get("has_vehicle"):
                    stats["with_vehicles"] += 1
                    stats["object_types"].add("vehicle")
                if meta.get("alert_triggered"):
                    stats["with_alerts"] += 1
                    stats["alert_frames"] += 1

                zone = meta.get("location_zone", "unknown")
                stats["by_zone"][zone] = stats["by_zone"].get(zone, 0) + 1
                stats["zones"].add(zone)

        # Convert sets to lists for JSON serialization
        stats["zones"] = list(stats["zones"])
        stats["object_types"] = list(stats["object_types"])

        return stats

    def clear(self):
        """Clear all data from the vector store."""
        self.client.delete_collection(self.COLLECTION_NAME)
        self.collection = self.client.create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.embedding_function,
            metadata={"description": "Security video frame embeddings"}
        )

    def clear_collection(self):
        """Alias for clear() - clear all data from the vector store."""
        self.clear()

    def get_frame_by_id(self, frame_id: int) -> Optional[Dict]:
        """
        Get a frame by its ID.

        Args:
            frame_id: Frame ID to retrieve

        Returns:
            Frame data as dict or None if not found
        """
        results = self.collection.get(
            where={"frame_id": frame_id},
            include=["documents", "metadatas"]
        )

        if results["ids"]:
            metadata = results["metadatas"][0]
            objects = []
            if metadata.get("objects_json"):
                try:
                    objects = json.loads(metadata["objects_json"])
                except:
                    pass

            return {
                "frame_id": metadata.get("frame_id"),
                "timestamp": metadata.get("timestamp"),
                "description": results["documents"][0],
                "location_name": metadata.get("location_name"),
                "location_zone": metadata.get("location_zone"),
                "objects": objects,
                "alert_triggered": metadata.get("alert_triggered", False)
            }

        return None

    def delete_frame(self, frame_id: int) -> bool:
        """
        Delete a frame by its ID.

        Args:
            frame_id: Frame ID to delete

        Returns:
            True if deleted, False if not found
        """
        # Get the document ID first
        results = self.collection.get(
            where={"frame_id": frame_id},
            include=[]
        )

        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            return True

        return False

    def get_all_frames(self, limit: int = 100) -> List[VectorSearchResult]:
        """Get all frames (up to limit)."""
        results = self.collection.get(
            limit=limit,
            include=["documents", "metadatas"]
        )

        frames = []
        if results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i]
                objects = []
                if metadata.get("objects_json"):
                    try:
                        objects = json.loads(metadata["objects_json"])
                    except:
                        pass

                frames.append(VectorSearchResult(
                    frame_id=metadata.get("frame_id", 0),
                    timestamp=metadata.get("timestamp", ""),
                    location=metadata.get("location_name", ""),
                    description=results["documents"][i],
                    objects=objects,
                    similarity_score=1.0,
                    metadata=metadata
                ))

        return frames


class SemanticQueryParser:
    """
    Parses natural language queries into structured search parameters.
    """

    OBJECT_KEYWORDS = {
        "person": ["person", "people", "someone", "man", "woman", "worker", "intruder", "human"],
        "vehicle": ["vehicle", "car", "truck", "van", "suv", "pickup", "automobile", "ford", "toyota"],
        "animal": ["animal", "dog", "cat", "bird", "wildlife"]
    }

    LOCATION_KEYWORDS = {
        "perimeter": ["perimeter", "fence", "boundary", "edge"],
        "parking": ["parking", "lot", "parked"],
        "storage": ["warehouse", "storage", "depot"],
        "main": ["office", "building", "main", "entrance"],
        "vehicles": ["garage", "vehicle bay"]
    }

    TIME_KEYWORDS = {
        "night": (0, 5),
        "morning": (6, 11),
        "afternoon": (12, 16),
        "evening": (17, 21),
        "midnight": (0, 2),
        "dawn": (5, 7),
        "dusk": (18, 20)
    }

    @classmethod
    def parse(cls, query: str) -> Dict[str, Any]:
        """
        Parse a natural language query.

        Args:
            query: Natural language query

        Returns:
            Dictionary with parsed parameters
        """
        query_lower = query.lower()
        params = {"raw_query": query}

        # Detect object type
        for obj_type, keywords in cls.OBJECT_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                params["object_type"] = obj_type
                break

        # Detect location
        for zone, keywords in cls.LOCATION_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                params["location_zone"] = zone
                break

        # Detect time
        for time_name, (start, end) in cls.TIME_KEYWORDS.items():
            if time_name in query_lower:
                params["time_range"] = (start, end)
                break

        # Detect alert-related queries
        if any(word in query_lower for word in ["alert", "warning", "incident", "suspicious", "security"]):
            params["alerts_only"] = True

        # Detect similarity queries
        if any(word in query_lower for word in ["similar", "like", "same as"]):
            params["similarity_search"] = True

        return params

    @classmethod
    def parse_query(cls, query: str) -> Dict[str, Any]:
        """
        Alias for parse() method for backward compatibility.

        Args:
            query: Natural language query

        Returns:
            Dictionary with parsed parameters
        """
        result = cls.parse(query)
        # Add normalized_query field for tests
        result["normalized_query"] = query.lower()
        return result


# Convenience function for quick searches
def quick_search(query: str, vector_store: Optional[FrameVectorStore] = None) -> List[Dict]:
    """
    Perform a quick semantic search.

    Args:
        query: Natural language query
        vector_store: Optional existing vector store instance

    Returns:
        List of matching frames as dictionaries
    """
    if vector_store is None:
        vector_store = FrameVectorStore()

    # Parse query
    params = SemanticQueryParser.parse(query)

    # Perform hybrid search
    results = vector_store.hybrid_search(
        query=params["raw_query"],
        object_type=params.get("object_type"),
        location_zone=params.get("location_zone"),
        time_range=params.get("time_range"),
        alerts_only=params.get("alerts_only", False),
        n_results=10
    )

    return [r.to_dict() for r in results]


if __name__ == "__main__":
    # Test the vector store
    print("=" * 60)
    print("VECTOR STORE TEST")
    print("=" * 60)

    store = FrameVectorStore()

    # Add test frames
    test_frames = [
        {
            "frame_id": 1,
            "timestamp": datetime.now().replace(hour=10),
            "description": "Blue Ford F150 pickup truck entering through main gate",
            "objects": [{"type": "vehicle", "subtype": "truck", "color": "blue", "make": "Ford"}],
            "location_name": "Main Gate",
            "location_zone": "perimeter",
            "alert_triggered": False
        },
        {
            "frame_id": 2,
            "timestamp": datetime.now().replace(hour=2),
            "description": "Person in dark clothing walking near warehouse at night",
            "objects": [{"type": "person", "subtype": "unknown", "attributes": {"suspicious": True}}],
            "location_name": "Warehouse",
            "location_zone": "storage",
            "alert_triggered": True
        },
        {
            "frame_id": 3,
            "timestamp": datetime.now().replace(hour=14),
            "description": "Red Toyota sedan parked in visitor area",
            "objects": [{"type": "vehicle", "subtype": "sedan", "color": "red", "make": "Toyota"}],
            "location_name": "Parking Lot",
            "location_zone": "parking",
            "alert_triggered": False
        },
        {
            "frame_id": 4,
            "timestamp": datetime.now().replace(hour=3),
            "description": "Person loitering near main gate at night",
            "objects": [{"type": "person", "attributes": {"loitering": True}}],
            "location_name": "Main Gate",
            "location_zone": "perimeter",
            "alert_triggered": True
        },
    ]

    print("\nAdding test frames...")
    for frame in test_frames:
        store.add_frame(**frame)
        print(f"  Added frame {frame['frame_id']}: {frame['description'][:50]}...")

    # Test queries
    print("\n" + "=" * 60)
    print("SEMANTIC SEARCH TESTS")
    print("=" * 60)

    queries = [
        "show me all trucks",
        "suspicious activity at night",
        "vehicles at the gate",
        "people near warehouse",
        "security alerts",
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")
        results = store.semantic_search(query, n_results=3)
        for r in results:
            print(f"  [{r.similarity_score:.2f}] Frame {r.frame_id}: {r.description[:60]}...")

    # Test hybrid search
    print("\n" + "=" * 60)
    print("HYBRID SEARCH TEST")
    print("=" * 60)

    print("\nQuery: 'activity' + filter: vehicles only")
    results = store.hybrid_search("activity", object_type="vehicle")
    for r in results:
        print(f"  Frame {r.frame_id}: {r.description[:60]}...")

    # Statistics
    print("\n" + "=" * 60)
    stats = store.get_statistics()
    print(f"Statistics: {json.dumps(stats, indent=2)}")
