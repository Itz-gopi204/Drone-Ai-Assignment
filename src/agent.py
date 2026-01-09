"""
LangChain Security Analyst Agent

An AI agent that processes security events, answers queries,
and provides intelligent analysis of surveillance data.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import Tool, StructuredTool
from pydantic import BaseModel, Field

# LangChain v1 agent imports - using create_react_agent (recommended over deprecated create_openai_functions_agent)
try:
    from langchain.agents import create_react_agent, AgentExecutor
    LANGCHAIN_AGENT_AVAILABLE = True
except ImportError:
    try:
        # Fallback to langchain_classic for backward compatibility
        from langchain_classic.agents import AgentExecutor, create_openai_functions_agent as create_react_agent
        LANGCHAIN_AGENT_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AGENT_AVAILABLE = False
        create_react_agent = None
        AgentExecutor = None

# Groq support (open-source LLMs with fast inference)
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    ChatGroq = None

from .config import (
    OPENAI_API_KEY, MODEL_NAME, AGENT_CONFIG,
    LLM_PROVIDER, GROQ_API_KEY, GROQ_MODEL_NAME, OPENAI_MODEL_NAME
)
from .database import SecurityDatabase, FrameRecord, parse_natural_query
from .analyzer import FrameAnalyzer, AnalysisResult, ObjectTracker
from .alert_engine import AlertEngine, Alert, AlertFormatter

# Optional vector store import
try:
    from .vector_store import FrameVectorStore
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    VECTOR_STORE_AVAILABLE = False


# ==================== Tool Input Schemas ====================

class AnalyzeFrameInput(BaseModel):
    """Input schema for frame analysis tool."""
    frame_id: int = Field(description="Unique identifier for the frame")
    description: str = Field(description="Text description of the frame content")
    location_name: str = Field(description="Name of the location where frame was captured")
    location_zone: str = Field(description="Zone category (e.g., perimeter, parking)")


class QueryFramesInput(BaseModel):
    """Input schema for querying frames."""
    query: str = Field(description="Natural language query or search term")
    limit: int = Field(default=10, description="Maximum number of results to return")


class TimeRangeQueryInput(BaseModel):
    """Input schema for time-range queries."""
    start_time: str = Field(description="Start time in ISO format or relative (e.g., '2 hours ago')")
    end_time: str = Field(default="now", description="End time in ISO format or 'now'")
    object_type: Optional[str] = Field(default=None, description="Filter by object type (vehicle, person, animal)")


class GetAlertsInput(BaseModel):
    """Input schema for getting alerts."""
    priority: Optional[str] = Field(default=None, description="Filter by priority (LOW, MEDIUM, HIGH)")
    limit: int = Field(default=10, description="Maximum number of alerts to return")


class GenerateSummaryInput(BaseModel):
    """Input schema for generating summaries."""
    time_period: str = Field(default="today", description="Time period: 'today', 'last_hour', 'last_24h'")


class SemanticSearchInput(BaseModel):
    """Input schema for semantic search queries."""
    query: str = Field(description="Natural language query for semantic search")
    n_results: int = Field(default=10, description="Maximum number of results")
    object_type: Optional[str] = Field(default=None, description="Filter by object type")
    location_zone: Optional[str] = Field(default=None, description="Filter by location zone")
    alerts_only: bool = Field(default=False, description="Only return frames that triggered alerts")


class FindSimilarInput(BaseModel):
    """Input schema for finding similar frames."""
    frame_id: int = Field(description="Frame ID to find similar frames for")
    n_results: int = Field(default=5, description="Number of similar frames to return")


# ==================== Security Analyst Agent ====================

class SecurityAnalystAgent:
    """
    LangChain-based security analyst agent.

    Provides intelligent analysis of surveillance data through
    natural language interactions and tool-based operations.
    """

    SYSTEM_PROMPT = """You are an AI Security Analyst Agent monitoring a property via drone surveillance.

Your responsibilities:
1. Analyze video frames and telemetry data to detect security-relevant events
2. Generate alerts for suspicious activities based on predefined rules
3. Answer questions about detected objects, events, and patterns
4. Provide summaries of surveillance activity
5. Help property owners understand security events

You have access to the following capabilities:
- Analyze incoming video frames for objects and activities
- Query the frame index database for historical events
- Check and report security alerts
- Generate activity summaries
- Track recurring objects (vehicles, persons)

When analyzing events, consider:
- Time of day (night activity is more suspicious)
- Location context (perimeter zones are more sensitive)
- Object patterns (recurring vehicles, loitering persons)
- Historical context (what happened before)

Always provide clear, actionable insights. When you detect security concerns,
explain what was detected, where, and why it's significant.

Current date/time: {current_time}
"""

    def __init__(
        self,
        database: Optional[SecurityDatabase] = None,
        vector_store: Optional[Any] = None,
        use_api: bool = True
    ):
        """
        Initialize the security analyst agent.

        Args:
            database: Database for frame indexing and queries
            vector_store: ChromaDB vector store for semantic search
            use_api: Whether to use OpenAI API (requires API key)
        """
        self.database = database or SecurityDatabase()
        self.vector_store = vector_store
        self.tracker = ObjectTracker()
        self.alert_engine = AlertEngine(database=self.database)

        # Check API availability based on provider
        if LLM_PROVIDER == "groq":
            self.use_api = use_api and bool(GROQ_API_KEY) and GROQ_AVAILABLE
        else:
            self.use_api = use_api and bool(OPENAI_API_KEY)

        self.analyzer = FrameAnalyzer(use_api=self.use_api)

        self.conversation_history: List[Dict] = []
        self.context_window: List[Dict] = []

        if self.use_api:
            self._init_langchain_agent()
        else:
            self.agent_executor = None

    def _init_langchain_agent(self):
        """Initialize the LangChain agent with tools."""
        if not LANGCHAIN_AGENT_AVAILABLE:
            self.agent_executor = None
            return

        # Create LLM based on provider
        if LLM_PROVIDER == "groq" and GROQ_AVAILABLE:
            self.llm = ChatGroq(
                model=GROQ_MODEL_NAME,
                temperature=AGENT_CONFIG["temperature"],
                api_key=GROQ_API_KEY
            )
        else:
            self.llm = ChatOpenAI(
                model=OPENAI_MODEL_NAME,
                temperature=AGENT_CONFIG["temperature"],
                api_key=OPENAI_API_KEY
            )

        # Create tools
        tools = self._create_tools()

        # Create prompt for ReAct agent (LangChain v1 format)
        # The ReAct prompt requires specific placeholders for the agent loop
        from langchain_core.prompts import PromptTemplate

        react_prompt = PromptTemplate.from_template(
            self.SYSTEM_PROMPT + """

You have access to the following tools:
{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
        )

        try:
            # Create agent using create_react_agent (LangChain v1 recommended)
            agent = create_react_agent(self.llm, tools, react_prompt)

            # Create executor
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5
            )
        except Exception as e:
            print(f"Warning: Could not create LangChain agent: {e}")
            self.agent_executor = None

    def _create_tools(self) -> List[Tool]:
        """Create LangChain tools for the agent."""
        tools = [
            StructuredTool.from_function(
                func=self._tool_analyze_frame,
                name="analyze_frame",
                description="Analyze a video frame to detect objects, people, vehicles, and security-relevant events",
                args_schema=AnalyzeFrameInput
            ),
            StructuredTool.from_function(
                func=self._tool_query_frames,
                name="query_frames",
                description="Search the frame database for events matching a query (e.g., 'trucks at gate', 'person near warehouse')",
                args_schema=QueryFramesInput
            ),
            StructuredTool.from_function(
                func=self._tool_query_by_time,
                name="query_by_time",
                description="Query frames within a specific time range, optionally filtered by object type",
                args_schema=TimeRangeQueryInput
            ),
            StructuredTool.from_function(
                func=self._tool_get_alerts,
                name="get_alerts",
                description="Get recent security alerts, optionally filtered by priority level",
                args_schema=GetAlertsInput
            ),
            StructuredTool.from_function(
                func=self._tool_generate_summary,
                name="generate_summary",
                description="Generate a summary of security activity for a time period",
                args_schema=GenerateSummaryInput
            ),
            Tool(
                name="get_statistics",
                func=self._tool_get_statistics,
                description="Get overall statistics about detections, alerts, and indexed frames"
            ),
            Tool(
                name="get_recurring_objects",
                func=self._tool_get_recurring_objects,
                description="Get list of objects (vehicles, persons) that have been seen multiple times"
            ),
        ]

        # Add semantic search tools if vector store is available
        if self.vector_store:
            tools.extend([
                StructuredTool.from_function(
                    func=self._tool_semantic_search,
                    name="semantic_search",
                    description="Perform semantic similarity search to find frames matching a natural language query. Better than keyword search for finding conceptually similar events.",
                    args_schema=SemanticSearchInput
                ),
                StructuredTool.from_function(
                    func=self._tool_find_similar,
                    name="find_similar_frames",
                    description="Find frames similar to a given frame. Useful for finding related security events.",
                    args_schema=FindSimilarInput
                ),
            ])

        return tools

    # ==================== Tool Implementations ====================

    def _tool_analyze_frame(
        self,
        frame_id: int,
        description: str,
        location_name: str,
        location_zone: str
    ) -> str:
        """Analyze a video frame."""
        timestamp = datetime.now()
        location = {"name": location_name, "zone": location_zone}

        # Perform analysis
        result = self.analyzer.analyze_frame(
            frame_id=frame_id,
            timestamp=timestamp,
            frame_description=description,
            location_context=location
        )

        # Track objects
        for obj in result.detected_objects:
            self.tracker.track_object(obj, timestamp, location_name)

        # Check for alerts
        alerts = self.alert_engine.evaluate(
            frame_id=frame_id,
            timestamp=timestamp,
            detected_objects=result.detected_objects,
            location=location,
            description=result.description
        )

        # Index frame in database
        frame_record = FrameRecord(
            frame_id=frame_id,
            timestamp=timestamp,
            location_name=location_name,
            location_zone=location_zone,
            latitude=0.0,
            longitude=0.0,
            description=result.description,
            objects=result.detected_objects,
            alert_triggered=bool(alerts)
        )
        self.database.index_frame(frame_record)

        # Format response
        response = {
            "frame_id": frame_id,
            "analysis": result.description,
            "objects_detected": result.detected_objects,
            "security_relevant": result.security_relevant,
            "alerts_triggered": [a.to_dict() for a in alerts]
        }

        return json.dumps(response, indent=2, default=str)

    def _tool_query_frames(self, query: str, limit: int = 10) -> str:
        """Query frames by natural language or keyword."""
        # Parse natural language query
        params = parse_natural_query(query)

        if params:
            results = self.database.query_frames_complex(
                **params,
                limit=limit
            )
        else:
            # Fallback to description search
            results = self.database.query_frames_by_description(query)[:limit]

        if not results:
            return json.dumps({"message": "No matching frames found", "query": query})

        formatted_results = [
            {
                "frame_id": r.frame_id,
                "timestamp": r.timestamp.isoformat(),
                "location": r.location_name,
                "description": r.description,
                "objects": r.objects
            }
            for r in results
        ]

        return json.dumps({
            "query": query,
            "result_count": len(formatted_results),
            "frames": formatted_results
        }, indent=2)

    def _tool_query_by_time(
        self,
        start_time: str,
        end_time: str = "now",
        object_type: Optional[str] = None
    ) -> str:
        """Query frames by time range."""
        # Parse time strings
        start = self._parse_time_string(start_time)
        end = self._parse_time_string(end_time)

        results = self.database.query_frames_complex(
            start_time=start,
            end_time=end,
            object_type=object_type,
            limit=50
        )

        formatted_results = [
            {
                "frame_id": r.frame_id,
                "timestamp": r.timestamp.isoformat(),
                "location": r.location_name,
                "description": r.description
            }
            for r in results
        ]

        return json.dumps({
            "time_range": {"start": start.isoformat(), "end": end.isoformat()},
            "object_filter": object_type,
            "result_count": len(formatted_results),
            "frames": formatted_results
        }, indent=2)

    def _tool_get_alerts(
        self,
        priority: Optional[str] = None,
        limit: int = 10
    ) -> str:
        """Get recent alerts."""
        alerts = self.database.get_alerts(priority=priority, limit=limit)

        formatted_alerts = [
            {
                "alert_id": a.alert_id,
                "timestamp": a.timestamp.isoformat(),
                "priority": a.priority,
                "description": a.description,
                "location": a.location,
                "status": a.status
            }
            for a in alerts
        ]

        return json.dumps({
            "filter": {"priority": priority},
            "alert_count": len(formatted_alerts),
            "alerts": formatted_alerts
        }, indent=2)

    def _tool_generate_summary(self, time_period: str = "today") -> str:
        """Generate activity summary."""
        # Determine time range
        now = datetime.now()
        if time_period == "last_hour":
            start = now - timedelta(hours=1)
        elif time_period == "last_24h":
            start = now - timedelta(hours=24)
        else:  # today
            start = now.replace(hour=0, minute=0, second=0)

        # Get data
        frames = self.database.query_frames_by_time(start, now)
        alerts = self.database.get_alerts(limit=100)
        stats = self.database.get_statistics()

        # Filter alerts by time
        period_alerts = [
            a for a in alerts
            if a.timestamp >= start
        ]

        # Generate summary
        summary = {
            "time_period": time_period,
            "start_time": start.isoformat(),
            "end_time": now.isoformat(),
            "total_frames_processed": len(frames),
            "alerts_generated": len(period_alerts),
            "high_priority_alerts": len([a for a in period_alerts if a.priority == "HIGH"]),
            "detections_by_type": stats.get("detections_by_type", {}),
            "key_events": self._extract_key_events(frames, period_alerts)
        }

        return json.dumps(summary, indent=2, default=str)

    def _tool_get_statistics(self, _: str = "") -> str:
        """Get overall statistics."""
        stats = self.database.get_statistics()
        analysis_summary = self.analyzer.get_analysis_summary()
        recurring = self.tracker.get_recurring_objects()

        return json.dumps({
            "database_stats": stats,
            "analysis_stats": analysis_summary,
            "recurring_objects": len(recurring)
        }, indent=2)

    def _tool_get_recurring_objects(self, _: str = "") -> str:
        """Get recurring objects."""
        recurring = self.tracker.get_recurring_objects()

        return json.dumps({
            "recurring_objects_count": len(recurring),
            "objects": recurring
        }, indent=2, default=str)

    def _tool_semantic_search(
        self,
        query: str,
        n_results: int = 10,
        object_type: Optional[str] = None,
        location_zone: Optional[str] = None,
        alerts_only: bool = False
    ) -> str:
        """Perform semantic search using ChromaDB vector store."""
        if not self.vector_store:
            return json.dumps({"error": "Vector store not available"})

        try:
            results = self.vector_store.hybrid_search(
                query=query,
                object_type=object_type,
                location_zone=location_zone,
                alerts_only=alerts_only,
                n_results=n_results
            )

            formatted_results = []
            for result in results:
                # Handle both VectorSearchResult objects and dicts
                if hasattr(result, 'to_dict'):
                    result = result.to_dict()
                formatted_results.append({
                    "frame_id": result.get("frame_id") if isinstance(result, dict) else getattr(result, 'frame_id', None),
                    "timestamp": result.get("timestamp") if isinstance(result, dict) else getattr(result, 'timestamp', None),
                    "description": result.get("description") if isinstance(result, dict) else getattr(result, 'description', None),
                    "location": result.get("location") if isinstance(result, dict) else getattr(result, 'location', None),
                    "zone": result.get("location_zone", result.get("zone")) if isinstance(result, dict) else getattr(result, 'location_zone', None),
                    "objects": result.get("objects", []) if isinstance(result, dict) else getattr(result, 'objects', []),
                    "similarity_score": result.get("similarity_score", 0) if isinstance(result, dict) else getattr(result, 'similarity_score', 0)
                })

            return json.dumps({
                "query": query,
                "filters": {
                    "object_type": object_type,
                    "location_zone": location_zone,
                    "alerts_only": alerts_only
                },
                "result_count": len(formatted_results),
                "frames": formatted_results
            }, indent=2, default=str)

        except Exception as e:
            return json.dumps({"error": f"Semantic search failed: {str(e)}"})

    def _tool_find_similar(self, frame_id: int, n_results: int = 5) -> str:
        """Find frames similar to a given frame."""
        if not self.vector_store:
            return json.dumps({"error": "Vector store not available"})

        try:
            results = self.vector_store.find_similar_frames(
                frame_id=frame_id,
                n_results=n_results
            )

            formatted_results = []
            for result in results:
                # Handle both VectorSearchResult objects and dicts
                if hasattr(result, 'to_dict'):
                    result = result.to_dict()
                formatted_results.append({
                    "frame_id": result.get("frame_id") if isinstance(result, dict) else getattr(result, 'frame_id', None),
                    "timestamp": result.get("timestamp") if isinstance(result, dict) else getattr(result, 'timestamp', None),
                    "description": result.get("description") if isinstance(result, dict) else getattr(result, 'description', None),
                    "location": result.get("location") if isinstance(result, dict) else getattr(result, 'location', None),
                    "similarity_score": result.get("similarity_score", 0) if isinstance(result, dict) else getattr(result, 'similarity_score', 0)
                })

            return json.dumps({
                "reference_frame_id": frame_id,
                "similar_frames_count": len(formatted_results),
                "similar_frames": formatted_results
            }, indent=2, default=str)

        except Exception as e:
            return json.dumps({"error": f"Find similar failed: {str(e)}"})

    # ==================== Helper Methods ====================

    def _parse_time_string(self, time_str: str) -> datetime:
        """Parse a time string into datetime."""
        time_str = time_str.lower().strip()

        if time_str == "now":
            return datetime.now()

        # Handle relative times
        if "ago" in time_str:
            parts = time_str.replace("ago", "").strip().split()
            if len(parts) >= 2:
                amount = int(parts[0])
                unit = parts[1]

                if "hour" in unit:
                    return datetime.now() - timedelta(hours=amount)
                elif "minute" in unit:
                    return datetime.now() - timedelta(minutes=amount)
                elif "day" in unit:
                    return datetime.now() - timedelta(days=amount)

        # Try ISO format
        try:
            return datetime.fromisoformat(time_str)
        except:
            pass

        # Default to start of today
        return datetime.now().replace(hour=0, minute=0, second=0)

    def _extract_key_events(
        self,
        frames: List[FrameRecord],
        alerts: List
    ) -> List[Dict]:
        """Extract key events from frames and alerts."""
        key_events = []

        # Add all alerts as key events
        for alert in alerts[:5]:
            key_events.append({
                "type": "alert",
                "time": alert.timestamp.isoformat(),
                "description": alert.description,
                "priority": alert.priority
            })

        # Add frames with detected objects
        for frame in frames:
            if frame.objects and frame.alert_triggered:
                key_events.append({
                    "type": "detection",
                    "time": frame.timestamp.isoformat(),
                    "description": frame.description,
                    "location": frame.location_name
                })

        # Sort by time and limit
        key_events.sort(key=lambda x: x["time"], reverse=True)
        return key_events[:10]

    # ==================== Main Interface ====================

    def process_frame(
        self,
        frame_id: int,
        timestamp: datetime,
        description: str,
        location: Dict,
        telemetry: Dict
    ) -> Dict:
        """
        Process an incoming frame through the full pipeline.

        Args:
            frame_id: Frame identifier
            timestamp: Frame timestamp
            description: Frame description
            location: Location data
            telemetry: Telemetry data

        Returns:
            Processing result with analysis and alerts
        """
        # Analyze frame
        analysis = self.analyzer.analyze_frame(
            frame_id=frame_id,
            timestamp=timestamp,
            frame_description=description,
            location_context=location
        )

        # Track objects
        tracked_objects = []
        for obj in analysis.detected_objects:
            tracked = self.tracker.track_object(obj, timestamp, location.get("name", "Unknown"))
            tracked_objects.append(tracked)

        # Check for alerts
        alerts = self.alert_engine.evaluate(
            frame_id=frame_id,
            timestamp=timestamp,
            detected_objects=tracked_objects,
            location=location,
            description=analysis.description
        )

        # Index frame in SQLite database
        frame_record = FrameRecord(
            frame_id=frame_id,
            timestamp=timestamp,
            location_name=location.get("name", "Unknown"),
            location_zone=location.get("zone", ""),
            latitude=telemetry.get("latitude", 0.0),
            longitude=telemetry.get("longitude", 0.0),
            description=analysis.description,
            objects=tracked_objects,
            alert_triggered=bool(alerts)
        )
        self.database.index_frame(frame_record)

        # Also index in vector store for semantic search
        if self.vector_store:
            try:
                self.vector_store.add_frame(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    description=analysis.description,
                    objects=tracked_objects,
                    location_name=location.get("name", "Unknown"),
                    location_zone=location.get("zone", ""),
                    alert_triggered=bool(alerts)
                )
            except Exception as e:
                # Log but don't fail if vector store indexing fails
                pass

        # Log detections
        for obj in tracked_objects:
            from .database import DetectionRecord
            detection = DetectionRecord(
                detection_id=None,
                timestamp=timestamp,
                frame_id=frame_id,
                object_type=obj.get("type", "unknown"),
                object_subtype=obj.get("subtype", ""),
                object_attributes=obj.get("attributes", {}),
                location_zone=location.get("zone", ""),
                confidence=analysis.confidence
            )
            self.database.log_detection(detection)

        return {
            "frame_id": frame_id,
            "timestamp": timestamp.isoformat(),
            "analysis": analysis.to_dict(),
            "tracked_objects": tracked_objects,
            "alerts": [a.to_dict() for a in alerts],
            "alert_triggered": bool(alerts)
        }

    def chat(self, user_message: str) -> str:
        """
        Chat with the agent using natural language.

        Args:
            user_message: User's question or command

        Returns:
            Agent's response
        """
        if self.use_api and self.agent_executor:
            try:
                result = self.agent_executor.invoke({
                    "input": user_message,
                    "chat_history": self._get_chat_history(),
                    "current_time": datetime.now().isoformat()
                })

                response = result.get("output", "I couldn't process that request.")

                # Update conversation history
                self.conversation_history.append({
                    "role": "user",
                    "content": user_message
                })
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response
                })

                return response

            except Exception as e:
                return f"Error processing request: {str(e)}"

        else:
            # Fallback to simple query handling without API
            return self._handle_query_offline(user_message)

    def _get_chat_history(self) -> List:
        """Get formatted chat history for agent."""
        history = []
        for msg in self.conversation_history[-10:]:  # Last 10 messages
            if msg["role"] == "user":
                history.append(HumanMessage(content=msg["content"]))
            else:
                history.append(AIMessage(content=msg["content"]))
        return history

    def _handle_query_offline(self, query: str) -> str:
        """Handle queries without API access."""
        query_lower = query.lower()

        # Simple keyword-based handling
        if any(word in query_lower for word in ["truck", "vehicle", "car"]):
            results = self.database.query_frames_by_object_type("vehicle")
            if results:
                return f"Found {len(results)} frames with vehicles. Most recent: {results[0].description}"
            return "No vehicle detections found."

        elif any(word in query_lower for word in ["person", "people", "someone"]):
            results = self.database.query_frames_by_object_type("person")
            if results:
                return f"Found {len(results)} frames with people. Most recent: {results[0].description}"
            return "No person detections found."

        elif any(word in query_lower for word in ["alert", "alerts", "warning"]):
            alerts = self.database.get_alerts(limit=5)
            if alerts:
                return f"Found {len(alerts)} recent alerts. Latest: {alerts[0].description} ({alerts[0].priority})"
            return "No alerts found."

        elif any(word in query_lower for word in ["summary", "overview", "status"]):
            stats = self.database.get_statistics()
            return f"System Status: {stats['total_frames']} frames indexed, {stats['total_alerts']} alerts generated, {stats['total_detections']} detections logged."

        else:
            # Generic search
            results = self.database.query_frames_by_description(query)
            if results:
                return f"Found {len(results)} matching frames. Most recent: {results[0].description}"

            return "I couldn't find specific information about that. Try asking about vehicles, people, alerts, or request a summary."

    def get_context_summary(self) -> str:
        """Get a summary of current context for the agent."""
        stats = self.database.get_statistics()
        recurring = self.tracker.get_recurring_objects()

        return f"""
Current Context:
- Frames indexed: {stats['total_frames']}
- Total alerts: {stats['total_alerts']} (High priority: {stats['high_priority_alerts']})
- Detections: {stats['total_detections']}
- Recurring objects tracked: {len(recurring)}
"""


# ==================== Standalone Usage ====================

if __name__ == "__main__":
    print("=" * 60)
    print("SECURITY ANALYST AGENT - TEST")
    print("=" * 60)

    # Initialize agent (without API for testing)
    agent = SecurityAnalystAgent(use_api=False)

    # Test frame processing
    test_frames = [
        {
            "frame_id": 1,
            "description": "Blue Ford F150 pickup truck entering through main gate",
            "location": {"name": "Main Gate", "zone": "perimeter"},
        },
        {
            "frame_id": 2,
            "description": "Person in dark clothing walking near warehouse",
            "location": {"name": "Warehouse", "zone": "storage"},
        },
    ]

    print("\nProcessing test frames...")
    for frame_data in test_frames:
        result = agent.process_frame(
            frame_id=frame_data["frame_id"],
            timestamp=datetime.now(),
            description=frame_data["description"],
            location=frame_data["location"],
            telemetry={"latitude": 37.77, "longitude": -122.41}
        )

        print(f"\nFrame {result['frame_id']}:")
        print(f"  Analysis: {result['analysis']['description']}")
        print(f"  Objects: {len(result['tracked_objects'])}")
        print(f"  Alerts: {len(result['alerts'])}")

    # Test queries
    print("\n" + "=" * 60)
    print("Testing queries...")

    queries = [
        "Show me all truck events",
        "Any alerts today?",
        "Give me a summary"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        response = agent.chat(query)
        print(f"Response: {response}")

    print("\n" + "=" * 60)
    print(agent.get_context_summary())
