"""
LangGraph-based Security Analyst Agent

A production-grade multi-agent system using LangGraph for orchestrating
specialized agents with:
- Parallel execution for performance
- Human-in-the-loop for critical alerts
- Stateful workflows with persistence
- Conditional routing based on context
- Tool calling with structured outputs

Architecture follows the Supervisor pattern with specialized worker agents.
"""

import json
import operator
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Annotated, TypedDict, Literal, Sequence, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool, StructuredTool
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

# Groq support (open-source LLMs with fast inference)
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    ChatGroq = None

try:
    from langgraph.graph import StateGraph, END, START
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True

    # ToolNode import - langgraph.prebuilt still works but may move to langchain.agents
    try:
        from langgraph.prebuilt import ToolNode
    except ImportError:
        # Fallback for newer versions where it may move
        ToolNode = None
except ImportError:
    LANGGRAPH_AVAILABLE = False
    ToolNode = None
    print("LangGraph not installed. Run: pip install langgraph")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .config import (
    OPENAI_API_KEY, MODEL_NAME, AGENT_CONFIG,
    LLM_PROVIDER, GROQ_API_KEY, GROQ_MODEL_NAME, OPENAI_MODEL_NAME
)
from .database import SecurityDatabase, FrameRecord, parse_natural_query
from .analyzer import FrameAnalyzer, ObjectTracker
from .alert_engine import AlertEngine

# Optional vector store import
try:
    from .vector_store import FrameVectorStore
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    VECTOR_STORE_AVAILABLE = False


# ==================== Enums and Constants ====================

class AlertPriority(str, Enum):
    """Alert priority levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class AgentType(str, Enum):
    """Types of specialized agents."""
    ANALYZER = "analyzer"
    ALERTER = "alerter"
    SEARCHER = "searcher"
    SUMMARIZER = "summarizer"
    SUPERVISOR = "supervisor"


# ==================== State Definitions ====================

class AgentState(TypedDict):
    """
    State that flows through the multi-agent graph.

    This state is shared between all agents and accumulates
    results as the workflow progresses.
    """
    # Input
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_query: str

    # Routing
    intent: str  # 'analyze', 'search', 'alert', 'summarize', 'chat'
    next_agent: str  # Next agent to call
    agent_sequence: List[str]  # Sequence of agents to call

    # Frame processing
    frame_data: Optional[Dict]
    analysis_result: Optional[Dict]

    # Search results
    search_results: List[Dict]

    # Alerts
    alerts: List[Dict]
    critical_alert: bool  # Flag for human-in-the-loop
    requires_human_review: bool

    # Summary
    summary: Optional[str]

    # Final response
    response: str
    confidence: float  # Confidence in the response

    # Metadata
    timestamp: str
    iteration: int
    max_iterations: int

    # Error handling
    errors: List[str]


class SupervisorDecision(BaseModel):
    """Supervisor's routing decision."""
    next_agent: Literal["analyzer", "alerter", "searcher", "summarizer", "finish"]
    reasoning: str = Field(description="Brief explanation for the routing decision")
    parallel_agents: List[str] = Field(default=[], description="Agents to run in parallel if any")


# ==================== Intent Classification ====================

class IntentClassifier:
    """Classifies user intent to route to appropriate agent."""

    INTENT_KEYWORDS = {
        'analyze': ['analyze', 'process', 'detect', 'identify', 'what is', 'describe'],
        'search': ['find', 'search', 'show', 'list', 'query', 'look for', 'where', 'when', 'detection', 'detections'],
        'alert': ['alert', 'alerts', 'warning', 'warnings', 'suspicious', 'danger', 'threat'],
        'summarize': ['summary', 'summarize', 'overview', 'report', 'brief', 'recap'],
    }

    # Higher priority intents that should win in case of ties
    PRIORITY_ORDER = ['alert', 'summarize', 'search', 'analyze']

    @classmethod
    def classify(cls, query: str) -> str:
        """Classify the intent of a user query."""
        query_lower = query.lower()

        # Check for frame processing (special case)
        if 'frame_id' in query_lower or 'process frame' in query_lower:
            return 'analyze'

        # Score each intent
        scores = {}
        for intent, keywords in cls.INTENT_KEYWORDS.items():
            scores[intent] = sum(1 for kw in keywords if kw in query_lower)

        # Return highest scoring intent, using priority order for ties
        max_score = max(scores.values()) if scores else 0
        if max_score > 0:
            # Get all intents with max score and pick by priority
            tied_intents = [i for i, s in scores.items() if s == max_score]
            for priority_intent in cls.PRIORITY_ORDER:
                if priority_intent in tied_intents:
                    return priority_intent
            return tied_intents[0]

        return 'chat'


# ==================== Agent Nodes ====================

class AnalyzerAgent:
    """Agent specialized in analyzing video frames."""

    SYSTEM_PROMPT = """You are a Video Frame Analyzer specializing in security surveillance.
Your job is to analyze frame descriptions and identify:
- Objects (vehicles, people, animals)
- Activities (movement, loitering, suspicious behavior)
- Security-relevant details (time of day, location context)

Provide structured analysis with detected objects and security assessment."""

    def __init__(self, analyzer: FrameAnalyzer, tracker: ObjectTracker):
        self.analyzer = analyzer
        self.tracker = tracker

    def __call__(self, state: AgentState) -> Dict:
        """Process a frame and return analysis results."""
        frame_data = state.get('frame_data')

        if not frame_data:
            return {
                'analysis_result': None,
                'response': "No frame data provided for analysis."
            }

        # Perform analysis
        analysis = self.analyzer.analyze_frame(
            frame_id=frame_data.get('frame_id', 0),
            timestamp=datetime.fromisoformat(frame_data.get('timestamp', datetime.now().isoformat())),
            frame_description=frame_data.get('description', ''),
            location_context=frame_data.get('location', {})
        )

        # Track objects
        tracked_objects = []
        for obj in analysis.detected_objects:
            tracked = self.tracker.track_object(
                obj,
                datetime.fromisoformat(frame_data.get('timestamp', datetime.now().isoformat())),
                frame_data.get('location', {}).get('name', 'Unknown')
            )
            tracked_objects.append(tracked)

        result = {
            'frame_id': frame_data.get('frame_id'),
            'description': analysis.description,
            'objects': tracked_objects,
            'security_relevant': analysis.security_relevant,
            'confidence': analysis.confidence
        }

        return {
            'analysis_result': result,
            'response': f"Analyzed frame {result['frame_id']}: {len(tracked_objects)} objects detected. Security relevant: {result['security_relevant']}"
        }


class AlerterAgent:
    """Agent specialized in evaluating security alerts."""

    SYSTEM_PROMPT = """You are a Security Alert Evaluator.
Your job is to evaluate detected objects and activities against security rules:
- R001: Person detected during night hours (00:00-05:00) -> HIGH
- R002: Loitering detected (same zone > 5 min) -> HIGH
- R003: Activity in perimeter zone -> MEDIUM
- R004: Same vehicle seen > 2 times in 24h -> LOW
- R005: Unknown vehicle in restricted area -> MEDIUM

Generate alerts with priority levels and actionable descriptions."""

    def __init__(self, alert_engine: AlertEngine):
        self.alert_engine = alert_engine

    def __call__(self, state: AgentState) -> Dict:
        """Evaluate state and generate alerts."""
        analysis = state.get('analysis_result')
        frame_data = state.get('frame_data')

        if not analysis or not frame_data:
            return {'alerts': [], 'response': "No data to evaluate for alerts."}

        # Evaluate alerts
        alerts = self.alert_engine.evaluate(
            frame_id=analysis.get('frame_id', 0),
            timestamp=datetime.fromisoformat(frame_data.get('timestamp', datetime.now().isoformat())),
            detected_objects=analysis.get('objects', []),
            location=frame_data.get('location', {}),
            description=analysis.get('description', '')
        )

        alert_dicts = [a.to_dict() for a in alerts]

        if alerts:
            alert_summary = f"Generated {len(alerts)} alert(s): " + ", ".join(
                f"[{a.priority}] {a.description[:30]}..." for a in alerts
            )
        else:
            alert_summary = "No alerts triggered."

        return {
            'alerts': alert_dicts,
            'response': alert_summary
        }


class SearcherAgent:
    """Agent specialized in searching the frame database."""

    SYSTEM_PROMPT = """You are a Security Database Searcher.
Your job is to find relevant security events based on queries:
- Semantic search for conceptually similar events
- Filter by object type (vehicle, person)
- Filter by location/zone
- Filter by time range
- Find similar incidents

Return relevant frames with context."""

    def __init__(self, database: SecurityDatabase, vector_store=None):
        self.database = database
        self.vector_store = vector_store

    def __call__(self, state: AgentState) -> Dict:
        """Search for relevant frames based on query."""
        query = state.get('user_query', '')

        results = []

        # Try semantic search first if vector store available
        if self.vector_store:
            try:
                from .vector_store import SemanticQueryParser
                params = SemanticQueryParser.parse(query)

                vector_results = self.vector_store.hybrid_search(
                    query=query,
                    object_type=params.get('object_type'),
                    location_zone=params.get('location_zone'),
                    alerts_only=params.get('alerts_only', False),
                    n_results=10
                )

                for r in vector_results:
                    results.append({
                        'frame_id': r.frame_id,
                        'timestamp': r.timestamp,
                        'description': r.description,
                        'location': r.location,
                        'similarity_score': r.similarity_score,
                        'source': 'semantic'
                    })
            except Exception as e:
                pass  # Fall back to SQL search

        # Also try SQL-based search
        if not results:
            sql_results = self.database.query_frames_by_description(query)[:10]
            for r in sql_results:
                results.append({
                    'frame_id': r.frame_id,
                    'timestamp': r.timestamp.isoformat(),
                    'description': r.description,
                    'location': r.location_name,
                    'source': 'keyword'
                })

        response = f"Found {len(results)} matching frames."
        if results:
            response += f" Most relevant: {results[0]['description'][:50]}..."

        return {
            'search_results': results,
            'response': response
        }


class SummarizerAgent:
    """Agent specialized in generating summaries."""

    SYSTEM_PROMPT = """You are a Security Report Summarizer.
Your job is to generate concise, actionable summaries of:
- Recent activity patterns
- Alert statistics
- Notable events
- Recommendations

Focus on what matters for security decision-making."""

    def __init__(self, database: SecurityDatabase):
        self.database = database

    def __call__(self, state: AgentState) -> Dict:
        """Generate a summary of security activity."""
        stats = self.database.get_statistics()
        alerts = self.database.get_alerts(limit=5)

        # Build summary
        summary_parts = [
            f"Security Summary as of {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"",
            f"📊 Statistics:",
            f"  • Total frames indexed: {stats['total_frames']}",
            f"  • Total alerts: {stats['total_alerts']}",
            f"  • High priority alerts: {stats['high_priority_alerts']}",
            f"  • Total detections: {stats['total_detections']}",
        ]

        if stats.get('detections_by_type'):
            summary_parts.append(f"")
            summary_parts.append(f"🔍 Detections by Type:")
            for obj_type, count in stats['detections_by_type'].items():
                summary_parts.append(f"  • {obj_type}: {count}")

        if alerts:
            summary_parts.append(f"")
            summary_parts.append(f"⚠️ Recent Alerts:")
            for alert in alerts[:3]:
                summary_parts.append(f"  • [{alert.priority}] {alert.description[:40]}...")

        summary = "\n".join(summary_parts)

        return {
            'summary': summary,
            'response': summary
        }


class ChatAgent:
    """General chat agent for handling miscellaneous queries."""

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm

    def __call__(self, state: AgentState) -> Dict:
        """Handle general chat queries."""
        query = state.get('user_query', '')

        if self.llm:
            # Use LLM for response
            try:
                response = self.llm.invoke([
                    SystemMessage(content="You are a helpful security analyst assistant. Answer questions about the surveillance system."),
                    HumanMessage(content=query)
                ])
                return {'response': response.content}
            except:
                pass

        # Fallback response
        return {
            'response': f"I understand you're asking about '{query}'. Try commands like: 'search for trucks', 'show alerts', 'give me a summary', or provide frame data to analyze."
        }


# ==================== Supervisor Agent ====================

class SupervisorAgent:
    """
    Supervisor agent that uses LLM for intelligent routing decisions.

    The supervisor:
    1. Analyzes the user query/task
    2. Decides which worker agent(s) to invoke
    3. Can run agents in parallel when appropriate
    4. Handles human-in-the-loop for critical decisions
    """

    SUPERVISOR_PROMPT = """You are a supervisor for a security analysis team. Route tasks to the right agent.

AGENTS:
- analyzer: Analyze video frames, detect objects/people/vehicles
- alerter: Evaluate security rules, generate alerts
- searcher: Search database for events
- summarizer: Generate reports/summaries
- finish: Task complete

ROUTING RULES:
- Frame data present → analyzer
- Search/find requests → searcher
- Summary/report requests → summarizer
- Alert evaluation → alerter

REQUEST: {query}
STATE: {state_summary}

Respond ONLY with valid JSON on a single line:
{{"next_agent": "analyzer", "reasoning": "frame data needs analysis"}}"""

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm

    def __call__(self, state: AgentState) -> Dict:
        """Make routing decision based on state."""
        query = state.get('user_query', '')

        # If no LLM, use rule-based routing
        if not self.llm:
            return self._rule_based_routing(state)

        # Build state summary for context
        state_summary = self._build_state_summary(state)

        try:
            # Get LLM decision
            prompt = self.SUPERVISOR_PROMPT.format(
                query=query,
                state_summary=state_summary
            )

            response = self.llm.invoke([
                SystemMessage(content=prompt)
            ])

            # Parse response - handle various JSON formats from LLMs
            import re
            content = response.content.strip()

            # Try multiple patterns to extract JSON
            json_patterns = [
                r'```json\s*(\{.*?\})\s*```',  # JSON in code block
                r'```\s*(\{.*?\})\s*```',      # JSON in generic code block
                r'(\{[^{}]*"next_agent"[^{}]*\})',  # Simple JSON with next_agent
                r'(\{.*?\})',  # Any JSON object
            ]

            decision = None
            for pattern in json_patterns:
                json_match = re.search(pattern, content, re.DOTALL)
                if json_match:
                    try:
                        decision = json.loads(json_match.group(1) if '(' in pattern else json_match.group())
                        if 'next_agent' in decision:
                            break
                    except json.JSONDecodeError:
                        continue

            # If still no JSON, try to extract next_agent directly
            if not decision:
                agent_match = re.search(r'"next_agent"\s*:\s*"(\w+)"', content)
                if agent_match:
                    decision = {'next_agent': agent_match.group(1)}

            if decision and 'next_agent' in decision:
                return {
                    'next_agent': decision.get('next_agent', 'chat'),
                    'agent_sequence': decision.get('parallel_agents', []),
                    'response': decision.get('reasoning', '')
                }
        except Exception as e:
            logger.warning(f"Supervisor LLM failed: {e}, falling back to rules")

        return self._rule_based_routing(state)

    def _rule_based_routing(self, state: AgentState) -> Dict:
        """Fallback rule-based routing."""
        query = state.get('user_query', '').lower()
        frame_data = state.get('frame_data')

        # Frame analysis takes priority
        if frame_data:
            return {'next_agent': 'analyzer', 'agent_sequence': ['analyzer', 'alerter']}

        # Classify intent
        intent = IntentClassifier.classify(query)

        routing = {
            'analyze': {'next_agent': 'analyzer', 'agent_sequence': ['analyzer', 'alerter']},
            'search': {'next_agent': 'searcher', 'agent_sequence': ['searcher']},
            'alert': {'next_agent': 'alerter', 'agent_sequence': ['alerter']},
            'summarize': {'next_agent': 'summarizer', 'agent_sequence': ['summarizer']},
        }

        return routing.get(intent, {'next_agent': 'chat', 'agent_sequence': []})

    def _build_state_summary(self, state: AgentState) -> str:
        """Build a summary of current state for LLM context."""
        parts = []

        if state.get('frame_data'):
            parts.append(f"Frame data provided: {state['frame_data'].get('description', '')[:50]}...")

        if state.get('analysis_result'):
            parts.append(f"Analysis complete: {len(state['analysis_result'].get('objects', []))} objects detected")

        if state.get('alerts'):
            parts.append(f"Alerts generated: {len(state['alerts'])}")

        if state.get('search_results'):
            parts.append(f"Search results: {len(state['search_results'])} found")

        return " | ".join(parts) if parts else "No prior results"


# ==================== Human-in-the-Loop ====================

class HumanReviewNode:
    """
    Node for human-in-the-loop review of critical alerts.

    When a CRITICAL or HIGH priority alert is detected, this node
    can pause execution for human review before proceeding.
    """

    CRITICAL_KEYWORDS = ['intruder', 'breach', 'weapon', 'fire', 'emergency']

    def __init__(self, callback=None):
        """
        Initialize with optional callback for human input.

        Args:
            callback: Function to call for human input. If None, auto-approves.
        """
        self.callback = callback
        self.pending_reviews = []

    def __call__(self, state: AgentState) -> Dict:
        """Check if human review is needed."""
        alerts = state.get('alerts', [])

        # Check for critical alerts
        critical_alerts = [
            a for a in alerts
            if a.get('priority') in ['CRITICAL', 'HIGH']
        ]

        # Check for critical keywords in description
        frame_data = state.get('frame_data', {})
        description = frame_data.get('description', '').lower()
        has_critical_content = any(kw in description for kw in self.CRITICAL_KEYWORDS)

        requires_review = bool(critical_alerts) or has_critical_content

        if requires_review:
            review_reason = []
            if critical_alerts:
                review_reason.append(f"{len(critical_alerts)} high-priority alerts")
            if has_critical_content:
                review_reason.append("critical content detected")

            # If callback provided, request human input
            if self.callback:
                human_response = self.callback({
                    'alerts': critical_alerts,
                    'description': description,
                    'reason': ", ".join(review_reason)
                })

                return {
                    'requires_human_review': False,  # Review completed
                    'response': f"Human review completed: {human_response}"
                }

            # No callback - flag for review but continue
            self.pending_reviews.append({
                'timestamp': datetime.now().isoformat(),
                'alerts': critical_alerts,
                'reason': ", ".join(review_reason)
            })

            return {
                'critical_alert': True,
                'requires_human_review': True,
                'response': f"⚠️ HUMAN REVIEW REQUIRED: {', '.join(review_reason)}"
            }

        return {
            'critical_alert': False,
            'requires_human_review': False
        }

    def get_pending_reviews(self) -> List[Dict]:
        """Get list of pending human reviews."""
        return self.pending_reviews

    def acknowledge_review(self, index: int) -> bool:
        """Acknowledge a pending review."""
        if 0 <= index < len(self.pending_reviews):
            self.pending_reviews.pop(index)
            return True
        return False


# ==================== Tool Definitions for ReAct Pattern ====================

class SecurityTools:
    """Collection of tools for ReAct-style agent execution."""

    def __init__(self, database: SecurityDatabase, vector_store=None, analyzer=None, alert_engine=None):
        self.database = database
        self.vector_store = vector_store
        self.analyzer = analyzer
        self.alert_engine = alert_engine

    def get_tools(self) -> List[Tool]:
        """Get list of tools for agent use."""
        tools = [
            Tool(
                name="search_frames",
                description="Search for video frames matching a query. Use for finding specific events.",
                func=self._search_frames
            ),
            Tool(
                name="get_alerts",
                description="Get recent security alerts. Optionally filter by priority.",
                func=self._get_alerts
            ),
            Tool(
                name="get_statistics",
                description="Get system statistics including frame count, alert count, and detections.",
                func=self._get_statistics
            ),
            Tool(
                name="analyze_description",
                description="Analyze a text description to extract objects and assess security relevance.",
                func=self._analyze_description
            ),
        ]

        if self.vector_store:
            tools.append(Tool(
                name="semantic_search",
                description="Search using semantic similarity. Better for conceptual queries like 'suspicious activity'.",
                func=self._semantic_search
            ))

        return tools

    def _search_frames(self, query: str) -> str:
        """Search frames by keyword."""
        results = self.database.query_frames_by_description(query)[:10]
        if not results:
            return json.dumps({"message": "No frames found", "count": 0})

        return json.dumps({
            "count": len(results),
            "frames": [
                {"frame_id": r.frame_id, "description": r.description[:100], "location": r.location_name}
                for r in results
            ]
        }, indent=2)

    def _get_alerts(self, priority: str = None) -> str:
        """Get alerts, optionally filtered."""
        alerts = self.database.get_alerts(priority=priority if priority else None, limit=10)
        return json.dumps({
            "count": len(alerts),
            "alerts": [
                {"priority": a.priority, "description": a.description, "location": a.location}
                for a in alerts
            ]
        }, indent=2)

    def _get_statistics(self, _: str = "") -> str:
        """Get system statistics."""
        stats = self.database.get_statistics()
        return json.dumps(stats, indent=2)

    def _analyze_description(self, description: str) -> str:
        """Analyze a description for objects."""
        if self.analyzer:
            result = self.analyzer.analyze_frame(
                frame_id=0,
                timestamp=datetime.now(),
                frame_description=description,
                location_context={}
            )
            return json.dumps({
                "objects": result.detected_objects,
                "security_relevant": result.security_relevant,
                "confidence": result.confidence
            }, indent=2)
        return json.dumps({"error": "Analyzer not available"})

    def _semantic_search(self, query: str) -> str:
        """Perform semantic search."""
        if not self.vector_store:
            return json.dumps({"error": "Vector store not available"})

        results = self.vector_store.semantic_search(query, n_results=10)
        return json.dumps({
            "count": len(results),
            "results": [
                {"frame_id": r.frame_id, "description": r.description[:100], "similarity": r.similarity_score}
                for r in results
            ]
        }, indent=2)


# ==================== Graph Builder ====================

class SecurityAgentGraph:
    """
    LangGraph-based multi-agent system for security analysis.

    Architecture:
    ```
                    ┌─────────────┐
                    │   Router    │
                    └──────┬──────┘
                           │
           ┌───────┬───────┼───────┬───────┐
           ▼       ▼       ▼       ▼       ▼
      ┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐
      │Analyzer││Alerter ││Searcher││Summary ││  Chat  │
      └───┬────┘└───┬────┘└───┬────┘└───┬────┘└───┬────┘
          │         │         │         │         │
          └────────┴─────────┴─────────┴─────────┘
                              │
                    ┌─────────▼─────────┐
                    │   Response Builder │
                    └───────────────────┘
    ```
    """

    def __init__(
        self,
        database: Optional[SecurityDatabase] = None,
        vector_store=None,
        use_api: bool = True
    ):
        if not LANGGRAPH_AVAILABLE:
            raise ImportError("LangGraph is required. Install with: pip install langgraph")

        self.database = database or SecurityDatabase()
        self.vector_store = vector_store

        # Check API availability based on provider
        if LLM_PROVIDER == "groq":
            self.use_api = use_api and bool(GROQ_API_KEY) and GROQ_AVAILABLE
        else:
            self.use_api = use_api and bool(OPENAI_API_KEY)

        # Initialize components
        self.analyzer = FrameAnalyzer(use_api=self.use_api)
        self.tracker = ObjectTracker()
        self.alert_engine = AlertEngine(database=self.database)

        # Initialize LLM based on provider
        self.llm = None
        if self.use_api:
            if LLM_PROVIDER == "groq" and GROQ_AVAILABLE:
                self.llm = ChatGroq(
                    model=GROQ_MODEL_NAME,
                    temperature=AGENT_CONFIG["temperature"],
                    api_key=GROQ_API_KEY
                )
                logger.info(f"Using Groq LLM: {GROQ_MODEL_NAME}")
            else:
                self.llm = ChatOpenAI(
                    model=OPENAI_MODEL_NAME,
                    temperature=AGENT_CONFIG["temperature"],
                    api_key=OPENAI_API_KEY
                )
                logger.info(f"Using OpenAI LLM: {OPENAI_MODEL_NAME}")

        # Initialize agent nodes
        self.analyzer_agent = AnalyzerAgent(self.analyzer, self.tracker)
        self.alerter_agent = AlerterAgent(self.alert_engine)
        self.searcher_agent = SearcherAgent(self.database, self.vector_store)
        self.summarizer_agent = SummarizerAgent(self.database)
        self.chat_agent = ChatAgent(self.llm)

        # Initialize supervisor and human review
        self.supervisor = SupervisorAgent(self.llm)
        self.human_review = HumanReviewNode()

        # Initialize security tools for ReAct pattern
        self.security_tools = SecurityTools(
            database=self.database,
            vector_store=self.vector_store,
            analyzer=self.analyzer,
            alert_engine=self.alert_engine
        )

        # Build the graph
        self.graph = self._build_graph()
        self.memory = MemorySaver()
        self.app = self.graph.compile(checkpointer=self.memory)

        logger.info("SecurityAgentGraph initialized successfully")

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow with supervisor pattern.

        Architecture:
        ```
                        ┌─────────────────┐
                        │   Supervisor    │
                        │  (LLM Routing)  │
                        └────────┬────────┘
                                 │
                ┌────────┬───────┼───────┬────────┐
                ▼        ▼       ▼       ▼        ▼
           ┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐
           │Analyzer││Alerter ││Searcher││Summary ││  Chat  │
           └───┬────┘└───┬────┘└───┬────┘└───┬────┘└───┬────┘
               │         │         │         │         │
               └────┬────┴─────────┴─────────┴─────────┘
                    │
           ┌────────▼────────┐
           │  Human Review   │ (for critical alerts)
           └────────┬────────┘
                    │
           ┌────────▼────────┐
           │Response Builder │
           └─────────────────┘
        ```
        """
        # Create graph with state schema
        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("supervisor", self.supervisor)
        graph.add_node("analyzer", self.analyzer_agent)
        graph.add_node("alerter", self.alerter_agent)
        graph.add_node("searcher", self.searcher_agent)
        graph.add_node("summarizer", self.summarizer_agent)
        graph.add_node("chat", self.chat_agent)
        graph.add_node("human_review", self.human_review)
        graph.add_node("response_builder", self._build_response)

        # Add edges from START to supervisor
        graph.add_edge(START, "supervisor")

        # Add conditional edges from supervisor
        graph.add_conditional_edges(
            "supervisor",
            self._get_next_node,
            {
                "analyzer": "analyzer",
                "alerter": "alerter",
                "searcher": "searcher",
                "summarizer": "summarizer",
                "chat": "chat"
            }
        )

        # Analysis flows to alert check
        graph.add_edge("analyzer", "alerter")

        # Alerter flows to human review check
        graph.add_edge("alerter", "human_review")

        # Human review flows to response builder
        graph.add_edge("human_review", "response_builder")

        # Other agents go directly to response builder
        graph.add_edge("searcher", "response_builder")
        graph.add_edge("summarizer", "response_builder")
        graph.add_edge("chat", "response_builder")

        # Add edge from response builder to END
        graph.add_edge("response_builder", END)

        return graph

    def _route_intent(self, state: AgentState) -> Dict:
        """Route to appropriate agent based on intent."""
        query = state.get('user_query', '')
        intent = IntentClassifier.classify(query)

        # Special case: if frame_data provided, always analyze
        if state.get('frame_data'):
            intent = 'analyze'

        return {
            'intent': intent,
            'timestamp': datetime.now().isoformat()
        }

    def _get_next_node(self, state: AgentState) -> str:
        """Determine next node based on supervisor decision."""
        # Use next_agent from supervisor, fallback to intent-based routing
        next_agent = state.get('next_agent')
        if next_agent:
            return next_agent

        intent = state.get('intent', 'chat')

        # Map intent to node
        intent_map = {
            'analyze': 'analyzer',
            'alert': 'alerter',
            'search': 'searcher',
            'summarize': 'summarizer',
            'chat': 'chat'
        }

        return intent_map.get(intent, 'chat')

    def _build_response(self, state: AgentState) -> Dict:
        """Build final response from agent outputs."""
        responses = []

        # Collect responses from different agents
        if state.get('analysis_result'):
            responses.append(f"Analysis: {state.get('response', '')}")

        if state.get('alerts'):
            alert_count = len(state['alerts'])
            high_priority = sum(1 for a in state['alerts'] if a.get('priority') == 'HIGH')
            responses.append(f"Alerts: {alert_count} total ({high_priority} high priority)")

        if state.get('search_results'):
            responses.append(f"Search: Found {len(state['search_results'])} results")

        if state.get('summary'):
            responses.append(state['summary'])

        # Default to the response field if nothing specific
        if not responses and state.get('response'):
            responses.append(state['response'])

        final_response = "\n\n".join(responses) if responses else "I couldn't process that request."

        return {
            'response': final_response,
            'messages': [AIMessage(content=final_response)]
        }

    # ==================== Public Interface ====================

    def process_frame(
        self,
        frame_id: int,
        timestamp: datetime,
        description: str,
        location: Dict,
        telemetry: Dict
    ) -> Dict:
        """
        Process a video frame through the analysis pipeline.

        Args:
            frame_id: Unique frame identifier
            timestamp: Frame timestamp
            description: Frame description
            location: Location data
            telemetry: Telemetry data

        Returns:
            Processing results with analysis and alerts
        """
        # Prepare state
        initial_state = {
            'messages': [HumanMessage(content=f"Analyze frame {frame_id}")],
            'user_query': f"Analyze frame {frame_id}",
            'frame_data': {
                'frame_id': frame_id,
                'timestamp': timestamp.isoformat(),
                'description': description,
                'location': location,
                'telemetry': telemetry
            },
            'intent': '',
            'next_agent': '',
            'agent_sequence': [],
            'analysis_result': None,
            'search_results': [],
            'alerts': [],
            'critical_alert': False,
            'requires_human_review': False,
            'summary': None,
            'response': '',
            'confidence': 0.0,
            'timestamp': '',
            'iteration': 0,
            'max_iterations': 5,
            'errors': []
        }

        # Run the graph
        config = {"configurable": {"thread_id": f"frame_{frame_id}"}}
        result = self.app.invoke(initial_state, config)

        # Index in database
        if result.get('analysis_result'):
            analysis = result['analysis_result']
            frame_record = FrameRecord(
                frame_id=frame_id,
                timestamp=timestamp,
                location_name=location.get('name', 'Unknown'),
                location_zone=location.get('zone', ''),
                latitude=telemetry.get('latitude', 0.0),
                longitude=telemetry.get('longitude', 0.0),
                description=analysis.get('description', description),
                objects=analysis.get('objects', []),
                alert_triggered=bool(result.get('alerts'))
            )
            self.database.index_frame(frame_record)

            # Also index in vector store
            if self.vector_store:
                try:
                    self.vector_store.add_frame(
                        frame_id=frame_id,
                        timestamp=timestamp,
                        description=analysis.get('description', description),
                        objects=analysis.get('objects', []),
                        location_name=location.get('name', 'Unknown'),
                        location_zone=location.get('zone', ''),
                        alert_triggered=bool(result.get('alerts'))
                    )
                except:
                    pass

        return {
            'frame_id': frame_id,
            'timestamp': timestamp.isoformat(),
            'analysis': result.get('analysis_result'),
            'tracked_objects': result.get('analysis_result', {}).get('objects', []),
            'alerts': result.get('alerts', []),
            'response': result.get('response', '')
        }

    def process_image(
        self,
        image_data,
        location: Dict = None,
        timestamp: datetime = None,
        frame_id: int = None
    ) -> Dict:
        """
        Process an actual image through the vision pipeline.

        This is the BEST approach - sends actual image to GPT-4 Vision
        for complete security analysis in ONE API call.

        Args:
            image_data: PIL Image, numpy array, or path to image file
            location: Location info {"name": "...", "zone": "..."}
            timestamp: Frame timestamp (defaults to now)
            frame_id: Frame identifier (auto-generated if not provided)

        Returns:
            Complete analysis including objects, alerts, and threat level
        """
        try:
            from .vision_pipeline import DirectVisionPipeline, PipelineConfig
        except ImportError:
            logger.warning("Vision pipeline not available, falling back to description-based analysis")
            return {"error": "Vision pipeline not available"}

        # Set defaults
        location = location or {"name": "Unknown", "zone": "unknown"}
        timestamp = timestamp or datetime.now()
        frame_id = frame_id or 1

        # Create pipeline with appropriate provider
        provider = "direct" if self.use_api else "simulated"
        config = PipelineConfig(
            provider=provider,
            store_to_database=True
        )

        pipeline = DirectVisionPipeline(config=config, database=self.database)

        # Analyze the image
        result = pipeline.analyze_frame(
            frame_data=image_data,
            location=location,
            timestamp=timestamp,
            frame_id=frame_id
        )

        # Index in vector store if available
        if self.vector_store:
            try:
                self.vector_store.add_frame(
                    frame_id=result.frame_id,
                    timestamp=result.timestamp,
                    description=result.description,
                    objects=result.objects,
                    location_name=location.get('name', 'Unknown'),
                    location_zone=location.get('zone', ''),
                    alert_triggered=bool(result.alerts)
                )
            except Exception as e:
                logger.warning(f"Failed to index in vector store: {e}")

        return {
            'frame_id': result.frame_id,
            'timestamp': result.timestamp.isoformat(),
            'description': result.description,
            'objects': result.objects,
            'alerts': result.alerts,
            'analysis': result.analysis,
            'threat_level': result.threat_level,
            'provider': result.provider,
            'processing_time_ms': result.processing_time_ms
        }

    def process_video(
        self,
        video_path: str,
        frame_interval: int = 5,
        max_frames: int = 50,
        progress_callback=None
    ) -> List[Dict]:
        """
        Process a video file through the vision pipeline.

        Extracts frames and sends each to GPT-4 Vision for complete analysis.

        Args:
            video_path: Path to video file
            frame_interval: Seconds between frame extraction
            max_frames: Maximum frames to extract
            progress_callback: Optional callback(current, total, result)

        Returns:
            List of analysis results for each frame
        """
        try:
            from .vision_pipeline import DirectVisionPipeline, PipelineConfig
        except ImportError:
            logger.warning("Vision pipeline not available")
            return [{"error": "Vision pipeline not available"}]

        # Create pipeline
        provider = "direct" if self.use_api else "simulated"
        config = PipelineConfig(
            provider=provider,
            frame_interval_seconds=frame_interval,
            max_frames=max_frames,
            store_to_database=True
        )

        pipeline = DirectVisionPipeline(config=config, database=self.database)

        # Process video
        results = pipeline.process_video(video_path, progress_callback)

        # Index in vector store
        if self.vector_store:
            for result in results:
                try:
                    self.vector_store.add_frame(
                        frame_id=result.frame_id,
                        timestamp=result.timestamp,
                        description=result.description,
                        objects=result.objects,
                        location_name=result.location.get('name', 'Unknown'),
                        location_zone=result.location.get('zone', ''),
                        alert_triggered=bool(result.alerts)
                    )
                except Exception as e:
                    logger.warning(f"Failed to index frame {result.frame_id} in vector store: {e}")

        return [r.to_dict() for r in results]

    def chat(self, user_message: str) -> str:
        """
        Chat with the agent using natural language.

        Args:
            user_message: User's question or command

        Returns:
            Agent's response
        """
        # Prepare state
        initial_state = {
            'messages': [HumanMessage(content=user_message)],
            'user_query': user_message,
            'frame_data': None,
            'intent': '',
            'analysis_result': None,
            'search_results': [],
            'alerts': [],
            'summary': None,
            'response': '',
            'timestamp': '',
            'iteration': 0
        }

        # Run the graph
        config = {"configurable": {"thread_id": "chat"}}
        result = self.app.invoke(initial_state, config)

        return result.get('response', "I couldn't process that request.")

    def get_context_summary(self) -> str:
        """Get a summary of current system context."""
        stats = self.database.get_statistics()

        return f"""
Current Context (LangGraph Agent):
- Frames indexed: {stats['total_frames']}
- Total alerts: {stats['total_alerts']} (High priority: {stats['high_priority_alerts']})
- Detections: {stats['total_detections']}
- Vector store: {'Available' if self.vector_store else 'Not configured'}
- LLM API: {'Connected' if self.use_api else 'Offline mode'}
"""


# ==================== Convenience Factory ====================

def create_security_agent(
    database: Optional[SecurityDatabase] = None,
    vector_store=None,
    use_api: bool = True,
    use_langgraph: bool = True
):
    """
    Factory function to create the appropriate agent.

    Args:
        database: Security database instance
        vector_store: ChromaDB vector store instance
        use_api: Whether to use OpenAI API
        use_langgraph: Whether to use LangGraph (recommended)

    Returns:
        Agent instance (SecurityAgentGraph or SecurityAnalystAgent)
    """
    if use_langgraph and LANGGRAPH_AVAILABLE:
        return SecurityAgentGraph(
            database=database,
            vector_store=vector_store,
            use_api=use_api
        )
    else:
        # Fall back to original agent
        from .agent import SecurityAnalystAgent
        return SecurityAnalystAgent(
            database=database,
            vector_store=vector_store,
            use_api=use_api
        )


# ==================== Standalone Test ====================

if __name__ == "__main__":
    print("=" * 60)
    print("LANGGRAPH SECURITY AGENT TEST")
    print("=" * 60)

    if not LANGGRAPH_AVAILABLE:
        print("LangGraph not installed. Run: pip install langgraph")
        exit(1)

    # Create agent
    agent = SecurityAgentGraph(use_api=False)

    # Test frame processing
    print("\n--- Frame Processing Test ---")
    result = agent.process_frame(
        frame_id=1,
        timestamp=datetime.now(),
        description="Blue Ford F150 pickup truck at main gate",
        location={"name": "Main Gate", "zone": "perimeter"},
        telemetry={"latitude": 37.77, "longitude": -122.41}
    )
    print(f"Result: {result['response']}")

    # Test chat queries
    print("\n--- Chat Test ---")
    queries = [
        "Show me all truck events",
        "Give me a summary",
        "Any alerts today?",
        "Search for vehicles at the gate"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        response = agent.chat(query)
        print(f"Response: {response[:200]}...")

    print("\n" + "=" * 60)
    print(agent.get_context_summary())
