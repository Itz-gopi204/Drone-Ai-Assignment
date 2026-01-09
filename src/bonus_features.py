"""
Bonus Features for Drone Security Analyst Agent

This module implements additional features for bonus points:
1. Video Summarization - Generate concise summaries of surveillance sessions
2. Follow-up Q&A - Answer questions about detected objects and events
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

from .database import SecurityDatabase, FrameRecord
from .config import (
    OPENAI_API_KEY, MODEL_NAME, LLM_PROVIDER,
    GROQ_API_KEY, GROQ_MODEL_NAME
)


def get_llm():
    """Get the configured LLM based on provider setting."""
    # Try Groq first if configured
    if LLM_PROVIDER == "groq" and GROQ_API_KEY:
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=GROQ_MODEL_NAME,
                temperature=0.3,
                api_key=GROQ_API_KEY
            )
        except ImportError:
            print("langchain-groq not installed. Run: pip install langchain-groq")

    # Fall back to OpenAI
    if OPENAI_API_KEY:
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=MODEL_NAME,
                temperature=0.3,
                api_key=OPENAI_API_KEY
            )
        except ImportError:
            print("langchain-openai not installed")

    return None


def has_api_key():
    """Check if any LLM API key is configured."""
    if LLM_PROVIDER == "groq":
        return bool(GROQ_API_KEY)
    return bool(OPENAI_API_KEY)


class VideoSummarizer:
    """
    Generates intelligent summaries of surveillance video sessions.

    Analyzes indexed frames and detected events to produce
    human-readable summaries of security activity.
    """

    def __init__(self, database: SecurityDatabase, use_api: bool = True):
        """
        Initialize the video summarizer.

        Args:
            database: Security database instance
            use_api: Whether to use LLM API for enhanced summaries
        """
        self.database = database
        self.use_api = use_api and has_api_key()
        self.llm = None

        if self.use_api:
            self._init_llm()

    def _init_llm(self):
        """Initialize LLM for enhanced summarization."""
        try:
            self.llm = get_llm()
            if not self.llm:
                self.use_api = False
        except Exception as e:
            print(f"Failed to initialize LLM: {e}")
            self.use_api = False

    def summarize_session(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        max_length: int = 200
    ) -> str:
        """
        Generate a summary of a surveillance session.

        Args:
            start_time: Start of session (defaults to last 24 hours)
            end_time: End of session (defaults to now)
            max_length: Maximum summary length in words

        Returns:
            Human-readable summary string
        """
        # Set default time range
        end_time = end_time or datetime.now()
        start_time = start_time or (end_time - timedelta(hours=24))

        # Get frames and alerts
        frames = self.database.query_frames_by_time(start_time, end_time)
        alerts = self.database.get_alerts(limit=100)

        # Filter alerts by time
        session_alerts = [
            a for a in alerts
            if start_time <= a.timestamp <= end_time
        ]

        if not frames:
            return "No surveillance data available for the specified time period."

        if self.use_api:
            return self._generate_llm_summary(frames, session_alerts, start_time, end_time)
        else:
            return self._generate_rule_based_summary(frames, session_alerts, start_time, end_time)

    def _generate_llm_summary(
        self,
        frames: List[FrameRecord],
        alerts: List,
        start_time: datetime,
        end_time: datetime
    ) -> str:
        """Generate summary using LLM."""
        # Prepare context
        frame_descriptions = [f.description for f in frames[:20]]  # Limit for context
        alert_descriptions = [a.description for a in alerts[:10]]

        # Count objects
        object_counts = self._count_objects(frames)

        prompt = f"""Summarize this drone surveillance session in 1-3 sentences.

Time Period: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%H:%M')}
Total Frames: {len(frames)}
Alerts Generated: {len(alerts)}

Object Detections:
{json.dumps(object_counts, indent=2)}

Sample Frame Descriptions:
{chr(10).join(f'- {d}' for d in frame_descriptions[:10])}

Alerts:
{chr(10).join(f'- {d}' for d in alert_descriptions) if alert_descriptions else '- No alerts'}

Provide a concise, professional security summary focusing on:
1. Overall activity level
2. Key objects detected (especially vehicles and people)
3. Any security concerns or alerts
"""

        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return self._generate_rule_based_summary(frames, alerts, start_time, end_time)

    def _generate_rule_based_summary(
        self,
        frames: List[FrameRecord],
        alerts: List,
        start_time: datetime,
        end_time: datetime
    ) -> str:
        """Generate summary using rule-based approach."""
        # Count objects
        object_counts = self._count_objects(frames)

        # Build summary parts
        parts = []

        # Time period
        duration_hours = (end_time - start_time).total_seconds() / 3600
        parts.append(f"Surveillance session: {duration_hours:.1f} hours, {len(frames)} frames processed.")

        # Objects detected
        if object_counts:
            obj_parts = []
            if object_counts.get("vehicle", 0) > 0:
                obj_parts.append(f"{object_counts['vehicle']} vehicle(s)")
            if object_counts.get("person", 0) > 0:
                obj_parts.append(f"{object_counts['person']} person(s)")
            if obj_parts:
                parts.append(f"Detected: {', '.join(obj_parts)}.")

        # Alerts
        if alerts:
            high_priority = len([a for a in alerts if a.priority == "HIGH"])
            if high_priority > 0:
                parts.append(f"{len(alerts)} alerts generated ({high_priority} high priority).")
            else:
                parts.append(f"{len(alerts)} alerts generated.")
        else:
            parts.append("No security alerts triggered.")

        return " ".join(parts)

    def _count_objects(self, frames: List[FrameRecord]) -> Dict[str, int]:
        """Count objects across all frames."""
        counts = {}
        for frame in frames:
            for obj in frame.objects:
                obj_type = obj.get("type", "unknown")
                counts[obj_type] = counts.get(obj_type, 0) + 1
        return counts

    def generate_daily_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive daily security report.

        Returns:
            Dictionary with report sections
        """
        today_start = datetime.now().replace(hour=0, minute=0, second=0)
        today_end = datetime.now()

        frames = self.database.query_frames_by_time(today_start, today_end)
        alerts = self.database.get_alerts(limit=100)
        today_alerts = [a for a in alerts if a.timestamp >= today_start]
        stats = self.database.get_statistics()

        # Group frames by hour
        hourly_activity = {}
        for frame in frames:
            hour = frame.timestamp.hour
            hourly_activity[hour] = hourly_activity.get(hour, 0) + 1

        # Find peak activity hour
        peak_hour = max(hourly_activity.items(), key=lambda x: x[1])[0] if hourly_activity else None

        return {
            "date": today_start.strftime("%Y-%m-%d"),
            "summary": self.summarize_session(today_start, today_end),
            "statistics": {
                "total_frames": len(frames),
                "total_alerts": len(today_alerts),
                "high_priority_alerts": len([a for a in today_alerts if a.priority == "HIGH"]),
                "detections_by_type": self._count_objects(frames)
            },
            "hourly_activity": hourly_activity,
            "peak_activity_hour": peak_hour,
            "alerts": [
                {
                    "time": a.timestamp.strftime("%H:%M"),
                    "priority": a.priority,
                    "description": a.description
                }
                for a in today_alerts[:10]
            ]
        }


class SecurityQA:
    """
    Question-Answering system for security queries.

    Provides natural language interface for querying
    surveillance data and security events.
    """

    def __init__(self, database: SecurityDatabase, use_api: bool = True):
        """
        Initialize the Q&A system.

        Args:
            database: Security database instance
            use_api: Whether to use LLM for enhanced responses
        """
        self.database = database
        self.use_api = use_api and has_api_key()
        self.llm = None

        if self.use_api:
            self._init_llm()

    def _init_llm(self):
        """Initialize LLM for Q&A."""
        try:
            self.llm = get_llm()
            if not self.llm:
                self.use_api = False
        except Exception as e:
            print(f"Failed to initialize LLM: {e}")
            self.use_api = False

    def answer(self, question: str) -> str:
        """
        Answer a security-related question.

        Args:
            question: Natural language question

        Returns:
            Answer string
        """
        question_lower = question.lower()

        # Determine question type and gather relevant data
        if self._is_object_query(question_lower):
            return self._answer_object_query(question_lower)
        elif self._is_time_query(question_lower):
            return self._answer_time_query(question_lower)
        elif self._is_alert_query(question_lower):
            return self._answer_alert_query(question_lower)
        elif self._is_location_query(question_lower):
            return self._answer_location_query(question_lower)
        elif self._is_summary_query(question_lower):
            return self._answer_summary_query()
        else:
            return self._answer_general_query(question)

    def _is_object_query(self, q: str) -> bool:
        """Check if question is about objects."""
        keywords = ["vehicle", "car", "truck", "person", "people", "object", "what was", "who was"]
        return any(k in q for k in keywords)

    def _is_time_query(self, q: str) -> bool:
        """Check if question is about time."""
        keywords = ["when", "what time", "today", "yesterday", "last hour", "morning", "night"]
        return any(k in q for k in keywords)

    def _is_alert_query(self, q: str) -> bool:
        """Check if question is about alerts."""
        keywords = ["alert", "warning", "security", "incident", "suspicious"]
        return any(k in q for k in keywords)

    def _is_location_query(self, q: str) -> bool:
        """Check if question is about locations."""
        keywords = ["where", "location", "gate", "parking", "warehouse", "fence"]
        return any(k in q for k in keywords)

    def _is_summary_query(self, q: str) -> bool:
        """Check if question is asking for summary."""
        keywords = ["summary", "overview", "report", "what happened", "status"]
        return any(k in q for k in keywords)

    def _answer_object_query(self, question: str) -> str:
        """Answer questions about detected objects."""
        # Determine object type
        if any(w in question for w in ["truck", "car", "vehicle"]):
            obj_type = "vehicle"
            results = self.database.query_frames_by_object_type("vehicle")
        elif any(w in question for w in ["person", "people", "someone"]):
            obj_type = "person"
            results = self.database.query_frames_by_object_type("person")
        else:
            # Search in descriptions
            results = self.database.get_all_frames(limit=50)
            results = [r for r in results if r.objects]

        if not results:
            return f"No {obj_type if 'obj_type' in dir() else 'objects'} detected in the surveillance data."

        # Format response
        response_parts = [f"Found {len(results)} relevant detections:"]

        for r in results[:5]:
            time_str = r.timestamp.strftime("%H:%M")
            response_parts.append(f"• [{time_str}] {r.description} at {r.location_name}")

        if len(results) > 5:
            response_parts.append(f"... and {len(results) - 5} more.")

        return "\n".join(response_parts)

    def _answer_time_query(self, question: str) -> str:
        """Answer questions about timing of events."""
        # Determine time range
        now = datetime.now()

        if "today" in question:
            start = now.replace(hour=0, minute=0, second=0)
            period = "today"
        elif "yesterday" in question:
            start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0)
            now = start.replace(hour=23, minute=59)
            period = "yesterday"
        elif "last hour" in question:
            start = now - timedelta(hours=1)
            period = "in the last hour"
        elif "night" in question:
            start = now.replace(hour=0, minute=0, second=0)
            now = now.replace(hour=6, minute=0)
            period = "during night hours"
        else:
            start = now - timedelta(hours=24)
            period = "in the last 24 hours"

        frames = self.database.query_frames_by_time(start, now)

        if not frames:
            return f"No events recorded {period}."

        # Find frames with detections
        with_objects = [f for f in frames if f.objects]

        response = f"Activity {period}:\n"
        response += f"• {len(frames)} frames processed\n"
        response += f"• {len(with_objects)} frames with detections\n"

        if with_objects:
            response += "\nKey events:\n"
            for f in with_objects[:5]:
                response += f"• [{f.timestamp.strftime('%H:%M')}] {f.description}\n"

        return response

    def _answer_alert_query(self, question: str) -> str:
        """Answer questions about alerts."""
        alerts = self.database.get_alerts(limit=20)

        if not alerts:
            return "No security alerts have been generated."

        high = [a for a in alerts if a.priority == "HIGH"]
        medium = [a for a in alerts if a.priority == "MEDIUM"]
        low = [a for a in alerts if a.priority == "LOW"]

        response = f"Security Alert Summary:\n"
        response += f"• Total alerts: {len(alerts)}\n"
        response += f"• High priority: {len(high)}\n"
        response += f"• Medium priority: {len(medium)}\n"
        response += f"• Low priority: {len(low)}\n"

        if high:
            response += "\nHigh priority alerts:\n"
            for a in high[:3]:
                response += f"• [{a.timestamp.strftime('%H:%M')}] {a.description}\n"

        return response

    def _answer_location_query(self, question: str) -> str:
        """Answer questions about specific locations."""
        # Extract location from question
        locations = ["gate", "parking", "warehouse", "fence", "garage", "office"]
        target_location = None

        for loc in locations:
            if loc in question:
                target_location = loc
                break

        if target_location:
            results = self.database.query_frames_by_description(target_location)
        else:
            # Get activity by zone
            stats = self.database.get_statistics()
            return f"Database contains {stats['total_frames']} frames across all locations. Ask about a specific location (gate, parking, warehouse, etc.)."

        if not results:
            return f"No activity recorded at {target_location}."

        response = f"Activity at {target_location}:\n"
        response += f"• {len(results)} events recorded\n"

        for r in results[:5]:
            response += f"• [{r.timestamp.strftime('%H:%M')}] {r.description}\n"

        return response

    def _answer_summary_query(self) -> str:
        """Answer summary/overview questions."""
        stats = self.database.get_statistics()

        response = "Security System Summary:\n"
        response += f"• Frames indexed: {stats['total_frames']}\n"
        response += f"• Total alerts: {stats['total_alerts']}\n"
        response += f"• High priority alerts: {stats['high_priority_alerts']}\n"
        response += f"• Total detections: {stats['total_detections']}\n"

        if stats['detections_by_type']:
            response += "\nDetections by type:\n"
            for obj_type, count in stats['detections_by_type'].items():
                response += f"• {obj_type}: {count}\n"

        return response

    def _answer_general_query(self, question: str) -> str:
        """Handle general queries using LLM if available."""
        if self.use_api:
            # Gather context
            stats = self.database.get_statistics()
            recent_frames = self.database.get_all_frames(limit=10)
            recent_alerts = self.database.get_alerts(limit=5)

            context = f"""
Statistics: {json.dumps(stats)}
Recent Frames: {[f.description for f in recent_frames]}
Recent Alerts: {[a.description for a in recent_alerts]}
"""

            prompt = f"""You are a security analyst AI. Answer this question based on the surveillance data:

Question: {question}

Context:
{context}

Provide a helpful, concise answer."""

            try:
                response = self.llm.invoke(prompt)
                return response.content
            except:
                pass

        return "I couldn't find specific information about that. Try asking about:\n• Vehicles or people detected\n• Security alerts\n• Activity at specific locations\n• Summary or status"


# Example usage and demonstration
def demo_bonus_features():
    """Demonstrate bonus features."""
    from .database import SecurityDatabase

    print("=" * 60)
    print("BONUS FEATURES DEMONSTRATION")
    print("=" * 60)

    db = SecurityDatabase()

    # Video Summarization
    print("\n1. VIDEO SUMMARIZATION")
    print("-" * 40)

    summarizer = VideoSummarizer(db, use_api=False)
    summary = summarizer.summarize_session()
    print(f"Session Summary: {summary}")

    report = summarizer.generate_daily_report()
    print(f"\nDaily Report:")
    print(f"  Date: {report['date']}")
    print(f"  Summary: {report['summary']}")
    print(f"  Statistics: {report['statistics']}")

    # Q&A System
    print("\n2. QUESTION & ANSWER SYSTEM")
    print("-" * 40)

    qa = SecurityQA(db, use_api=False)

    questions = [
        "What vehicles were detected today?",
        "Any security alerts?",
        "What happened at the gate?",
        "Give me a summary"
    ]

    for q in questions:
        print(f"\nQ: {q}")
        answer = qa.answer(q)
        print(f"A: {answer}")

    db.close()


if __name__ == "__main__":
    demo_bonus_features()
