"""
Drone Security Analyst Agent - Main Entry Point

This is the main application that orchestrates all components:
- Telemetry and video frame simulation
- VLM-based frame analysis
- Security alert generation
- Frame indexing and querying
- Interactive agent interface
"""

import argparse
import sys
import time
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich import print as rprint

from .simulator import DroneSimulator, SimulatedEvent
from .database import SecurityDatabase
from .analyzer import FrameAnalyzer
from .alert_engine import AlertEngine, AlertFormatter, Alert
from .agent import SecurityAnalystAgent

# Optional ChromaDB vector store
try:
    from .vector_store import FrameVectorStore
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    VECTOR_STORE_AVAILABLE = False
    print("Note: ChromaDB not available. Install with: pip install chromadb sentence-transformers")

# Optional LangGraph agent
try:
    from .graph_agent import SecurityAgentGraph, create_security_agent, LANGGRAPH_AVAILABLE
except ImportError:
    LANGGRAPH_AVAILABLE = False


console = Console()


class DroneSecuritySystem:
    """
    Main system orchestrator for the Drone Security Analyst Agent.

    Coordinates all components and provides the main application interface.
    """

    def __init__(self, use_api: bool = True, verbose: bool = True, use_langgraph: bool = True):
        """
        Initialize the security system.

        Args:
            use_api: Whether to use OpenAI API for enhanced analysis
            verbose: Whether to print detailed output
            use_langgraph: Whether to use LangGraph for multi-agent orchestration
        """
        self.verbose = verbose
        self.use_api = use_api
        self.use_langgraph = use_langgraph and LANGGRAPH_AVAILABLE

        # Initialize components
        console.print("[bold blue]Initializing Drone Security System...[/bold blue]")

        self.database = SecurityDatabase()
        console.print("  [green][OK][/green] Database initialized")

        # Initialize vector store if available
        self.vector_store = None
        if VECTOR_STORE_AVAILABLE:
            try:
                self.vector_store = FrameVectorStore()
                console.print("  [green][OK][/green] ChromaDB Vector Store initialized")
            except Exception as e:
                console.print(f"  [yellow]![/yellow] Vector store unavailable: {e}")

        self.simulator = DroneSimulator()
        console.print("  [green][OK][/green] Simulator initialized")

        # Choose agent type based on configuration
        if self.use_langgraph:
            try:
                self.agent = SecurityAgentGraph(
                    database=self.database,
                    vector_store=self.vector_store,
                    use_api=use_api
                )
                console.print("  [green][OK][/green] LangGraph Multi-Agent System initialized")
            except Exception as e:
                console.print(f"  [yellow]![/yellow] LangGraph unavailable: {e}, falling back to standard agent")
                self.agent = SecurityAnalystAgent(
                    database=self.database,
                    vector_store=self.vector_store,
                    use_api=use_api
                )
                console.print("  [green][OK][/green] Security Agent initialized")
        else:
            self.agent = SecurityAnalystAgent(
                database=self.database,
                vector_store=self.vector_store,
                use_api=use_api
            )
            console.print("  [green][OK][/green] Security Agent initialized")

        if use_api:
            console.print("  [yellow]![/yellow] Using OpenAI API for enhanced analysis")
        else:
            console.print("  [yellow]![/yellow] Running in offline mode (simulated analysis)")

        console.print("[bold green]System ready![/bold green]\n")

    def run_demo(self, num_events: int = 20):
        """
        Run a demonstration of the system with simulated events.

        Args:
            num_events: Number of events to simulate
        """
        console.print(Panel.fit(
            "[bold]DRONE SECURITY ANALYST - DEMO MODE[/bold]\n"
            f"Processing {num_events} simulated events",
            border_style="blue"
        ))

        events = list(self.simulator.generate_stream(num_events=num_events, include_special=True))

        all_alerts = []

        for i, event in enumerate(events):
            # Process frame
            result = self.agent.process_frame(
                frame_id=event.frame.frame_id,
                timestamp=event.frame.timestamp,
                description=event.frame.description,
                location={
                    "name": event.frame.location_name,
                    "zone": event.frame.location_zone
                },
                telemetry={
                    "latitude": event.telemetry.latitude,
                    "longitude": event.telemetry.longitude
                }
            )

            # Display frame info
            self._display_frame_result(event, result)

            # Collect alerts
            if result["alerts"]:
                all_alerts.extend(result["alerts"])

            # Small delay for readability
            if self.verbose:
                time.sleep(0.3)

        # Display summary
        self._display_demo_summary(events, all_alerts)

    def run_curated_demo(self):
        """Run a curated demo scenario showcasing key features."""
        console.print(Panel.fit(
            "[bold]DRONE SECURITY ANALYST - CURATED DEMO[/bold]\n"
            "Showcasing detection, alerting, and indexing capabilities",
            border_style="green"
        ))

        events = self.simulator.generate_demo_scenario()
        all_alerts = []

        for event in events:
            result = self.agent.process_frame(
                frame_id=event.frame.frame_id,
                timestamp=event.frame.timestamp,
                description=event.frame.description,
                location={
                    "name": event.frame.location_name,
                    "zone": event.frame.location_zone
                },
                telemetry={
                    "latitude": event.telemetry.latitude,
                    "longitude": event.telemetry.longitude
                }
            )

            self._display_frame_result(event, result)

            if result["alerts"]:
                all_alerts.extend(result["alerts"])

            time.sleep(0.5)

        self._display_demo_summary(events, all_alerts)

        # Demonstrate querying
        self._demonstrate_queries()

    def _display_frame_result(self, event: SimulatedEvent, result: dict):
        """Display the result of processing a frame."""
        timestamp = event.frame.timestamp.strftime("%H:%M:%S")
        location = event.frame.location_name

        # Frame info
        console.print(f"\n[dim]{timestamp}[/dim] [bold]{location}[/bold]")
        console.print(f"  {event.frame.description}")

        # Objects detected
        if result["tracked_objects"]:
            objects_str = ", ".join(
                f"{obj.get('type', 'unknown')}" +
                (f" ({obj.get('color', '')} {obj.get('subtype', '')})" if obj.get('color') or obj.get('subtype') else "")
                for obj in result["tracked_objects"]
            )
            console.print(f"  [cyan]Detected:[/cyan] {objects_str}")

        # Alerts
        if result["alerts"]:
            for alert in result["alerts"]:
                priority = alert.get("priority", "MEDIUM")
                color = {"HIGH": "red", "MEDIUM": "yellow", "LOW": "blue"}.get(priority, "white")
                console.print(f"  [bold {color}][ALERT - {priority}][/bold {color}] {alert.get('description', '')}")

    def _display_demo_summary(self, events: list, alerts: list):
        """Display summary after demo."""
        console.print("\n" + "=" * 60)

        # Statistics table
        stats = self.database.get_statistics()

        table = Table(title="Demo Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Frames Processed", str(len(events)))
        table.add_row("Frames Indexed", str(stats["total_frames"]))
        table.add_row("Total Alerts", str(len(alerts)))
        table.add_row("High Priority Alerts", str(len([a for a in alerts if a.get("priority") == "HIGH"])))
        table.add_row("Detections Logged", str(stats["total_detections"]))

        console.print(table)

        # Detection breakdown
        if stats["detections_by_type"]:
            console.print("\n[bold]Detections by Type:[/bold]")
            for obj_type, count in stats["detections_by_type"].items():
                console.print(f"  • {obj_type}: {count}")

        # Recurring objects
        recurring = self.agent.tracker.get_recurring_objects()
        if recurring:
            console.print(f"\n[bold]Recurring Objects Detected:[/bold] {len(recurring)}")
            for obj in recurring[:5]:
                console.print(f"  • {obj['key']}: seen {obj['sighting_count']} times")

    def _demonstrate_queries(self):
        """Demonstrate the query capabilities."""
        console.print("\n" + "=" * 60)
        console.print(Panel.fit("[bold]QUERY DEMONSTRATION[/bold]", border_style="cyan"))

        queries = [
            ("All truck events", "truck"),
            ("Activity at gate", "gate"),
            ("Person detections", "person"),
        ]

        for query_name, search_term in queries:
            console.print(f"\n[bold cyan]Query:[/bold cyan] {query_name}")

            results = self.database.query_frames_by_description(search_term)

            if results:
                console.print(f"  Found {len(results)} results:")
                for r in results[:3]:
                    console.print(f"    • Frame {r.frame_id}: {r.description[:50]}...")
            else:
                console.print("  No results found")

    def interactive_mode(self):
        """Run interactive query mode."""
        console.print(Panel.fit(
            "[bold]INTERACTIVE MODE[/bold]\n"
            "Ask questions about security events\n"
            "Type 'quit' to exit, 'help' for commands",
            border_style="magenta"
        ))

        while True:
            try:
                user_input = console.input("\n[bold green]You:[/bold green] ")

                if user_input.lower() in ['quit', 'exit', 'q']:
                    console.print("[yellow]Exiting interactive mode...[/yellow]")
                    break

                if user_input.lower() == 'help':
                    self._show_help()
                    continue

                if user_input.lower() == 'stats':
                    self._show_stats()
                    continue

                if user_input.lower() == 'alerts':
                    self._show_alerts()
                    continue

                # Process with agent
                response = self.agent.chat(user_input)
                console.print(f"\n[bold blue]Agent:[/bold blue] {response}")

            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Type 'quit' to exit.[/yellow]")

    def _show_help(self):
        """Show help message."""
        help_text = """
[bold]Available Commands:[/bold]
  • [cyan]quit[/cyan] - Exit interactive mode
  • [cyan]stats[/cyan] - Show system statistics
  • [cyan]alerts[/cyan] - Show recent alerts
  • [cyan]help[/cyan] - Show this help message

[bold]Example Queries:[/bold]
  • "Show all truck events"
  • "Any activity near the gate?"
  • "What vehicles were seen today?"
  • "Show alerts from the last hour"
  • "Give me a summary"
  • "Any recurring vehicles?"
"""
        console.print(help_text)

    def _show_stats(self):
        """Show system statistics."""
        stats = self.database.get_statistics()

        table = Table(title="System Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Frames", str(stats["total_frames"]))
        table.add_row("Total Alerts", str(stats["total_alerts"]))
        table.add_row("High Priority Alerts", str(stats["high_priority_alerts"]))
        table.add_row("Total Detections", str(stats["total_detections"]))

        console.print(table)

    def _show_alerts(self):
        """Show recent alerts."""
        alerts = self.database.get_alerts(limit=10)

        if not alerts:
            console.print("[yellow]No alerts found.[/yellow]")
            return

        table = Table(title="Recent Alerts")
        table.add_column("Time", style="dim")
        table.add_column("Priority", style="bold")
        table.add_column("Description")
        table.add_column("Location")

        for alert in alerts:
            priority_color = {"HIGH": "red", "MEDIUM": "yellow", "LOW": "blue"}.get(alert.priority, "white")
            table.add_row(
                alert.timestamp.strftime("%H:%M:%S"),
                f"[{priority_color}]{alert.priority}[/{priority_color}]",
                alert.description[:40] + "..." if len(alert.description) > 40 else alert.description,
                alert.location
            )

        console.print(table)

    def cleanup(self):
        """Clean up resources."""
        self.database.close()
        console.print("[dim]System shutdown complete.[/dim]")


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="Drone Security Analyst Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main --demo              Run demo with random events
  python -m src.main --curated           Run curated demo scenario
  python -m src.main --interactive       Interactive query mode
  python -m src.main --demo --events 50  Demo with 50 events
        """
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration mode with simulated events"
    )

    parser.add_argument(
        "--curated",
        action="store_true",
        help="Run curated demo scenario"
    )

    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start interactive query mode"
    )

    parser.add_argument(
        "--events", "-n",
        type=int,
        default=20,
        help="Number of events to simulate in demo mode (default: 20)"
    )

    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Run without OpenAI API (simulated analysis only)"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )

    parser.add_argument(
        "--clear-db",
        action="store_true",
        help="Clear database before running"
    )

    parser.add_argument(
        "--no-langgraph",
        action="store_true",
        help="Disable LangGraph multi-agent system (use simple agent)"
    )

    args = parser.parse_args()

    # Initialize system
    try:
        system = DroneSecuritySystem(
            use_api=not args.no_api,
            verbose=not args.quiet,
            use_langgraph=not args.no_langgraph
        )

        if args.clear_db:
            system.database.clear_all_data()
            console.print("[yellow]Database cleared.[/yellow]")

        # Determine mode
        if args.curated:
            system.run_curated_demo()

        elif args.demo:
            system.run_demo(num_events=args.events)

        elif args.interactive:
            # Run demo first to populate data, then interactive
            system.run_demo(num_events=10)
            system.interactive_mode()

        else:
            # Default: run curated demo then interactive
            system.run_curated_demo()
            system.interactive_mode()

        system.cleanup()

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        sys.exit(0)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise


if __name__ == "__main__":
    main()
