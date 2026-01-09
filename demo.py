#!/usr/bin/env python3
"""
Quick Demo Script for Drone Security Analyst Agent
==================================================
This script demonstrates the core capabilities of the system:
1. Simulated video frame generation
2. Object detection and analysis
3. Alert generation based on security rules
4. Database querying

Run: python demo.py
"""

import sys
import os
import time
from datetime import datetime, timedelta

# Fix Windows console encoding
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich import box

console = Console(force_terminal=True, legacy_windows=False)

# Simulated frame data for demonstration
DEMO_FRAMES = [
    {
        "frame_id": 1,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "location": "Main Gate",
        "zone": "perimeter",
        "description": "Blue Ford F150 pickup truck entering through main gate",
        "objects": [{"type": "vehicle", "subtype": "pickup truck", "color": "blue", "make": "Ford", "model": "F150"}],
        "alert": None
    },
    {
        "frame_id": 2,
        "timestamp": (datetime.now() + timedelta(seconds=5)).strftime("%Y-%m-%d %H:%M:%S"),
        "location": "Parking Lot",
        "zone": "parking",
        "description": "Vehicle parking in designated area, driver exiting",
        "objects": [
            {"type": "vehicle", "subtype": "pickup truck", "color": "blue"},
            {"type": "person", "description": "male, work clothes"}
        ],
        "alert": None
    },
    {
        "frame_id": 3,
        "timestamp": (datetime.now() + timedelta(seconds=10)).strftime("%Y-%m-%d %H:%M:%S"),
        "location": "Warehouse",
        "zone": "storage",
        "description": "Person in dark clothing walking near warehouse entrance",
        "objects": [{"type": "person", "description": "dark clothing, unknown identity"}],
        "alert": {"rule": "R003", "priority": "MEDIUM", "message": "Activity detected near storage zone"}
    },
    {
        "frame_id": 4,
        "timestamp": (datetime.now() + timedelta(seconds=15)).strftime("%Y-%m-%d %H:%M:%S"),
        "location": "Back Fence",
        "zone": "perimeter",
        "description": "Two individuals near perimeter fence, one carrying equipment",
        "objects": [
            {"type": "person", "description": "carrying toolbox"},
            {"type": "person", "description": "standing nearby"}
        ],
        "alert": {"rule": "R003", "priority": "MEDIUM", "message": "Multiple persons detected at perimeter"}
    },
    {
        "frame_id": 5,
        "timestamp": datetime.now().replace(hour=2, minute=30).strftime("%Y-%m-%d %H:%M:%S"),
        "location": "Main Gate",
        "zone": "perimeter",
        "description": "Person detected at main gate during night hours",
        "objects": [{"type": "person", "description": "unknown, dark clothing"}],
        "alert": {"rule": "R001", "priority": "HIGH", "message": "Person detected at Main Gate during restricted hours (02:30)"}
    },
]

def print_banner():
    """Print the demo banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║       🚁 DRONE SECURITY ANALYST AGENT - LIVE DEMO 🚁         ║
    ║                                                              ║
    ║   Simulating drone patrol with AI-powered security analysis  ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")

def display_frame(frame, delay=2.0):
    """Display a single frame with its analysis."""

    # Frame header
    console.print(f"\n{'='*60}", style="dim")
    console.print(f"📹 [bold]Frame #{frame['frame_id']}[/bold] | {frame['timestamp']}", style="cyan")
    console.print(f"📍 Location: [bold]{frame['location']}[/bold] (Zone: {frame['zone']})")
    console.print(f"{'='*60}", style="dim")

    # Raw frame description (what VLM would see)
    console.print("\n[yellow]▶ RAW FRAME INPUT (Simulated VLM):[/yellow]")
    console.print(Panel(frame['description'], border_style="yellow"))

    # Object detection results
    console.print("\n[green]▶ DETECTED OBJECTS:[/green]")
    obj_table = Table(box=box.ROUNDED)
    obj_table.add_column("Type", style="cyan")
    obj_table.add_column("Details", style="white")

    for obj in frame['objects']:
        obj_type = obj.get('type', 'unknown')
        if obj_type == 'vehicle':
            details = f"{obj.get('color', '')} {obj.get('make', '')} {obj.get('model', '')} {obj.get('subtype', '')}".strip()
        else:
            details = obj.get('description', 'No details')
        obj_table.add_row(obj_type.upper(), details)

    console.print(obj_table)

    # Alert if triggered
    if frame['alert']:
        alert = frame['alert']
        priority_color = "red" if alert['priority'] == "HIGH" else "yellow"
        console.print(f"\n[{priority_color}]⚠️  ALERT TRIGGERED![/{priority_color}]")
        alert_panel = Panel(
            f"[bold]Rule:[/bold] {alert['rule']}\n"
            f"[bold]Priority:[/bold] {alert['priority']}\n"
            f"[bold]Message:[/bold] {alert['message']}",
            title="🚨 Security Alert",
            border_style=priority_color
        )
        console.print(alert_panel)
    else:
        console.print("\n[green]✅ No alerts triggered[/green]")

    time.sleep(delay)

def display_summary():
    """Display a summary of the demo session."""
    console.print("\n" + "="*60, style="bold")
    console.print("📊 [bold cyan]DEMO SESSION SUMMARY[/bold cyan]")
    console.print("="*60, style="bold")

    summary_table = Table(box=box.DOUBLE)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="white")

    alerts_count = sum(1 for f in DEMO_FRAMES if f['alert'])
    high_alerts = sum(1 for f in DEMO_FRAMES if f['alert'] and f['alert']['priority'] == 'HIGH')

    summary_table.add_row("Total Frames Processed", str(len(DEMO_FRAMES)))
    summary_table.add_row("Objects Detected", str(sum(len(f['objects']) for f in DEMO_FRAMES)))
    summary_table.add_row("Alerts Generated", str(alerts_count))
    summary_table.add_row("High Priority Alerts", f"[red]{high_alerts}[/red]")
    summary_table.add_row("Locations Patrolled", str(len(set(f['location'] for f in DEMO_FRAMES))))

    console.print(summary_table)

def demonstrate_query():
    """Demonstrate the query capability."""
    console.print("\n" + "="*60, style="bold")
    console.print("🔍 [bold cyan]QUERY DEMONSTRATION[/bold cyan]")
    console.print("="*60, style="bold")

    queries = [
        ("Show all vehicle detections", ["Frame #1: Blue Ford F150 at Main Gate", "Frame #2: Vehicle in Parking Lot"]),
        ("Find events at perimeter", ["Frame #1: Main Gate - Vehicle entry", "Frame #4: Back Fence - Multiple persons", "Frame #5: Main Gate - Night activity"]),
        ("List all HIGH priority alerts", ["Frame #5: Person at Main Gate during restricted hours (02:30)"])
    ]

    for query, results in queries:
        console.print(f"\n[yellow]Query:[/yellow] \"{query}\"")
        console.print("[green]Results:[/green]")
        for result in results:
            console.print(f"  • {result}")
        time.sleep(1)

def main():
    """Run the demo."""
    print_banner()

    console.print("\n[bold]This demo shows what the Drone Security Agent does:[/bold]")
    console.print("  1. 📹 Receives video frames from drone camera")
    console.print("  2. 🤖 Analyzes frames using AI (simulated VLM)")
    console.print("  3. 🎯 Detects objects (persons, vehicles, etc.)")
    console.print("  4. ⚠️  Triggers alerts based on security rules")
    console.print("  5. 💾 Stores everything for later querying")

    console.print("\n[dim]Press Enter to start the live demo...[/dim]")
    input()

    console.print("\n[bold cyan]🎬 Starting Live Drone Patrol Simulation...[/bold cyan]\n")
    time.sleep(1)

    # Process each frame
    for frame in DEMO_FRAMES:
        display_frame(frame, delay=2.5)

    # Show summary
    display_summary()

    # Demonstrate queries
    demonstrate_query()

    # Final message
    console.print("\n" + "="*60, style="bold green")
    console.print("[bold green]✅ DEMO COMPLETE[/bold green]")
    console.print("="*60, style="bold green")
    console.print("\n[bold]Next steps:[/bold]")
    console.print("  • Run [cyan]streamlit run streamlit_app.py[/cyan] for interactive UI")
    console.print("  • Run [cyan]python -m src.main --curated[/cyan] for full system demo")
    console.print("  • Run [cyan]pytest tests/[/cyan] to verify all 142 test cases")
    console.print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
        sys.exit(0)
