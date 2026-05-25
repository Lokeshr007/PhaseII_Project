"""
Report generator for security managers and executives.

Produces daily/weekly summaries in plain language that non-technical
stakeholders can understand. Shows trends, top attackers, detection coverage,
and system health at a glance.
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from collections import Counter, defaultdict
import textwrap

from deepdefend.utils.logging import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """
    Generate human-readable security reports.
    
    Reports are designed for:
    - Security managers (understand threat landscape)
    - IT directors (justify security investment)
    - Auditors (evidence of active monitoring)
    
    Two report types:
    - Executive summary (one page, key metrics only)
    - Detailed technical report (full breakdown)
    """
    
    def __init__(self, organization_name: str = "Enterprise"):
        self.organization_name = organization_name
        self._report_data: List[Dict] = []  # Accumulated report data
    
    def add_incident(self, incident_data: Dict):
        """Add an incident to the report accumulator."""
        self._report_data.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **incident_data
        })
    
    def generate_daily_report(self, 
                              metrics: Dict = None,
                              incidents: List[Dict] = None) -> str:
        """
        Generate daily security report.
        """
        incidents = incidents or self._report_data
        metrics = metrics or {}
        
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        lines = []
        lines.append("=" * 60)
        lines.append(f"DEEPDEFEND NIDS — DAILY SECURITY REPORT")
        lines.append(f"Organization: {self.organization_name}")
        lines.append(f"Date: {today}")
        lines.append("=" * 60)
        lines.append("")
        
        # Executive summary
        total_alerts = len(incidents)
        severity_counts = Counter(i.get("severity", "unknown") for i in incidents)
        attack_counts = Counter(i.get("attack_class", "unknown") for i in incidents)
        
        lines.append("EXECUTIVE SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Total alerts: {total_alerts}")
        lines.append(f"  Critical: {severity_counts.get('critical', 0)}")
        lines.append(f"  High:     {severity_counts.get('high', 0)}")
        lines.append(f"  Medium:   {severity_counts.get('medium', 0)}")
        lines.append(f"  Low:      {severity_counts.get('low', 0)}")
        lines.append("")
        
        if attack_counts:
            lines.append("Attack types detected:")
            for attack, count in attack_counts.most_common():
                pct = (count / total_alerts * 100) if total_alerts > 0 else 0
                lines.append(f"  {attack.upper():<8} {count:>5} ({pct:.0f}%)")
            lines.append("")
        
        # Top sources
        ip_counts = Counter(i.get("src_ip", "unknown") for i in incidents)
        if ip_counts:
            lines.append("Top source IPs:")
            for ip, count in ip_counts.most_common(5):
                lines.append(f"  {ip:<20} {count} alerts")
            lines.append("")
        
        # Top targets
        dst_counts = Counter(i.get("dst_ip", "unknown") for i in incidents)
        if dst_counts:
            lines.append("Most targeted assets:")
            for ip, count in dst_counts.most_common(5):
                lines.append(f"  {ip:<20} {count} alerts")
            lines.append("")
        
        # System health
        if metrics:
            lines.append("SYSTEM HEALTH")
            lines.append("-" * 40)
            lines.append(f"Uptime: {metrics.get('uptime_seconds', 0)/3600:.1f} hours")
            lines.append(f"Packets processed: {metrics.get('packets_processed', 0):,}")
            lines.append(f"Flows analyzed: {metrics.get('flows_processed', 0):,}")
            lines.append(f"Active blocks: {metrics.get('blocks_active', 0)}")
            lines.append(f"CPU usage: {metrics.get('cpu_percent', 0):.1f}%")
            lines.append(f"Memory: {metrics.get('memory_mb', 0):.0f} MB")
            lines.append("")
        
        # Recommendations
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 40)
        
        if severity_counts.get('critical', 0) > 0:
            lines.append("⚠️  Critical alerts detected — review immediately.")
        
        if attack_counts.get('u2r', 0) > 0:
            lines.append("⚠️  Privilege escalation (U2R) detected — check affected systems.")
        
        if attack_counts.get('probe', 0) > 10:
            lines.append("📌 High reconnaissance activity — review firewall rules.")
        
        if total_alerts == 0:
            lines.append("✅ No alerts today. System is operating normally.")
        
        lines.append("")
        lines.append("=" * 60)
        lines.append("End of Report")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def generate_weekly_report(self,
                               daily_incidents: List[List[Dict]] = None,
                               daily_metrics: List[Dict] = None) -> str:
        """Generate weekly summary report."""
        today = datetime.now(timezone.utc)
        week_start = (today - timedelta(days=today.weekday())).strftime("%Y-%m-%d")
        week_end = today.strftime("%Y-%m-%d")
        
        lines = []
        lines.append("=" * 60)
        lines.append(f"DEEPDEFEND NIDS — WEEKLY SECURITY REPORT")
        lines.append(f"Organization: {self.organization_name}")
        lines.append(f"Period: {week_start} to {week_end}")
        lines.append("=" * 60)
        lines.append("")
        
        # Aggregate statistics
        if daily_incidents:
            all_incidents = []
            for day_incidents in daily_incidents:
                all_incidents.extend(day_incidents)
            
            total_weekly = len(all_incidents)
            weekly_severity = Counter(i.get("severity", "unknown") for i in all_incidents)
            weekly_attacks = Counter(i.get("attack_class", "unknown") for i in all_incidents)
            
            lines.append(f"Total weekly alerts: {total_weekly}")
            lines.append(f"Average daily alerts: {total_weekly / 7:.1f}")
            lines.append("")
            
            # Trend
            daily_counts = [len(day) for day in daily_incidents]
            lines.append("Daily alert trend:")
            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            max_count = max(daily_counts) if daily_counts else 1
            for day, count in zip(days[:len(daily_counts)], daily_counts):
                bar = "█" * int((count / max_count) * 30) if max_count > 0 else ""
                lines.append(f"  {day}: {bar} {count}")
            lines.append("")
        
        lines.append("=" * 60)
        lines.append("End of Weekly Report")
        lines.append("=" * 60)
        
        return "\n".join(lines)