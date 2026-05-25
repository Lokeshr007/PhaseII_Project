"""
DeepDefend NIDS - Web Dashboard
Flask-based web interface for monitoring alerts, metrics, and system health.
Runs alongside the detection engine.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template, jsonify, request
import threading
import time
import queue

app = Flask(__name__)

# Alert storage (in-memory for demo; use Redis/DB for production)
alert_store = []
alert_store_lock = threading.Lock()
MAX_ALERTS = 500

# Blocked IPs
blocked_ips = []

# System status
system_status = {
    "engine_running": False,
    "snort_running": False,
    "models_loaded": False,
    "uptime_seconds": 0,
    "start_time": datetime.now().isoformat(),
    "alerts_total": 0,
    "alerts_by_severity": {"critical": 0, "high": 0, "medium": 0, "low": 0},
    "alerts_by_class": defaultdict(int),
    "packets_processed": 0,
    "flows_analyzed": 0,
    "false_positive_rate": 0.0,
    "attack_detection_rate": 0.0,
}


def add_alert(alert_data):
    """Add an alert to the store."""
    with alert_store_lock:
        alert_store.append({
            "timestamp": datetime.now().isoformat(),
            **alert_data
        })
        
        # Keep only last N alerts
        if len(alert_store) > MAX_ALERTS:
            alert_store.pop(0)
        
        # Update counters
        severity = alert_data.get("severity", "low")
        system_status["alerts_total"] += 1
        system_status["alerts_by_severity"][severity] = \
            system_status["alerts_by_severity"].get(severity, 0) + 1
        
        attack_class = alert_data.get("attack_class", "unknown")
        system_status["alerts_by_class"][attack_class] += 1


def get_recent_alerts(limit=50):
    """Get most recent alerts."""
    with alert_store_lock:
        return list(reversed(alert_store[-limit:]))


def get_alerts_by_severity():
    """Get alert counts by severity."""
    with alert_store_lock:
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for alert in alert_store:
            sev = alert.get("severity", "low")
            counts[sev] = counts.get(sev, 0) + 1
        return counts


def get_alerts_by_class():
    """Get alert counts by attack class."""
    with alert_store_lock:
        counts = defaultdict(int)
        for alert in alert_store:
            cls = alert.get("attack_class", "unknown")
            counts[cls] += 1
        return dict(counts)


# ─── Routes ────────────────────────────────────────────────

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('dashboard.html')


@app.route('/alerts')
def alerts_page():
    """Alerts detail page."""
    return render_template('alerts.html')


@app.route('/metrics')
def metrics_page():
    """Metrics page."""
    return render_template('metrics.html')


@app.route('/about')
def about_page():
    """About page."""
    return render_template('about.html')


# ─── API Endpoints ─────────────────────────────────────────

@app.route('/api/status')
def api_status():
    """Get system status."""
    return jsonify(system_status)


@app.route('/api/alerts')
def api_alerts():
    """Get recent alerts."""
    limit = request.args.get('limit', 50, type=int)
    severity = request.args.get('severity', None)
    attack_class = request.args.get('class', None)
    
    alerts = get_recent_alerts(limit=200)
    
    # Filter
    if severity:
        alerts = [a for a in alerts if a.get('severity') == severity]
    if attack_class:
        alerts = [a for a in alerts if a.get('attack_class') == attack_class]
    
    return jsonify(alerts[:limit])


@app.route('/api/alerts/summary')
def api_alerts_summary():
    """Get alert summary statistics."""
    return jsonify({
        "by_severity": get_alerts_by_severity(),
        "by_class": get_alerts_by_class(),
        "total": len(alert_store),
        "recent_5": get_recent_alerts(limit=5),
        "blocked_ips": blocked_ips
    })


@app.route('/api/metrics')
def api_metrics():
    """Get metrics (proxy to Prometheus or local)."""
    try:
        import requests
        resp = requests.get('http://localhost:9090/metrics', timeout=2)
        return resp.text, 200, {'Content-Type': 'text/plain'}
    except:
        return jsonify({
            "error": "Metrics endpoint not available",
            "note": "Ensure the DeepDefend engine is running"
        }), 503


@app.route('/api/health')
def api_health():
    """Health check endpoint."""
    try:
        import requests
        resp = requests.get('http://localhost:8080/health', timeout=2)
        return jsonify(resp.json())
    except:
        return jsonify({
            "status": "unknown",
            "engine": "not reachable",
            "dashboard": "running"
        })


@app.route('/api/demo/generate')
def api_demo_generate():
    """Generate a demo alert for testing the dashboard."""
    import random
    import uuid
    
    req_class = request.args.get('class')
    req_severity = request.args.get('severity')
    
    if req_class:
        attack_class = req_class
    else:
        attack_classes = ["dos", "probe", "r2l", "u2r"]
        attack_class = random.choices(attack_classes, weights=[0.4, 0.3, 0.2, 0.1])[0]
        
    if req_severity:
        severity = req_severity
    else:
        if attack_class == "u2r": severity = "critical"
        elif attack_class == "r2l": severity = "high"
        elif attack_class == "dos": severity = "medium"
        else: severity = random.choice(["low", "medium"])
    
    alert = {
        "incident_id": f"INC-{uuid.uuid4().hex[:8].upper()}",
        "severity": severity,
        "attack_class": attack_class,
        "confidence": round(random.uniform(0.3, 0.99), 2),
        "anomaly_score": round(random.uniform(0.5, 0.99), 4),
        "src_ip": f"10.0.0.{random.randint(1, 254)}",
        "src_port": random.randint(30000, 65535),
        "dst_ip": f"192.168.1.{random.randint(1, 254)}",
        "dst_port": random.choice([22, 80, 443, 3306, 8080]),
        "protocol": random.choice(["tcp", "udp"]),
        "detection_source": ["anomaly_ml", "specialist_ml"],
        "action_taken": "logged",
        "recommended_followup": "Investigate source IP and review target logs.",
        "evidence_summary": {
            "ML Anomaly Score": round(random.uniform(0.5, 0.99), 4),
            "Connection Rate": f"{random.randint(10, 500)}/sec",
            "Unique Ports": str(random.randint(1, 50)),
        }
    }
    
    add_alert(alert)
    return jsonify({"status": "generated", "alert": alert})


# ─── Alert Collector (called by engine) ────────────────────

@app.route('/api/alerts/collect', methods=['POST'])
def api_collect_alert():
    """API endpoint for the engine to push alerts via HTTP."""
    try:
        alert_data = request.json
        if not alert_data:
            return jsonify({"error": "No data"}), 400
            
        collect_alert(alert_data)
        return jsonify({"status": "collected"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/status/update', methods=['POST'])
def api_update_status():
    """API endpoint for the engine to update status via HTTP."""
    try:
        status_data = request.json
        if not status_data:
            return jsonify({"error": "No data"}), 400
            
        for key, value in status_data.items():
            update_status(key, value)
        return jsonify({"status": "updated"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def collect_alert(alert_data):
    """External interface for the engine to push alerts."""
    # Normalize alert data
    normalized = {
        "incident_id": alert_data.get("incident_id", "UNKNOWN"),
        "severity": alert_data.get("severity", "low"),
        "attack_class": alert_data.get("attack_class", "unknown"),
        "confidence": alert_data.get("confidence", 0.0),
        "anomaly_score": alert_data.get("anomaly_score", 0.0),
        "src_ip": alert_data.get("src_ip", "0.0.0.0"),
        "src_port": alert_data.get("src_port", 0),
        "dst_ip": alert_data.get("dst_ip", "0.0.0.0"),
        "dst_port": alert_data.get("dst_port", 0),
        "protocol": alert_data.get("protocol", "unknown"),
        "detection_source": alert_data.get("detection_source", []),
        "action_taken": alert_data.get("action_taken", "logged"),
        "recommended_followup": alert_data.get("recommended_followup", "Investigate."),
        "evidence_summary": alert_data.get("evidence_summary", {}),
        "snort_rule_id": alert_data.get("snort_rule_id"),
    }
    
    # Auto-block high severity
    if normalized["severity"] in ["high", "critical"] and normalized["action_taken"] != "blocked":
        normalized["action_taken"] = "blocked"
        
    if normalized["action_taken"] == "blocked":
        src_ip = normalized["src_ip"]
        if not any(b['ip'] == src_ip for b in blocked_ips):
            import datetime
            blocked_ips.insert(0, {
                "ip": src_ip,
                "reason": f"Auto-mitigation for {normalized['attack_class'].upper()} attack",
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            # Keep only last 50
            while len(blocked_ips) > 50:
                blocked_ips.pop()
                
    add_alert(normalized)


def update_status(key, value):
    """Update system status from engine."""
    system_status[key] = value


# ─── Start Dashboard ───────────────────────────────────────

def start_dashboard(host='0.0.0.0', port=5000, debug=False):
    """Start the Flask dashboard server."""
    print(f"\n{'='*60}")
    print(f"  DeepDefend Web Dashboard")
    print(f"  http://localhost:{port}")
    print(f"{'='*60}")
    print(f"  Pages:")
    print(f"    /         - Main Dashboard")
    print(f"    /alerts   - Alert Details")
    print(f"    /metrics  - System Metrics")
    print(f"    /about    - About DeepDefend")
    print(f"{'='*60}\n")
    
    app.run(host=host, port=port, debug=debug, use_reloader=False)


if __name__ == '__main__':
    start_dashboard(debug=True)