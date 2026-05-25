#!/usr/bin/env python3
"""
DeepDefend Interactive Attack Simulator (CLI)

This tool allows you to manually trigger synthetic network attacks 
from the command line, bypassing the engine's standard traffic capture,
and injecting the alert directly into the SOC Dashboard for demonstration.
"""

import time
import sys
import uuid
import random
import requests
import argparse

# Colors for hacker aesthetic
class C:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    PURPLE = '\033[95m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_slow(text, delay=0.03):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def trigger_attack(attack_type, target_ip, source_ip):
    print_slow(f"\n{C.BOLD}{C.CYAN}[*] INITIALIZING EXPLOIT FRAMEWORK v2.4...{C.END}", 0.01)
    time.sleep(0.5)
    
    print(f"{C.YELLOW}[*] Target Acquired:{C.END} {target_ip}")
    print(f"{C.YELLOW}[*] Source IP Spoofed:{C.END} {source_ip}")
    print(f"{C.YELLOW}[*] Selected Vector:{C.END} {attack_type.upper()}")
    
    time.sleep(0.8)
    
    if attack_type == 'dos':
        print_slow(f"{C.GREEN}[+] Launching SYN Flood. Generating 10,000 packets/sec...{C.END}")
        port = 80
        severity = "medium"
        evidence = {
            "Connection Rate": "10,244/sec",
            "TCP Flags": "SYN",
            "ML Anomaly Score": 0.8942
        }
    elif attack_type == 'probe':
        print_slow(f"{C.GREEN}[+] Initiating Nmap Stealth Scan (SYN/FIN/XMAS)...{C.END}")
        port = 0
        severity = "low"
        evidence = {
            "Unique Ports Scanned": "1,024",
            "Scan Speed": "Insane (T5)",
            "ML Anomaly Score": 0.6511
        }
    elif attack_type == 'r2l':
        print_slow(f"{C.PURPLE}[+] Attempting FTP Buffer Overflow / Auth Bypass...{C.END}")
        port = 21
        severity = "high"
        evidence = {
            "Payload Pattern": "NOOP Sled Detected",
            "Authentication": "Bypassed",
            "ML Anomaly Score": 0.9421
        }
    elif attack_type == 'u2r':
        print_slow(f"{C.RED}[+] Injecting Linux Kernel Privilege Escalation Exploit (DirtyCOW)...{C.END}")
        port = 22
        severity = "critical"
        evidence = {
            "Shellcode": "Present",
            "Privilege Esc": "Root Shell spawned",
            "ML Anomaly Score": 0.9984
        }
    else:
        print("Unknown attack type.")
        return
        
    # Simulate work
    for i in range(1, 101, 15):
        sys.stdout.write(f"\r{C.CYAN}[*] Sending payloads... {i}%{C.END}")
        sys.stdout.flush()
        time.sleep(0.2)
    print(f"\r{C.GREEN}[*] Payloads delivered successfully. 100%{C.END}")
    
    # Wait for "Detection"
    time.sleep(1.0)
    print_slow(f"\n{C.BOLD}{C.RED}[!] WARNING: DEEPDEFEND NIDS HAS DETECTED THE INTRUSION!{C.END}", 0.02)
    print_slow(f"{C.YELLOW}[!] Connection terminated by DeepDefend Active Firewall.{C.END}")
    
    # Send to dashboard
    alert = {
        "incident_id": f"INC-{uuid.uuid4().hex[:8].upper()}",
        "severity": severity,
        "attack_class": attack_type,
        "confidence": round(random.uniform(0.85, 0.99), 2),
        "anomaly_score": evidence.get("ML Anomaly Score"),
        "src_ip": source_ip,
        "src_port": random.randint(30000, 65535),
        "dst_ip": target_ip,
        "dst_port": port,
        "protocol": "tcp",
        "detection_source": ["anomaly_ml", "specialist_ml"],
        "action_taken": "blocked" if severity in ["high", "critical"] else "logged",
        "recommended_followup": "Review firewall rules immediately.",
        "evidence_summary": evidence
    }
    
    try:
        requests.post('http://localhost:5000/api/alerts/collect', json=alert)
        print(f"\n{C.BOLD}{C.GREEN}[✔] Incident successfully pushed to SOC Dashboard.{C.END}")
        print(f"{C.BOLD}Check your Web Dashboard Live Feed now!{C.END}\n")
    except Exception as e:
        print(f"\n{C.RED}[x] Failed to connect to Dashboard (is run_dashboard.py running?){C.END}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepDefend Interactive Attack CLI")
    parser.add_argument("--type", choices=['dos', 'probe', 'r2l', 'u2r'], required=True, help="Type of attack to simulate")
    parser.add_argument("--target", default="192.168.1.100", help="Target IP address")
    parser.add_argument("--source", default="10.0.66.6", help="Spoofed Source IP")
    
    args = parser.parse_args()
    trigger_attack(args.type, args.target, args.source)
