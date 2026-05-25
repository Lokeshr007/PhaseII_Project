"""
Production flow aggregator for DeepDefend NIDS.

Transforms raw network packets into windowed NSL-KDD-compatible
feature vectors with behavioral extensions for anomaly detection.

Thread-safe producer/consumer architecture for real-time processing.
"""

import time
import struct
import socket
import math
import threading
from typing import Dict, Tuple, Optional, List, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum


# ─── Protocol Definitions ────────────────────────────────────

class Protocol(Enum):
    TCP = 6
    UDP = 17
    ICMP = 1


class TCPFlags:
    FIN = 0x01
    SYN = 0x02
    RST = 0x04
    PSH = 0x08
    ACK = 0x10
    URG = 0x20
    
    @staticmethod
    def to_string(flags_byte: int) -> str:
        names = []
        for mask, name in [(0x01, "FIN"), (0x02, "SYN"), (0x04, "RST"),
                          (0x08, "PSH"), (0x10, "ACK"), (0x20, "URG")]:
            if flags_byte & mask:
                names.append(name)
        return ",".join(names) if names else "NONE"
    
    @staticmethod
    def is_syn_only(flags_byte: int) -> bool:
        return (flags_byte & (TCPFlags.SYN | TCPFlags.ACK)) == TCPFlags.SYN


# ─── Service Detection ────────────────────────────────────────

SERVICE_MAP = {
    21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp",
    53: "domain", 80: "http", 110: "pop_3", 143: "imap4",
    443: "https", 993: "imaps", 995: "pop3s", 3306: "mysql",
    5432: "postgresql", 6379: "redis", 27017: "mongodb",
    8080: "http_alt", 8443: "https_alt"
}

def detect_service(port: int) -> str:
    return SERVICE_MAP.get(port, f"other_{port}")


# ─── Data Structures ──────────────────────────────────────────

@dataclass
class Packet:
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: Protocol
    tcp_flags: int
    payload_size: int
    is_fragment: bool = False
    
    @property
    def flow_key(self) -> Tuple:
        return (self.src_ip, self.dst_ip, self.src_port,
                self.dst_port, self.protocol.name)


class ConnectionState(Enum):
    INIT = "init"
    SYN_SENT = "syn_sent"
    SYN_ACK = "syn_ack"
    ESTABLISHED = "established"
    FIN_WAIT = "fin_wait"
    CLOSED = "closed"


@dataclass
class ConnectionTracker:
    state: ConnectionState = ConnectionState.INIT
    syn_count: int = 0
    syn_ack_count: int = 0
    rst_count: int = 0
    fin_count: int = 0
    
    def update(self, flags: int):
        if (flags & (TCPFlags.SYN | TCPFlags.ACK)) == TCPFlags.SYN:
            self.syn_count += 1
            if self.state == ConnectionState.INIT:
                self.state = ConnectionState.SYN_SENT
        elif (flags & (TCPFlags.SYN | TCPFlags.ACK)) == (TCPFlags.SYN | TCPFlags.ACK):
            self.syn_ack_count += 1
            if self.state == ConnectionState.SYN_SENT:
                self.state = ConnectionState.ESTABLISHED
        elif flags & TCPFlags.RST:
            self.rst_count += 1
            self.state = ConnectionState.CLOSED
        elif flags & TCPFlags.FIN:
            self.fin_count += 1
            if self.state == ConnectionState.ESTABLISHED:
                self.state = ConnectionState.FIN_WAIT


@dataclass
class FlowRecord:
    packet_count: int = 0
    total_src_bytes: int = 0
    total_dst_bytes: int = 0
    start_time: float = 0.0
    last_time: float = 0.0
    syn_count: int = 0
    rst_count: int = 0
    error_count: int = 0
    wrong_fragment_count: int = 0
    urgent_packet_count: int = 0
    connection_tracker: ConnectionTracker = field(default_factory=ConnectionTracker)
    service: str = ""
    src_host: str = ""
    dst_host: str = ""
    
    @property
    def duration(self) -> float:
        return self.last_time - self.start_time if self.start_time else 0.0
    
    def update(self, packet: Packet):
        if self.packet_count == 0:
            self.start_time = packet.timestamp
            self.src_host = packet.src_ip
            self.dst_host = packet.dst_ip
            self.service = detect_service(packet.dst_port)
        
        self.packet_count += 1
        self.total_src_bytes += packet.payload_size
        self.last_time = packet.timestamp
        
        if packet.protocol == Protocol.TCP:
            self.connection_tracker.update(packet.tcp_flags)
            if TCPFlags.is_syn_only(packet.tcp_flags):
                self.syn_count += 1
            if packet.tcp_flags & TCPFlags.RST:
                self.rst_count += 1
                self.error_count += 1
            if packet.tcp_flags & TCPFlags.URG:
                self.urgent_packet_count += 1
        
        if packet.is_fragment:
            self.wrong_fragment_count += 1


# ─── Feature Vector ───────────────────────────────────────────

@dataclass
class FlowFeatures:
    duration: float = 0.0
    protocol_type: str = ""
    service: str = ""
    flag: str = ""
    src_bytes: int = 0
    dst_bytes: int = 0
    land: int = 0
    wrong_fragment: int = 0
    urgent: int = 0
    hot: int = 0
    num_failed_logins: int = 0
    logged_in: int = 0
    num_compromised: int = 0
    root_shell: int = 0
    su_attempted: int = 0
    num_root: int = 0
    num_file_creations: int = 0
    num_shells: int = 0
    num_access_files: int = 0
    num_outbound_cmds: int = 0
    is_host_login: int = 0
    is_guest_login: int = 0
    count: int = 0
    srv_count: int = 0
    serror_rate: float = 0.0
    srv_serror_rate: float = 0.0
    rerror_rate: float = 0.0
    srv_rerror_rate: float = 0.0
    same_srv_rate: float = 0.0
    diff_srv_rate: float = 0.0
    srv_diff_host_rate: float = 0.0
    dst_host_count: int = 0
    dst_host_srv_count: int = 0
    dst_host_same_srv_rate: float = 0.0
    dst_host_diff_srv_rate: float = 0.0
    dst_host_same_src_port_rate: float = 0.0
    dst_host_srv_diff_host_rate: float = 0.0
    dst_host_serror_rate: float = 0.0
    dst_host_srv_serror_rate: float = 0.0
    dst_host_rerror_rate: float = 0.0
    dst_host_srv_rerror_rate: float = 0.0
    connection_rate_per_sec: float = 0.0
    unique_dst_ips: int = 0
    unique_dst_ports: int = 0
    port_entropy: float = 0.0
    byte_ratio_out_in: float = 0.0
    src_ip: str = ""
    dst_ip: str = ""
    src_port: int = 0
    dst_port: int = 0
    
    def to_dict(self) -> dict:
        return {
            "duration": self.duration,
            "protocol_type": self.protocol_type,
            "service": self.service,
            "flag": self.flag,
            "src_bytes": self.src_bytes,
            "dst_bytes": self.dst_bytes,
            "land": self.land,
            "wrong_fragment": self.wrong_fragment,
            "urgent": self.urgent,
            "hot": self.hot,
            "num_failed_logins": self.num_failed_logins,
            "logged_in": self.logged_in,
            "num_compromised": self.num_compromised,
            "root_shell": self.root_shell,
            "su_attempted": self.su_attempted,
            "num_root": self.num_root,
            "num_file_creations": self.num_file_creations,
            "num_shells": self.num_shells,
            "num_access_files": self.num_access_files,
            "num_outbound_cmds": self.num_outbound_cmds,
            "is_host_login": self.is_host_login,
            "is_guest_login": self.is_guest_login,
            "count": self.count,
            "srv_count": self.srv_count,
            "serror_rate": self.serror_rate,
            "srv_serror_rate": self.srv_serror_rate,
            "rerror_rate": self.rerror_rate,
            "srv_rerror_rate": self.srv_rerror_rate,
            "same_srv_rate": self.same_srv_rate,
            "diff_srv_rate": self.diff_srv_rate,
            "srv_diff_host_rate": self.srv_diff_host_rate,
            "dst_host_count": self.dst_host_count,
            "dst_host_srv_count": self.dst_host_srv_count,
            "dst_host_same_srv_rate": self.dst_host_same_srv_rate,
            "dst_host_diff_srv_rate": self.dst_host_diff_srv_rate,
            "dst_host_same_src_port_rate": self.dst_host_same_src_port_rate,
            "dst_host_srv_diff_host_rate": self.dst_host_srv_diff_host_rate,
            "dst_host_serror_rate": self.dst_host_serror_rate,
            "dst_host_srv_serror_rate": self.dst_host_srv_serror_rate,
            "dst_host_rerror_rate": self.dst_host_rerror_rate,
            "dst_host_srv_rerror_rate": self.dst_host_srv_rerror_rate,
            "connection_rate_per_sec": self.connection_rate_per_sec,
            "unique_dst_ips": self.unique_dst_ips,
            "unique_dst_ports": self.unique_dst_ports,
            "port_entropy": self.port_entropy,
            "byte_ratio_out_in": self.byte_ratio_out_in,
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
        }


# ─── Main Aggregator ──────────────────────────────────────────

class FlowAggregator:
    """
    Real-time flow aggregator with sliding window.
    
    Thread-safe producer/consumer architecture.
    Producer: packet capture thread calls process_packet()
    Consumer: inference engine pulls from output_queue
    
    Performance targets:
    - 100K packets/sec on single CPU core
    - Memory: <100MB for ring buffer
    - Graceful degradation under overload (drop oldest windows)
    """
    
    def __init__(self, window_seconds: float = 2.0, slide_seconds: float = 1.0):
        self.window_seconds = window_seconds
        self.slide_seconds = slide_seconds
        
        self._active_flows: Dict[Tuple, FlowRecord] = {}
        self._host_stats: Dict[str, dict] = defaultdict(lambda: defaultdict(set))
        
        self._window_start: float = 0.0
        self._next_window: float = 0.0
        
        self.output_queue: deque = deque(maxlen=10000)
        self._lock = threading.Lock()
        
        self.on_window_complete: Optional[Callable] = None
        
        self._packets_processed: int = 0
        self._windows_completed: int = 0
    
    def process_packet(self, packet: Packet):
        """Thread-safe packet ingestion."""
        with self._lock:
            self._process_packet_unsafe(packet)
            self._packets_processed += 1
    
    def _process_packet_unsafe(self, packet: Packet):
        if self._window_start == 0.0:
            self._window_start = packet.timestamp
            self._next_window = self._window_start + self.window_seconds
        
        while packet.timestamp >= self._next_window:
            self._close_window()
            self._window_start = self._next_window
            self._next_window += self.slide_seconds
        
        key = packet.flow_key
        if key not in self._active_flows:
            self._active_flows[key] = FlowRecord()
        
        self._active_flows[key].update(packet)
    
    def _close_window(self):
        if not self._active_flows:
            return
        
        # Build host statistics
        src_host_stats = defaultdict(lambda: {"connections": 0, "unique_dsts": set(), 
                                                "syn_errors": 0, "rst_errors": 0, "services": set()})
        dst_host_stats = defaultdict(lambda: {"connections": 0, "unique_srcs": set(),
                                                "syn_errors": 0, "rst_errors": 0, "services": set()})
        
        for flow in self._active_flows.values():
            src = flow.src_host
            dst = flow.dst_host
            
            src_host_stats[src]["connections"] += 1
            src_host_stats[src]["unique_dsts"].add(dst)
            src_host_stats[src]["syn_errors"] += flow.syn_count
            src_host_stats[src]["rst_errors"] += flow.rst_count
            src_host_stats[src]["services"].add(flow.service)
            
            dst_host_stats[dst]["connections"] += 1
            dst_host_stats[dst]["unique_srcs"].add(src)
            dst_host_stats[dst]["syn_errors"] += flow.syn_count
            dst_host_stats[dst]["rst_errors"] += flow.rst_count
            dst_host_stats[dst]["services"].add(flow.service)
        
        # Build features
        features_batch = []
        
        for key, flow in self._active_flows.items():
            src_ip, dst_ip, src_port, dst_port, protocol = key
            
            feat = FlowFeatures()
            feat.duration = flow.duration
            feat.protocol_type = protocol.lower()
            feat.service = flow.service
            feat.flag = TCPFlags.to_string(flow.connection_tracker.syn_count)
            feat.src_bytes = flow.total_src_bytes
            feat.dst_bytes = flow.total_dst_bytes
            feat.land = 1 if src_ip == dst_ip and src_port == dst_port else 0
            feat.wrong_fragment = flow.wrong_fragment_count
            feat.urgent = flow.urgent_packet_count
            feat.hot = flow.syn_count
            feat.logged_in = 1 if flow.connection_tracker.state == ConnectionState.ESTABLISHED else 0
            feat.src_ip = src_ip
            feat.dst_ip = dst_ip
            feat.src_port = src_port
            feat.dst_port = dst_port
            
            # Host-based features
            src_stats = src_host_stats.get(src_ip)
            if src_stats:
                feat.count = src_stats["connections"]
                feat.srv_count = sum(1 for f in self._active_flows.values() 
                                    if f.src_host == src_ip and f.service == flow.service)
                total = src_stats["connections"]
                if total > 0:
                    feat.serror_rate = src_stats["syn_errors"] / total
                    feat.rerror_rate = src_stats["rst_errors"] / total
                    same_srv = feat.srv_count
                    feat.same_srv_rate = same_srv / total
                    feat.diff_srv_rate = 1.0 - feat.same_srv_rate
                    feat.srv_diff_host_rate = len(src_stats["unique_dsts"]) / total
            
            dst_stats = dst_host_stats.get(dst_ip)
            if dst_stats:
                feat.dst_host_count = dst_stats["connections"]
                feat.dst_host_srv_count = sum(1 for f in self._active_flows.values()
                                             if f.dst_host == dst_ip and f.service == flow.service)
                total = dst_stats["connections"]
                if total > 0:
                    feat.dst_host_same_srv_rate = feat.dst_host_srv_count / total
                    feat.dst_host_diff_srv_rate = 1.0 - feat.dst_host_same_srv_rate
                    same_src = sum(1 for f in self._active_flows.values()
                                  if f.dst_host == dst_ip and f.src_host == src_ip)
                    feat.dst_host_same_src_port_rate = same_src / total
                    feat.dst_host_srv_diff_host_rate = len(dst_stats["unique_srcs"]) / total
            
            # Behavioral features
            if src_stats:
                feat.connection_rate_per_sec = src_stats["connections"] / self.window_seconds
                feat.unique_dst_ips = len(src_stats["unique_dsts"])
                feat.unique_dst_ports = len(src_stats["services"])
                
                service_counts = defaultdict(int)
                for f in self._active_flows.values():
                    if f.src_host == src_ip:
                        service_counts[f.service] += 1
                if service_counts:
                    total = sum(service_counts.values())
                    probs = [c/total for c in service_counts.values()]
                    feat.port_entropy = -sum(p * math.log2(p) for p in probs if p > 0)
            
            if feat.dst_bytes > 0:
                feat.byte_ratio_out_in = feat.src_bytes / feat.dst_bytes
            
            features_batch.append(feat.to_dict())
        
        if features_batch:
            result = {
                "window_end": self._next_window,
                "window_duration": self.window_seconds,
                "flow_count": len(features_batch),
                "features": features_batch
            }
            self.output_queue.append(result)
            
            if self.on_window_complete:
                self.on_window_complete(features_batch)
        
        self._windows_completed += 1
        self._active_flows.clear()
    
    def get_next_batch(self, timeout: float = 5.0) -> Optional[dict]:
        start = time.time()
        while time.time() - start < timeout:
            if self.output_queue:
                return self.output_queue.popleft()
            time.sleep(0.01)
        return None
    
    def get_stats(self) -> dict:
        return {
            "packets_processed": self._packets_processed,
            "windows_completed": self._windows_completed,
            "queue_depth": len(self.output_queue),
            "active_flows": len(self._active_flows),
        }


class PacketParser:
    """Parse raw bytes into Packet objects."""
    
    @staticmethod
    def parse(raw_data: bytes, timestamp: float) -> Optional[Packet]:
        if len(raw_data) < 20:
            return None
        
        version_ihl = raw_data[0]
        if (version_ihl >> 4) != 4:
            return None
        
        ihl = version_ihl & 0x0F
        header_length = ihl * 4
        
        if len(raw_data) < header_length:
            return None
        
        total_length = struct.unpack('!H', raw_data[2:4])[0]
        protocol_num = raw_data[9]
        src_ip = socket.inet_ntoa(raw_data[12:16])
        dst_ip = socket.inet_ntoa(raw_data[16:20])
        
        flags_offset = struct.unpack('!H', raw_data[6:8])[0]
        is_fragment = (flags_offset & 0x3FFF) != 0
        
        try:
            protocol = Protocol(protocol_num)
        except ValueError:
            return None
        
        src_port = 0
        dst_port = 0
        tcp_flags = 0
        payload_size = 0
        transport_start = header_length
        
        if protocol in (Protocol.TCP, Protocol.UDP):
            if len(raw_data) < transport_start + 4:
                return None
            src_port = struct.unpack('!H', raw_data[transport_start:transport_start+2])[0]
            dst_port = struct.unpack('!H', raw_data[transport_start+2:transport_start+4])[0]
            
            if protocol == Protocol.TCP:
                if len(raw_data) >= transport_start + 14:
                    tcp_flags = raw_data[transport_start + 13]
                    tcp_header_length = ((raw_data[transport_start + 12] >> 4) & 0x0F) * 4
                    payload_start = transport_start + tcp_header_length
                else:
                    payload_start = transport_start + 20
            else:
                payload_start = transport_start + 8
            
            payload_size = max(0, total_length - payload_start)
        
        return Packet(
            timestamp=timestamp,
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=src_port,
            dst_port=dst_port,
            protocol=protocol,
            tcp_flags=tcp_flags,
            payload_size=payload_size,
            is_fragment=is_fragment
        )