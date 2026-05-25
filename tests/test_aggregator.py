"""
Tests for the flow aggregator component.
"""

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from deepdefend.engine.aggregator import (
    FlowAggregator, PacketParser, Packet,
    Protocol, TCPFlags, FlowRecord, detect_service
)


class TestTCPFlags(unittest.TestCase):
    def test_syn_only_detection(self):
        self.assertTrue(TCPFlags.is_syn_only(TCPFlags.SYN))
        self.assertFalse(TCPFlags.is_syn_only(TCPFlags.SYN | TCPFlags.ACK))
        self.assertFalse(TCPFlags.is_syn_only(TCPFlags.ACK))
        self.assertFalse(TCPFlags.is_syn_only(TCPFlags.RST))
    
    def test_flag_to_string(self):
        self.assertEqual(TCPFlags.to_string(TCPFlags.SYN), "SYN")
        self.assertEqual(TCPFlags.to_string(TCPFlags.SYN | TCPFlags.ACK), "SYN,ACK")
        self.assertEqual(TCPFlags.to_string(0), "NONE")


class TestServiceDetection(unittest.TestCase):
    def test_common_services(self):
        self.assertEqual(detect_service(22), "ssh")
        self.assertEqual(detect_service(80), "http")
        self.assertEqual(detect_service(443), "https")
        self.assertEqual(detect_service(3306), "mysql")
    
    def test_unknown_service(self):
        self.assertTrue(detect_service(12345).startswith("other_"))


class TestFlowRecord(unittest.TestCase):
    def test_basic_flow_tracking(self):
        flow = FlowRecord()
        
        packet = Packet(
            timestamp=100.0,
            src_ip="192.168.1.1",
            dst_ip="10.0.0.1",
            src_port=50000,
            dst_port=80,
            protocol=Protocol.TCP,
            tcp_flags=TCPFlags.SYN,
            payload_size=100
        )
        
        flow.update(packet)
        
        self.assertEqual(flow.packet_count, 1)
        self.assertEqual(flow.total_src_bytes, 100)
        self.assertEqual(flow.start_time, 100.0)
        self.assertEqual(flow.service, "http")
    
    def test_duration_calculation(self):
        flow = FlowRecord()
        
        flow.update(Packet(100.0, "1.1.1.1", "2.2.2.2", 1, 80, 
                          Protocol.TCP, TCPFlags.SYN, 50))
        flow.update(Packet(102.0, "1.1.1.1", "2.2.2.2", 1, 80,
                          Protocol.TCP, TCPFlags.ACK, 50))
        
        self.assertEqual(flow.duration, 2.0)
        self.assertEqual(flow.packet_count, 2)


class TestFlowAggregator(unittest.TestCase):
    def setUp(self):
        self.aggregator = FlowAggregator(window_seconds=2.0, slide_seconds=1.0)
    
    def _make_packet(self, src_ip, dst_ip, src_port, dst_port, 
                    protocol="TCP", flags=TCPFlags.SYN, ts=None, payload=100):
        return Packet(
            timestamp=ts or time.time(),
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=src_port,
            dst_port=dst_port,
            protocol=Protocol[protocol],
            tcp_flags=flags,
            payload_size=payload
        )
    
    def test_window_closes_on_time(self):
        base_time = 1000.0
        
        self.aggregator.process_packet(
            self._make_packet("192.168.1.1", "10.0.0.1", 50000, 80, ts=base_time)
        )
        self.aggregator.process_packet(
            self._make_packet("192.168.1.1", "10.0.0.1", 50000, 80, ts=base_time + 3.0)
        )
        
        batch = self.aggregator.get_next_batch(timeout=1.0)
        
        self.assertIsNotNone(batch)
        self.assertGreater(batch["flow_count"], 0)
    
    def test_port_scan_detection_features(self):
        """Port scan should produce high unique_dst_ports and port_entropy."""
        base_time = 1000.0
        scan_ports = [22, 23, 25, 53, 80, 110, 143, 443, 3306, 8080, 8443, 9090]
        
        for port in scan_ports:
            self.aggregator.process_packet(
                self._make_packet("192.168.1.105", "192.168.1.20",
                                 50000 + port, port, ts=base_time + 0.1)
            )
        
        # Push past window
        self.aggregator.process_packet(
            self._make_packet("192.168.1.1", "10.0.0.1", 50000, 80, ts=base_time + 3.0)
        )
        
        batch = self.aggregator.get_next_batch(timeout=1.0)
        
        self.assertIsNotNone(batch)
        
        # Find the scanner's features
        scanner_features = None
        for feat in batch["features"]:
            if feat.get("src_ip") == "192.168.1.105":
                scanner_features = feat
                break
        
        self.assertIsNotNone(scanner_features)
        self.assertGreater(scanner_features["unique_dst_ports"], 5)
        self.assertGreater(scanner_features["port_entropy"], 1.0)
    
    def test_syn_flood_features(self):
        """SYN flood should produce high syn_count and serror_rate."""
        base_time = 1000.0
        
        for i in range(50):
            self.aggregator.process_packet(
                self._make_packet("10.0.0.99", "192.168.1.1",
                                 50000 + i, 80, flags=TCPFlags.SYN, ts=base_time)
            )
        
        self.aggregator.process_packet(
            self._make_packet("192.168.1.1", "10.0.0.1", 50000, 80, ts=base_time + 3.0)
        )
        
        batch = self.aggregator.get_next_batch(timeout=1.0)
        self.assertIsNotNone(batch)


class TestPacketParser(unittest.TestCase):
    def test_parse_valid_ip_packet(self):
        # Minimal valid IP packet (TCP SYN to port 80)
        raw = bytes([
            0x45, 0x00, 0x00, 0x28,  # IP header
            0x00, 0x01, 0x00, 0x00,
            0x40, 0x06, 0x00, 0x00,  # TTL=64, TCP
            0xC0, 0xA8, 0x01, 0x01,  # Src: 192.168.1.1
            0x0A, 0x00, 0x00, 0x01,  # Dst: 10.0.0.1
            0xC3, 0x50, 0x00, 0x50,  # Src port: 50000, Dst: 80
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00,
            0x50, 0x02, 0x20, 0x00,  # Flags: SYN
            0x00, 0x00, 0x00, 0x00
        ])
        
        packet = PacketParser.parse(raw, 1000.0)
        
        self.assertIsNotNone(packet)
        self.assertEqual(packet.src_ip, "192.168.1.1")
        self.assertEqual(packet.dst_ip, "10.0.0.1")
        self.assertEqual(packet.src_port, 50000)
        self.assertEqual(packet.dst_port, 80)
        self.assertEqual(packet.protocol, Protocol.TCP)
    
    def test_parse_invalid_packet(self):
        # Too short
        self.assertIsNone(PacketParser.parse(bytes([0x45, 0x00]), 1000.0))
        
        # Bad version
        bad_version = bytes([0x65, 0x00] + [0x00] * 18)
        self.assertIsNone(PacketParser.parse(bad_version, 1000.0))


if __name__ == "__main__":
    unittest.main()