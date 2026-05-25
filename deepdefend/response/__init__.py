"""
Automated response system for DeepDefend NIDS.
Handles blocking, quarantine, incident generation, and notifications.
"""

from deepdefend.response.firewall import FirewallManager
from deepdefend.response.quarantine import QuarantineManager
from deepdefend.response.incident import IncidentGenerator
from deepdefend.response.notify import NotificationManager

__all__ = [
    "FirewallManager",
    "QuarantineManager",
    "IncidentGenerator",
    "NotificationManager",
]