
import sys
from pathlib import Path

# Add project root and webapp to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'webapp'))

from webapp.app import start_dashboard

if __name__ == '__main__':
    start_dashboard(host='0.0.0.0', port=5000, debug=False)