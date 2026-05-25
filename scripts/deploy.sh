#!/bin/bash
#
# DeepDefend NIDS Deployment Script
# Deploys the system in production or development mode.
#
# Usage:
#   ./scripts/deploy.sh [production|development]
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

MODE="${1:-production}"
CONFIG_FILE="$PROJECT_ROOT/config/${MODE}.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ─── Prerequisites Check ──────────────────────────────────────

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    log_info "Python version: $PYTHON_VERSION"
    
    # Check for required packages
    python3 -c "import numpy, pandas, sklearn, xgboost" 2>/dev/null || {
        log_warn "Required Python packages not found. Installing..."
        pip install -r "$PROJECT_ROOT/requirements.txt"
    }
    
    # Check iptables
    if ! command -v iptables &> /dev/null; then
        log_warn "iptables not found. Firewall blocking will not work."
    fi
    
    # Check Snort
    if [ "$MODE" = "production" ]; then
        if ! command -v snort &> /dev/null; then
            log_warn "Snort not found. Signature-based detection will be disabled."
        fi
    fi
    
    log_info "Prerequisites check complete"
}

# ─── Directory Setup ──────────────────────────────────────────

setup_directories() {
    log_info "Setting up directories..."
    
    mkdir -p "$PROJECT_ROOT/data/nsl_kdd"
    mkdir -p "$PROJECT_ROOT/models/saved"
    mkdir -p "$PROJECT_ROOT/logs"
    
    if [ "$MODE" = "production" ]; then
        sudo mkdir -p /var/lib/deepdefend
        sudo mkdir -p /var/log/deepdefend
        sudo mkdir -p /etc/deepdefend
        
        sudo chown -R "$USER:$USER" /var/lib/deepdefend
        sudo chown -R "$USER:$USER" /var/log/deepdefend
        
        # Copy config
        sudo cp "$CONFIG_FILE" /etc/deepdefend/config.yaml
        
        # Copy Snort config if exists
        if [ -f "$PROJECT_ROOT/config/snort/snort.conf" ]; then
            sudo mkdir -p /etc/snort
            sudo cp "$PROJECT_ROOT/config/snort/snort.conf" /etc/snort/
            sudo cp "$PROJECT_ROOT/config/snort/custom.rules" /etc/snort/
        fi
    fi
    
    log_info "Directories created"
}

# ─── Download and Prepare Data ────────────────────────────────

prepare_data() {
    log_info "Preparing NSL-KDD dataset..."
    
    cd "$PROJECT_ROOT"
    python3 -c "
from deepdefend.data.loader import NSLKDLoader
loader = NSLKDLoader(data_dir='./data/nsl_kdd')
loader.download_if_needed()
print('NSL-KDD dataset ready')
"
    
    log_info "Data preparation complete"
}

# ─── Train Models ────────────────────────────────────────────

train_models() {
    log_info "Training models (mode: $MODE)..."
    
    cd "$PROJECT_ROOT"
    python3 scripts/train.py --mode "$MODE"
    
    log_info "Model training complete"
}

# ─── Verify Models ───────────────────────────────────────────

verify_models() {
    log_info "Verifying trained models..."
    
    cd "$PROJECT_ROOT"
    python3 scripts/evaluate.py --model-path "$PROJECT_ROOT/models/saved"
    
    log_info "Model verification complete"
}

# ─── Start Engine ────────────────────────────────────────────

start_engine() {
    log_info "Starting DeepDefend engine..."
    
    cd "$PROJECT_ROOT"
    
    if [ "$MODE" = "production" ]; then
        # Use systemd if available
        if command -v systemctl &> /dev/null; then
            log_info "Using systemd service"
            sudo cp "$PROJECT_ROOT/deepdefend.service" /etc/systemd/system/
            sudo systemctl daemon-reload
            sudo systemctl enable deepdefend
            sudo systemctl start deepdefend
            sudo systemctl status deepdefend
        else
            # Run directly
            nohup python3 scripts/run_engine.py --mode "$MODE" \
                > logs/deepdefend.log 2>&1 &
            echo $! > "$PROJECT_ROOT/deepdefend.pid"
            log_info "Engine started with PID $(cat $PROJECT_ROOT/deepdefend.pid)"
        fi
    else
        # Development mode — run in foreground
        python3 scripts/run_engine.py --mode "$MODE"
    fi
}

# ─── Health Check ────────────────────────────────────────────

health_check() {
    log_info "Running health check..."
    
    sleep 5
    
    if curl -s http://localhost:8080/healthz > /dev/null 2>&1; then
        log_info "Health check passed ✓"
    else
        log_warn "Health check endpoint not responding (engine may still be starting)"
    fi
    
    if curl -s http://localhost:9090/metrics > /dev/null 2>&1; then
        log_info "Metrics endpoint responding ✓"
    else
        log_warn "Metrics endpoint not responding"
    fi
}

# ─── Main ────────────────────────────────────────────────────

main() {
    echo ""
    echo "=========================================="
    echo "  DeepDefend NIDS Deployment"
    echo "  Mode: $MODE"
    echo "=========================================="
    echo ""
    
    check_prerequisites
    setup_directories
    
    # Check if models exist
    if [ ! -f "$PROJECT_ROOT/models/saved/attack_ensemble.pkl" ]; then
        log_info "No trained models found. Starting training pipeline..."
        prepare_data
        train_models
        verify_models
    else
        log_info "Existing models found. Skipping training."
        log_info "To retrain, delete models/saved/ and re-run."
    fi
    
    # Start engine
    start_engine
    
    # Health check (background)
    health_check
    
    echo ""
    log_info "Deployment complete!"
    echo ""
    echo "Health check:  http://localhost:8080/healthz"
    echo "Metrics:       http://localhost:9090/metrics"
    echo "Logs:          $PROJECT_ROOT/logs/"
    echo ""
    echo "To stop:"
    if [ "$MODE" = "production" ] && command -v systemctl &> /dev/null; then
        echo "  sudo systemctl stop deepdefend"
    else
        echo "  kill \$(cat $PROJECT_ROOT/deepdefend.pid)"
    fi
    echo ""
}

main "$@"