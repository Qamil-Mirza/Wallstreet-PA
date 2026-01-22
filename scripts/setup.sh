#!/bin/bash
# =============================================================================
# Newsletter Bot Setup Script
# =============================================================================
# This script sets up the newsletter bot environment on a fresh server.
# 
# Usage:
#   chmod +x scripts/setup.sh
#   ./scripts/setup.sh
#
# Requirements:
#   - Python 3.11 (Coqui TTS requirement)
#   - ffmpeg (for audio processing)
#   - Ollama (for LLM summarization)
# =============================================================================

set -e  # Exit on error

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

# =============================================================================
# Check Python version
# =============================================================================
check_python() {
    log_info "Checking Python version..."
    
    # Try python3.11 first, then python3
    if command -v python3.11 &> /dev/null; then
        PYTHON_CMD="python3.11"
    elif command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    else
        log_error "Python 3 not found. Please install Python 3.11."
        exit 1
    fi
    
    # Check version
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [ "$PYTHON_MAJOR" -ne 3 ] || [ "$PYTHON_MINOR" -lt 9 ] || [ "$PYTHON_MINOR" -gt 11 ]; then
        log_error "Python 3.9-3.11 required for Coqui TTS. Found: $PYTHON_VERSION"
        log_info "Install Python 3.11:"
        log_info "  macOS: brew install python@3.11"
        log_info "  Ubuntu: sudo apt install python3.11 python3.11-venv"
        exit 1
    fi
    
    log_info "Found Python $PYTHON_VERSION ✓"
}

# =============================================================================
# Check system dependencies
# =============================================================================
check_system_deps() {
    log_info "Checking system dependencies..."
    
    # Check ffmpeg
    if ! command -v ffmpeg &> /dev/null; then
        log_warn "ffmpeg not found. Installing..."
        if [[ "$OSTYPE" == "darwin"* ]]; then
            brew install ffmpeg
        elif [[ -f /etc/debian_version ]]; then
            sudo apt-get update && sudo apt-get install -y ffmpeg
        elif [[ -f /etc/redhat-release ]]; then
            sudo dnf install -y ffmpeg
        else
            log_error "Please install ffmpeg manually"
            exit 1
        fi
    fi
    log_info "ffmpeg installed ✓"
    
    # Check espeak (optional but recommended for TTS)
    if ! command -v espeak-ng &> /dev/null && ! command -v espeak &> /dev/null; then
        log_warn "espeak-ng not found. Some TTS models may not work."
        log_info "Install with: sudo apt install espeak-ng (Ubuntu) or brew install espeak (macOS)"
    fi
}

# =============================================================================
# Check Ollama
# =============================================================================
check_ollama() {
    log_info "Checking Ollama..."
    
    if ! command -v ollama &> /dev/null; then
        log_warn "Ollama not installed."
        log_info "Install Ollama from: https://ollama.ai/download"
        log_info "Or run: curl -fsSL https://ollama.ai/install.sh | sh"
        read -p "Continue without Ollama? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        log_info "Ollama installed ✓"
        
        # Check if llama3 model is available
        if ollama list 2>/dev/null | grep -q "llama3"; then
            log_info "llama3 model available ✓"
        else
            log_warn "llama3 model not found. Pull it with: ollama pull llama3"
        fi
    fi
}

# =============================================================================
# Setup virtual environment
# =============================================================================
setup_venv() {
    log_info "Setting up virtual environment..."
    
    # Remove old venv if it exists
    if [ -d "venv" ]; then
        log_warn "Existing venv found. Removing..."
        rm -rf venv
    fi
    
    # Create new venv
    $PYTHON_CMD -m venv venv
    
    # Activate venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip wheel setuptools
    
    log_info "Virtual environment created ✓"
}

# =============================================================================
# Install dependencies
# =============================================================================
install_deps() {
    log_info "Installing Python dependencies..."
    
    # Ensure venv is activated
    source venv/bin/activate
    
    # Install PyTorch (CPU version for smaller size)
    log_info "Installing PyTorch..."
    pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
    
    # Install remaining dependencies
    log_info "Installing remaining dependencies..."
    pip install -r requirements.txt
    
    log_info "Dependencies installed ✓"
}

# =============================================================================
# Setup configuration
# =============================================================================
setup_config() {
    log_info "Setting up configuration..."
    
    if [ ! -f ".env" ]; then
        if [ -f "env.example" ]; then
            cp env.example .env
            log_info "Created .env from env.example"
            log_warn "Please edit .env with your API keys and configuration!"
        else
            log_error "env.example not found"
            exit 1
        fi
    else
        log_info ".env already exists ✓"
    fi
    
    # Create output directories
    mkdir -p audio_output logs
    log_info "Output directories created ✓"
}

# =============================================================================
# Verify installation
# =============================================================================
verify_install() {
    log_info "Verifying installation..."
    
    source venv/bin/activate
    
    # Try importing key modules
    if python -c "from news_bot.tts_engine import TTSEngine; print('TTS engine OK')" 2>/dev/null; then
        log_info "TTS engine imports successfully ✓"
    else
        log_error "TTS engine import failed. Check error above."
        exit 1
    fi
    
    # Run tests
    log_info "Running tests..."
    if python -m pytest tests/ -v --tb=short; then
        log_info "All tests passed ✓"
    else
        log_warn "Some tests failed. Check output above."
    fi
}

# =============================================================================
# Main
# =============================================================================
main() {
    echo "=============================================="
    echo "  Newsletter Bot Setup"
    echo "=============================================="
    echo
    
    check_python
    check_system_deps
    check_ollama
    setup_venv
    install_deps
    setup_config
    verify_install
    
    echo
    echo "=============================================="
    echo "  Setup Complete!"
    echo "=============================================="
    echo
    log_info "To run the newsletter bot:"
    log_info "  1. Activate venv: source venv/bin/activate"
    log_info "  2. Edit .env with your configuration"
    log_info "  3. Start Ollama: ollama serve"
    log_info "  4. Run: python -m news_bot.main"
    echo
    log_info "For scheduled runs, add to crontab:"
    log_info "  0 7 * * * cd $(pwd) && ./scripts/run.sh"
    echo
}

main "$@"
