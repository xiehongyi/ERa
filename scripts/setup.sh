#!/bin/bash

# Setup script for ERa Attack
# This script attempts to configure the environment
# Note: Full setup requires additional files not included

echo "=========================================="
echo "ERa Attack Environment Setup"
echo "=========================================="
echo ""

# Color codes
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

# Check OS
echo "Checking operating system..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="Mac"
else
    echo -e "${RED}Unsupported OS: $OSTYPE${NC}"
    echo "This code is tested only on Linux"
    exit 1
fi
echo "OS: $OS"

# Check Python version
echo ""
echo "Checking Python version..."
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [ "$PYTHON_VERSION" != "3.8" ]; then
    echo -e "${YELLOW}Warning: Python $PYTHON_VERSION detected${NC}"
    echo "Python 3.8 is required for optimal performance"
    echo "Continue? (y/n)"
    read response
    if [ "$response" != "y" ]; then
        exit 1
    fi
else
    echo -e "${GREEN}Python 3.8 detected${NC}"
fi

# Check CUDA
echo ""
echo "Checking CUDA installation..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep release | awk '{print $6}' | cut -c2-)
    echo "CUDA $CUDA_VERSION detected"
    if [ "$CUDA_VERSION" != "11.1" ]; then
        echo -e "${YELLOW}Warning: CUDA 11.1 is recommended${NC}"
    fi
else
    echo -e "${YELLOW}CUDA not found - will run on CPU (slow)${NC}"
fi

# Create directories
echo ""
echo "Creating directory structure..."
mkdir -p models
mkdir -p calibration
mkdir -p results
mkdir -p logs
mkdir -p data

# Check for required files
echo ""
echo "Checking for required files..."

MISSING_FILES=()

# Check model shards
for i in {0..4}; do
    if [ ! -f "models/emgnet_shard_$i.pth" ]; then
        MISSING_FILES+=("models/emgnet_shard_$i.pth")
    fi
done

# Check calibration files
if [ ! -f "calibration/device_profile.json" ]; then
    MISSING_FILES+=("calibration/device_profile.json")
fi

if [ ! -f "calibration/channel_gains.npy" ]; then
    MISSING_FILES+=("calibration/channel_gains.npy")
fi

# Check optimal params
if [ ! -f "optimal_params.py" ]; then
    MISSING_FILES+=("optimal_params.py")
fi

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo -e "${YELLOW}Missing files:${NC}"
    for file in "${MISSING_FILES[@]}"; do
        echo "  - $file"
    done
    echo ""
    echo -e "${YELLOW}System will run in degraded mode${NC}"
    echo "Contact authors for complete files"
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
echo "This may take several minutes..."

pip install -r requirements_strict.txt

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to install dependencies${NC}"
    echo "Please install manually:"
    echo "  pip install -r requirements_strict.txt"
    exit 1
fi

# Check GNU Radio (optional)
echo ""
echo "Checking GNU Radio installation..."
if command -v gnuradio-config-info &> /dev/null; then
    GR_VERSION=$(gnuradio-config-info --version)
    echo "GNU Radio $GR_VERSION detected"
else
    echo -e "${YELLOW}GNU Radio not found${NC}"
    echo "GNU Radio is required for physical attacks"
    echo "Install with:"
    echo "  sudo apt-get install gnuradio"
fi

# Check HackRF (optional)
echo ""
echo "Checking HackRF tools..."
if command -v hackrf_info &> /dev/null; then
    echo "HackRF tools detected"
    hackrf_info 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}No HackRF device connected${NC}"
    fi
else
    echo -e "${YELLOW}HackRF tools not found${NC}"
    echo "Install with:"
    echo "  sudo apt-get install hackrf"
fi

# Generate dummy calibration if missing
if [ ! -f "calibration/device_profile.json" ]; then
    echo ""
    echo "Generating dummy calibration..."
    cat > calibration/device_profile.json << EOF
{
    "device_id": "DUMMY_DEVICE",
    "carrier_freq": 435000000,
    "channel_gains": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "timestamp": $(date +%s)
}
EOF
    echo -e "${YELLOW}Using dummy calibration (suboptimal)${NC}"
fi

echo ""
echo "=========================================="
echo "Setup Summary"
echo "=========================================="

if [ ${#MISSING_FILES[@]} -eq 0 ]; then
    echo -e "${GREEN}All files present${NC}"
else
    echo -e "${YELLOW}Missing ${#MISSING_FILES[@]} files${NC}"
fi

echo ""
echo "To run the attack:"
echo "  python run_validation.py"
echo ""
echo "Or for quick test:"
echo "  python main.py --attack fc_pgd --samples 5"
echo ""
echo -e "${YELLOW}Note: Without complete files, results will not match paper${NC}"
echo ""
echo "Setup complete!"