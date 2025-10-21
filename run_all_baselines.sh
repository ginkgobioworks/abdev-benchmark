#!/bin/bash
# Run all baseline predictions
# This script iterates through all baseline directories and runs their prediction tasks

set -e  # Exit on error

# Color output for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# List of all baselines to run
BASELINES=(
    "tap_linear"
    "tap_single_features"
    "aggrescan3d"
    "antifold"
    "saprot_vh"
    "deepviscosity"
)

echo "================================"
echo "Running All Baseline Predictions"
echo "================================"
echo ""

# Track successes and failures
SUCCESSES=()
FAILURES=()

for baseline in "${BASELINES[@]}"; do
    echo -e "${YELLOW}[*] Processing: $baseline${NC}"
    
    if [ ! -d "baselines/$baseline" ]; then
        echo -e "${RED}[!] Directory not found: baselines/$baseline${NC}"
        FAILURES+=("$baseline")
        continue
    fi
    
    cd "baselines/$baseline"
    
    # Install dependencies
    echo "    Installing dependencies..."
    if pixi install 2>&1 | grep -q "Nothing to do"; then
        echo "    Dependencies already installed"
    fi
    
    # Run prediction
    echo "    Running predictions..."
    if pixi run predict; then
        echo -e "${GREEN}[✓] Success: $baseline${NC}"
        SUCCESSES+=("$baseline")
    else
        echo -e "${RED}[✗] Failed: $baseline${NC}"
        FAILURES+=("$baseline")
    fi
    
    cd "$SCRIPT_DIR"
    echo ""
done

# Summary
echo "================================"
echo "Summary"
echo "================================"
echo -e "${GREEN}Successful: ${#SUCCESSES[@]}/${#BASELINES[@]}${NC}"
for baseline in "${SUCCESSES[@]}"; do
    echo -e "  ${GREEN}✓${NC} $baseline"
done

if [ ${#FAILURES[@]} -gt 0 ]; then
    echo ""
    echo -e "${RED}Failed: ${#FAILURES[@]}/${#BASELINES[@]}${NC}"
    for baseline in "${FAILURES[@]}"; do
        echo -e "  ${RED}✗${NC} $baseline"
    done
    exit 1
fi

echo ""
echo -e "${GREEN}All baselines completed successfully!${NC}"
echo "Predictions saved to: predictions/"

