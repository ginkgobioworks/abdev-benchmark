#!/bin/bash
# Run all baseline predictions
# This script trains models and generates predictions for all baselines

set -e  # Exit on error

# Color output for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse command line arguments
SKIP_TRAIN=false
RUN_DIR="${SCRIPT_DIR}/runs"

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --run-dir)
            RUN_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-train    Skip training step (use existing models)"
            echo "  --run-dir DIR   Directory for model artifacts (default: ./runs)"
            echo "  --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Discover all baselines by scanning the baselines directory
# A baseline is valid if it has a pixi.toml file
BASELINES=()
for dir in "${SCRIPT_DIR}/baselines"/*; do
    if [ -d "$dir" ] && [ -f "$dir/pixi.toml" ]; then
        baseline_name=$(basename "$dir")
        BASELINES+=("$baseline_name")
    fi
done

# Sort baselines alphabetically for consistent output
IFS=$'\n' BASELINES=($(sort <<<"${BASELINES[*]}"))
unset IFS

echo "================================"
echo "Running All Baselines"
echo "================================"
echo "Mode: $([ "$SKIP_TRAIN" = true ] && echo "Predict only" || echo "Train + Predict")"
echo "Run directory: $RUN_DIR"
echo "Discovered ${#BASELINES[@]} baselines: ${BASELINES[*]}"
echo ""

# Data paths
TRAIN_DATA="${SCRIPT_DIR}/data/GDPa1_v1.2_20250814.csv"
HELDOUT_DATA="${SCRIPT_DIR}/data/heldout-set-sequences.csv"
PRED_DIR="${SCRIPT_DIR}/predictions"

# Track successes and failures
TRAIN_SUCCESSES=()
TRAIN_FAILURES=()
PREDICT_SUCCESSES=()
PREDICT_FAILURES=()

for baseline in "${BASELINES[@]}"; do
    echo -e "${BLUE}[*] Processing: $baseline${NC}"
    
    if [ ! -d "baselines/$baseline" ]; then
        echo -e "${RED}[!] Directory not found: baselines/$baseline${NC}"
        TRAIN_FAILURES+=("$baseline")
        PREDICT_FAILURES+=("$baseline")
        continue
    fi
    
    cd "baselines/$baseline"
    
    # Install dependencies
    echo "    Installing dependencies..."
    if ! pixi install 2>&1; then
        echo -e "${RED}[✗] Failed to install dependencies${NC}"
        TRAIN_FAILURES+=("$baseline")
        PREDICT_FAILURES+=("$baseline")
        cd "$SCRIPT_DIR"
        continue
    fi
    
    # Train model
    if [ "$SKIP_TRAIN" = false ]; then
        echo "    Training model..."
        baseline_module=$(echo "$baseline" | tr '-' '_')
        baseline_run_dir="${RUN_DIR}/${baseline}"
        
        if pixi run python -m "$baseline_module" train \
            --data "$TRAIN_DATA" \
            --run-dir "$baseline_run_dir" \
            --seed 42 > /dev/null 2>&1; then
            echo -e "${GREEN}    ✓ Training complete${NC}"
            TRAIN_SUCCESSES+=("$baseline")
        else
            echo -e "${RED}    ✗ Training failed${NC}"
            TRAIN_FAILURES+=("$baseline")
            cd "$SCRIPT_DIR"
            continue
        fi
    else
        TRAIN_SUCCESSES+=("$baseline (skipped)")
    fi
    
    # Predict on training data
    echo "    Generating predictions (training set)..."
    baseline_module=$(echo "$baseline" | tr '-' '_')
    baseline_run_dir="${RUN_DIR}/${baseline}"
    out_dir_train="${PRED_DIR}/GDPa1_cross_validation/${baseline}"
    
    if pixi run python -m "$baseline_module" predict \
        --data "$TRAIN_DATA" \
        --run-dir "$baseline_run_dir" \
        --out-dir "$out_dir_train" > /dev/null 2>&1; then
        echo -e "${GREEN}    ✓ Predictions (training)${NC}"
    else
        echo -e "${RED}    ✗ Predictions (training) failed${NC}"
        PREDICT_FAILURES+=("$baseline")
        cd "$SCRIPT_DIR"
        continue
    fi
    
    # Predict on heldout data
    echo "    Generating predictions (heldout set)..."
    out_dir_heldout="${PRED_DIR}/heldout_test/${baseline}"
    
    if pixi run python -m "$baseline_module" predict \
        --data "$HELDOUT_DATA" \
        --run-dir "$baseline_run_dir" \
        --out-dir "$out_dir_heldout" > /dev/null 2>&1; then
        echo -e "${GREEN}    ✓ Predictions (heldout)${NC}"
        PREDICT_SUCCESSES+=("$baseline")
    else
        echo -e "${RED}    ✗ Predictions (heldout) failed${NC}"
        PREDICT_FAILURES+=("$baseline")
        cd "$SCRIPT_DIR"
        continue
    fi
    
    echo -e "${GREEN}[✓] Success: $baseline${NC}"
    cd "$SCRIPT_DIR"
    echo ""
done

# Summary
echo "================================"
echo "Summary"
echo "================================"

if [ "$SKIP_TRAIN" = false ]; then
    echo -e "${BLUE}Training:${NC}"
    echo -e "${GREEN}  Successful: ${#TRAIN_SUCCESSES[@]}/${#BASELINES[@]}${NC}"
    for baseline in "${TRAIN_SUCCESSES[@]}"; do
        echo -e "    ${GREEN}✓${NC} $baseline"
    done
    
    if [ ${#TRAIN_FAILURES[@]} -gt 0 ]; then
        echo -e "${RED}  Failed: ${#TRAIN_FAILURES[@]}/${#BASELINES[@]}${NC}"
        for baseline in "${TRAIN_FAILURES[@]}"; do
            echo -e "    ${RED}✗${NC} $baseline"
        done
    fi
    echo ""
fi

echo -e "${BLUE}Prediction:${NC}"
echo -e "${GREEN}  Successful: ${#PREDICT_SUCCESSES[@]}/${#BASELINES[@]}${NC}"
for baseline in "${PREDICT_SUCCESSES[@]}"; do
    echo -e "    ${GREEN}✓${NC} $baseline"
done

if [ ${#PREDICT_FAILURES[@]} -gt 0 ]; then
    echo -e "${RED}  Failed: ${#PREDICT_FAILURES[@]}/${#BASELINES[@]}${NC}"
    for baseline in "${PREDICT_FAILURES[@]}"; do
        echo -e "    ${RED}✗${NC} $baseline"
    done
    exit 1
fi

echo ""
echo -e "${GREEN}All baselines completed successfully!${NC}"
echo "Model artifacts: $RUN_DIR"
echo "Predictions: $PRED_DIR"

