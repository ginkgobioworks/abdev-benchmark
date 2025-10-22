#!/bin/bash
# Run all baseline predictions with cross-validation and evaluation
# This script trains models, generates predictions, and evaluates performance

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
SKIP_EVAL=false
RUN_DIR="${SCRIPT_DIR}/runs"
NUM_FOLDS=5

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --skip-eval)
            SKIP_EVAL=true
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
            echo "  --skip-eval     Skip evaluation step"
            echo "  --run-dir DIR   Directory for model artifacts (default: ./runs)"
            echo "  --help          Show this help message"
            echo ""
            echo "This script performs:"
            echo "  1. Cross-validation training (5-fold) on GDPa1 dataset"
            echo "  2. Prediction generation for each fold"
            echo "  3. Heldout test set predictions"
            echo "  4. Evaluation and metric computation"
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
echo "Mode: $([ "$SKIP_TRAIN" = true ] && echo "Predict only" || echo "Train + Predict + Eval")"
echo "Evaluation: $([ "$SKIP_EVAL" = true ] && echo "Disabled" || echo "Enabled")"
echo "Run directory: $RUN_DIR"
echo "Discovered ${#BASELINES[@]} baselines: ${BASELINES[*]}"
echo ""

# Data paths
TRAIN_DATA="${SCRIPT_DIR}/data/GDPa1_v1.2_20250814.csv"
HELDOUT_DATA="${SCRIPT_DIR}/data/heldout-set-sequences.csv"
PRED_DIR="${SCRIPT_DIR}/predictions"
EVAL_DIR="${SCRIPT_DIR}/evaluation_results"
TEMP_DIR="${SCRIPT_DIR}/.tmp_cv_splits"

# Create directories
mkdir -p "$PRED_DIR" "$EVAL_DIR" "$TEMP_DIR"

# Track successes and failures
TRAIN_SUCCESSES=()
TRAIN_FAILURES=()
PREDICT_SUCCESSES=()
PREDICT_FAILURES=()
EVAL_SUCCESSES=()
EVAL_FAILURES=()

for baseline in "${BASELINES[@]}"; do
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}[*] Processing: $baseline${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    if [ ! -d "baselines/$baseline" ]; then
        echo -e "${RED}[!] Directory not found: baselines/$baseline${NC}"
        TRAIN_FAILURES+=("$baseline")
        PREDICT_FAILURES+=("$baseline")
        EVAL_FAILURES+=("$baseline")
        continue
    fi
    
    cd "baselines/$baseline"
    baseline_module=$(echo "$baseline" | tr '-' '_')
    
    # Install dependencies
    echo "  Installing dependencies..."
    if ! pixi install > /dev/null 2>&1; then
        echo -e "${RED}  [✗] Failed to install dependencies${NC}"
        TRAIN_FAILURES+=("$baseline")
        PREDICT_FAILURES+=("$baseline")
        EVAL_FAILURES+=("$baseline")
        cd "$SCRIPT_DIR"
        continue
    fi
    echo -e "${GREEN}  [✓] Dependencies installed${NC}"
    
    # =================================================================
    # PART 1: Cross-Validation Training and Prediction
    # =================================================================
    echo ""
    echo -e "${YELLOW}[1/3] Cross-Validation (5-fold)${NC}"
    
    cv_failed=false
    for fold in $(seq 0 $((NUM_FOLDS - 1))); do
        echo "  Fold $fold:"
        
        # Train on all folds except current fold
        if [ "$SKIP_TRAIN" = false ]; then
            echo "    Training on folds != $fold..."
            
            # Create fold-specific training split
            fold_train_data="${TEMP_DIR}/${baseline}_fold${fold}_train.csv"
            export PATH="/home/sritter_ginkgobioworks_com/.pixi/bin:$PATH"
            python -c "from abdev_core.utils import split_data_by_fold; split_data_by_fold('$TRAIN_DATA', $fold, '$fold_train_data')" > /dev/null 2>&1
            
            # Train model
            fold_run_dir="${RUN_DIR}/${baseline}/fold_${fold}"
            if pixi run python -m "$baseline_module" train \
                --data "$fold_train_data" \
                --run-dir "$fold_run_dir" \
                --seed 42 > /dev/null 2>&1; then
                echo -e "${GREEN}    ✓ Training complete${NC}"
            else
                echo -e "${RED}    ✗ Training failed${NC}"
                cv_failed=true
                break
            fi
        fi
        
        # Predict on ALL data (including held-out fold)
        echo "    Predicting on all data..."
        fold_run_dir="${RUN_DIR}/${baseline}/fold_${fold}"
        fold_pred_dir="${PRED_DIR}/.tmp_cv/${baseline}/fold_${fold}"
        
        if pixi run python -m "$baseline_module" predict \
            --data "$TRAIN_DATA" \
            --run-dir "$fold_run_dir" \
            --out-dir "$fold_pred_dir" > /dev/null 2>&1; then
            echo -e "${GREEN}    ✓ Predictions complete${NC}"
        else
            echo -e "${RED}    ✗ Predictions failed${NC}"
            cv_failed=true
            break
        fi
    done
    
    if [ "$cv_failed" = true ]; then
        TRAIN_FAILURES+=("$baseline (CV)")
        PREDICT_FAILURES+=("$baseline (CV)")
        cd "$SCRIPT_DIR"
        continue
    fi
    
    # Merge CV predictions: take fold i predictions from model trained without fold i
    echo "  Merging CV predictions..."
    cv_merged_pred="${PRED_DIR}/GDPa1_cross_validation/${baseline}/predictions.csv"
    mkdir -p "$(dirname "$cv_merged_pred")"
    
    export PATH="/home/sritter_ginkgobioworks_com/.pixi/bin:$PATH"
    python -c "
import pandas as pd
from pathlib import Path
from abdev_core import FOLD_COL

# Read ground truth to get fold assignments
df_truth = pd.read_csv('$TRAIN_DATA')
fold_col = FOLD_COL

# Read all fold predictions
fold_preds = []
for fold in range($NUM_FOLDS):
    pred_file = Path('${PRED_DIR}/.tmp_cv/${baseline}/fold_' + str(fold) + '/predictions.csv')
    df_pred = pd.read_csv(pred_file)
    df_pred['_fold'] = fold
    fold_preds.append(df_pred)

# Concatenate all predictions
df_all_preds = pd.concat(fold_preds, ignore_index=True)

# Merge with truth to get fold assignments
df_merged = df_truth[['antibody_name', fold_col]].merge(
    df_all_preds, on='antibody_name', how='left'
)

# For each antibody, keep only the prediction from the model that didn't see it
df_cv = df_merged[df_merged[fold_col] == df_merged['_fold']].copy()
df_cv = df_cv.drop(columns=['_fold'])

# Save
df_cv.to_csv('$cv_merged_pred', index=False)
print(f'Merged CV predictions: {len(df_cv)} samples')
" > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}  [✓] CV predictions merged${NC}"
        TRAIN_SUCCESSES+=("$baseline")
    else
        echo -e "${RED}  [✗] Failed to merge CV predictions${NC}"
        PREDICT_FAILURES+=("$baseline (CV merge)")
        cd "$SCRIPT_DIR"
        continue
    fi
    
    # =================================================================
    # PART 2: Heldout Test Predictions
    # =================================================================
    echo ""
    echo -e "${YELLOW}[2/3] Heldout Test Set${NC}"
    
    # Train on ALL training data
    if [ "$SKIP_TRAIN" = false ]; then
        echo "  Training on full GDPa1 dataset..."
        full_run_dir="${RUN_DIR}/${baseline}/full"
        
        if pixi run python -m "$baseline_module" train \
            --data "$TRAIN_DATA" \
            --run-dir "$full_run_dir" \
            --seed 42 > /dev/null 2>&1; then
            echo -e "${GREEN}  [✓] Training complete${NC}"
        else
            echo -e "${RED}  [✗] Training failed${NC}"
            TRAIN_FAILURES+=("$baseline (full)")
            cd "$SCRIPT_DIR"
            continue
        fi
    fi
    
    # Predict on heldout data
    echo "  Predicting on heldout test set..."
    full_run_dir="${RUN_DIR}/${baseline}/full"
    heldout_pred_dir="${PRED_DIR}/heldout_test/${baseline}"
    
    if pixi run python -m "$baseline_module" predict \
        --data "$HELDOUT_DATA" \
        --run-dir "$full_run_dir" \
        --out-dir "$heldout_pred_dir" > /dev/null 2>&1; then
        echo -e "${GREEN}  [✓] Heldout predictions complete${NC}"
        PREDICT_SUCCESSES+=("$baseline")
    else
        echo -e "${RED}  [✗] Heldout predictions failed${NC}"
        PREDICT_FAILURES+=("$baseline (heldout)")
        cd "$SCRIPT_DIR"
        continue
    fi
    
    # =================================================================
    # PART 3: Evaluation
    # =================================================================
    if [ "$SKIP_EVAL" = false ]; then
        echo ""
        echo -e "${YELLOW}[3/3] Evaluation${NC}"
        
        # Evaluate CV predictions
        echo "  Evaluating CV predictions..."
        cv_eval_output="${EVAL_DIR}/${baseline}_cv.csv"
        
        if pixi run -C "${SCRIPT_DIR}/evaluation" score \
            --pred "$cv_merged_pred" \
            --truth "$TRAIN_DATA" \
            --model-name "$baseline" \
            --dataset "GDPa1_cross_validation" \
            --output "$cv_eval_output" > /dev/null 2>&1; then
            echo -e "${GREEN}  [✓] CV evaluation complete${NC}"
            EVAL_SUCCESSES+=("$baseline")
        else
            echo -e "${RED}  [✗] CV evaluation failed${NC}"
            EVAL_FAILURES+=("$baseline")
        fi
    else
        echo ""
        echo -e "${YELLOW}[3/3] Evaluation (skipped)${NC}"
    fi
    
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}[✓] $baseline complete${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    cd "$SCRIPT_DIR"
done

# Clean up temporary files
echo ""
echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR" "${PRED_DIR}/.tmp_cv"

# =================================================================
# Final Summary
# =================================================================
echo ""
echo "================================"
echo "SUMMARY"
echo "================================"
echo ""

# Training Summary
if [ "$SKIP_TRAIN" = false ]; then
    echo -e "${BLUE}Training (CV + Full):${NC}"
    if [ ${#TRAIN_SUCCESSES[@]} -eq ${#BASELINES[@]} ]; then
        echo -e "${GREEN}  ✓ All baselines: ${#TRAIN_SUCCESSES[@]}/${#BASELINES[@]}${NC}"
    else
        echo -e "  Successful: ${#TRAIN_SUCCESSES[@]}/${#BASELINES[@]}"
    fi
    
    if [ ${#TRAIN_FAILURES[@]} -gt 0 ]; then
        echo -e "${RED}  Failed: ${#TRAIN_FAILURES[@]}${NC}"
        for baseline in "${TRAIN_FAILURES[@]}"; do
            echo -e "    ${RED}✗${NC} $baseline"
        done
    fi
    echo ""
fi

# Prediction Summary
echo -e "${BLUE}Predictions (CV + Heldout):${NC}"
if [ ${#PREDICT_SUCCESSES[@]} -eq ${#BASELINES[@]} ]; then
    echo -e "${GREEN}  ✓ All baselines: ${#PREDICT_SUCCESSES[@]}/${#BASELINES[@]}${NC}"
else
    echo -e "  Successful: ${#PREDICT_SUCCESSES[@]}/${#BASELINES[@]}"
fi

if [ ${#PREDICT_FAILURES[@]} -gt 0 ]; then
    echo -e "${RED}  Failed: ${#PREDICT_FAILURES[@]}${NC}"
    for baseline in "${PREDICT_FAILURES[@]}"; do
        echo -e "    ${RED}✗${NC} $baseline"
    done
fi
echo ""

# Evaluation Summary
if [ "$SKIP_EVAL" = false ]; then
    echo -e "${BLUE}Evaluation:${NC}"
    if [ ${#EVAL_SUCCESSES[@]} -eq ${#BASELINES[@]} ]; then
        echo -e "${GREEN}  ✓ All baselines: ${#EVAL_SUCCESSES[@]}/${#BASELINES[@]}${NC}"
    else
        echo -e "  Successful: ${#EVAL_SUCCESSES[@]}/${#BASELINES[@]}"
    fi
    
    if [ ${#EVAL_FAILURES[@]} -gt 0 ]; then
        echo -e "${RED}  Failed: ${#EVAL_FAILURES[@]}${NC}"
        for baseline in "${EVAL_FAILURES[@]}"; do
            echo -e "    ${RED}✗${NC} $baseline"
        done
    fi
    echo ""
fi

# Output locations
echo -e "${BLUE}Output Locations:${NC}"
echo "  Model artifacts: $RUN_DIR"
echo "  Predictions:     $PRED_DIR"
if [ "$SKIP_EVAL" = false ]; then
    echo "  Evaluations:     $EVAL_DIR"
fi
echo ""

# Exit status
if [ ${#TRAIN_FAILURES[@]} -gt 0 ] || [ ${#PREDICT_FAILURES[@]} -gt 0 ] || [ ${#EVAL_FAILURES[@]} -gt 0 ]; then
    echo -e "${RED}✗ Some baselines failed${NC}"
    exit 1
else
    echo -e "${GREEN}✓ All baselines completed successfully!${NC}"
    exit 0
fi

