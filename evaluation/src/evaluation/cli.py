"""Command-line interface for evaluation tasks."""

import argparse
from pathlib import Path
import sys
import pandas as pd

from evaluation.metrics import evaluate_model
from evaluation.validate import validate_prediction_file


def score_command(args):
    """Score predictions against ground truth."""
    pred_path = Path(args.pred)
    target_path = Path(args.truth)
    
    # Validate prediction file
    is_valid, errors = validate_prediction_file(pred_path)
    if not is_valid:
        print("Validation errors:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    
    # Evaluate
    model_name = args.model_name or pred_path.stem
    dataset_name = args.dataset
    
    try:
        results = evaluate_model(pred_path, target_path, model_name, dataset_name)
        
        # Print results
        df_results = pd.DataFrame(results)
        print(df_results.to_string(index=False))
        
        # Save if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_results.to_csv(output_path, index=False)
            print(f"\nResults saved to {output_path}")
        
        return 0
    except Exception as e:
        print(f"Evaluation failed: {str(e)}", file=sys.stderr)
        return 1


def validate_command(args):
    """Validate prediction file format."""
    pred_path = Path(args.pred)
    
    is_valid, errors = validate_prediction_file(pred_path)
    
    if is_valid:
        print(f"✓ {pred_path} is valid")
        return 0
    else:
        print(f"✗ {pred_path} has validation errors:")
        for error in errors:
            print(f"  - {error}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Antibody Developability Benchmark Evaluation"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Score command
    score_parser = subparsers.add_parser("score", help="Score predictions")
    score_parser.add_argument("--pred", required=True, help="Path to predictions CSV")
    score_parser.add_argument("--truth", required=True, help="Path to ground truth CSV")
    score_parser.add_argument("--model-name", help="Model name (defaults to filename)")
    score_parser.add_argument("--dataset", help="Dataset name (e.g., GDPa1_cross_validation)")
    score_parser.add_argument("--output", "-o", help="Path to save results CSV")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate prediction format")
    validate_parser.add_argument("--pred", required=True, help="Path to predictions CSV")
    
    args = parser.parse_args()
    
    if args.command == "score":
        return score_command(args)
    elif args.command == "validate":
        return validate_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

