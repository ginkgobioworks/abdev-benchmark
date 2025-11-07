#!/usr/bin/env python3
"""Orchestrate cross-validation training and evaluation for all model.

This script performs:
1. 5-fold cross-validation training on GDPa1
2. Prediction generation for CV and heldout sets
3. Evaluation metric computation

The orchestrator handles all CV logic, while model implement simple
train/predict interfaces.

Usage:
    pixi run all                    # Run with default config
    pixi run all-skip-train         # Skip training
    python run_all_models.py --help  # See all options
    python run_all_models.py --config configs/custom.toml
"""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import toml
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import numpy as np

from abdev_core import assign_random_folds, split_data_by_fold, evaluate_model

console = Console()
app = typer.Typer(add_completion=False)


def load_config(config_path: Path) -> Dict:
    """Load configuration from TOML file."""
    if not config_path.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        sys.exit(1)
    return toml.load(config_path)


def discover_model(model_dir: Path, include: List[str], exclude: List[str]) -> List[str]:
    """Discover valid model based on include/exclude filters."""
    all_model = []
    for model_path in model_dir.iterdir():
        if model_path.is_dir() and (model_path / "pixi.toml").exists():
            all_model.append(model_path.name)
    
    # Apply filters
    if include:
        model = [b for b in all_model if b in include]
    else:
        model = all_model
    
    if exclude:
        model = [b for b in model if b not in exclude]
    
    return sorted(model)


def run_command(cmd: List[str], cwd: Path, capture: bool = True) -> tuple[bool, str]:
    """Run a command and return success status and output."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=capture,
            text=True
        )
        return True, result.stdout if capture else ""
    except subprocess.CalledProcessError as e:
        error_msg = f"{e.stdout}\n{e.stderr}" if capture else str(e)
        return False, error_msg


def merge_cv_predictions(
    model_name: str,
    train_data_path: Path,
    pred_dir: Path,
    num_folds: int,
    fold_col: str,
    verbose: bool = False
) -> tuple[bool, Optional[Path]]:
    """Merge cross-validation predictions from all folds.
    
    For each sample, keep the prediction from the fold that didn't train on it.
    """
    try:
        df_truth = pd.read_csv(train_data_path)
        
        # Collect all fold predictions
        fold_preds = []
        for fold in range(num_folds):
            pred_file = pred_dir / f".tmp_cv/{model_name}/fold_{fold}/predictions.csv"
            if not pred_file.exists():
                if verbose:
                    console.print(f"[yellow]Warning: Missing prediction file: {pred_file}[/yellow]")
                return False, None
            
            df_pred = pd.read_csv(pred_file)
            df_pred['_fold'] = fold
            fold_preds.append(df_pred)
        
        # Merge and filter to out-of-fold predictions only
        df_all_preds = pd.concat(fold_preds, ignore_index=True)
        
        # Extract only the fold column from truth data to avoid duplicate column issues
        df_truth_folds = df_truth[['antibody_name', fold_col]].copy()
        
        # Merge predictions with fold assignments
        # Drop fold_col from predictions if it exists to avoid _x and _y suffix issues
        df_all_preds_clean = df_all_preds.drop(columns=[fold_col], errors='ignore')
        
        df_merged = df_truth_folds.merge(
            df_all_preds_clean, on='antibody_name', how='left'
        )
        
        # Keep only out-of-fold predictions: where fold_col matches _fold
        df_cv = df_merged[df_merged[fold_col] == df_merged['_fold']].copy()
        df_cv = df_cv.drop(columns=['_fold'])
        
        # Save merged predictions
        output_file = pred_dir / f"GDPa1_cross_validation/{model_name}/predictions.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df_cv.to_csv(output_file, index=False)
        
        return True, output_file
    except Exception as e:
        if verbose:
            console.print(f"[red]Error merging predictions: {e}[/red]")
        return False, None


def merge_cv_train_predictions(
    model_name: str,
    train_data_path: Path,
    pred_dir: Path,
    num_folds: int,
    fold_col: str,
    verbose: bool = False
) -> tuple[bool, Optional[Path]]:
    """Merge cross-validation train predictions from all folds.
    
    For each sample, keep the prediction from the folds that DID train on it.
    """
    try:
        df_truth = pd.read_csv(train_data_path)
        
        # Collect all fold predictions
        fold_preds = []
        for fold in range(num_folds):
            pred_file = pred_dir / f".tmp_cv/{model_name}/fold_{fold}/predictions.csv"
            if not pred_file.exists():
                if verbose:
                    console.print(f"[yellow]Warning: Missing prediction file: {pred_file}[/yellow]")
                return False, None
            
            df_pred = pd.read_csv(pred_file)
            df_pred['_fold'] = fold
            fold_preds.append(df_pred)
        
        # Merge and filter to in-fold predictions (training data for that fold)
        df_all_preds = pd.concat(fold_preds, ignore_index=True)
        
        # Extract only the fold column from truth data to avoid duplicate column issues
        df_truth_folds = df_truth[['antibody_name', fold_col]].copy()
        
        # Merge predictions with fold assignments
        # Drop fold_col from predictions if it exists to avoid _x and _y suffix issues
        df_all_preds_clean = df_all_preds.drop(columns=[fold_col], errors='ignore')
        
        df_merged = df_truth_folds.merge(
            df_all_preds_clean, on='antibody_name', how='left'
        )
        
        # Keep only in-fold predictions: where fold_col != _fold (trained on these samples)
        df_train = df_merged[df_merged[fold_col] != df_merged['_fold']].copy()
        df_train = df_train.drop(columns=['_fold'])
        
        # Save merged predictions
        output_file = pred_dir / f".tmp_cv_train/{model_name}/predictions.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df_train.to_csv(output_file, index=False)
        
        return True, output_file
    except Exception as e:
        if verbose:
            console.print(f"[red]Error merging train predictions: {e}[/red]")
        return False, None


@app.command()
def main(
    config: Path = typer.Option(
        Path("configs/default.toml"),
        "--config", "-c",
        help="Path to configuration file"
    ),
    skip_train: Optional[bool] = typer.Option(
        None, "--skip-train",
        help="Skip training (overrides config)"
    ),
    skip_eval: Optional[bool] = typer.Option(
        None, "--skip-eval",
        help="Skip evaluation (overrides config)"
    ),
    verbose: Optional[bool] = typer.Option(
        None, "--verbose", "-v",
        help="Show detailed output (overrides config)"
    ),
    run_dir: Optional[Path] = typer.Option(
        None, "--run-dir",
        help="Directory for model artifacts (overrides config)"
    ),
):
    """Run all model with cross-validation and evaluation."""
    
    # Load and merge config with CLI overrides
    script_dir = Path(__file__).parent
    cfg = load_config(script_dir / config)
    
    # Override config with CLI arguments
    if skip_train is not None:
        cfg['execution']['skip_train'] = skip_train
    if skip_eval is not None:
        cfg['execution']['skip_eval'] = skip_eval
    if verbose is not None:
        cfg['execution']['verbose'] = verbose
    if run_dir is not None:
        cfg['paths']['run_dir'] = str(run_dir)
    
    # Setup paths (all relative to script_dir)
    model_dir = script_dir / cfg['model']['model_dir']
    train_data = script_dir / cfg['data']['train_file']
    test_data = script_dir / cfg['data']['test_file']
    run_dir = script_dir / cfg['paths']['run_dir']
    pred_dir = script_dir / cfg['paths']['predictions_dir']
    eval_dir = script_dir / cfg['paths']['evaluation_dir']
    temp_dir = script_dir / cfg['paths']['temp_dir']
    
    # Create directories
    for directory in [run_dir, pred_dir, eval_dir, temp_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Discover model
    model = discover_model(
        model_dir,
        cfg['model'].get('include', []),
        cfg['model'].get('exclude', [])
    )
    
    if not model:
        console.print("[red]No model found![/red]")
        sys.exit(1)
    
    # Print header
    mode_desc = []
    if not cfg['execution']['skip_train']:
        mode_desc.append("Train")
    mode_desc.append("Predict")
    if not cfg['execution']['skip_eval']:
        mode_desc.append("Eval")
    
    console.print(Panel.fit(
        f"[bold cyan]Running All Model[/bold cyan]\n\n"
        f"Config: {config}\n"
        f"Mode: {' + '.join(mode_desc)}\n"
        f"Model: {len(model)} discovered\n"
        f"  {', '.join(model)}\n\n"
        f"Run directory: {run_dir}",
        border_style="cyan"
    ))
    
    # Track results
    results = {
        'success': [],
        'failed_train': [],
        'failed_predict': [],
        'failed_eval': [],
        'metrics': {}  # model -> metrics DataFrame
    }
    
    num_folds = cfg['cross_validation']['num_folds']
    seed = cfg['cross_validation']['seed']
    fold_col = cfg['cross_validation']['fold_col']
    verbose = cfg['execution']['verbose']
    
    # Handle fold assignments
    if fold_col == "":
        # Generate random folds
        if verbose:
            console.print(f"[yellow]Generating random {num_folds}-fold splits...[/yellow]")
        df_train = pd.read_csv(train_data)
        fold_col = "fold"  # Use this name for generated folds
        df_train = assign_random_folds(df_train, num_folds=num_folds, seed=seed, fold_col=fold_col)
        # Save with fold assignments
        train_data_with_folds = temp_dir / "train_with_folds.csv"
        df_train.to_csv(train_data_with_folds, index=False)
        train_data = train_data_with_folds
        if verbose:
            console.print(f"[green]✓ Random folds generated and saved to {train_data_with_folds}[/green]")
    
    # Process each model
    for model_idx, model_name in enumerate(model, 1):
        console.rule(f"[bold cyan][{model_idx}/{len(model)}] {model_name}[/bold cyan]")
        
        model_dir_path = model_dir / model_name
        model_module = model_name.replace('-', '_')
        model_failed = False
        
        # Install dependencies
        console.print("  [dim]Installing dependencies...[/dim]")
        success, _ = run_command(["pixi", "install"], model_dir_path, capture=not verbose)
        if not success:
            console.print("  [red]✗ Failed to install dependencies[/red]")
            results['failed_train'].append(model_name)
            continue
        if verbose:
            console.print("  [green]✓ Dependencies installed[/green]")
        
        # ===== CROSS-VALIDATION =====
        console.print(f"  [yellow]Cross-Validation ({num_folds}-fold)[/yellow]")
        
        for fold in range(num_folds):
            if verbose:
                console.print(f"    Fold {fold}:")
            
            # Train on folds != current
            if not cfg['execution']['skip_train']:
                # Split data
                fold_train_data = temp_dir / f"{model_name}_fold{fold}_train.csv"
                try:
                    split_data_by_fold(train_data, fold, fold_col, fold_train_data)
                except Exception as e:
                    console.print(f"    [red]✗ Failed to split data for fold {fold}: {e}[/red]")
                    model_failed = True
                    break
                
                # Train model
                fold_run_dir = run_dir / model_name / f"fold_{fold}"
                cmd = [
                    "pixi", "run", "python", "-m", model_module, "train",
                    "--data", str(fold_train_data),
                    "--run-dir", str(fold_run_dir),
                    "--seed", str(seed)
                ]
                success, output = run_command(cmd, model_dir_path, capture=not verbose)
                
                if not success:
                    console.print(f"    [red]✗ Training failed on fold {fold}[/red]")
                    if verbose:
                        console.print(f"    [dim]{output}[/dim]")
                    model_failed = True
                    break
            
            # Predict on all data
            fold_run_dir = run_dir / model_name / f"fold_{fold}"
            fold_pred_dir = pred_dir / f".tmp_cv/{model_name}/fold_{fold}"
            fold_pred_dir.mkdir(parents=True, exist_ok=True)
            
            cmd = [
                "pixi", "run", "python", "-m", model_module, "predict",
                "--data", str(train_data),
                "--run-dir", str(fold_run_dir),
                "--out-dir", str(fold_pred_dir)
            ]
            success, output = run_command(cmd, model_dir_path, capture=not verbose)
            
            if not success:
                console.print(f"    [red]✗ Predictions failed on fold {fold}[/red]")
                if verbose:
                    console.print(f"    [dim]{output}[/dim]")
                model_failed = True
                break
            
            if verbose:
                console.print(f"    [green]✓ Fold {fold} predictions saved[/green]")
        
        if model_failed:
            results['failed_train'].append(model_name)
            continue
        
        # Merge CV predictions (test)
        if verbose:
            console.print("    Merging CV test predictions...")
        success, cv_test_file = merge_cv_predictions(model_name, train_data, pred_dir, num_folds, fold_col, verbose)
        if not success:
            console.print("  [red]✗ Failed to merge CV test predictions[/red]")
            results['failed_predict'].append(model_name)
            continue
        
        # Merge CV predictions (train)
        if verbose:
            console.print("    Merging CV train predictions...")
        success, cv_train_file = merge_cv_train_predictions(model_name, train_data, pred_dir, num_folds, fold_col, verbose)
        if not success:
            console.print("  [red]✗ Failed to merge CV train predictions[/red]")
            results['failed_predict'].append(model_name)
            continue
        
        console.print("  [green]✓ Cross-validation complete[/green]")
        
        # ===== FULL MODEL + TEST SET =====
        console.print("  [yellow]Test Set[/yellow]")
        
        # Train on all data
        if not cfg['execution']['skip_train']:
            full_run_dir = run_dir / model_name / "full"
            cmd = [
                "pixi", "run", "python", "-m", model_module, "train",
                "--data", str(train_data),
                "--run-dir", str(full_run_dir),
                "--seed", str(seed)
            ]
            success, output = run_command(cmd, model_dir_path, capture=not verbose)
            
            if not success:
                console.print("  [red]✗ Full training failed[/red]")
                if verbose:
                    console.print(f"  [dim]{output}[/dim]")
                results['failed_train'].append(model_name)
                continue
        
        # Predict on test set
        full_run_dir = run_dir / model_name / "full"
        test_pred_dir = pred_dir / f"heldout_test/{model_name}"
        test_pred_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "pixi", "run", "python", "-m", model_module, "predict",
            "--data", str(test_data),
            "--run-dir", str(full_run_dir),
            "--out-dir", str(test_pred_dir)
        ]
        success, output = run_command(cmd, model_dir_path, capture=not verbose)
        
        if not success:
            console.print("  [red]✗ Test predictions failed[/red]")
            if verbose:
                console.print(f"  [dim]{output}[/dim]")
            results['failed_predict'].append(model_name)
            continue
        
        if verbose:
            console.print("  [green]✓ Test predictions saved[/green]")
        
        console.print("  [green]✓ Test predictions complete[/green]")
        
        # ===== EVALUATION =====
        if not cfg['execution']['skip_eval']:
            console.print("  [yellow]Evaluation[/yellow]")
            
            cv_test_pred_file = pred_dir / f"GDPa1_cross_validation/{model_name}/predictions.csv"
            cv_train_pred_file = pred_dir / f".tmp_cv_train/{model_name}/predictions.csv"
            cv_eval_output = eval_dir / f"{model_name}_cv.csv"
            
            try:
                # Evaluate test predictions
                if verbose:
                    console.print("    Evaluating test predictions...")
                test_results_list = evaluate_model(
                    cv_test_pred_file,
                    train_data,
                    model_name,
                    cfg['evaluation']['cv_dataset_name'],
                    fold_col=fold_col,
                    num_folds=num_folds,
                    split="test"
                )
                
                # Evaluate train predictions
                if verbose:
                    console.print("    Evaluating train predictions...")
                train_results_list = evaluate_model(
                    cv_train_pred_file,
                    train_data,
                    model_name,
                    cfg['evaluation']['cv_dataset_name'],
                    fold_col=fold_col,
                    num_folds=num_folds,
                    split="train"
                )
                
                # Combine results
                all_results = test_results_list + train_results_list
                df_results = pd.DataFrame(all_results)
                df_results.to_csv(cv_eval_output, index=False)
                
                # Store metrics for summary (test split only)
                df_test_results = df_results[df_results['split'] == 'test']
                results['metrics'][model_name] = df_test_results
                
                console.print("  [green]✓ Evaluation complete[/green]")
            except Exception as e:
                console.print(f"  [red]✗ Evaluation failed: {e}[/red]")
                results['failed_eval'].append(model_name)
                continue
        
        results['success'].append(model_name)
        console.print(f"[bold green]✓ {model_name} complete[/bold green]\n")
    
    # Cleanup
    if verbose:
        console.print("\n[dim]Cleaning up temporary files...[/dim]")
    shutil.rmtree(temp_dir, ignore_errors=True)
    shutil.rmtree(pred_dir / ".tmp_cv", ignore_errors=True)
    shutil.rmtree(pred_dir / ".tmp_cv_train", ignore_errors=True)
    
    # ===== SUMMARY =====
    console.print("\n")
    console.rule("[bold cyan]SUMMARY[/bold cyan]")
    
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Stage", style="cyan", width=20)
    table.add_column("Success", justify="right", style="green")
    table.add_column("Failed", justify="right", style="red")
    
    total_model = len(model)
    train_failed = len(results['failed_train'])
    predict_failed = len(results['failed_predict'])
    eval_failed = len(results['failed_eval'])
    
    if not cfg['execution']['skip_train']:
        table.add_row("Training", str(total_model - train_failed), str(train_failed))
    
    table.add_row("Prediction", str(total_model - predict_failed), str(predict_failed))
    
    if not cfg['execution']['skip_eval']:
        table.add_row("Evaluation", str(total_model - eval_failed), str(eval_failed))
    
    console.print(table)
    
    # List failures if any
    if train_failed or predict_failed or eval_failed:
        console.print("\n[bold red]Failed Model:[/bold red]")
        for model_name in set(results['failed_train'] + results['failed_predict'] + results['failed_eval']):
            console.print(f"  • {model_name}")
    
    # ===== METRICS SUMMARY =====
    if not cfg['execution']['skip_eval'] and results['metrics']:
        console.print("\n")
        console.rule("[bold cyan]METRICS SUMMARY[/bold cyan]")
        
        # Organize metrics by model, assay, and metric type
        spearman_by_model = {}
        recall_by_model = {}
        all_assays = set()
        
        for model_name, df_metrics in results['metrics'].items():
            # Filter to "average" fold and "test" split for summary
            df_summary = df_metrics[(df_metrics['fold'] == 'average') & (df_metrics['split'] == 'test')]
            
            if len(df_summary) > 0:
                spearman_by_model[model_name] = {}
                recall_by_model[model_name] = {}
                
                for _, row in df_summary.iterrows():
                    assay = row['assay']
                    all_assays.add(assay)
                    spearman_by_model[model_name][assay] = f"{row['spearman']:.3f}"
                    recall_by_model[model_name][assay] = f"{row['top_10_recall']:.3f}"
        
        # Sort assays
        sorted_assays = sorted(all_assays)
        
        # Sort model by average Spearman (descending)
        sorted_model = sorted(
            spearman_by_model.keys(),
            key=lambda b: np.mean([float(spearman_by_model[b].get(a, "0")) for a in sorted_assays]),
            reverse=True
        )
        
        # Create Spearman table
        spearman_table = Table(show_header=True, header_style="bold cyan", title="Spearman ρ (Test folds of training set, Average Fold)")
        spearman_table.add_column("Model", style="cyan")
        for assay in sorted_assays:
            spearman_table.add_column(assay, justify="right", style="green")
        
        for model_name in sorted_model:
            row_data = [model_name]
            for assay in sorted_assays:
                row_data.append(spearman_by_model[model_name].get(assay, "—"))
            spearman_table.add_row(*row_data)
        
        console.print(spearman_table)
        
        # Create Top 10% Recall table
        console.print()
        recall_table = Table(show_header=True, header_style="bold yellow", title="Top 10% Recall (Test folds of training set, Average Fold)")
        recall_table.add_column("Model", style="cyan")
        for assay in sorted_assays:
            recall_table.add_column(assay, justify="right", style="yellow")
        
        for model_name in sorted_model:
            row_data = [model_name]
            for assay in sorted_assays:
                row_data.append(recall_by_model[model_name].get(assay, "—"))
            recall_table.add_row(*row_data)
        
        console.print(recall_table)
        console.print(f"\n[dim]Note: Using 'average' fold and 'test' split. See {eval_dir} for per-fold/per-property/per-split results.[/dim]")
    
    # Output locations
    console.print(f"\n[bold cyan]Output Locations:[/bold cyan]")
    console.print(f"  Models:      {run_dir}")
    console.print(f"  Predictions: {pred_dir}")
    if not cfg['execution']['skip_eval']:
        console.print(f"  Evaluations: {eval_dir}")
    
    # Final status
    total_failed = len(set(results['failed_train'] + results['failed_predict'] + results['failed_eval']))
    console.print()
    if total_failed > 0:
        console.print(f"[red]✗ {total_failed}/{total_model} model(s) failed[/red]")
        sys.exit(1)
    else:
        console.print(f"[green]✓ All {total_model} model completed successfully![/green]")
        sys.exit(0)


if __name__ == "__main__":
    app()
