"""CLI interface for TAP Linear baseline."""

from pathlib import Path
import typer
import pandas as pd

from .model import TapLinearModel


app = typer.Typer(add_completion=False, help="TAP Linear baseline - Ridge regression on TAP features")


@app.command()
def train(
    data: Path = typer.Option(..., help="Path to training data CSV"),
    run_dir: Path = typer.Option(..., help="Directory to save model artifacts"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
):
    """Train Ridge regression models on TAP features.
    
    Features are loaded automatically from the centralized feature store.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    
    typer.echo(f"Loading data from {data}...")
    df = pd.read_csv(data)
    
    typer.echo("Training TAP Linear model...")
    model = TapLinearModel()
    model.train(df, run_dir, seed=seed)
    
    typer.echo(f"✓ Training complete. Models saved to {run_dir}")


@app.command()
def predict(
    data: Path = typer.Option(..., help="Path to input data CSV"),
    run_dir: Path = typer.Option(..., help="Directory containing trained models"),
    out_dir: Path = typer.Option(..., help="Directory to write predictions.csv"),
):
    """Generate predictions using trained Ridge regression models.
    
    Features are loaded automatically from the centralized feature store.
    """
    if not run_dir.exists():
        typer.echo(f"Error: run_dir does not exist: {run_dir}", err=True)
        raise typer.Exit(1)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    typer.echo(f"Loading data from {data}...")
    df = pd.read_csv(data)
    
    typer.echo("Generating predictions...")
    model = TapLinearModel()
    model.predict(df, run_dir, out_dir)
    
    typer.echo(f"✓ Predictions saved to {out_dir / 'predictions.csv'}")


if __name__ == "__main__":
    app()

