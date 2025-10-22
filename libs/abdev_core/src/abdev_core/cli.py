"""CLI utilities for baseline models."""

from pathlib import Path
import tempfile
import uuid
import typer
import pandas as pd
from typing import Optional

from .base import BaseModel


def create_cli_app(model_class: type[BaseModel], model_name: str) -> typer.Typer:
    """Create a standardized Typer CLI app for a baseline model.
    
    Args:
        model_class: The BaseModel implementation class
        model_name: Name of the model for display purposes
        
    Returns:
        Configured Typer application with train and predict commands
        
    Example:
        >>> from abdev_core.cli import create_cli_app
        >>> from .model import MyModel
        >>> app = create_cli_app(MyModel, "my_model")
        >>> if __name__ == "__main__":
        >>>     app()
    """
    app = typer.Typer(add_completion=False, help=f"{model_name} baseline model")
    
    @app.command()
    def train(
        data: Path = typer.Option(..., help="Path to training data CSV"),
        run_dir: Optional[Path] = typer.Option(
            None, 
            help="Directory to save model artifacts (default: random temp dir)"
        ),
        seed: int = typer.Option(42, help="Random seed for reproducibility"),
    ):
        """Train the model and save artifacts."""
        # Use temp directory with UUID if run_dir not specified
        if run_dir is None:
            run_dir = Path(tempfile.gettempdir()) / f"{model_name}_{uuid.uuid4().hex[:8]}"
            typer.echo(f"Using temporary run directory: {run_dir}")
        
        run_dir.mkdir(parents=True, exist_ok=True)
        
        typer.echo(f"Loading data from {data}...")
        df = pd.read_csv(data)
        
        typer.echo(f"Training {model_name}...")
        model = model_class()
        model.train(df, run_dir, seed=seed)
        
        typer.echo(f"✓ Training complete. Artifacts saved to {run_dir}")
    
    @app.command()
    def predict(
        data: Path = typer.Option(..., help="Path to input data CSV"),
        run_dir: Path = typer.Option(..., help="Directory containing model artifacts"),
        out_dir: Path = typer.Option(..., help="Directory to write predictions.csv"),
    ):
        """Generate predictions using trained model."""
        if not run_dir.exists():
            typer.echo(f"Error: run_dir does not exist: {run_dir}", err=True)
            raise typer.Exit(1)
        
        out_dir.mkdir(parents=True, exist_ok=True)
        
        typer.echo(f"Loading data from {data}...")
        df = pd.read_csv(data)
        
        typer.echo(f"Generating predictions with {model_name}...")
        model = model_class()
        model.predict(df, run_dir, out_dir)
        
        typer.echo(f"✓ Predictions saved to {out_dir / 'predictions.csv'}")
    
    return app


def validate_data_path(path: Path) -> Path:
    """Validate that a data path exists and is a CSV file.
    
    Args:
        path: Path to validate
        
    Returns:
        The validated path
        
    Raises:
        typer.BadParameter: If path doesn't exist or isn't a CSV
    """
    if not path.exists():
        raise typer.BadParameter(f"File does not exist: {path}")
    if path.suffix.lower() != ".csv":
        raise typer.BadParameter(f"File must be a CSV: {path}")
    return path


def validate_dir_path(path: Path, must_exist: bool = False) -> Path:
    """Validate a directory path.
    
    Args:
        path: Path to validate
        must_exist: If True, directory must already exist
        
    Returns:
        The validated path
        
    Raises:
        typer.BadParameter: If validation fails
    """
    if must_exist and not path.exists():
        raise typer.BadParameter(f"Directory does not exist: {path}")
    if path.exists() and not path.is_dir():
        raise typer.BadParameter(f"Path exists but is not a directory: {path}")
    return path

