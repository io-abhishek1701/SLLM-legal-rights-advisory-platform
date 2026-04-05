"""
JurisAI - Model Downloader
Downloads the base model from Hugging Face to local storage.

Usage:
    python scripts/download_model.py
    python scripts/download_model.py --token YOUR_HF_TOKEN
"""

import sys
import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console

console = Console()


def download_model(token: str = None):
    """Download the base model from Hugging Face."""
    from huggingface_hub import snapshot_download
    from src.data.data_utils import load_config
    
    config = load_config("model_config.yaml")
    model_name = config["base_model"]["name"]
    local_dir = config["base_model"].get("local_dir", f"./models/base/{model_name.split('/')[-1]}")
    
    console.print("\n[bold cyan]═══════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]  JurisAI - Model Downloader[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════[/bold cyan]")
    console.print(f"\n  Model: [bold]{model_name}[/bold]")
    console.print(f"  Local dir: {local_dir}")
    
    local_path = Path(PROJECT_ROOT) / local_dir
    local_path.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    if (local_path / "config.json").exists():
        console.print(f"\n[yellow]⚠ Model appears to already exist at {local_path}[/yellow]")
        console.print("[yellow]  Delete the directory to re-download.[/yellow]")
        return str(local_path)
    
    console.print(f"\n[blue]Downloading {model_name}...[/blue]")
    console.print("[dim]This may take several minutes depending on your connection.[/dim]")
    
    try:
        path = snapshot_download(
            repo_id=model_name,
            local_dir=str(local_path),
            token=token,
            resume_download=True,
        )
        
        # Calculate size
        total_size = sum(
            f.stat().st_size for f in local_path.rglob("*") if f.is_file()
        )
        size_gb = total_size / (1024 ** 3)
        
        console.print(f"\n[green]✓ Model downloaded successfully![/green]")
        console.print(f"  Location: {path}")
        console.print(f"  Size: {size_gb:.2f} GB")
        
        return str(local_path)
        
    except Exception as e:
        console.print(f"\n[red]✗ Download failed: {e}[/red]")
        console.print("[yellow]Tips:[/yellow]")
        console.print("[yellow]  - Check your internet connection[/yellow]")
        console.print("[yellow]  - Set HF_TOKEN if the model requires authentication[/yellow]")
        console.print("[yellow]  - Try: huggingface-cli login[/yellow]")
        return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download JurisAI base model")
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token"
    )
    args = parser.parse_args()
    
    token = args.token or os.environ.get("HF_TOKEN")
    download_model(token=token)
