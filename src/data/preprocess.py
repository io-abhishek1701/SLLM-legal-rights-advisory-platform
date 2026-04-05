"""
JurisAI - Data Preprocessor
Cleans, filters, and chunks raw legal text for training.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_from_disk, Dataset, DatasetDict
from rich.console import Console
from rich.progress import track

from src.data.data_utils import (
    load_config, clean_text, has_legal_keywords,
    compute_text_hash, ensure_dirs, save_jsonl
)

console = Console()


def load_raw_datasets(raw_dir: str) -> List[Dataset]:
    """Load all raw datasets from disk."""
    datasets = []
    raw_path = Path(raw_dir)
    
    if not raw_path.exists():
        console.print(f"[red]✗ Raw data directory not found: {raw_path}[/red]")
        console.print("[yellow]  Run download_datasets.py first![/yellow]")
        return datasets
    
    for ds_dir in raw_path.iterdir():
        if ds_dir.is_dir() and (ds_dir / "dataset_info.json").exists():
            try:
                ds = load_from_disk(str(ds_dir))
                console.print(f"[green]✓ Loaded {ds_dir.name}: {len(ds)} rows[/green]")
                datasets.append(ds)
            except Exception as e:
                console.print(f"[red]✗ Failed to load {ds_dir.name}: {e}[/red]")
    
    return datasets


def detect_columns(dataset: Dataset) -> Dict[str, str]:
    """Auto-detect which columns map to instruction/input/output."""
    columns = dataset.column_names
    mapping = {}
    
    # Common column name patterns
    instruction_names = ["instruction", "question", "query", "prompt", "input_text"]
    context_names = ["input", "context", "passage", "document", "text"]
    output_names = ["output", "answer", "response", "completion", "target"]
    
    for col in columns:
        col_lower = col.lower()
        if col_lower in instruction_names:
            mapping["instruction"] = col
        elif col_lower in context_names and "instruction" not in mapping:
            mapping["instruction"] = col
        elif col_lower in context_names:
            mapping["context"] = col
        elif col_lower in output_names:
            mapping["output"] = col
    
    # Fallback: if we have 'text' only, it's a pretraining dataset
    if not mapping and "text" in columns:
        mapping["text"] = "text"
    
    return mapping


def clean_and_filter(
    datasets: List[Dataset],
    min_length: int = 50,
    max_length: int = 8000,
    require_legal: bool = True,
) -> Tuple[List[Dict], List[Dict]]:
    """Clean datasets and split into pretraining vs instruction data.
    
    Returns:
        (pretrain_data, instruct_data) — lists of cleaned examples
    """
    pretrain_data = []
    instruct_data = []
    seen_hashes = set()
    
    stats = {
        "total": 0,
        "too_short": 0,
        "too_long": 0,
        "no_legal_keywords": 0,
        "duplicates": 0,
        "kept_pretrain": 0,
        "kept_instruct": 0,
    }
    
    for ds in datasets:
        col_mapping = detect_columns(ds)
        console.print(f"[dim]Column mapping: {col_mapping}[/dim]")
        
        for row in track(ds, description="Processing rows..."):
            stats["total"] += 1
            
            # Handle instruction-format data
            if "instruction" in col_mapping and "output" in col_mapping:
                instruction = clean_text(str(row.get(col_mapping["instruction"], "")))
                context = clean_text(str(row.get(col_mapping.get("context", ""), "")))
                output = clean_text(str(row.get(col_mapping["output"], "")))
                
                # Combine for length/quality checks
                combined = f"{instruction} {context} {output}".strip()
                
                if len(combined) < min_length:
                    stats["too_short"] += 1
                    continue
                if len(combined) > max_length:
                    stats["too_long"] += 1
                    # Truncate output rather than skip
                    output = output[:max_length - len(instruction) - len(context) - 100]
                
                if require_legal and not has_legal_keywords(combined):
                    stats["no_legal_keywords"] += 1
                    continue
                
                # Dedup
                text_hash = compute_text_hash(combined)
                if text_hash in seen_hashes:
                    stats["duplicates"] += 1
                    continue
                seen_hashes.add(text_hash)
                
                entry = {
                    "instruction": instruction,
                    "input": context if context else "",
                    "output": output,
                }
                instruct_data.append(entry)
                stats["kept_instruct"] += 1
                
                # Also add to pretrain corpus
                pretrain_data.append({"text": combined})
                stats["kept_pretrain"] += 1
            
            # Handle raw text data (pretraining only)
            elif "text" in col_mapping:
                text = clean_text(str(row.get(col_mapping["text"], "")))
                
                if len(text) < min_length:
                    stats["too_short"] += 1
                    continue
                
                if require_legal and not has_legal_keywords(text):
                    stats["no_legal_keywords"] += 1
                    continue
                
                text_hash = compute_text_hash(text)
                if text_hash in seen_hashes:
                    stats["duplicates"] += 1
                    continue
                seen_hashes.add(text_hash)
                
                pretrain_data.append({"text": text})
                stats["kept_pretrain"] += 1
    
    # Print stats
    console.print("\n[bold]═══ Preprocessing Stats ═══[/bold]")
    for key, val in stats.items():
        console.print(f"  {key}: {val:,}")
    
    return pretrain_data, instruct_data


def create_splits(
    data: List[Dict],
    train_ratio: float = 0.85,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """Split data into train/validation/test sets."""
    import random
    random.seed(seed)
    
    shuffled = data.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    return {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def preprocess_all() -> None:
    """Run the full preprocessing pipeline."""
    console.print("\n[bold cyan]═══════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]  JurisAI - Data Preprocessor[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════[/bold cyan]")
    
    data_config = load_config("data_config.yaml")
    
    raw_dir = str(PROJECT_ROOT / "data" / "raw" / "huggingface")
    pretrain_dir = str(PROJECT_ROOT / "data" / "processed" / "pretrain")
    instruct_dir = str(PROJECT_ROOT / "data" / "processed" / "instruct")
    eval_dir = str(PROJECT_ROOT / "data" / "evaluation")
    
    ensure_dirs(pretrain_dir, instruct_dir, eval_dir)
    
    # Step 1: Load raw datasets
    console.print("\n[bold]Step 1: Loading raw datasets...[/bold]")
    datasets = load_raw_datasets(raw_dir)
    
    if not datasets:
        console.print("[red]No datasets found. Run download_datasets.py first![/red]")
        return
    
    # Step 2: Clean and filter
    console.print("\n[bold]Step 2: Cleaning and filtering...[/bold]")
    prep = data_config.get("preprocessing", {})
    pretrain_data, instruct_data = clean_and_filter(
        datasets,
        min_length=prep.get("min_text_length", 50),
        max_length=prep.get("max_text_length", 8000),
        require_legal=prep.get("require_legal_keywords", True),
    )
    
    # Step 3: Create splits
    console.print("\n[bold]Step 3: Creating data splits...[/bold]")
    splits_config = data_config.get("splits", {})
    
    if pretrain_data:
        pretrain_splits = create_splits(
            pretrain_data,
            train_ratio=splits_config.get("train", 0.85),
            val_ratio=splits_config.get("validation", 0.10),
            seed=splits_config.get("seed", 42),
        )
        
        for split_name, split_data in pretrain_splits.items():
            save_jsonl(split_data, f"{pretrain_dir}/{split_name}.jsonl")
    
    if instruct_data:
        instruct_splits = create_splits(
            instruct_data,
            train_ratio=splits_config.get("train", 0.85),
            val_ratio=splits_config.get("validation", 0.10),
            seed=splits_config.get("seed", 42),
        )
        
        for split_name, split_data in instruct_splits.items():
            save_jsonl(split_data, f"{instruct_dir}/{split_name}.jsonl")
        
        # Save test set for evaluation
        save_jsonl(instruct_splits["test"], f"{eval_dir}/test_questions.json")
    
    console.print("\n[bold green]✓ Preprocessing complete![/bold green]")
    console.print(f"  Pretrain: {len(pretrain_data):,} examples")
    console.print(f"  Instruct: {len(instruct_data):,} examples")


if __name__ == "__main__":
    preprocess_all()
