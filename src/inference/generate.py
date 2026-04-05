"""
JurisAI - Interactive Inference
Quick test script for generating legal responses.

Usage:
    python -m src.inference.generate
    python -m src.inference.generate --adapter-path ./models/adapters/instruct_v1/final
    python -m src.inference.generate --query "Explain Article 21"
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.data.data_utils import load_config

console = Console()


def generate(model, tokenizer, query: str, max_new_tokens: int = 1024) -> str:
    """Generate a response with streaming-like output."""
    from unsloth import FastLanguageModel
    
    # Load system prompt
    try:
        data_config = load_config("data_config.yaml")
        system_prompt = data_config.get("instruction_format", {}).get(
            "system_prompt",
            "You are JurisAI, an expert AI legal assistant specialized in Indian law."
        )
    except Exception:
        system_prompt = "You are JurisAI, an expert AI legal assistant specialized in Indian law."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    
    # Tokenize
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    
    # Set to inference mode
    FastLanguageModel.for_inference(model)
    
    # Generate
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1,
        use_cache=True,
    )
    
    # Decode response only
    response = tokenizer.decode(
        outputs[0][inputs.shape[-1]:],
        skip_special_tokens=True,
    )
    
    return response.strip()


def interactive_mode(model, tokenizer):
    """Run interactive Q&A loop."""
    console.print("\n[bold cyan]═══════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]  JurisAI - Interactive Legal Assistant[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════[/bold cyan]")
    console.print("[dim]Type your legal query and press Enter.[/dim]")
    console.print("[dim]Type 'quit' or 'exit' to stop.[/dim]")
    console.print("[dim]Type 'clear' to clear the screen.[/dim]\n")
    
    while True:
        try:
            query = console.input("[bold green]You > [/bold green]").strip()
            
            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                console.print("[yellow]Goodbye! 👋[/yellow]")
                break
            if query.lower() == "clear":
                console.clear()
                continue
            
            console.print("\n[dim]Thinking...[/dim]")
            response = generate(model, tokenizer, query)
            
            # Display response in a nice panel
            console.print()
            console.print(Panel(
                Markdown(response),
                title="[bold blue]JurisAI[/bold blue]",
                border_style="blue",
                padding=(1, 2),
            ))
            console.print()
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Goodbye! 👋[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="JurisAI Interactive Inference")
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to fine-tuned adapter directory"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Single query to answer (non-interactive mode)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate"
    )
    args = parser.parse_args()
    
    # Load model
    from src.training.train_utils import load_model_and_tokenizer, clear_gpu_memory
    
    model_config = load_config("model_config.yaml")
    
    if args.adapter_path:
        model_config_copy = model_config.copy()
        model_config_copy["base_model"] = model_config["base_model"].copy()
        model_config_copy["base_model"]["name"] = args.adapter_path
    else:
        # Try to find the latest adapter
        adapter_dirs = [
            PROJECT_ROOT / "models" / "adapters" / "instruct_v1" / "final",
            PROJECT_ROOT / "models" / "adapters" / "pretrain_v1" / "final",
        ]
        
        model_config_copy = model_config.copy()
        for adapter_dir in adapter_dirs:
            if adapter_dir.exists():
                console.print(f"[green]Found adapter: {adapter_dir}[/green]")
                model_config_copy["base_model"] = model_config["base_model"].copy()
                model_config_copy["base_model"]["name"] = str(adapter_dir)
                break
        else:
            console.print("[yellow]No fine-tuned adapter found. Using base model.[/yellow]")
    
    console.print("[bold]Loading model...[/bold]")
    model, tokenizer = load_model_and_tokenizer(model_config_copy)
    
    if args.query:
        # Single query mode
        response = generate(model, tokenizer, args.query, args.max_tokens)
        console.print(Panel(
            Markdown(response),
            title="[bold blue]JurisAI[/bold blue]",
            border_style="blue",
            padding=(1, 2),
        ))
    else:
        # Interactive mode
        interactive_mode(model, tokenizer)
    
    clear_gpu_memory()


if __name__ == "__main__":
    main()
