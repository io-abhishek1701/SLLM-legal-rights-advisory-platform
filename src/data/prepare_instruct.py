"""
JurisAI - Instruction Data Formatter
Converts preprocessed data into ChatML format for Qwen2.5 fine-tuning.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.progress import track

from src.data.data_utils import (
    load_config, load_jsonl, save_jsonl, format_messages,
    ensure_dirs, get_cross_reference
)

console = Console()


# System prompt for JurisAI
SYSTEM_PROMPT = """You are JurisAI, an expert AI legal assistant specialized in Indian law. You provide accurate, well-cited legal information based on Indian statutes, constitutional provisions, and court judgments.

IMPORTANT GUIDELINES:
- Always cite the specific section, act, or case you reference
- Clearly distinguish between old laws (IPC, CrPC, Indian Evidence Act) and new laws (BNS, BNSS, BSA) that replaced them on July 1, 2024
- Include relevant cross-references (e.g., "Section 302 IPC, now Section 103 BNS") when applicable
- Add a disclaimer that this is informational only, not legal advice
- If unsure, say so rather than fabricating information
- Refuse to provide advice on committing illegal acts"""


def enhance_with_cross_references(text: str) -> str:
    """Add BNS/IPC cross-references to legal text.
    
    Scans text for section references and appends cross-references
    to help the model learn both old and new law mappings.
    """
    import re
    
    # Find IPC section references
    ipc_pattern = r'(?:Section\s+)?(\d+[A-Z]?)\s+(?:of\s+)?(?:the\s+)?(?:IPC|Indian Penal Code)'
    for match in re.finditer(ipc_pattern, text, re.IGNORECASE):
        section_num = match.group(1)
        ref = get_cross_reference(f"IPC {section_num}")
        if ref and ref not in text:
            # Append cross-reference note
            text += f"\n\n[Note: IPC Section {section_num} corresponds to {ref} under the new Bharatiya Nyaya Sanhita (BNS), 2023]"
    
    return text


def format_instruction_entry(
    entry: Dict,
    add_cross_refs: bool = True,
    add_disclaimer: bool = True,
) -> Dict:
    """Convert a single instruction entry to ChatML messages format.
    
    Input format:
        {"instruction": "...", "input": "...", "output": "..."}
    
    Output format:
        {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
    """
    instruction = entry.get("instruction", "").strip()
    context = entry.get("input", "").strip()
    output = entry.get("output", "").strip()
    
    # Build user message
    if context:
        user_msg = f"{instruction}\n\nContext: {context}"
    else:
        user_msg = instruction
    
    # Enhance output
    assistant_msg = output
    
    if add_cross_refs:
        assistant_msg = enhance_with_cross_references(assistant_msg)
    
    if add_disclaimer and "disclaimer" not in assistant_msg.lower() and "not legal advice" not in assistant_msg.lower():
        assistant_msg += "\n\n---\n*Disclaimer: This information is for educational purposes only and does not constitute legal advice. Please consult a qualified legal professional for specific legal matters.*"
    
    # Format as messages
    messages = format_messages(
        system=SYSTEM_PROMPT,
        user=user_msg,
        assistant=assistant_msg,
    )
    
    return {"messages": messages}


def prepare_instruction_dataset(
    input_dir: str,
    output_dir: str,
    add_cross_refs: bool = True,
    add_disclaimer: bool = True,
) -> None:
    """Convert all instruction data files to ChatML format."""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    ensure_dirs(str(output_path))
    
    for split_file in input_path.glob("*.jsonl"):
        split_name = split_file.stem  # train, validation, test
        console.print(f"\n[blue]Processing {split_name}...[/blue]")
        
        # Load raw instruction data
        raw_data = load_jsonl(str(split_file))
        
        if not raw_data:
            console.print(f"[yellow]  ⚠ No data in {split_file}[/yellow]")
            continue
        
        # Convert to ChatML format
        formatted = []
        skipped = 0
        
        for entry in track(raw_data, description=f"  Formatting {split_name}"):
            try:
                formatted_entry = format_instruction_entry(
                    entry,
                    add_cross_refs=add_cross_refs,
                    add_disclaimer=add_disclaimer,
                )
                formatted.append(formatted_entry)
            except Exception as e:
                skipped += 1
                continue
        
        # Save formatted data
        output_file = output_path / f"{split_name}_formatted.jsonl"
        save_jsonl(formatted, str(output_file))
        
        if skipped > 0:
            console.print(f"  [yellow]⚠ Skipped {skipped} malformed entries[/yellow]")
        
        # Show a sample
        if formatted:
            console.print(f"\n  [dim]Sample formatted entry:[/dim]")
            sample = formatted[0]
            for msg in sample["messages"]:
                role = msg["role"]
                content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                console.print(f"  [dim]  [{role}]: {content}[/dim]")


def prepare_all() -> None:
    """Run full instruction preparation pipeline."""
    console.print("\n[bold cyan]═══════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]  JurisAI - Instruction Data Formatter[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════[/bold cyan]")
    
    instruct_raw = str(PROJECT_ROOT / "data" / "processed" / "instruct")
    instruct_formatted = str(PROJECT_ROOT / "data" / "processed" / "instruct" / "formatted")
    
    prepare_instruction_dataset(
        input_dir=instruct_raw,
        output_dir=instruct_formatted,
        add_cross_refs=True,
        add_disclaimer=True,
    )
    
    console.print("\n[bold green]✓ Instruction data preparation complete![/bold green]")


if __name__ == "__main__":
    prepare_all()
