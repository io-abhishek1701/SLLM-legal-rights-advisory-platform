"""
JurisAI - Data Utilities
Shared helper functions for data processing pipeline.
"""

import os
import re
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml
from rich.console import Console

console = Console()


def load_config(config_name: str) -> Dict[str, Any]:
    """Load a YAML config file from the config/ directory."""
    config_path = Path(__file__).parent.parent.parent / "config" / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_text(text: str, remove_html: bool = True, normalize: bool = True) -> str:
    """Clean raw legal text."""
    if not text or not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    if remove_html:
        text = re.sub(r"<[^>]+>", " ", text)
    
    # Normalize unicode
    if normalize:
        import unicodedata
        text = unicodedata.normalize("NFKC", text)
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    # Remove control characters (except newlines)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    
    return text


def has_legal_keywords(text: str) -> bool:
    """Check if text contains Indian legal terminology."""
    legal_keywords = [
        # General legal terms
        "section", "act", "article", "constitution", "court", "judgment",
        "tribunal", "petition", "appeal", "bail", "prosecution", "accused",
        "plaintiff", "defendant", "advocate", "bench", "verdict", "statute",
        "ordinance", "amendment", "provision", "clause", "schedule",
        
        # Old Indian laws
        "ipc", "crpc", "indian penal code", "criminal procedure",
        "evidence act",
        
        # New Indian laws (2024)
        "bns", "bnss", "bsa",
        "bharatiya nyaya sanhita", "bharatiya nagarik suraksha sanhita",
        "bharatiya sakshya adhiniyam",
        
        # Courts
        "supreme court", "high court", "district court", "sessions court",
        
        # Hindi legal terms
        "dhara", "kanoon", "adalat", "nyayalaya", "dand", "aparadh",
        "nyaya", "vidhi", "sanhita",
    ]
    
    text_lower = text.lower()
    return any(kw in text_lower for kw in legal_keywords)


def compute_text_hash(text: str) -> str:
    """Compute MD5 hash for deduplication."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def ensure_dirs(*dirs: str) -> None:
    """Create directories if they don't exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def save_jsonl(data: List[Dict], filepath: str) -> None:
    """Save data as JSON Lines file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    console.print(f"[green]✓ Saved {len(data)} entries to {filepath}[/green]")


def load_jsonl(filepath: str) -> List[Dict]:
    """Load data from JSON Lines file."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_chatml(
    system: str,
    user: str,
    assistant: str
) -> str:
    """Format a conversation in ChatML format (Qwen2.5 native format).
    
    Returns the formatted string for tokenization.
    """
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>"
    )


def format_messages(
    system: str,
    user: str,
    assistant: str
) -> List[Dict[str, str]]:
    """Format as messages list for HuggingFace chat template."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    messages.append({"role": "assistant", "content": assistant})
    return messages


# IPC Section → BNS Section cross-reference mapping
# This is a critical differentiator for JurisAI
IPC_TO_BNS_MAPPING = {
    # Murder & Culpable Homicide
    "IPC 299": "BNS 100",   # Culpable homicide
    "IPC 300": "BNS 101",   # Murder
    "IPC 302": "BNS 103",   # Punishment for murder
    "IPC 304": "BNS 105",   # Punishment for culpable homicide
    "IPC 304A": "BNS 106",  # Death by negligence
    "IPC 304B": "BNS 80",   # Dowry death
    
    # Theft, Robbery, Dacoity
    "IPC 378": "BNS 303",   # Theft
    "IPC 379": "BNS 305",   # Punishment for theft
    "IPC 390": "BNS 309",   # Robbery
    "IPC 391": "BNS 310",   # Dacoity
    "IPC 395": "BNS 312",   # Punishment for dacoity
    
    # Cheating & Fraud
    "IPC 415": "BNS 318",   # Cheating
    "IPC 420": "BNS 316",   # Cheating and dishonestly inducing
    
    # Hurt & Grievous Hurt
    "IPC 319": "BNS 114",   # Hurt
    "IPC 320": "BNS 115",   # Grievous hurt
    "IPC 323": "BNS 118",   # Voluntarily causing hurt
    "IPC 325": "BNS 119",   # Voluntarily causing grievous hurt
    "IPC 326": "BNS 120",   # Grievous hurt by dangerous weapons
    
    # Kidnapping & Abduction
    "IPC 359": "BNS 137",   # Kidnapping
    "IPC 363": "BNS 138",   # Punishment for kidnapping
    "IPC 366": "BNS 139",   # Kidnapping woman to compel marriage
    
    # Sexual Offences
    "IPC 375": "BNS 63",    # Rape
    "IPC 376": "BNS 64",    # Punishment for rape
    
    # Criminal Intimidation & Defamation
    "IPC 499": "BNS 356",   # Defamation
    "IPC 500": "BNS 357",   # Punishment for defamation
    "IPC 503": "BNS 351",   # Criminal intimidation
    "IPC 506": "BNS 351",   # Criminal intimidation punishment
    
    # Forgery
    "IPC 463": "BNS 336",   # Forgery
    "IPC 465": "BNS 338",   # Punishment for forgery
    "IPC 468": "BNS 340",   # Forgery for cheating
    
    # Sedition & National Security (significant changes)
    "IPC 124A": "BNS 152",  # Sedition → Acts endangering sovereignty
    
    # Miscellaneous
    "IPC 34":  "BNS 3(5)",  # Common intention
    "IPC 107": "BNS 45",    # Abetment
    "IPC 120B": "BNS 61",   # Criminal conspiracy
    "IPC 149": "BNS 190",   # Unlawful assembly
    "IPC 153A": "BNS 196",  # Promoting enmity
    "IPC 279": "BNS 281",   # Rash driving
    "IPC 354": "BNS 74",    # Assault on woman
    "IPC 498A": "BNS 85",   # Cruelty by husband
    "IPC 509": "BNS 79",    # Word/gesture to insult modesty
}

# Reverse mapping
BNS_TO_IPC_MAPPING = {v: k for k, v in IPC_TO_BNS_MAPPING.items()}


def get_cross_reference(section: str) -> Optional[str]:
    """Get the cross-reference for an IPC/BNS section.
    
    Args:
        section: e.g., "IPC 302" or "BNS 103"
    
    Returns:
        The corresponding section in the other law, or None.
    """
    section = section.strip().upper()
    
    if section in IPC_TO_BNS_MAPPING:
        return IPC_TO_BNS_MAPPING[section]
    elif section in BNS_TO_IPC_MAPPING:
        return BNS_TO_IPC_MAPPING[section]
    
    return None
