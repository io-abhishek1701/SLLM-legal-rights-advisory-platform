# JurisAI — sLLM-Powered Legal Advisory Platform for Indian Law

> AI-powered legal assistant specialized in Indian law, built around a fine-tuned small Language Model (sLLM) with RAG and safety guardrails.

![Status](https://img.shields.io/badge/Phase-1%20Training-blue)
![Model](https://img.shields.io/badge/Model-Qwen2.5--1.5B-green)
![License](https://img.shields.io/badge/License-Apache%202.0-orange)

## 🏗️ Architecture

```
User Query
    ↓
Query Analyzer (intent, jurisdiction, risk)
    ↓
Retriever - RAG (Acts, Judgments, Sections)
    ↓
Fine-Tuned sLLM (QLoRA - Qwen2.5-1.5B)
    ↓
Citation Validator
    ↓
Safety & Refusal Layer
    ↓
Final Legal Response
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- NVIDIA GPU with CUDA support (RTX 3050 4GB+ tested)
- 64GB System RAM recommended

### Setup

```powershell
# 1. Run setup script
.\scripts\setup_env.ps1

# 2. Activate environment
.\venv\Scripts\Activate.ps1

# 3. Download base model
python scripts/download_model.py

# 4. Download & prepare datasets
python -m src.data.download_datasets
python -m src.data.preprocess
python -m src.data.prepare_instruct

# 5. Train (Stage 1: Pretraining + Stage 2: Fine-tuning)
python -m src.training.pretrain
python -m src.training.finetune --from-pretrained ./models/adapters/pretrain_v1/final --export-gguf

# 6. Test
python -m src.inference.generate

# 7. Evaluate
python -m src.evaluation.evaluate
```

## 📊 Model Details

| Property | Value |
|:---|:---|
| Base Model | Qwen2.5-1.5B-Instruct |
| Training Method | QLoRA (4-bit, rank 16) |
| Framework | Unsloth + HuggingFace |
| Hardware | RTX 3050 4GB VRAM |
| Context Length | 2048 tokens |
| Domain | Indian Law (IPC/BNS, CrPC/BNSS, Constitution) |

## 📁 Project Structure

```
JurisAI/
├── config/              # YAML configuration files
├── data/                # Datasets (raw, processed, evaluation)
├── models/              # Model weights and adapters
├── src/
│   ├── data/            # Data processing pipeline
│   ├── training/        # QLoRA training scripts
│   ├── evaluation/      # Metrics and evaluation suite
│   └── inference/       # Generation and inference
├── scripts/             # Setup and utility scripts
├── logs/                # Training logs
└── notebooks/           # Jupyter notebooks
```

## ⚖️ Indian Law Coverage

- **Old Laws**: IPC, CrPC, Indian Evidence Act
- **New Laws (2024)**: BNS, BNSS, BSA
- **Constitutional Law**: Fundamental Rights, DPSP, Articles
- **Cross-References**: IPC ↔ BNS section mappings

## ⚠️ Disclaimer

This tool is for **educational and informational purposes only**. It does not constitute legal advice. Always consult a qualified legal professional for specific legal matters.
