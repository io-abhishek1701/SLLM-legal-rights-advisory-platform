# =============================================================================
# JurisAI - Environment Setup Script (Windows PowerShell)
# =============================================================================
# Run: .\scripts\setup_env.ps1
# =============================================================================

Write-Host ""
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "  JurisAI - Environment Setup" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "[1/6] Checking Python..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
Write-Host "  Found: $pythonVersion" -ForegroundColor Green

# Check NVIDIA GPU
Write-Host "[2/6] Checking GPU..." -ForegroundColor Yellow
$nvidiaSmi = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  Found: $nvidiaSmi" -ForegroundColor Green
} else {
    Write-Host "  WARNING: No NVIDIA GPU detected!" -ForegroundColor Red
    Write-Host "  Training will be very slow on CPU." -ForegroundColor Red
}

# Create virtual environment
Write-Host "[3/6] Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "  Virtual environment already exists." -ForegroundColor Yellow
} else {
    python -m venv venv
    Write-Host "  Created: ./venv" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "[4/6] Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Install PyTorch with CUDA
Write-Host "[5/6] Installing PyTorch with CUDA support..." -ForegroundColor Yellow
Write-Host "  This may take several minutes..." -ForegroundColor DarkGray
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 2>&1 | Out-Null

# Verify CUDA
python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# Install all requirements
Write-Host "[6/6] Installing project dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt 2>&1 | Out-Null

# Create directory structure
Write-Host ""
Write-Host "Creating project directories..." -ForegroundColor Yellow
$dirs = @(
    "data/raw/huggingface",
    "data/raw/indian_legal_texts",
    "data/raw/judgments",
    "data/processed/pretrain",
    "data/processed/instruct/formatted",
    "data/evaluation",
    "models/base",
    "models/adapters/pretrain_v1",
    "models/adapters/instruct_v1",
    "models/merged",
    "logs",
    "notebooks"
)

foreach ($dir in $dirs) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
}
Write-Host "  Directories created." -ForegroundColor Green

# Create .gitkeep files
foreach ($dir in $dirs) {
    if (-not (Test-Path "$dir/.gitkeep")) {
        New-Item -ItemType File -Path "$dir/.gitkeep" -Force | Out-Null
    }
}

# Final verification
Write-Host ""
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Verifying installation..." -ForegroundColor Yellow
python -c @"
import torch
import transformers
import peft
import datasets
print(f'  PyTorch:      {torch.__version__}')
print(f'  CUDA:         {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"})')
print(f'  Transformers: {transformers.__version__}')
print(f'  PEFT:         {peft.__version__}')
print(f'  Datasets:     {datasets.__version__}')
try:
    import unsloth
    print(f'  Unsloth:      OK')
except ImportError:
    print(f'  Unsloth:      NOT INSTALLED (run: pip install unsloth)')
"@

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Activate venv:  .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  2. Download model: python scripts/download_model.py" -ForegroundColor White
Write-Host "  3. Download data:  python -m src.data.download_datasets" -ForegroundColor White
Write-Host "  4. Preprocess:     python -m src.data.preprocess" -ForegroundColor White
Write-Host "  5. Format data:    python -m src.data.prepare_instruct" -ForegroundColor White
Write-Host "  6. Pretrain:       python -m src.training.pretrain" -ForegroundColor White
Write-Host "  7. Fine-tune:      python -m src.training.finetune" -ForegroundColor White
Write-Host "  8. Test:           python -m src.inference.generate" -ForegroundColor White
Write-Host ""
