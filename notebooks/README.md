# Training Notebooks

Interactive tutorials for understanding and training Visual CoT models.

## Setup

### 1. Create Environment

```bash
# Using conda (recommended)
conda create -n sheikh-freemium python=3.10 -y
conda activate sheikh-freemium

# Install requirements
pip install -r requirements.txt

# Optional: Flash Attention (2-4x faster training)
pip install flash_attn --no-build-isolation
```

### 2. Login to HuggingFace

```bash
huggingface-cli login
```

## Notebooks

### 01. Setup and Imports (`01_setup_and_imports.py`)

Comprehensive setup covering all required imports:

- **PyTorch**: Core deep learning framework
- **Transformers**: Tokenizers, models, processors
- **PEFT**: LoRA/QLoRA for efficient fine-tuning
- **TRL**: SFTTrainer for VLM training
- **Datasets**: Data loading and processing
- **PIL**: Image handling

```bash
python notebooks/01_setup_and_imports.py
```

## Key Dependencies

| Package | Purpose | Version |
|---------|---------|--------|
| torch | Deep learning | >=2.1.0 |
| transformers | Models & tokenizers | >=4.40.0 |
| peft | LoRA/QLoRA | >=0.10.0 |
| trl | SFT training | >=0.8.0 |
| bitsandbytes | 4-bit quantization | >=0.43.0 |
| datasets | Data loading | >=2.18.0 |

## Quick Reference

### Essential Imports

```python
# Core
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

# Quantization
from transformers import BitsAndBytesConfig

# PEFT
from peft import LoraConfig, get_peft_model

# Training
from trl import SFTTrainer, SFTConfig

# Data
from datasets import load_dataset
from PIL import Image
```

### Model Loading with QLoRA

```python
# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load model
model = AutoModelForImageTextToText.from_pretrained(
    "model_id",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
```

## Resources

- [Zebra-CoT Repository](https://github.com/multimodal-reasoning-lab/Bagel-Zebra-CoT)
- [TRL VLM Fine-tuning Guide](https://huggingface.co/docs/trl/main/en/training_vlm_sft)
- [PEFT Documentation](https://huggingface.co/docs/peft)
