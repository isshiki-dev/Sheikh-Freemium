#!/usr/bin/env python3
"""
Sheikh-Freemium: Setup and Imports
===================================

This module covers the essential setup and imports for training Visual Chain of Thought
models. Based on research from:
- Zebra-CoT (multimodal-reasoning-lab/Bagel-Zebra-CoT)
- HuggingFace TRL VLM fine-tuning documentation
- PyTorch best practices for multimodal training

It's highly recommended to build a solid foundation in the attention mechanism
and tokenization before proceeding with Visual CoT training.
"""

# ============================================================================
# SECTION 1: Core Deep Learning Framework
# ============================================================================
"""
PyTorch is the deep learning framework used for training.
It provides:
- Tensor operations on GPU
- Automatic differentiation (autograd)
- Neural network building blocks
- Optimizers and loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler  # Mixed precision training

# ============================================================================
# SECTION 2: Transformers and Tokenization
# ============================================================================
"""
HuggingFace Transformers provides:
- Pre-trained models (LLMs, VLMs)
- Tokenizers for text processing
- Processors for multimodal inputs
- Training utilities (Trainer, TrainingArguments)
"""

from transformers import (
    # Tokenization
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    
    # Models
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,  # For VLMs
    
    # Processors (multimodal)
    AutoProcessor,
    
    # Configuration
    AutoConfig,
    BitsAndBytesConfig,  # Quantization
    
    # Training
    TrainingArguments,
    Trainer,
    
    # Optimization
    get_scheduler,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

# ============================================================================
# SECTION 3: Parameter-Efficient Fine-Tuning (PEFT)
# ============================================================================
"""
PEFT allows fine-tuning large models with minimal parameters:
- LoRA: Low-Rank Adaptation
- QLoRA: Quantized LoRA (4-bit training)
- Adapters: Small trainable modules

This is essential for training 7B+ parameter VLMs on consumer hardware.
"""

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,
)

# ============================================================================
# SECTION 4: TRL (Transformer Reinforcement Learning)
# ============================================================================
"""
TRL provides specialized training for:
- Supervised Fine-Tuning (SFT)
- Reinforcement Learning from Human Feedback (RLHF)
- Direct Preference Optimization (DPO)
- Vision Language Model training
"""

from trl import (
    SFTTrainer,
    SFTConfig,
    # DataCollatorForCompletionOnlyLM,  # For instruction tuning
)

# ============================================================================
# SECTION 5: Data Handling
# ============================================================================
"""
Datasets library for efficient data loading and processing.
Supports streaming, memory mapping, and various formats.
"""

from datasets import (
    load_dataset,
    Dataset as HFDataset,
    DatasetDict,
    Features,
    Image as ImageFeature,
    Value,
    Sequence,
)

# ============================================================================
# SECTION 6: Image Processing
# ============================================================================
"""
PIL and torchvision for image handling.
Used for processing input images and generated reasoning images.
"""

from PIL import Image
import io

try:
    import torchvision
    from torchvision import transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

# ============================================================================
# SECTION 7: Utilities
# ============================================================================
"""
Various utilities for training, logging, and data handling.
"""

import os
import sys
import json
import yaml
import math
import random
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

import numpy as np
from tqdm import tqdm, trange

# HuggingFace Hub for model/dataset management
from huggingface_hub import (
    login,
    HfApi,
    hf_hub_download,
    snapshot_download,
    upload_folder,
    list_repo_files,
)

# ============================================================================
# SECTION 8: Flash Attention (Optional but Recommended)
# ============================================================================
"""
Flash Attention provides:
- 2-4x faster attention computation
- Reduced memory usage
- Essential for training long sequences

Install with: pip install flash_attn --no-build-isolation
"""

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

# ============================================================================
# SECTION 9: Weights & Biases (Optional Logging)
# ============================================================================
"""
W&B for experiment tracking and visualization.
"""

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# ============================================================================
# SECTION 10: Environment Configuration
# ============================================================================

@dataclass
class EnvironmentInfo:
    """System environment information."""
    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_version: Optional[str]
    gpu_count: int
    gpu_names: List[str]
    flash_attn_available: bool
    transformers_version: str

def get_environment_info() -> EnvironmentInfo:
    """Gather environment information for reproducibility."""
    import transformers
    
    gpu_names = []
    cuda_version = None
    
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        for i in range(torch.cuda.device_count()):
            gpu_names.append(torch.cuda.get_device_name(i))
    
    return EnvironmentInfo(
        python_version=sys.version.split()[0],
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_version=cuda_version,
        gpu_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        gpu_names=gpu_names,
        flash_attn_available=FLASH_ATTN_AVAILABLE,
        transformers_version=transformers.__version__,
    )

def print_environment_info():
    """Print environment information."""
    info = get_environment_info()
    
    print("=" * 60)
    print("SHEIKH-FREEMIUM: Environment Information")
    print("=" * 60)
    print(f"Python Version:      {info.python_version}")
    print(f"PyTorch Version:     {info.torch_version}")
    print(f"Transformers:        {info.transformers_version}")
    print(f"CUDA Available:      {info.cuda_available}")
    
    if info.cuda_available:
        print(f"CUDA Version:        {info.cuda_version}")
        print(f"GPU Count:           {info.gpu_count}")
        for i, name in enumerate(info.gpu_names):
            print(f"  GPU {i}:             {name}")
    
    print(f"Flash Attention:     {'Yes' if info.flash_attn_available else 'No (pip install flash_attn)'}")
    print(f"W&B Available:       {'Yes' if WANDB_AVAILABLE else 'No'}")
    print(f"TorchVision:         {'Yes' if TORCHVISION_AVAILABLE else 'No'}")
    print("=" * 60)

# ============================================================================
# SECTION 11: Seed Setting for Reproducibility
# ============================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to: {seed}")

# ============================================================================
# SECTION 12: Model Loading Configurations
# ============================================================================

def get_bnb_config(load_in_4bit: bool = True) -> BitsAndBytesConfig:
    """
    Get BitsAndBytes configuration for quantized training.
    
    This enables training 7B+ models on consumer GPUs (24GB VRAM).
    
    Args:
        load_in_4bit: Whether to use 4-bit quantization (vs 8-bit)
    
    Returns:
        BitsAndBytesConfig for model loading
    """
    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )

def get_lora_config(
    r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: str = "all-linear",
) -> LoraConfig:
    """
    Get LoRA configuration for parameter-efficient fine-tuning.
    
    Args:
        r: LoRA rank (higher = more parameters, better quality)
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout for LoRA layers
        target_modules: Which modules to apply LoRA to
    
    Returns:
        LoraConfig for PEFT
    """
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=["lm_head", "embed_tokens"],
    )

# ============================================================================
# SECTION 13: Visual CoT Specific Imports
# ============================================================================

# Special tokens for Visual CoT
VISUAL_COT_SPECIAL_TOKENS = {
    "thought": "<|thought|>",
    "reasoning_image": "<|reasoning_image|>",
    "answer": "<|answer|>",
    "image_start": "<|image_start|>",
    "image_end": "<|image_end|>",
}

# Category definitions from Zebra-CoT
ZEBRA_COT_CATEGORIES = [
    "scientific",      # Geometry, physics, algorithms
    "visual_2d",       # Visual search, jigsaw puzzles
    "visual_3d",       # Multi-hop inference, embodied planning
    "logic_games",     # Chess, visual logic, strategic games
]

# ============================================================================
# SECTION 14: Main Execution
# ============================================================================

if __name__ == "__main__":
    # Print environment info
    print_environment_info()
    
    # Set seed
    set_seed(42)
    
    # Verify all imports work
    print("\n" + "=" * 60)
    print("IMPORT VERIFICATION")
    print("=" * 60)
    
    checks = [
        ("torch", torch.__version__),
        ("transformers", "Available"),
        ("peft", "Available"),
        ("trl", "Available"),
        ("datasets", "Available"),
        ("PIL", "Available"),
        ("huggingface_hub", "Available"),
    ]
    
    for name, status in checks:
        print(f"  [OK] {name}: {status}")
    
    print("\n" + "=" * 60)
    print("All imports successful! Ready for Visual CoT training.")
    print("=" * 60)
    
    print("""
    Next Steps:
    1. Run: python notebooks/02_tokenization.py (tokenization deep dive)
    2. Run: python notebooks/03_attention.py (attention mechanism)
    3. Run: python mlops/training/train.py --dry-run (test training)
    
    For full training:
    - Ensure HF_TOKEN is set: huggingface-cli login
    - Modify mlops/training/config.yaml for your setup
    - Run: python mlops/training/train.py
    """)
