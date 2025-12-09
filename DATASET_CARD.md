---
license: cc-by-nc-4.0
task_categories:
  - visual-question-answering
  - image-to-text
  - text-generation
language:
  - en
tags:
  - visual-reasoning
  - chain-of-thought
  - multimodal
  - visual-cot
  - interleaved-generation
  - zebra-cot
  - geometry
  - physics
  - chess
  - embodied-ai
size_categories:
  - 100K<n<1M
source_datasets:
  - multimodal-reasoning-lab/Zebra-CoT
pretty_name: Sheikh-Freemium Visual CoT Dataset
---

# Sheikh-Freemium: Visual Chain of Thought Dataset

<div align="center">

[![Based on Zebra-CoT](https://img.shields.io/badge/Based%20on-Zebra--CoT-blue)](https://huggingface.co/datasets/multimodal-reasoning-lab/Zebra-CoT)
[![Paper](https://img.shields.io/badge/arXiv-2507.16746-b31b1b)](https://arxiv.org/abs/2507.16746)
[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey)](https://creativecommons.org/licenses/by-nc/4.0/)

</div>

## Dataset Description

Sheikh-Freemium is a Visual Chain of Thought (Visual CoT) dataset for training multimodal models to generate interleaved text-image reasoning traces. Based on the [Zebra-CoT](https://huggingface.co/datasets/multimodal-reasoning-lab/Zebra-CoT) framework.

### Dataset Summary

- **Total Samples**: 182,384
- **Format**: Parquet / JSON
- **Modalities**: Text + Image
- **License**: CC BY-NC 4.0

## Dataset Structure

### Categories

| Category | Samples | Percentage | Description |
|----------|---------|------------|-------------|
| Visual Logic & Games | 66,854 | 36.7% | Chess, strategic games, visual logic |
| 2D Visual Reasoning | 51,899 | 28.5% | Visual search, jigsaw puzzles |
| 3D Visual Reasoning | 39,610 | 21.7% | Multi-hop inference, embodied planning |
| Scientific Reasoning | 24,021 | 13.2% | Geometry, physics, algorithms |

### Data Fields

```python
{
    "id": str,                    # Unique sample identifier
    "category": str,              # Task category
    "subcategory": str,           # Specific task type
    "question": str,              # Problem statement
    "input_images": List[Image],  # Problem images (optional)
    "reasoning_steps": [          # Interleaved reasoning chain
        {
            "step": int,
            "type": "text" | "image" | "interleaved",
            "content": str,
            "image_path": str,    # Generated visual (optional)
            "intermediate_result": str
        }
    ],
    "answer": str,                # Final solution
    "metadata": {
        "difficulty": str,
        "source": str,
        "tags": List[str]
    }
}
```

### Data Splits

| Split | Samples |
|-------|--------|
| train | 160,485 |
| test | 21,899 |

## Usage

### Loading the Dataset

```python
from datasets import load_dataset

# Load full dataset
dataset = load_dataset("shk-bd/Sheikh-Freemium")

# Load specific split
train_data = load_dataset("shk-bd/Sheikh-Freemium", split="train")

# Load specific category
scientific = dataset.filter(lambda x: x['category'] == 'scientific')
```

### Iterating Through Samples

```python
for sample in dataset['train']:
    print(f"Question: {sample['question']}")
    print(f"Category: {sample['category']}")
    
    for step in sample['reasoning_steps']:
        print(f"  Step {step['step']}: {step['content']}")
    
    print(f"Answer: {sample['answer']}")
    print("---")
```

### Training Example

```python
from transformers import Trainer, TrainingArguments

# Prepare for training
def preprocess(example):
    # Format interleaved reasoning trace
    trace = ""
    for step in example['reasoning_steps']:
        trace += f"THOUGHT {step['step']}: {step['content']}\n"
        if step.get('image_path'):
            trace += f"[reasoning_image_{step['step']}]\n"
    return {"text": f"{example['question']}\n{trace}\nANSWER: {example['answer']}"}

train_dataset = dataset['train'].map(preprocess)
```

## Performance Benchmarks

Models fine-tuned on this dataset show significant improvements:

| Model | Before | After | Gain |
|-------|--------|-------|------|
| Anole-7B | 4.2% | 16.9% | +12.7% |
| VLM Benchmarks | baseline | — | up to +13% |

## Considerations

### Intended Uses

- Training multimodal models for visual reasoning
- Research on chain-of-thought prompting
- Developing interleaved text-image generation systems

### Limitations

- English only
- Focused on 4 main reasoning categories
- Requires substantial compute for full training

### Licensing

This dataset is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Commercial use requires separate licensing.

## Citation

```bibtex
@misc{li2025zebracot,
  title={Zebra-CoT: A Dataset for Interleaved Vision Language Reasoning},
  author={Ang Li and Charles Wang and Kaiyu Yue and Zikui Cai and Ollie Liu and Deqing Fu and Peng Guo and Wang Bill Zhu and Vatsal Sharan and Robin Jia and Willie Neiswanger and Furong Huang and Tom Goldstein and Micah Goldblum},
  year={2025},
  eprint={2507.16746},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2507.16746},
}
```

## Related Resources

- [Zebra-CoT Dataset](https://huggingface.co/datasets/multimodal-reasoning-lab/Zebra-CoT)
- [Zebra-CoT Paper](https://arxiv.org/abs/2507.16746)
- [Anole-Zebra-CoT Model](https://huggingface.co/multimodal-reasoning-lab/Anole-Zebra-CoT)
- [Bagel-Zebra-CoT Model](https://huggingface.co/multimodal-reasoning-lab/Bagel-Zebra-CoT)
- [GitHub Repository](https://github.com/isshiki-dev/Sheikh-Freemium)
