---
pipeline_tag: image-text-to-text
library_name: transformers
license: cc-by-nc-4.0
language:
  - en
datasets:
  - multimodal-reasoning-lab/Zebra-CoT
base_model:
  - multimodal-reasoning-lab/Anole-Zebra-CoT
  - multimodal-reasoning-lab/Bagel-Zebra-CoT
tags:
  - visual-reasoning
  - chain-of-thought
  - multimodal
  - visual-cot
  - interleaved-generation
  - zebra-cot
metrics:
  - accuracy
---

# Sheikh-Freemium: Visual Chain of Thought Reasoning

<div align="center">

[![Dataset](https://img.shields.io/badge/🤗%20Dataset-Zebra--CoT-yellow)](https://huggingface.co/datasets/multimodal-reasoning-lab/Zebra-CoT)
[![Paper](https://img.shields.io/badge/arXiv-2507.16746-b31b1b)](https://arxiv.org/abs/2507.16746)
[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey)](https://creativecommons.org/licenses/by-nc/4.0/)

</div>

## Model Description

Sheikh-Freemium is a Visual Chain of Thought (Visual CoT) reasoning framework based on the [Zebra-CoT](https://huggingface.co/datasets/multimodal-reasoning-lab/Zebra-CoT) dataset. It enables multimodal models to generate interleaved text-image reasoning traces for complex problem-solving.

### Key Features

- **Mixture-of-Transformer-Experts (MoT)** architecture for diverse multimodal learning
- **Dual encoders** capturing pixel-level and semantic-level image features
- **Next Group of Token Prediction (NGTP)** paradigm for interleaved generation
- **182K+ training samples** across 4 reasoning categories

## Intended Use

### Primary Use Cases

- Scientific reasoning (geometry, physics, algorithms)
- 2D visual reasoning (visual search, jigsaw puzzles)
- 3D spatial reasoning (multi-hop inference, embodied planning)
- Strategic games and visual logic (chess, pattern recognition)

### Out-of-Scope Uses

- Real-time safety-critical applications
- Medical or legal decision-making without human oversight

## Usage

### Quick Start

```python
from transformers import AutoModelForCausalLM, AutoProcessor

# Load model and processor
model = AutoModelForCausalLM.from_pretrained("shk-bd/Sheikh-Freemium")
processor = AutoProcessor.from_pretrained("shk-bd/Sheikh-Freemium")

# Prepare input
inputs = processor(
    text="Solve this geometry problem step by step:",
    images=image,
    return_tensors="pt"
)

# Generate reasoning chain
outputs = model.generate(**inputs, max_new_tokens=512)
response = processor.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Using with Zebra-CoT Dataset

```python
from datasets import load_dataset

# Load the training data
dataset = load_dataset("multimodal-reasoning-lab/Zebra-CoT")

# Access a sample
sample = dataset['train'][0]
print(f"Problem: {sample['problem']}")
print(f"Answer: {sample['final_answer']}")
```

## Training Details

### Dataset

| Category | Samples | Percentage |
|----------|---------|------------|
| Visual Logic & Strategic Games | 66,854 | 36.7% |
| 2D Visual Reasoning | 51,899 | 28.5% |
| 3D Visual Reasoning | 39,610 | 21.7% |
| Scientific Reasoning | 24,021 | 13.2% |
| **Total** | **182,384** | **100%** |

### Architecture

- **Base**: Mixture-of-Transformer-Experts (MoT)
- **Encoders**: Dual (pixel-level + semantic-level)
- **Training Paradigm**: Next Group of Token Prediction

## Performance

### Evaluation Results

| Metric | Before Fine-tuning | After Fine-tuning | Improvement |
|--------|-------------------|-------------------|-------------|
| In-distribution Accuracy | 4.2% | 16.9% | **+12.7%** |
| VLM Benchmark (avg) | baseline | +13% | **+13%** |

### Capabilities

- ✅ Generates interleaved text-image reasoning chains
- ✅ Produces intermediate visual sketches/diagrams
- ✅ Handles multi-step logical reasoning
- ✅ Supports diverse visual reasoning tasks

## Limitations

- **Training Data**: Performance may vary on domains outside the 4 main categories
- **Image Generation**: Quality of visual reasoning images depends on base model capabilities
- **Computational Requirements**: Requires GPU for efficient inference
- **Language**: Primarily trained on English data

## Ethical Considerations

- Model outputs should be verified for accuracy in critical applications
- Visual reasoning may reflect biases present in training data
- Not intended for autonomous decision-making without human review

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

## References

- [Zebra-CoT Paper](https://arxiv.org/abs/2507.16746)
- [Zebra-CoT Dataset](https://huggingface.co/datasets/multimodal-reasoning-lab/Zebra-CoT)
- [Anole-Zebra-CoT Model](https://huggingface.co/multimodal-reasoning-lab/Anole-Zebra-CoT)
- [Bagel-Zebra-CoT Model](https://huggingface.co/multimodal-reasoning-lab/Bagel-Zebra-CoT)

## Model Card Contact

For questions or issues, please open an issue on the [GitHub repository](https://github.com/isshiki-dev/Sheikh-Freemium).
