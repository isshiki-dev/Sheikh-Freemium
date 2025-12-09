# Dataset Sources & References

This document tracks official and related visual reasoning datasets.

## Primary Dataset

### Zebra-CoT
- **Source**: [Hugging Face](https://huggingface.co/datasets/multimodal-reasoning-lab/Zebra-CoT)
- **Paper**: [arXiv:2507.16746](https://arxiv.org/abs/2507.16746)
- **Size**: 182,384 samples (58.9 GB)
- **License**: CC BY-NC 4.0
- **Modalities**: Image, Text
- **Use Case**: Training multimodal models for Visual Chain of Thought reasoning

## Related Datasets

### Visual-CoT (NeurIPS'24 Spotlight)
- **Source**: [Hugging Face](https://huggingface.co/datasets/deepcs233/Visual-CoT)
- **GitHub**: [deepcs233/Visual-CoT](https://github.com/deepcs233/Visual-CoT)
- **Size**: 438K question-answer pairs
- **Description**: Multi-turn processing pipeline for MLLMs with intermediate bounding boxes

### LLaVA-CoT-100k (ICCV 2025)
- **Source**: [Hugging Face](https://huggingface.co/datasets/Xkev/LLaVA-CoT-100k)
- **GitHub**: [PKU-YuanGroup/LLaVA-CoT](https://github.com/PKU-YuanGroup/LLaVA-CoT)
- **Description**: Visual language model with spontaneous systematic reasoning

### MM-CoT (Amazon Science)
- **GitHub**: [amazon-science/mm-cot](https://github.com/amazon-science/mm-cot)
- **Description**: Multimodal CoT with decoupled training framework for rationale generation

### MME-CoT Benchmark
- **GitHub**: [CaraJ7/MME-CoT](https://github.com/CaraJ7/MME-CoT)
- **Description**: Benchmark for evaluating CoT reasoning across math, science, OCR, logic, space-time

## Pre-trained Models

| Model | Base | Link |
|-------|------|------|
| Anole-Zebra-CoT | Anole-7B | [HuggingFace](https://huggingface.co/multimodal-reasoning-lab/Anole-Zebra-CoT) |
| Bagel-Zebra-CoT | Bagel-7B | [HuggingFace](https://huggingface.co/multimodal-reasoning-lab/Bagel-Zebra-CoT) |

## Adoption Status

- [x] Zebra-CoT integrated
- [ ] Visual-CoT samples pending
- [ ] Custom samples in development
