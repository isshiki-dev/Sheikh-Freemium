# Model Architecture

This document describes the architecture and training methodology for models fine-tuned on Zebra-CoT.

## Mixture-of-Transformer-Experts (MoT)

The architecture adopts a **Mixture-of-Transformer-Experts (MoT)** design to maximize the model's capacity to learn from richly diverse multimodal information.

### Key Design Principles

1. **Capacity Maximization**: MoT enables the model to handle the diversity of visual reasoning tasks across scientific, 2D, 3D, and logic/game domains.

2. **Expert Specialization**: Different experts can specialize in different types of reasoning patterns (geometric, spatial, strategic, etc.).

## Dual Encoder System

Following the capacity maximization principle, the architecture utilizes **two separate encoders**:

| Encoder | Purpose | Features Captured |
|---------|---------|-------------------|
| **Pixel-Level Encoder** | Low-level visual processing | Edges, textures, fine details |
| **Semantic-Level Encoder** | High-level understanding | Objects, relationships, concepts |

```
┌─────────────────────────────────────────────────────────┐
│                      Input Image                         │
└─────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
┌─────────────────────────┐   ┌─────────────────────────┐
│   Pixel-Level Encoder   │   │  Semantic-Level Encoder │
│   (Fine visual details) │   │  (High-level concepts)  │
└─────────────────────────┘   └─────────────────────────┘
              │                           │
              └─────────────┬─────────────┘
                            ▼
┌─────────────────────────────────────────────────────────┐
│              Mixture-of-Transformer-Experts              │
│                        (MoT)                             │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│           Next Group of Token Prediction                 │
└─────────────────────────────────────────────────────────┘
```

## Next Group of Token Prediction (NGTP)

The training paradigm follows **Next Group of Token Prediction**, where the model predicts the next group of language or visual tokens as a compression target.

### Advantages

- **Interleaved Generation**: Naturally supports generating interleaved text-image reasoning traces
- **Efficient Compression**: Groups of tokens provide better information density
- **Flexible Modality**: Can predict either language tokens or visual tokens based on context

### Token Groups

```
Input:  [Question] [Problem Image]
          │
          ▼
Step 1: [THOUGHT 1] ────────────────► Language tokens
          │
          ▼  
Step 2: [REASONING IMAGE 1] ────────► Visual tokens
          │
          ▼
Step 3: [THOUGHT 2] ────────────────► Language tokens
          │
          ▼
Output: [FINAL ANSWER]
```

## Performance Results

### In-Distribution Test Accuracy

| Model | Before Fine-tuning | After Fine-tuning | Improvement |
|-------|-------------------|-------------------|-------------|
| Anole-7B | 4.2% | 16.9% | **+12.7%** |
| Bagel-7B | — | High-quality interleaved chains | Qualitative |

### Benchmark Improvements

Fine-tuning on Zebra-CoT yields up to **+13%** performance gain on standard VLM benchmarks:

- Enhanced visual reasoning capabilities
- Improved chain-of-thought generation
- Better intermediate step visualization

## Category-Specific Performance

| Category | Description | Key Reasoning Skills |
|----------|-------------|---------------------|
| Scientific | Geometry, Physics, Algorithms | Diagram construction, step-by-step derivation |
| 2D Visual | Visual search, Jigsaw | Pattern recognition, spatial arrangement |
| 3D Visual | Multi-hop inference, Embodied planning | Depth perception, navigation |
| Logic/Games | Chess, Visual logic | Strategic thinking, rule application |

## References

- [Zebra-CoT Paper](https://arxiv.org/abs/2507.16746)
- [Anole-Zebra-CoT Model](https://huggingface.co/multimodal-reasoning-lab/Anole-Zebra-CoT)
- [Bagel-Zebra-CoT Model](https://huggingface.co/multimodal-reasoning-lab/Bagel-Zebra-CoT)
