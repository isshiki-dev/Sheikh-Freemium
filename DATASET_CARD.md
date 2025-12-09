---
pretty_name: Sheikh-Freemium Visual CoT Dataset
license: cc-by-nc-4.0
language:
  - en
size_categories:
  - 100K<n<1M
task_categories:
  - visual-question-answering
  - image-to-text
  - text-generation
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
  - image
  - text
  - datasets
  - pandas
  - arxiv:2507.16746
source_datasets:
  - multimodal-reasoning-lab/Zebra-CoT
configs:
  - config_name: default
    data_files:
      - split: train
        path: "dataset/samples/**/*.json"
---

# Sheikh-Freemium: Visual Chain of Thought Dataset

<div align="center">

[![Based on Zebra-CoT](https://img.shields.io/badge/Based%20on-Zebra--CoT-blue)](https://huggingface.co/datasets/multimodal-reasoning-lab/Zebra-CoT)
[![Paper](https://img.shields.io/badge/arXiv-2507.16746-b31b1b)](https://arxiv.org/abs/2507.16746)
[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey)](https://creativecommons.org/licenses/by-nc/4.0/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/isshiki-dev/Sheikh-Freemium)

</div>

## Dataset Description

- **Homepage**: [GitHub Repository](https://github.com/isshiki-dev/Sheikh-Freemium)
- **Repository**: [Hugging Face](https://huggingface.co/datasets/shk-bd/Sheikh-Freemium)
- **Paper**: [Zebra-CoT: A Dataset for Interleaved Vision Language Reasoning](https://arxiv.org/abs/2507.16746)
- **Point of Contact**: [GitHub Issues](https://github.com/isshiki-dev/Sheikh-Freemium/issues)

### Dataset Summary

Sheikh-Freemium is a Visual Chain of Thought (Visual CoT) dataset for training multimodal models to generate interleaved text-image reasoning traces. Based on the [Zebra-CoT](https://huggingface.co/datasets/multimodal-reasoning-lab/Zebra-CoT) framework, it enables models to use visual aids (diagrams, sketches) when solving complex problems.

| Statistic | Value |
|-----------|-------|
| **Total Samples** | 182,384 |
| **Download Size** | 58.9 GB |
| **Format** | Parquet / JSON |
| **Modalities** | Text + Image |
| **License** | CC BY-NC 4.0 |

### Supported Tasks

- `visual-question-answering`: Answering questions requiring visual reasoning
- `image-to-text`: Generating reasoning traces from images
- `text-generation`: Producing step-by-step solutions with visual aids

### Languages

English (`en`)

## Dataset Structure

### Data Instances

A typical data instance:

```json
{
  "id": "geo_triangle_001",
  "category": "scientific",
  "subcategory": "geometry",
  "question": "In triangle ABC, angle A = 60°, angle B = 45°. If side AB = 10 units, find the length of side BC.",
  "reasoning_steps": [
    {
      "step": 1,
      "type": "text",
      "content": "First, find angle C using the triangle angle sum property: C = 180° - 60° - 45° = 75°",
      "intermediate_result": "angle C = 75°"
    },
    {
      "step": 2,
      "type": "interleaved",
      "content": "Draw the triangle with labeled angles and the known side AB.",
      "image_path": "images/geo_triangle_001_step2.png",
      "intermediate_result": "Visual representation created"
    }
  ],
  "answer": "BC ≈ 8.97 units",
  "metadata": {
    "difficulty": "medium",
    "source": "synthetic",
    "tags": ["law-of-sines", "triangle", "trigonometry"]
  }
}
```

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique sample identifier |
| `category` | string | Task category (scientific, visual_2d, visual_3d, logic_games) |
| `subcategory` | string | Specific task type |
| `question` | string | Problem statement |
| `input_images` | list | Problem images (optional) |
| `reasoning_steps` | list | Interleaved text-image reasoning chain |
| `reasoning_steps.step` | int | Step number |
| `reasoning_steps.type` | string | "text", "image", or "interleaved" |
| `reasoning_steps.content` | string | Reasoning explanation |
| `reasoning_steps.image_path` | string | Path to generated visual (optional) |
| `reasoning_steps.intermediate_result` | string | Result from this step |
| `answer` | string | Final solution |
| `metadata` | object | Additional info (difficulty, source, tags) |

### Data Splits

| Split | Samples | Description |
|-------|---------|-------------|
| train | 160,485 | Training set |
| test | 21,899 | Held-out evaluation set |

### Categories

| Category | Samples | % | Subcategories |
|----------|---------|---|---------------|
| Visual Logic & Games | 66,854 | 36.7% | chess, visual_logic, strategic_games |
| 2D Visual Reasoning | 51,899 | 28.5% | visual_search, jigsaw_puzzles |
| 3D Visual Reasoning | 39,610 | 21.7% | multi_hop_inference, embodied_planning |
| Scientific Reasoning | 24,021 | 13.2% | geometry, physics, algorithms |

## Dataset Creation

### Curation Rationale

Humans often use visual aids (diagrams, sketches) when solving complex problems. Training multimodal models to do the same (Visual CoT) is challenging due to:
1. Poor off-the-shelf visual CoT performance, hindering reinforcement learning
2. Lack of high-quality visual CoT training data

This dataset addresses both challenges with 182K logically coherent interleaved text-image reasoning traces.

### Source Data

Based on [Zebra-CoT](https://huggingface.co/datasets/multimodal-reasoning-lab/Zebra-CoT), containing samples from:
- Scientific questions (geometry, physics, algorithms)
- 2D visual reasoning tasks (visual search, jigsaw)
- 3D reasoning tasks (multi-hop inference, embodied planning)
- Visual logic problems and strategic games (chess)

### Annotations

Each sample includes:
- Human-verified problem statements
- Step-by-step reasoning traces
- Generated visual aids for intermediate steps
- Final answers

## Considerations for Using the Data

### Social Impact

This dataset aims to improve AI visual reasoning capabilities, enabling more transparent and interpretable problem-solving.

### Discussion of Biases

- Primarily English language content
- Focus on structured reasoning domains (math, games, spatial)
- May not generalize to all visual reasoning scenarios

### Other Known Limitations

- Requires substantial compute for full training
- Image generation quality depends on base model
- Performance may vary outside the 4 main categories

## Usage

### Loading with Datasets Library

```python
from datasets import load_dataset

# Load full dataset
dataset = load_dataset("shk-bd/Sheikh-Freemium")

# Load specific split
train_data = load_dataset("shk-bd/Sheikh-Freemium", split="train")

# Filter by category
scientific = dataset['train'].filter(lambda x: x['category'] == 'scientific')
```

### Loading with Pandas

```python
import pandas as pd
from huggingface_hub import hf_hub_download

# Download and load
path = hf_hub_download(
    repo_id="shk-bd/Sheikh-Freemium",
    filename="dataset/samples/scientific/geometry_example.json",
    repo_type="dataset"
)
df = pd.read_json(path)
```

## Additional Information

### Licensing Information

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) - Non-commercial use only. Commercial licensing available upon request.

### Citation Information

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

### Contributions

Contributions welcome! Please open an issue or PR on the [GitHub repository](https://github.com/isshiki-dev/Sheikh-Freemium).
