# Zebra-CoT Dataset

[![Dataset on HuggingFace](https://img.shields.io/badge/🤗%20Dataset-Zebra--CoT-yellow)](https://huggingface.co/datasets/multimodal-reasoning-lab/Zebra-CoT)
[![Paper](https://img.shields.io/badge/arXiv-2507.16746-b31b1b)](https://arxiv.org/abs/2507.16746)
[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey)](https://creativecommons.org/licenses/by-nc/4.0/)

A diverse large-scale dataset for Visual Chain of Thought (Visual CoT) reasoning, containing logically coherent interleaved text-image reasoning traces.

## Official Resources

| Resource | Link |
|----------|------|
| 🤗 Dataset | [multimodal-reasoning-lab/Zebra-CoT](https://huggingface.co/datasets/multimodal-reasoning-lab/Zebra-CoT) |
| 📄 Paper | [arXiv:2507.16746](https://arxiv.org/abs/2507.16746) |
| 💻 Code | [multimodal-reasoning-lab/Bagel-Zebra-CoT](https://github.com/multimodal-reasoning-lab/Bagel-Zebra-CoT) |
| 🔮 Anole Model | [Anole-Zebra-CoT](https://huggingface.co/multimodal-reasoning-lab/Anole-Zebra-CoT) |
| 🔮 Bagel Model | [Bagel-Zebra-CoT](https://huggingface.co/multimodal-reasoning-lab/Bagel-Zebra-CoT) |

## Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Samples** | 182,384 |
| **Download Size** | 58.9 GB |
| **Format** | Parquet |
| **License** | CC BY-NC 4.0 |

## Categories

| Category | Samples | Percentage |
|----------|---------|------------|
| Visual Logic & Strategic Games | 66,854 | 36.7% |
| 2D Visual Reasoning | 51,899 | 28.5% |
| 3D Visual Reasoning | 39,610 | 21.7% |
| Scientific Reasoning | 24,021 | 13.2% |

### Category Details

#### 1. Scientific Reasoning (`samples/scientific/`)
- **Geometry**: Geometric proofs and spatial reasoning
- **Physics**: Physical simulations and force diagrams
- **Algorithms**: Algorithm visualization and step-by-step execution

#### 2. 2D Visual Reasoning (`samples/visual_2d/`)
- **Visual Search**: Finding patterns and objects
- **Jigsaw Puzzles**: Spatial arrangement tasks

#### 3. 3D Reasoning (`samples/visual_3d/`)
- **3D Multi-hop Inference**: Complex spatial reasoning
- **Embodied Planning**: Robot navigation and manipulation

#### 4. Visual Logic & Games (`samples/logic_games/`)
- **Chess**: Strategic game analysis
- **Visual Logic**: Pattern-based logical problems

## Sample Format

Each sample contains:

| Field | Description |
|-------|-------------|
| `problem` | Textual description of the question |
| `problem_image` | Zero or more images accompanying the problem |
| `reasoning_image` | Visual aids supporting intermediate reasoning steps |
| `text_reasoning_trace` | Sequence of thoughts with image placeholders |
| `final_answer` | Solution to the problem |

## Quick Start

### Load from Hugging Face

```python
from datasets import load_dataset

# Load the full dataset
dataset = load_dataset("multimodal-reasoning-lab/Zebra-CoT")

# Access a sample
sample = dataset['train'][0]
print(sample['problem'])
```

### Local Validation

```bash
# Validate local samples
python scripts/validate.py --input samples/
```

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

## License

This dataset is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
