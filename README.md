# Sheikh-Freemium

[![Validate Dataset](https://github.com/isshiki-dev/Sheikh-Freemium/actions/workflows/validate-dataset.yml/badge.svg)](https://github.com/isshiki-dev/Sheikh-Freemium/actions/workflows/validate-dataset.yml)
[![CI](https://github.com/isshiki-dev/Sheikh-Freemium/actions/workflows/ci.yml/badge.svg)](https://github.com/isshiki-dev/Sheikh-Freemium/actions/workflows/ci.yml)
[![Sync to Hugging Face](https://github.com/isshiki-dev/Sheikh-Freemium/actions/workflows/sync-huggingface.yml/badge.svg)](https://github.com/isshiki-dev/Sheikh-Freemium/actions/workflows/sync-huggingface.yml)

## Zebra-CoT Visual Reasoning Dataset

A repository for managing and developing Visual Chain of Thought (Visual CoT) datasets, based on the [Zebra-CoT](https://huggingface.co/datasets/multimodal-reasoning-lab/Zebra-CoT) framework.

### Quick Links

| Resource | Link |
|----------|------|
| 📂 Dataset | [dataset/](./dataset/) |
| 📖 Architecture | [dataset/docs/ARCHITECTURE.md](./dataset/docs/ARCHITECTURE.md) |
| 🔗 Sources | [dataset/SOURCES.md](./dataset/SOURCES.md) |
| 🤗 HuggingFace | [Zebra-CoT Dataset](https://huggingface.co/datasets/multimodal-reasoning-lab/Zebra-CoT) |

### Features

- **182K+ samples** across 4 reasoning categories
- **Interleaved text-image** reasoning traces
- **MoT architecture** support (Mixture-of-Transformer-Experts)
- **Automated validation** via GitHub Actions
- **HuggingFace sync** for dataset updates

### Categories

| Category | Samples | Description |
|----------|---------|-------------|
| Visual Logic & Games | 66,854 | Chess, strategic games, visual logic |
| 2D Visual Reasoning | 51,899 | Visual search, jigsaw puzzles |
| 3D Visual Reasoning | 39,610 | Multi-hop inference, embodied planning |
| Scientific Reasoning | 24,021 | Geometry, physics, algorithms |

### Getting Started

```bash
# Clone the repository
git clone https://github.com/isshiki-dev/Sheikh-Freemium.git
cd Sheikh-Freemium

# Validate local samples
python dataset/scripts/validate.py --input dataset/samples/
```

### License

Dataset content: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
