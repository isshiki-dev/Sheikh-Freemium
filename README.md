# Sheikh-Freemium

[![Train Model](https://github.com/isshiki-dev/Sheikh-Freemium/actions/workflows/train.yml/badge.svg)](https://github.com/isshiki-dev/Sheikh-Freemium/actions/workflows/train.yml)
[![Validate](https://github.com/isshiki-dev/Sheikh-Freemium/actions/workflows/validate.yml/badge.svg)](https://github.com/isshiki-dev/Sheikh-Freemium/actions/workflows/validate.yml)
[![Continuous Learning](https://github.com/isshiki-dev/Sheikh-Freemium/actions/workflows/continuous-learning.yml/badge.svg)](https://github.com/isshiki-dev/Sheikh-Freemium/actions/workflows/continuous-learning.yml)

## 🧠 What is Sheikh-Freemium?

**Sheikh-Freemium is not just a model. It's a self-updating learning system.**

| Traditional ML | Sheikh-Freemium |
|----------------|------------------|
| Manual training in notebooks | Automated via GitHub Actions |
| Ad-hoc weight management | Versioned, validated, promoted |
| Research lab chaos | DevOps discipline |
| Static models | Continuously learning |

### Core Principles

- 💻 **GitHub is the source of truth** — Data, prompts, configs live in version control
- ⚙️ **GitHub Actions is trainer + orchestrator** — No manual intervention needed
- 📦 **Weights are continuously adopted** — Validated, versioned, promoted automatically
- 🔄 **Training behaves like DevOps** — CI/CD for machine learning

## 🚀 How It Works

```
┌───────────────────────────┐
│  Code / Data / Prompts    │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│   GitHub Commit / PR      │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│     GitHub Actions        │
│  (Validate → Train)       │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│  Auto-training / Fine-tune│
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│    Weight Validation      │
│  (Accuracy ≥ 15%)         │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│  Weights Committed/Released│
│  (Versioned + HuggingFace)│
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│  Model Continues Learning │
│  (Weekly auto-iteration)  │
└───────────────────────────┘
```

## 📁 Repository Structure

```
Sheikh-Freemium/
├── dataset/                 # 📊 Training data (source of truth)
│   ├── samples/             # Categorized samples
│   ├── schemas/             # Data validation schemas
│   └── scripts/             # Data processing utilities
├── prompts/                 # 📝 Prompt templates
│   ├── training/            # Fine-tuning prompts
│   └── inference/           # Production prompts
├── mlops/                   # ⚙️ ML Operations
│   ├── pipeline.yaml        # Pipeline configuration
│   ├── training/            # Training scripts & config
│   └── validation/          # Weight validation
├── space/                   # 🌐 HuggingFace Space demo
└── .github/workflows/       # 🚀 Automation
    ├── train.yml            # Auto-training pipeline
    ├── validate.yml         # PR validation
    ├── release.yml          # Model releases
    └── continuous-learning.yml  # Weekly iterations
```

## 🎯 Triggers

| You Push... | System Does... |
|-------------|----------------|
| New samples to `dataset/samples/` | Validate → Queue training |
| Updated prompts to `prompts/` | Retrain with new templates |
| Config changes to `mlops/` | Full training run |
| Nothing (Sunday midnight) | Continuous learning iteration |

## 📊 Visual CoT Dataset

Based on [Zebra-CoT](https://huggingface.co/datasets/multimodal-reasoning-lab/Zebra-CoT):

| Category | Samples | Description |
|----------|---------|-------------|
| Visual Logic & Games | 66,854 | Chess, strategic games |
| 2D Visual Reasoning | 51,899 | Visual search, puzzles |
| 3D Visual Reasoning | 39,610 | Spatial reasoning |
| Scientific Reasoning | 24,021 | Geometry, physics |
| **Total** | **182,384** | |

## 🚀 Quick Start

### 1. Add Training Data

```bash
# Add new sample
cp my_sample.json dataset/samples/scientific/
git add . && git commit -m "Add new geometry sample"
git push  # → Triggers validation + training
```

### 2. Update Prompts

```bash
# Edit prompt template
vim prompts/training/visual_cot.txt
git add . && git commit -m "Improve reasoning prompt"
git push  # → Triggers retraining
```

### 3. Manual Release

```bash
# Trigger release workflow
gh workflow run release.yml -f version=v1.0.0 -f release_type=both
```

### 4. Monitor Training

```bash
# View training runs
gh run list --workflow=train.yml

# Watch live logs
gh run watch
```

## 🔗 Links

| Resource | URL |
|----------|-----|
| 🤗 Dataset | [shk-bd/Sheikh-Freemium](https://huggingface.co/datasets/shk-bd/Sheikh-Freemium) |
| 🤗 Model | [shk-bd/Sheikh-Freemium](https://huggingface.co/shk-bd/Sheikh-Freemium) |
| 🌐 Demo | [HuggingFace Space](https://huggingface.co/spaces/shk-bd/Sheikh-Freemium) |
| 📄 Paper | [arXiv:2507.16746](https://arxiv.org/abs/2507.16746) |

## 📈 Performance

| Metric | Before | After | Gain |
|--------|--------|-------|------|
| In-distribution Accuracy | 4.2% | 16.9% | **+12.7%** |
| VLM Benchmarks | baseline | — | **up to +13%** |

## 📜 License

Dataset & Model: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

---

<div align="center">

**Sheikh-Freemium: Where ML meets DevOps**

*Training as reliable as CI/CD*

</div>
