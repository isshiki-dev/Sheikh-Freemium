# MLOps Pipeline

Continuous training and deployment pipeline for Visual CoT models.

## Pipeline Architecture

```
Code / Data / Prompts
        ↓
 GitHub Commit / PR
        ↓
 GitHub Actions
        ↓
 Auto-training / Fine-tuning
        ↓
 Weight validation
        ↓
 Adopted weights committed or released
        ↓
 Model version auto-continues learning
```

## Workflow Triggers

| Trigger | Action |
|---------|--------|
| Push to `dataset/samples/` | Validate data, queue training |
| Push to `prompts/` | Update prompt templates, retrain |
| Push to `training/` | Run training pipeline |
| Manual dispatch | Full training run |
| Schedule (weekly) | Continuous learning iteration |

## Components

- `pipeline.yaml` - Main pipeline configuration
- `training/` - Training scripts and configs
- `validation/` - Weight validation scripts
- `prompts/` - Prompt templates for training

## Usage

### Trigger Training Manually

```bash
gh workflow run train.yml
```

### Monitor Training

```bash
gh run list --workflow=train.yml
```
