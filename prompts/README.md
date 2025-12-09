# Prompt Templates

Prompt templates for Visual CoT training and inference.

## Structure

- `training/` - Prompts used during fine-tuning
- `inference/` - Prompts for model inference
- `evaluation/` - Prompts for benchmarking

## Format

Prompts use Jinja2 templating with the following variables:

- `{{ question }}` - Problem statement
- `{{ category }}` - Task category
- `{{ image }}` - Image placeholder
- `{{ steps }}` - Reasoning steps
