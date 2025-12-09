# Zebra-CoT Dataset

A diverse large-scale dataset for Visual Chain of Thought (Visual CoT) reasoning, containing logically coherent interleaved text-image reasoning traces.

## Overview

This dataset supports training multimodal models to use visual aids (diagrams, sketches) when solving complex problems.

## Categories

### 1. Scientific (`samples/scientific/`)
- **Geometry**: Geometric proofs and spatial reasoning
- **Physics**: Physical simulations and force diagrams
- **Algorithms**: Algorithm visualization and step-by-step execution

### 2. 2D Visual Reasoning (`samples/visual_2d/`)
- **Visual Search**: Finding patterns and objects
- **Jigsaw Puzzles**: Spatial arrangement tasks

### 3. 3D Reasoning (`samples/visual_3d/`)
- **3D Multi-hop Inference**: Complex spatial reasoning
- **Embodied Planning**: Robot navigation and manipulation

### 4. Logic & Games (`samples/logic_games/`)
- **Chess**: Strategic game analysis
- **Visual Logic**: Pattern-based logical problems

## Sample Format

Each sample follows the schema defined in `schemas/sample_schema.json`:

```json
{
  "id": "unique_sample_id",
  "category": "scientific|visual_2d|visual_3d|logic_games",
  "subcategory": "geometry|physics|...",
  "question": "Problem statement",
  "reasoning_steps": [
    {
      "step": 1,
      "type": "text|image|interleaved",
      "content": "Reasoning content",
      "image_path": "optional/path/to/image.png"
    }
  ],
  "answer": "Final answer",
  "metadata": {}
}
```

## Usage

```bash
# Validate samples
python scripts/validate.py --input samples/
```

## License

See repository LICENSE for details.
