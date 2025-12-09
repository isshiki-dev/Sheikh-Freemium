#!/usr/bin/env python3
"""Validation script for Zebra-CoT dataset samples."""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False


def load_schema(schema_path: Path) -> dict:
    """Load JSON schema from file."""
    with open(schema_path, 'r') as f:
        return json.load(f)


def validate_sample(sample: dict, schema: dict) -> List[str]:
    """Validate a single sample against the schema."""
    errors = []
    
    if HAS_JSONSCHEMA:
        validator = jsonschema.Draft7Validator(schema)
        for error in validator.iter_errors(sample):
            errors.append(f"{error.json_path}: {error.message}")
    else:
        # Basic validation without jsonschema
        required = ['id', 'category', 'subcategory', 'question', 'reasoning_steps', 'answer']
        for field in required:
            if field not in sample:
                errors.append(f"Missing required field: {field}")
    
    return errors


def check_logical_coherence(sample: dict) -> List[str]:
    """Check if reasoning steps are logically coherent."""
    warnings = []
    steps = sample.get('reasoning_steps', [])
    
    # Check step numbering
    for i, step in enumerate(steps, 1):
        if step.get('step') != i:
            warnings.append(f"Step numbering mismatch: expected {i}, got {step.get('step')}")
    
    # Check for empty content
    for step in steps:
        if not step.get('content', '').strip():
            warnings.append(f"Step {step.get('step')} has empty content")
    
    return warnings


def validate_directory(input_dir: Path, schema: dict, strict: bool = False) -> Tuple[int, int]:
    """Validate all JSON files in a directory."""
    valid_count = 0
    error_count = 0
    
    for json_file in input_dir.rglob('*.json'):
        if json_file.name == 'sample_schema.json':
            continue
            
        try:
            with open(json_file, 'r') as f:
                sample = json.load(f)
            
            errors = validate_sample(sample, schema)
            warnings = check_logical_coherence(sample)
            
            if errors:
                print(f"❌ {json_file}: {len(errors)} error(s)")
                for err in errors:
                    print(f"   - {err}")
                error_count += 1
            elif warnings and strict:
                print(f"⚠️  {json_file}: {len(warnings)} warning(s)")
                for warn in warnings:
                    print(f"   - {warn}")
                error_count += 1
            else:
                print(f"✓ {json_file}")
                valid_count += 1
                
        except json.JSONDecodeError as e:
            print(f"❌ {json_file}: Invalid JSON - {e}")
            error_count += 1
        except Exception as e:
            print(f"❌ {json_file}: {e}")
            error_count += 1
    
    return valid_count, error_count


def main():
    parser = argparse.ArgumentParser(description='Validate Zebra-CoT dataset samples')
    parser.add_argument('--input', '-i', type=Path, required=True,
                        help='Input directory containing samples')
    parser.add_argument('--schema', '-s', type=Path, 
                        default=Path(__file__).parent.parent / 'schemas' / 'sample_schema.json',
                        help='Path to JSON schema')
    parser.add_argument('--strict', action='store_true',
                        help='Treat warnings as errors')
    
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"Error: Input directory not found: {args.input}")
        sys.exit(1)
    
    schema = {}
    if args.schema.exists():
        schema = load_schema(args.schema)
        print(f"Using schema: {args.schema}")
    else:
        print("Warning: Schema not found, using basic validation")
    
    print(f"Validating samples in: {args.input}\n")
    
    valid, errors = validate_directory(args.input, schema, args.strict)
    
    print(f"\nResults: {valid} valid, {errors} error(s)")
    sys.exit(0 if errors == 0 else 1)


if __name__ == '__main__':
    main()
