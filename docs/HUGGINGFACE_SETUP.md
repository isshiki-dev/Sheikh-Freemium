# Hugging Face Integration Setup

This guide covers different methods to sync this repository with Hugging Face Hub.

## Repository

- **HF Repo**: [shk-bd/Sheikh-Freemium](https://huggingface.co/shk-bd/Sheikh-Freemium)
- **Type**: Model (can also be used as dataset)

## Method 1: Python SDK (Recommended)

```python
from huggingface_hub import login, upload_folder

# Login with your Hugging Face credentials
login()

# Push your model files
upload_folder(
    folder_path=".",
    repo_id="shk-bd/Sheikh-Freemium",
    repo_type="model"
)
```

Or use the provided script:

```bash
pip install huggingface_hub
python scripts/upload_to_hf.py
```

## Method 2: Git-Xet (Large Files)

For repositories with large files (>10GB), use git-xet:

```bash
# Install git-xet
# See: https://hf.co/docs/hub/git-xet
git xet install

# Add HuggingFace remote
git remote add hf git@hf.co:shk-bd/Sheikh-Freemium

# Push (requires SSH key in HF settings)
git push -u hf main
```

### SSH Key Setup

1. Generate SSH key: `ssh-keygen -t ed25519`
2. Add to HuggingFace: https://huggingface.co/settings/keys

## Method 3: Hugging Face CLI

```bash
# Install CLI
curl -LsSf https://hf.co/cli/install.sh | bash

# Login
hf auth login

# Upload
hf upload shk-bd/Sheikh-Freemium .
```

Or use the provided script:

```bash
./scripts/sync_hf.sh model  # or 'dataset'
```

## Method 4: GitHub Actions (Automated)

The repository includes automated sync via GitHub Actions.

### Setup

1. Add `HF_TOKEN` secret to GitHub repository:
   - Go to **Settings** → **Secrets and variables** → **Actions**
   - Add secret named `HF_TOKEN` with your HuggingFace token

2. Trigger sync:
   - **Automatic**: Push to `main` branch (dataset/ or scripts/ changes)
   - **Manual**: Go to Actions → "Sync to Hugging Face" → Run workflow

### Get Your Token

1. Go to https://huggingface.co/settings/tokens
2. Create new token with `write` access
3. Copy and add as GitHub secret

## Verification

After upload, verify at:
- https://huggingface.co/shk-bd/Sheikh-Freemium

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Repository not found" | Ensure repo exists on HF or create it first |
| "Permission denied" | Check token has write access |
| Large file errors | Use git-xet or Git LFS |
| SSH errors | Verify SSH key in HF settings |
