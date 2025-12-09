#!/bin/bash
# Sync repository to Hugging Face Hub
# Usage: ./scripts/sync_hf.sh [model|dataset]

set -e

REPO_ID="shk-bd/Sheikh-Freemium"
REPO_TYPE="${1:-model}"

echo "=== Hugging Face Sync ==="
echo "Repo: $REPO_ID"
echo "Type: $REPO_TYPE"
echo ""

# Check if HF CLI is installed
if ! command -v hf &> /dev/null; then
    echo "Installing Hugging Face CLI..."
    curl -LsSf https://hf.co/cli/install.sh | bash
    export PATH="$HOME/.local/bin:$PATH"
fi

# Login if needed
if [ -n "$HF_TOKEN" ]; then
    echo "Using HF_TOKEN from environment"
    hf auth login --token "$HF_TOKEN"
else
    echo "No HF_TOKEN found. Running interactive login..."
    hf auth login
fi

# Upload
echo ""
echo "Uploading to Hugging Face..."
hf upload "$REPO_ID" . --repo-type "$REPO_TYPE"

echo ""
echo "Done! View at: https://huggingface.co/$REPO_ID"
