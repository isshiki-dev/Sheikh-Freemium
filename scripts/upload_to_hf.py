#!/usr/bin/env python3
"""Upload dataset/model to Hugging Face Hub."""

import os
import argparse
from huggingface_hub import login, upload_folder, HfApi


def main():
    parser = argparse.ArgumentParser(description="Upload to Hugging Face Hub")
    parser.add_argument(
        "--repo-id",
        default="shk-bd/Sheikh-Freemium",
        help="HuggingFace repo ID (default: shk-bd/Sheikh-Freemium)"
    )
    parser.add_argument(
        "--repo-type",
        choices=["model", "dataset", "space"],
        default="model",
        help="Repository type (default: model)"
    )
    parser.add_argument(
        "--folder",
        default=".",
        help="Folder to upload (default: current directory)"
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace token (or set HF_TOKEN env var)"
    )
    args = parser.parse_args()

    # Login
    token = args.token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    else:
        print("No token provided. Using cached credentials or interactive login.")
        login()

    # Verify connection
    api = HfApi()
    user = api.whoami()
    print(f"Logged in as: {user['name']}")

    # Upload
    print(f"Uploading {args.folder} to {args.repo_id} ({args.repo_type})...")
    upload_folder(
        folder_path=args.folder,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        ignore_patterns=["*.git*", "__pycache__", "*.pyc", ".env"]
    )
    print("Upload complete!")
    print(f"View at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
