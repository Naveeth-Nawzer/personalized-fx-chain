"""Push the built dataset folder to a Hugging Face dataset repository.

Run ``build_dataset.py`` first. The token is read from ``HF_TOKEN`` in the
environment or in the ``.env`` file set by ``paths.env_file`` in config.yaml.

Usage::

    python push_to_hub.py                   # repo/visibility from config.yaml
    python push_to_hub.py --repo-id USER/name --public
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

from build_dataset import HERE, SUMMARY_FILE, load_config


def read_token(env_file: Path) -> str:
    """HF_TOKEN from the environment, else from a KEY=VALUE .env file (spaces/quotes tolerated)."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token and env_file.is_file():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip().removeprefix("export ").strip() == "HF_TOKEN":
                token = value.strip().strip("'\"")
    if not token:
        sys.exit(f"HF_TOKEN not found in the environment or in {env_file}")
    return token


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--repo-id", help="override huggingface.repo_id")
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument("--public", action="store_true", help="create the repo as public")
    visibility.add_argument("--private", action="store_true", help="create the repo as private")
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    hf = config["huggingface"]
    repo_id = args.repo_id or hf["repo_id"]
    private = False if args.public else True if args.private else bool(hf.get("private", True))
    folder = config["paths"]["output_dir"]
    if not (folder / "metadata" / SUMMARY_FILE).is_file():
        sys.exit(f"No built dataset at {folder}. Run build_dataset.py first.")

    api = HfApi(token=read_token(config["paths"]["env_file"]))
    user = api.whoami()["name"]
    print(f"Authenticated as {user}")

    url = api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    print(f"Repository: {url} ({'private' if private else 'public'} on creation)")

    commit = api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=hf.get("commit_message", "Upload dataset"),
        # Remove files from earlier uploads that are no longer in the build (e.g. a dropped category).
        delete_patterns=["data/**", "metadata/**"],
    )
    print(f"Uploaded: {commit}")
    print(f"View at:  https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
