"""Step 4 — push the generated Phase 1 dataset to the Hugging Face Hub.

The token is read from ``HF_TOKEN`` in the environment or the ``.env`` file
set by ``paths.env_file`` in config.yaml.

Usage::

    .venv\\Scripts\\python push_phase1.py
    .venv\\Scripts\\python push_phase1.py --repo-id APINAJA/another-name
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi

from common import DEFAULT_CONFIG, SUMMARY_FILE, load_config, read_token


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--repo-id", help="override huggingface.repo_id")
    parser.add_argument("--message", help="commit message (default: huggingface.commit_message)")
    args = parser.parse_args()

    config = load_config(args.config)
    hf = config["huggingface"]
    repo_id = args.repo_id or hf["repo_id"]
    folder = config["paths"]["output_dir"]
    if not (folder / "metadata" / SUMMARY_FILE).is_file():
        sys.exit(f"No generated dataset at {folder}. Run generate_phase1.py first.")
    if not any((folder / "data").glob("*.parquet")):
        sys.exit("No Parquet shards found. Run export_parquet.py first (needed for Audio columns in the viewer).")

    api = HfApi(token=read_token(config["paths"]["env_file"]))
    print(f"Authenticated as {api.whoami()['name']}")
    url = api.create_repo(repo_id, repo_type="dataset", private=bool(hf.get("private", True)), exist_ok=True)
    print(f"Repository: {url}")

    commit = api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=args.message or hf.get("commit_message", "Upload Phase 1 dataset"),
        # Remove files left over from an earlier upload that are not in this build.
        delete_patterns=["data/**", "metadata/**"],
    )
    print(f"Uploaded: {commit}")
    print(f"View at:  https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
