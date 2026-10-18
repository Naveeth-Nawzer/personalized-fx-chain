"""Build and upload a dataset batch by batch: download -> generate -> check -> upload -> delete.

Only one batch of audio is on disk at a time, so the full effect-order dataset
(~37 GB of WAV + Parquet) can be built with a few GB free. For each batch of
``batch.songs_per_batch`` songs:

  1. download the batch's source stems from ``source.repo_id`` -> ``paths.batch_source_dir``
  2. render the dry clips + every chain -> ``<output_dir>/data/<split>/{dry,wet}/``
  3. delete the downloaded source stems
  4. check the metadata of all batches so far and the audio of this batch (incl. re-rendering)
  5. write this batch's Parquet shards (``data/<split>-bNN-*.parquet``) and the combined
     metadata CSVs, summary and dataset card
  6. upload to ``huggingface.repo_id`` (one commit per batch; earlier batches stay on the Hub)
  7. delete this batch's audio and Parquet locally (metadata stays)

Progress is kept in ``<output_dir>/batches/bNN.json``. Re-running the same command skips
uploaded batches and finishes an interrupted one. Every sample is seeded on its own,
so the result is identical to generating everything in one run.

Usage::

    .venv\\Scripts\\python run_batches.py --config config_order.yaml
    .venv\\Scripts\\python run_batches.py --config config_order.yaml --max-batches 1
    .venv\\Scripts\\python run_batches.py --config config_order.yaml --max-batches 1 --no-upload
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from common import DEFAULT_CONFIG, load_config, read_token
from export_parquet import build_schema, export_split, verify_split
from generate_phase1 import chain_mode, load_sources, process_sources, sample_columns, write_metadata
from validate_phase1 import check_audio, check_metadata, load_rows

BATCH_DIR = "batches"
RESULT_KEYS = ("records", "source_rows", "skipped", "failures")


def config_fingerprint(config: dict[str, Any]) -> str:
    """Hash of every setting that changes the generated data (all batches must share it)."""
    relevant = {k: v for k, v in config.items() if k not in ("paths", "huggingface", "batch")}
    relevant["songs_per_batch"] = config["batch"]["songs_per_batch"]
    return hashlib.sha256(json.dumps(relevant, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def plan_batches(sources: list[dict[str, Any]], songs_per_batch: int) -> list[list[dict[str, Any]]]:
    """Consecutive groups of songs (source_id order); all stems of a song stay in one batch."""
    song_ids = sorted({int(s["source_id"]) for s in sources})
    groups = [set(song_ids[i : i + songs_per_batch]) for i in range(0, len(song_ids), songs_per_batch)]
    return [[s for s in sources if int(s["source_id"]) in group] for group in groups]


def state_path(output_dir: Path, batch: int) -> Path:
    return output_dir / BATCH_DIR / f"b{batch:02d}.json"


def load_states(output_dir: Path) -> dict[int, dict[str, Any]]:
    return {int(p.stem[1:]): json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((output_dir / BATCH_DIR).glob("b*.json"))}


def save_state(output_dir: Path, state: dict[str, Any]) -> None:
    path = state_path(output_dir, state["batch"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def audio_present(output_dir: Path, state: dict[str, Any]) -> bool:
    files = {r["dry_audio"] for r in state["records"]} | {r["wet_audio"] for r in state["records"]}
    return all((output_dir / f).is_file() for f in files)


def download_batch(repo_id: str, batch: list[dict[str, Any]], target: Path, token: str) -> None:
    files = [s["file_name"] for s in batch]
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(target), token=token,
                      allow_patterns=files)
    missing = [f for f in files if not (target / f).is_file()]
    if missing:
        sys.exit(f"{len(missing)} source file(s) did not download, e.g. {missing[:3]}")


def merged(states: dict[int, dict[str, Any]]) -> dict[str, list]:
    return {k: [row for b in sorted(states) for row in states[b][k]] for k in RESULT_KEYS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-batches", type=int, help="process at most this many batches in this run")
    parser.add_argument("--no-upload", action="store_true",
                        help="stop before uploading (files are kept; a later run uploads them)")
    args = parser.parse_args()

    config = load_config(args.config)
    paths, hf = config["paths"], config["huggingface"]
    output_dir, source_dir = paths["output_dir"], paths["batch_source_dir"]
    songs_per_batch = int(config["batch"]["songs_per_batch"])
    rerender = int(config["batch"].get("rerender_per_batch", 10))
    token = read_token(paths["env_file"])
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_hub_download(config["source"]["repo_id"], "metadata/samples.csv", repo_type="dataset",
                    local_dir=str(source_dir), token=token)
    batches = plan_batches(load_sources(config, source_dir), songs_per_batch)
    fingerprint = config_fingerprint(config)
    states = load_states(output_dir)
    if any(s["fingerprint"] != fingerprint for s in states.values()):
        sys.exit(f"{output_dir} holds batches built with different settings. "
                 "Restore the settings or use a new output_dir.")
    columns = sample_columns(chain_mode(config))
    schema = build_schema(columns)
    api = HfApi(token=token)
    print(f"{len(batches)} batches of {songs_per_batch} songs -> {hf['repo_id']} "
          f"({sum(s['status'] == 'uploaded' for s in states.values())} already uploaded)")

    processed = 0
    for b, batch in enumerate(batches, start=1):
        state = states.get(b)
        if state and state["status"] == "uploaded":
            continue
        if args.max_batches and processed >= args.max_batches:
            break
        song_ids = sorted({int(s["source_id"]) for s in batch})
        name = f"batch {b}/{len(batches)} (songs {song_ids[0]}-{song_ids[-1]}, {len(batch)} sources)"
        free_gb = shutil.disk_usage(output_dir).free / 1024**3
        print(f"\n=== {name} - {free_gb:.1f} GB free ===")

        # 1-3. download, generate, delete the source stems (skipped if a previous run got this far)
        if not (state and state["status"] == "generated" and audio_present(output_dir, state)):
            shutil.rmtree(output_dir / "data", ignore_errors=True)  # leftovers of an interrupted batch
            download_batch(config["source"]["repo_id"], batch, source_dir, token)
            result = process_sources(config, batch, source_dir, output_dir)
            shutil.rmtree(source_dir / "data", ignore_errors=True)
            state = {"batch": b, "songs": song_ids, "status": "generated", "fingerprint": fingerprint, **result}
            save_state(output_dir, state)
            states[b] = state

        # 4-5. combined metadata, checks, this batch's Parquet shards
        uploaded = sum(s["status"] == "uploaded" for s in states.values())
        write_metadata(config, **merged(states), extra={"batches": {
            "completed": uploaded + 1, "total": len(batches), "songs_per_batch": songs_per_batch}})
        rows = load_rows(output_dir, config["splits"])
        batch_ids = {s["sample_id"] for s in batch}
        batch_rows = [r for r in rows if r["source_sample_id"] in batch_ids]
        errors = check_metadata(rows, config)
        audio_errors, n_rendered, max_diff = check_audio(batch_rows, output_dir, rerender)
        errors += audio_errors
        print(f"Checked {len(rows)} pairs (metadata) and {len(batch_rows)} pairs (audio); "
              f"re-rendered {n_rendered} (max abs diff {max_diff:.2e}); "
              f"skipped {len(state['skipped'])} sources, {len(state['failures'])} failed pairs")
        if errors:
            print(f"FAILED with {len(errors)} problem(s) - nothing uploaded:")
            for e in errors[:50]:
                print("  -", e)
            raise SystemExit(1)

        for old in (output_dir / "data").glob("*.parquet"):
            old.unlink()
        shards = []
        for split in config["splits"]:
            split_rows = [r for r in batch_rows if r["split"] == split]
            if split_rows:
                written = export_split(split, split_rows, output_dir, schema, prefix=f"-b{b:02d}")
                verify_split(written, split_rows, output_dir, columns)
                shards += written
        print(f"Parquet: {len(shards)} shard(s), {sum(p.stat().st_size for p in shards) / 1024**3:.2f} GB, verified")

        if args.no_upload:
            print(f"--no-upload: stopping before the upload; files kept in {output_dir}")
            return

        # 6. upload (no deletions: earlier batches stay), then confirm the shards are on the Hub
        api.create_repo(hf["repo_id"], repo_type="dataset", private=bool(hf.get("private", True)), exist_ok=True)
        print(f"Uploading {name} ...")
        commit = api.upload_folder(folder_path=str(output_dir), repo_id=hf["repo_id"], repo_type="dataset",
                                   commit_message=f"Add {name}", ignore_patterns=[f"{BATCH_DIR}/**"])
        expected = [p.relative_to(output_dir).as_posix() for p in shards] + [batch_rows[0]["wet_audio"]]
        found = {info.path for info in api.get_paths_info(hf["repo_id"], expected, repo_type="dataset")}
        if missing := sorted(set(expected) - found):
            sys.exit(f"Upload incomplete, not on the Hub: {missing}. Re-run to retry; local files are kept.")
        state.update(status="uploaded", commit=str(commit))
        save_state(output_dir, state)
        print(f"Uploaded: {commit}")

        # 7. free the disk for the next batch
        shutil.rmtree(output_dir / "data", ignore_errors=True)
        processed += 1

    states = load_states(output_dir)
    done = sum(s["status"] == "uploaded" for s in states.values())
    print(f"\n{done}/{len(batches)} batches uploaded to https://huggingface.co/datasets/{hf['repo_id']}")
    if done == len(batches):
        print("All batches done. Final check: validate_phase1.py --config <config> --metadata-only")


if __name__ == "__main__":
    main()
