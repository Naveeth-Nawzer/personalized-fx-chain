"""Phase 1 dry/wet dataset generation.

Pipeline:
    discover -> split songs -> select sources per split -> preprocess dry clip
    -> render every effect combination (canonical order) -> write audio
    -> write metadata + summary

Randomness is derived from the single ``generation.seed`` via independent
``SeedSequence`` streams per stage, so results do not depend on processing
order or on failures elsewhere.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.data.discovery import AudioFileInfo, discover_audio_files
from src.data.labels import combination_name, effect_combinations, label_columns, label_vector
from src.data.metadata import (
    CONFIG_USED_FILE,
    FAILURE_COLUMNS,
    FAILURES_FILE,
    OWNED_METADATA_FILES,
    SELECTED_SOURCES_COLUMNS,
    SELECTED_SOURCES_FILE,
    SKIPPED_COLUMNS,
    SKIPPED_FILE,
    SUMMARY_FILE,
    build_summary,
    write_json,
    write_split_csvs,
    write_table,
    write_yaml,
)
from src.data.preprocessing import PreparedSource, PreprocessingSettings, SourceRejected, prepare_source
from src.data.selection import SelectionResult, select_sources
from src.data.splitting import split_groups
from src.effects import BACKENDS, CANONICAL_ORDER, ParameterSampler, canonical_order, create_effect
from src.utils.audio import amplitude_to_db, peak, residual_db, write_audio
from src.utils.config import SPLITS, enabled_effects, public_config, validate_config

logger = logging.getLogger(__name__)

# Independent random streams (SeedSequence entropy suffixes).
_STREAM_SPLIT = 1
_STREAM_SELECTION = 2
_STREAM_PARAMETERS = 3


class GenerationError(RuntimeError):
    """Raised when generation cannot start or cannot produce a usable dataset."""


@dataclass
class GenerationReport:
    """Everything produced by a generation run."""

    num_discovered: int
    selection: SelectionResult
    records: list[dict[str, Any]]
    failures: list[dict[str, Any]]
    summary: dict[str, Any]
    output_dir: Path
    metadata_dir: Path
    warnings: list[str] = field(default_factory=list)


def sample_seed(seed: int, source_index: int, combo_index: int) -> int:
    """Deterministic per-sample seed for parameter sampling."""
    return int(np.random.SeedSequence([seed, _STREAM_PARAMETERS, source_index, combo_index]).generate_state(1)[0])


def sample_id_for(source_index: int, combo_index: int, num_combos: int) -> str:
    """Stable sample ID: depends only on the source position and the combination index."""
    return f"sample_{source_index * num_combos + combo_index + 1:06d}"


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


class DatasetGenerator:
    """Generates the Phase 1 effect-type dataset from a configuration mapping."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        validate_config(config, CANONICAL_ORDER, tuple(BACKENDS))
        self.config = config
        self.settings = PreprocessingSettings.from_config(config)
        gen = config["generation"]
        self.seed: int = int(gen["seed"])
        self.min_change_db: float = float(gen["min_effect_change_db"])
        self.max_retries: int = int(gen["max_retries"])
        self.sampler = ParameterSampler.from_config(config["parameters"], gen["parameter_mode"])
        self.backend: str = config["effects"]["backend"]
        self.subtype: str = config["audio"]["output_subtype"]
        self.enabled = canonical_order(enabled_effects(config))
        self.combinations = effect_combinations(self.enabled)
        options = config["effects"].get("scipy_reverb", {})
        self.effects = {name: create_effect(name, self.backend, options) for name in self.enabled}

    # ------------------------------------------------------------------ public

    def run(self, input_dir: Path, output_dir: Path, metadata_dir: Path, overwrite: bool = False) -> GenerationReport:
        """Run the full pipeline and write audio + metadata."""
        input_dir, output_dir, metadata_dir = Path(input_dir), Path(output_dir), Path(metadata_dir)
        self._prepare_output_dirs(input_dir, output_dir, metadata_dir, overwrite)

        ds_cfg = self.config["dataset"]
        files = discover_audio_files(
            input_dir,
            ds_cfg["audio_extensions"],
            ds_cfg.get("include_categories"),
            ds_cfg.get("exclude_categories") or (),
        )
        logger.info("Discovered %d audio file(s) under %s", len(files), input_dir)
        if not files:
            raise GenerationError(f"No audio files with extensions {ds_cfg['audio_extensions']} found in {input_dir}")
        categories = Counter(f.category for f in files)
        logger.info("Discovered categories: %s", dict(sorted(categories.items())))

        selection = self._select(files)
        self._log_selection(selection)

        records, failures = self._generate_all(selection, output_dir)
        summary = self._write_metadata(selection, records, failures, input_dir, len(files), metadata_dir)
        report = GenerationReport(len(files), selection, records, failures, summary, output_dir, metadata_dir,
                                  list(selection.warnings))
        log_generation_summary(report)
        return report

    def render(
        self, dry: np.ndarray, sample_rate: int, chain: Sequence[str], rng: np.random.Generator
    ) -> tuple[np.ndarray, dict[str, dict[str, float]], dict[str, float], list[str]]:
        """Apply ``chain`` (canonical order) to ``dry`` with sampled parameters.

        Each effect is rendered step by step so the audibility guard can
        re-sample that effect's parameters (random mode) when its change to
        the signal is below ``min_effect_change_db``. The result is identical
        to ``EffectChain(steps).process(dry)`` with the returned parameters.

        Returns:
            ``(wet, parameters, effect_change_db, low_effect_change)``.
        """
        x = dry
        parameters: dict[str, dict[str, float]] = {}
        changes: dict[str, float] = {}
        low: list[str] = []
        for name in canonical_order(chain):
            effect = self.effects[name]
            for attempt in range(self.max_retries + 1):
                params = self.sampler.sample(name, rng)
                y = effect.process(x, sample_rate, params)
                change = residual_db(x, y)
                if change >= self.min_change_db or not self.sampler.is_random:
                    break
                logger.debug("%s change %.1f dB below %.1f dB (attempt %d), re-sampling",
                             name, change, self.min_change_db, attempt + 1)
            if change < self.min_change_db:
                low.append(name)
            parameters[name] = params
            changes[name] = round(change, 2)
            x = y
        return x, parameters, changes, low

    # ----------------------------------------------------------------- stages

    def _prepare_output_dirs(self, input_dir: Path, output_dir: Path, metadata_dir: Path, overwrite: bool) -> None:
        if not input_dir.is_dir():
            raise GenerationError(f"Input directory not found: {input_dir}")
        for label, target in (("output", output_dir), ("metadata", metadata_dir)):
            if _is_within(target, input_dir):
                raise GenerationError(
                    f"The {label} directory ({target}) is inside the input dataset ({input_dir}). "
                    "Generated data must be stored separately from the original dataset."
                )
        existing = [p for s in SPLITS for kind in ("dry", "wet") for p in (output_dir / s / kind).glob("sample_*.wav")]
        existing += [metadata_dir / name for name in OWNED_METADATA_FILES
                     if (metadata_dir / name).is_file() and name != "generation.log"]
        if existing and not overwrite:
            raise GenerationError(
                f"Found {len(existing)} file(s) from a previous run in {output_dir} / {metadata_dir}. "
                "Use --overwrite to replace them, or choose another --output-dir / --metadata-dir."
            )
        if existing:
            logger.warning("--overwrite: removing %d previously generated file(s)", len(existing))
            for path in existing:
                path.unlink()
        for split in SPLITS:
            for kind in ("dry", "wet"):
                (output_dir / split / kind).mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

    def _select(self, files: Sequence[AudioFileInfo]) -> SelectionResult:
        ratios = {s: float(self.config["split"][s]) for s in SPLITS}
        split_rng = np.random.default_rng(np.random.SeedSequence([self.seed, _STREAM_SPLIT]))
        song_splits = split_groups((f.song for f in files), ratios, split_rng)
        songs_per_split = Counter(song_splits.values())
        logger.info("Songs per split: %s", {s: songs_per_split.get(s, 0) for s in SPLITS})

        def validator(info: AudioFileInfo) -> str | None:
            try:
                prepare_source(info, self.settings, self.seed)
            except SourceRejected as exc:
                return str(exc)
            return None

        select_rng = np.random.default_rng(np.random.SeedSequence([self.seed, _STREAM_SELECTION]))
        return select_sources(
            files,
            song_splits,
            int(self.config["dataset"]["num_source_files"]),
            ratios,
            select_rng,
            validator,
            stratify=bool(self.config["dataset"].get("stratify_by_category", True)),
        )

    def _generate_all(
        self, selection: SelectionResult, output_dir: Path
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        records: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        total = len(selection.selected)
        for k, source in enumerate(selection.selected, start=1):
            logger.info("[%d/%d] %s (%s) %s", k, total, source.source_id, source.split, source.file.rel_path)
            try:
                prepared = prepare_source(source.file, self.settings, self.seed)
            except Exception as exc:  # noqa: BLE001 — record and continue
                logger.error("Preprocessing failed for %s: %s", source.file.rel_path, exc)
                failures.append({"stage": "preprocess", "source_id": source.source_id,
                                 "source_file": source.file.rel_path, "error": str(exc)})
                continue
            for combo_index, chain in enumerate(self.combinations):
                sid = sample_id_for(source.source_index, combo_index, len(self.combinations))
                try:
                    records.append(self._generate_sample(source, prepared, chain, combo_index, sid, output_dir))
                except Exception as exc:  # noqa: BLE001 — record and continue
                    logger.error("Generation failed for %s %s: %s", sid, combination_name(chain), exc)
                    failures.append({"stage": "render", "source_id": source.source_id,
                                     "source_file": source.file.rel_path, "sample_id": sid,
                                     "effect_chain": list(chain), "error": str(exc)})
        return records, failures

    def _generate_sample(
        self, source, prepared: PreparedSource, chain: Sequence[str], combo_index: int, sid: str, output_dir: Path
    ) -> dict[str, Any]:
        seed = sample_seed(self.seed, source.source_index, combo_index)
        rng = np.random.default_rng(seed)
        wet, parameters, changes, low = self.render(prepared.audio, prepared.sample_rate, chain, rng)
        if low:
            logger.warning("%s: effect(s) %s changed the signal by less than %.1f dB",
                           sid, low, self.min_change_db)

        dry_rel = f"{source.split}/dry/{sid}.wav"
        wet_rel = f"{source.split}/wet/{sid}.wav"
        write_audio(output_dir / dry_rel, prepared.audio, prepared.sample_rate, self.subtype)
        write_audio(output_dir / wet_rel, wet, prepared.sample_rate, self.subtype)

        wet_peak = peak(wet)
        order = canonical_order(chain)
        return {
            "sample_id": sid,
            "source_id": source.source_id,
            "source_file": source.file.rel_path,
            "source_song": source.file.song,
            "source_category": source.file.category,
            "split": source.split,
            "dry_path": dry_rel,
            "wet_path": wet_rel,
            "effect_chain": list(chain),
            "processing_order": order,
            "combination": combination_name(chain),
            "label": label_vector(chain),
            **label_columns(chain),
            "parameters": parameters,
            "parameters_normalized": {n: self.sampler.normalize(n, p) for n, p in parameters.items()},
            "effect_change_db": changes,
            "low_effect_change": low,
            "sample_rate": prepared.sample_rate,
            "num_channels": int(prepared.audio.shape[0]),
            "duration_s": round(prepared.audio.shape[1] / prepared.sample_rate, 6),
            "wet_peak_dbfs": round(amplitude_to_db(wet_peak), 3),
            "wet_clipped": int(self.subtype != "FLOAT" and wet_peak > 1.0),
            "dsp_backend": self.backend,
            "parameter_mode": self.sampler.mode,
            "random_seed": self.seed,
            "sample_seed": seed,
            "preprocessing": prepared.record.to_dict(),
        }

    def _write_metadata(
        self,
        selection: SelectionResult,
        records: list[dict[str, Any]],
        failures: list[dict[str, Any]],
        input_dir: Path,
        num_discovered: int,
        metadata_dir: Path,
    ) -> dict[str, Any]:
        preprocessing_by_source = {r["source_id"]: r["preprocessing"] for r in records}
        sources = [
            {
                "source_id": s.source_id,
                "split": s.split,
                "source_file": s.file.rel_path,
                "source_song": s.file.song,
                "source_category": s.file.category,
                "preprocessing": preprocessing_by_source.get(s.source_id),
            }
            for s in selection.selected
        ]
        skipped = [
            {"source_file": s.rel_path, "source_song": s.song, "source_category": s.category, "reason": s.reason}
            for s in selection.skipped
        ]
        write_split_csvs(records, metadata_dir)
        write_table(metadata_dir / SELECTED_SOURCES_FILE, sources, SELECTED_SOURCES_COLUMNS)
        write_table(metadata_dir / SKIPPED_FILE, skipped, SKIPPED_COLUMNS)
        write_table(metadata_dir / FAILURES_FILE, failures, FAILURE_COLUMNS)
        write_yaml(metadata_dir / CONFIG_USED_FILE, public_config(self.config))
        summary = build_summary(
            records, sources, self.config, self.combinations,
            input_dir_name=input_dir.name, num_discovered=num_discovered,
            num_skipped=len(selection.skipped), failures=failures,
        )
        write_json(metadata_dir / SUMMARY_FILE, summary)
        return summary

    @staticmethod
    def _log_selection(selection: SelectionResult) -> None:
        logger.info("Selected %d source file(s); skipped %d candidate(s)",
                    len(selection.selected), len(selection.skipped))
        if selection.skipped:
            reasons = Counter(s.reason.split(" (")[0] for s in selection.skipped)
            for reason, count in reasons.most_common():
                logger.info("  skipped x%d: %s", count, reason)


def log_generation_summary(report: GenerationReport) -> None:
    """Log the end-of-run summary block."""
    s = report.summary
    lines = [
        "=" * 64,
        "PHASE 1 DATASET GENERATION SUMMARY",
        "=" * 64,
        f"Discovered audio files : {report.num_discovered}",
        f"Selected source files  : {s['num_source_files']}",
        f"Skipped candidates     : {s['num_skipped_files']}",
        f"Generated pairs        : {s['num_generated_pairs']} / {s['num_expected_pairs']} expected",
        f"Failed generations     : {s['num_failed_generations']}",
        f"Train / Val / Test     : {s['num_train_pairs']} / {s['num_validation_pairs']} / {s['num_test_pairs']} pairs",
        "Sources per split      : " + ", ".join(
            f"{k}={v['num_source_files']} ({v['num_songs']} songs)" for k, v in s["splits"].items()),
        f"Effect distribution    : {s['effect_distribution']}",
        f"Category distribution  : {s['source_category_distribution']}",
        f"Low-change samples     : {s['num_low_effect_change_samples']}",
        f"Parameter mode / seed  : {s['parameter_mode']} / {s['random_seed']}  (backend: {s['dsp_backend']})",
        f"Audio output           : {report.output_dir}",
        f"Metadata               : {report.metadata_dir}",
        "=" * 64,
    ]
    for line in lines:
        logger.info(line)
