"""Validation of a generated Phase 1 dataset.

Errors make the dataset unusable (missing files, wrong labels, leakage, ...).
Warnings flag things worth a look (wet nearly identical to dry, clipping).
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.data.labels import LABEL_COLUMNS, LABEL_NAMES, label_vector
from src.data.metadata import CSV_COLUMNS, SUMMARY_FILE, read_split_csvs
from src.data.splitting import find_leakage
from src.effects.base import EffectError
from src.effects.registry import canonical_order
from src.utils.audio import load_audio, residual_db
from src.utils.config import SPLITS

logger = logging.getLogger(__name__)

SAMPLE_ID_PATTERN = re.compile(r"^sample_\d{6}$")

#: Every check, in report order.
CHECKS: tuple[str, ...] = (
    "metadata_files",
    "columns",
    "sample_ids",
    "split_consistency",
    "paths",
    "labels",
    "effect_chain",
    "parameters",
    "audio_files",
    "audio_compatibility",
    "source_leakage",
    "completeness",
    "summary_consistency",
)


@dataclass(frozen=True)
class ValidationIssue:
    level: str  # "error" | "warning"
    check: str
    message: str
    sample_id: str | None = None


@dataclass
class ValidationReport:
    num_samples: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)
    checks_run: list[str] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def has_issue(self, check: str, level: str = "error") -> bool:
        return any(i.check == check and i.level == level for i in self.issues)

    def format(self, max_per_check: int = 20) -> str:
        """Human-readable report grouped by check."""
        lines = ["=" * 64, "PHASE 1 DATASET VALIDATION REPORT", "=" * 64, f"Samples checked: {self.num_samples}"]
        by_check: dict[str, list[ValidationIssue]] = defaultdict(list)
        for issue in self.issues:
            by_check[issue.check].append(issue)
        for check in self.checks_run:
            issues = by_check.get(check, [])
            n_err = sum(i.level == "error" for i in issues)
            n_warn = len(issues) - n_err
            status = "FAIL" if n_err else ("WARN" if n_warn else "PASS")
            lines.append(f"[{status}] {check}" + (f"  ({n_err} error(s), {n_warn} warning(s))" if issues else ""))
            for issue in issues[:max_per_check]:
                prefix = f"{issue.sample_id}: " if issue.sample_id else ""
                lines.append(f"    - {issue.level.upper()}: {prefix}{issue.message}")
            if len(issues) > max_per_check:
                lines.append(f"    ... {len(issues) - max_per_check} more")
        lines.append("=" * 64)
        lines.append(f"RESULT: {'PASSED' if self.ok else 'FAILED'} - "
                     f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)")
        lines.append("=" * 64)
        return "\n".join(lines)


class _Collector:
    def __init__(self, report: ValidationReport) -> None:
        self.report = report

    def error(self, check: str, message: str, sample_id: str | None = None) -> None:
        self.report.issues.append(ValidationIssue("error", check, message, sample_id))

    def warning(self, check: str, message: str, sample_id: str | None = None) -> None:
        self.report.issues.append(ValidationIssue("warning", check, message, sample_id))


def validate_dataset(
    output_dir: str | Path,
    metadata_dir: str | Path,
    check_audio: bool = True,
    duration_tolerance_s: float = 1e-3,
    identical_threshold_db: float = -60.0,
) -> ValidationReport:
    """Validate generated audio and metadata.

    Args:
        output_dir: Dataset root holding ``{split}/dry`` and ``{split}/wet``.
        metadata_dir: Directory with the split CSVs and ``dataset_summary.json``.
        check_audio: Decode every audio file (slower, but catches corrupt files).
        duration_tolerance_s: Allowed deviation from the configured clip duration.
        identical_threshold_db: Warn when wet-vs-dry residual is below this.
    """
    output_dir, metadata_dir = Path(output_dir), Path(metadata_dir)
    report = ValidationReport(checks_run=list(CHECKS))
    out = _Collector(report)

    summary = _load_summary(metadata_dir, out)
    def stop(message: str) -> ValidationReport:
        # Nothing else can be checked; only report the check that actually ran.
        out.error("metadata_files", message)
        report.checks_run = ["metadata_files"]
        return report

    try:
        by_split = read_split_csvs(metadata_dir)
    except FileNotFoundError as exc:
        return stop(str(exc))
    except (ValueError, json.JSONDecodeError) as exc:
        return stop(f"Cannot parse metadata CSV: {exc}")

    rows = [row for split in SPLITS for row in by_split[split]]
    report.num_samples = len(rows)
    if not rows:
        return stop("Metadata contains no samples")

    _check_columns(by_split, metadata_dir, out)
    _check_sample_ids(rows, out)
    _check_split_consistency(by_split, out)
    _check_paths(rows, output_dir, out)
    _check_labels(rows, out)
    _check_effect_chain(rows, out)
    _check_parameters(rows, out)
    if check_audio:
        _check_audio(rows, output_dir, summary, duration_tolerance_s, identical_threshold_db, out)
    else:
        report.checks_run = [c for c in report.checks_run if c not in ("audio_files", "audio_compatibility")]
    _check_leakage(rows, out)
    _check_completeness(rows, summary, out)
    _check_summary(rows, by_split, summary, out)
    return report


# --------------------------------------------------------------------- checks


def _load_summary(metadata_dir: Path, out: _Collector) -> dict[str, Any] | None:
    path = metadata_dir / SUMMARY_FILE
    if not path.is_file():
        out.error("metadata_files", f"Summary file not found: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        out.error("metadata_files", f"Cannot parse {path}: {exc}")
        return None


def _check_columns(by_split: Mapping[str, list[dict]], metadata_dir: Path, out: _Collector) -> None:
    for split, rows in by_split.items():
        if rows:
            missing = [c for c in CSV_COLUMNS if c not in rows[0]]
            if missing:
                out.error("columns", f"{metadata_dir / (split + '.csv')} is missing columns {missing}")


def _check_sample_ids(rows: Sequence[Mapping[str, Any]], out: _Collector) -> None:
    counts = Counter(r.get("sample_id") for r in rows)
    for sid, n in counts.items():
        if n > 1:
            out.error("sample_ids", f"duplicate sample_id ({n} rows)", sid)
        if not sid or not SAMPLE_ID_PATTERN.match(str(sid)):
            out.error("sample_ids", f"malformed sample_id {sid!r}")


def _check_split_consistency(by_split: Mapping[str, list[dict]], out: _Collector) -> None:
    for split, rows in by_split.items():
        for r in rows:
            if r.get("split") != split:
                out.error("split_consistency", f"row in {split}.csv has split={r.get('split')!r}", r.get("sample_id"))
            for col, kind in (("dry_path", "dry"), ("wet_path", "wet")):
                if not str(r.get(col, "")).startswith(f"{split}/{kind}/"):
                    out.error("split_consistency", f"{col} {r.get(col)!r} is not under {split}/{kind}/",
                              r.get("sample_id"))


def _check_paths(rows: Sequence[Mapping[str, Any]], output_dir: Path, out: _Collector) -> None:
    root = output_dir.resolve()
    for r in rows:
        for col in ("dry_path", "wet_path"):
            rel = r.get(col)
            if not rel:
                out.error("paths", f"{col} is empty", r.get("sample_id"))
                continue
            path = (output_dir / rel).resolve()
            try:
                path.relative_to(root)
            except ValueError:
                out.error("paths", f"{col} {rel!r} points outside the dataset root", r.get("sample_id"))
                continue
            if not path.is_file():
                out.error("paths", f"{col} file does not exist: {rel}", r.get("sample_id"))


def _check_labels(rows: Sequence[Mapping[str, Any]], out: _Collector) -> None:
    for r in rows:
        sid = r.get("sample_id")
        label = r.get("label")
        if not isinstance(label, list) or len(label) != len(LABEL_NAMES):
            out.error("labels", f"label must have exactly {len(LABEL_NAMES)} values (got {label!r})", sid)
            continue
        if any(v not in (0, 1) for v in label):
            out.error("labels", f"label values must be binary (got {label})", sid)
            continue
        if sum(label) == 0:
            out.error("labels", "no effect is active in the label", sid)
        columns = [r.get(LABEL_COLUMNS[name]) for name in LABEL_NAMES]
        if columns != label:
            out.error("labels", f"label {label} disagrees with label columns {columns}", sid)
        chain = r.get("effect_chain")
        try:
            expected = label_vector(chain or [])
        except EffectError as exc:
            out.error("labels", f"cannot derive label from effect_chain: {exc}", sid)
            continue
        if expected != label:
            out.error("labels", f"label {label} does not match effect_chain {chain} (expected {expected})", sid)


def _check_effect_chain(rows: Sequence[Mapping[str, Any]], out: _Collector) -> None:
    for r in rows:
        sid = r.get("sample_id")
        chain = r.get("effect_chain")
        if not isinstance(chain, list) or not chain:
            out.error("effect_chain", f"effect_chain must be a non-empty list (got {chain!r})", sid)
            continue
        try:
            expected_order = canonical_order(chain)
        except EffectError as exc:
            out.error("effect_chain", str(exc), sid)
            continue
        if r.get("processing_order") != expected_order:
            out.error("effect_chain",
                      f"processing_order {r.get('processing_order')} is not the canonical order {expected_order}", sid)


def _check_parameters(rows: Sequence[Mapping[str, Any]], out: _Collector) -> None:
    for r in rows:
        sid = r.get("sample_id")
        params = r.get("parameters")
        chain = r.get("effect_chain") or []
        if not isinstance(params, dict):
            out.error("parameters", f"parameters must be a JSON object (got {params!r})", sid)
            continue
        if sorted(params) != sorted(chain):
            out.error("parameters", f"parameter effects {sorted(params)} do not match effect_chain {sorted(chain)}",
                      sid)
        for effect, values in params.items():
            if not isinstance(values, dict) or not values:
                out.error("parameters", f"{effect} has no parameter values", sid)
            elif not all(isinstance(v, (int, float)) and np.isfinite(v) for v in values.values()):
                out.error("parameters", f"{effect} has non-numeric parameter values {values}", sid)
        if r.get("low_effect_change"):
            out.warning("parameters", f"effect(s) {r['low_effect_change']} barely changed the signal", sid)


def _check_audio(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    summary: Mapping[str, Any] | None,
    duration_tolerance_s: float,
    identical_threshold_db: float,
    out: _Collector,
) -> None:
    expected = (summary or {}).get("audio", {})
    exp_sr, exp_ch, exp_dur = expected.get("sample_rate"), expected.get("channels"), expected.get("duration_s")
    for r in rows:
        sid = r.get("sample_id")
        dry_path, wet_path = output_dir / str(r.get("dry_path")), output_dir / str(r.get("wet_path"))
        if not (dry_path.is_file() and wet_path.is_file()):
            continue  # already reported by the paths check
        try:
            dry, dry_sr = load_audio(dry_path)
            wet, wet_sr = load_audio(wet_path)
        except Exception as exc:  # noqa: BLE001
            out.error("audio_files", f"cannot load audio: {exc}", sid)
            continue

        check = "audio_compatibility"
        (dry_ch, dry_n), (wet_ch, wet_n) = dry.shape, wet.shape
        if dry_sr != wet_sr:
            out.error(check, f"sample rates differ (dry {dry_sr}, wet {wet_sr})", sid)
        if exp_sr is not None and dry_sr != exp_sr:
            out.error(check, f"sample rate {dry_sr} != configured {exp_sr}", sid)
        if r.get("sample_rate") is not None and dry_sr != r["sample_rate"]:
            out.error(check, f"sample rate {dry_sr} != metadata {r['sample_rate']}", sid)
        if dry_ch != wet_ch:
            out.error(check, f"channel counts differ (dry {dry_ch}, wet {wet_ch})", sid)
        if exp_ch is not None and dry_ch != exp_ch:
            out.error(check, f"channels {dry_ch} != configured {exp_ch}", sid)
        if dry_n != wet_n:
            out.error(check, f"lengths differ (dry {dry_n}, wet {wet_n} samples)", sid)
        if dry_n == 0:
            out.error(check, "audio has zero length", sid)
        if exp_dur is not None and dry_sr and abs(dry_n / dry_sr - exp_dur) > duration_tolerance_s:
            out.error(check, f"duration {dry_n / dry_sr:.4f}s != configured {exp_dur}s", sid)
        if not (np.all(np.isfinite(dry)) and np.all(np.isfinite(wet))):
            out.error(check, "audio contains NaN/inf samples", sid)
            continue
        if dry.shape == wet.shape and dry.size:
            change = residual_db(dry, wet)
            if change < identical_threshold_db:
                out.warning(check, f"wet is nearly identical to dry (residual {change:.1f} dB)", sid)
        if wet.size and float(np.max(np.abs(wet))) > 1.0:
            out.warning(check, f"wet peak {float(np.max(np.abs(wet))):.3f} exceeds full scale", sid)


def _check_leakage(rows: Sequence[Mapping[str, Any]], out: _Collector) -> None:
    for key in ("source_file", "source_song", "source_id"):
        for value, splits in find_leakage([(str(r.get(key)), str(r.get("split"))) for r in rows]).items():
            out.error("source_leakage", f"{key} {value!r} appears in multiple splits: {splits}")


def _check_completeness(rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any] | None, out: _Collector) -> None:
    combos = (summary or {}).get("effect_combinations")
    if not combos:
        out.warning("completeness", "effect_combinations missing from summary; skipped completeness check")
        return
    expected = Counter(combos)
    by_source: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by_source[str(r.get("source_id"))][str(r.get("combination"))] += 1
    for source_id, got in sorted(by_source.items()):
        if got != expected:
            missing = sorted(set(expected) - set(got))
            extra = sorted(c for c, n in got.items() if n > expected.get(c, 0))
            out.error("completeness", f"source {source_id}: missing {missing}, unexpected/duplicate {extra}")


def _check_summary(
    rows: Sequence[Mapping[str, Any]],
    by_split: Mapping[str, list[dict]],
    summary: Mapping[str, Any] | None,
    out: _Collector,
) -> None:
    if summary is None:
        return
    check = "summary_consistency"
    if summary.get("num_generated_pairs") != len(rows):
        out.error(check, f"summary num_generated_pairs={summary.get('num_generated_pairs')} but CSVs have {len(rows)}")
    for split in SPLITS:
        key = f"num_{split}_pairs"
        if summary.get(key) != len(by_split[split]):
            out.error(check, f"summary {key}={summary.get(key)} but {split}.csv has {len(by_split[split])} rows")
    n_sources = len({r.get("source_id") for r in rows})
    if summary.get("num_source_files") != n_sources:
        out.error(check, f"summary num_source_files={summary.get('num_source_files')} but CSVs reference "
                         f"{n_sources} source(s)")
    if summary.get("num_failed_generations"):
        out.warning(check, f"{summary['num_failed_generations']} generation failure(s) recorded (see failures.csv)")
