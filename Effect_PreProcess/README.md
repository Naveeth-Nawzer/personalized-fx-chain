# Phase 1 Effect Dataset — Inverse FX Chain Predictor

Generates paired **dry / wet** audio for Phase 1 (effect-type prediction) from the
source dataset [`APINAJA/inverse-fx-source-audio-cleaned`](https://huggingface.co/datasets/APINAJA/inverse-fx-source-audio-cleaned)
using **Pedalboard**, then pushes it to
[`APINAJA/inverse-fx-phase1-effect-type`](https://huggingface.co/datasets/APINAJA/inverse-fx-phase1-effect-type).

```
source (1-Bass, 1-Drums, 1-Mixture, ...)          60 dry sources (20 songs x 3 categories)
   └─ 10 s non-silent clip, mono, -20 LUFS        -> data/<split>/dry/1-Bass.wav
        └─ EQ -> Compressor -> Reverb (7 combos)  -> data/<split>/wet/1-Bass-EQ-Compressor.wav
                                                   = 420 dry/wet pairs
```

| # | effect_chain | label `[EQ, Comp, Rev]` |
|---|---|---|
| 1 | EQ | `[1,0,0]` |
| 2 | Compressor | `[0,1,0]` |
| 3 | Reverb | `[0,0,1]` |
| 4 | EQ + Compressor | `[1,1,0]` |
| 5 | EQ + Reverb | `[1,0,1]` |
| 6 | Compressor + Reverb | `[0,1,1]` |
| 7 | EQ + Compressor + Reverb | `[1,1,1]` |

**Split** — taken from the source dataset, which is split by **song**:
train 14 songs (70 %), validation 3 (15 %), test 3 (15 %) → 294 / 63 / 63 pairs.
All 7 versions of a clip stay in the same split as their dry audio, so no song leaks.

**Reproducible** — `generation.seed: 42` controls the clip window and every effect
parameter. Running again gives byte-identical audio and metadata.

---

## Files

| file | purpose |
|---|---|
| `config.yaml` | all settings: repos, categories, audio, preprocessing, seed, parameter ranges |
| `download_source.py` | step 1: download the source dataset to `data/source/` (read only) |
| `generate_phase1.py` | step 2: build dry clips + 7 wet versions + metadata in `output/phase1_effect_type/` |
| `validate_phase1.py` | step 3: check labels, ids, splits, audio, and re-render samples from stored parameters |
| `export_parquet.py` | step 4: write Parquet shards with `dry_audio` / `wet_audio` as `Audio` (playable in the Hub viewer) |
| `push_phase1.py` | step 5: upload `output/phase1_effect_type/` to the Hugging Face Hub |
| `fx.py` | Pedalboard effects, parameter sampling, combinations, labels |
| `audio_utils.py` | audio I/O and dry-clip preprocessing |
| `common.py` | config, token, CSV helpers |

The Hugging Face token is read from `../Dataset/.env` (`HF_TOKEN=...`).

---

## Guide (Windows Command Prompt)

### 0. Open cmd in this folder

```bat
cd /d "D:\Y4S1(jul-Dec-2026)\Research-Project\Effect_PreProcess"
```

### 1. Virtual environment (one time only)

A `.venv` (Python 3.12) with all libraries is already created here. To recreate it on
another machine:

```bat
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```

Activate it for the current cmd window (afterwards `python` means the venv's Python):

```bat
.venv\Scripts\activate
```

Check Pedalboard works:

```bat
python -c "import pedalboard; print(pedalboard.__version__)"
```

### 2. Download the source dataset

```bat
python download_source.py
```

Expected last line: `OK: 60 source samples available`. Re-running only downloads missing files.

### 3. Generate the Phase 1 dataset

Optional quick test (first 3 sources, 21 pairs):

```bat
python generate_phase1.py --limit 3 --overwrite
```

Full dataset (about 30 s, ~810 MB):

```bat
python generate_phase1.py --overwrite
```

Expected summary:

```
Pairs generated : 420 / 420 expected (0 sources skipped, 0 failed)
  train      songs=14  (70%)  dry=42   pairs=294
  validation songs=3   (15%)  dry=9    pairs=63
  test       songs=3   (15%)  dry=9    pairs=63
Labels          : {'EQ': 240, 'Compressor': 240, 'Reverb': 240}
```

### 4. Validate

```bat
python validate_phase1.py
```

Expected: `All checks passed.`

### 5. Export Parquet for the Dataset Viewer

```bat
python export_parquet.py
```

Writes `data/<split>-*.parquet` where `dry_audio` and `wet_audio` are Hugging Face `Audio`
columns (WAV bytes + original path), all other columns unchanged, and points the dataset
card's `configs` at them. Every value is read back and checked against the CSV/WAV files.
Without this step the viewer shows the two columns as plain `string` paths.

### 6. Push to Hugging Face

```bat
python push_phase1.py
```

Creates the **private** dataset repo `APINAJA/inverse-fx-phase1-effect-type` (if needed) and
uploads everything. To use a different name:

```bat
python push_phase1.py --repo-id APINAJA/your-repo-name
```

Re-running after a regeneration replaces the old files on the Hub.

---

## Output layout

```
output/phase1_effect_type/
├── README.md                              dataset card (shown on the Hub)
├── data/
│   ├── train-0000N-of-00004.parquet       viewer / load_dataset (Audio columns)
│   ├── validation-00000-of-00001.parquet, test-00000-of-00001.parquet
│   ├── train/dry/2-Bass.wav               dry clip (shared by its 7 wet versions)
│   ├── train/wet/2-Bass-EQ.wav ... 2-Bass-EQ-Compressor-Reverb.wav
│   ├── validation/...
│   └── test/...
└── metadata/
    ├── train.csv / validation.csv / test.csv   one row per dry/wet pair
    ├── sources.csv                             how each dry clip was cut and normalised
    ├── skipped.csv / failures.csv              should be empty
    ├── config_used.yaml
    └── dataset_summary.json
```

Main columns of `metadata/<split>.csv` (lists/dicts are JSON strings):

| column | example |
|---|---|
| `sample_id` | `1-Bass-EQ-Compressor` |
| `source_sample_id` | `1-Bass` |
| `source_id` | `1` |
| `source_category` | `Bass` |
| `dry_audio` / `wet_audio` | `data/test/dry/1-Bass.wav` / `data/test/wet/1-Bass-EQ-Compressor.wav` |
| `effect_chain` | `["EQ","Compressor"]` |
| `label` | `[1,1,0]` |
| `eq_label`, `compressor_label`, `reverb_label` | `1`, `1`, `0` |
| `eq_params` | `{"frequency":1188.0914,"gain":-6.3806,"q":1.6983}` |
| `compressor_params` | `{"threshold":-37.53,"ratio":4.95,"attack":1.26,"release":180.13,"makeup_gain":0.24}` |
| `reverb_params` | empty (effect not in the chain) |
| `parameters` / `parameters_normalized` | all used parameters, physical and scaled to [0, 1] |
| `effect_change_db` | how much each effect changed the signal (audibility check) |
| `sample_seed` | per-sample seed derived from `generation.seed` |

Pedalboard mapping: EQ = `PeakFilter(cutoff_frequency_hz, gain_db, q)`;
Compressor = `Compressor(threshold_db, ratio, attack_ms, release_ms)` + `Gain(makeup_gain)`;
Reverb = `Reverb(room_size, damping, wet_level=wet_mix, dry_level=dry_mix, width)`.

---

## Common changes (edit `config.yaml`, then run steps 3–6 again)

| want | change |
|---|---|
| only Bass + Drums | `source.categories: [Bass, Drums]` |
| longer clips | `audio.duration: 20.0` |
| stereo | `audio.channels: 2` |
| same parameters for every sample | `generation.parameter_mode: fixed` |
| different random parameters | `generation.seed: 123` |
| public repo | `huggingface.private: false` (only applies when the repo is first created) |

## Troubleshooting

- **`HF_TOKEN not found`** — check `../Dataset/.env` contains `HF_TOKEN=hf_...`, or run `set HF_TOKEN=hf_...` first.
- **`already contains data`** — add `--overwrite` to regenerate.
- **`Source dataset not found`** — run `python download_source.py` first.
- **`pyarrow` fails with "An Application Control policy has blocked this file"** — newer pyarrow
  builds are blocked on this machine; keep the pinned `pyarrow==21.0.0` from `requirements.txt`.
- **`import pedalboard` fails with a DLL error** — Windows Smart App Control / antivirus may block the
  native module; allow it, or recreate the venv with `py -3.12`.
