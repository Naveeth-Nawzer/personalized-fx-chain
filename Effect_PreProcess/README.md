# Phase 1 Effect Dataset — Inverse FX Chain Predictor

Generates paired **dry / wet** audio for Phase 1 (effect-type prediction) from the
source dataset [`APINAJA/inverse-fx-source-audio-cleaned`](https://huggingface.co/datasets/APINAJA/inverse-fx-source-audio-cleaned)
using **Pedalboard**, then pushes it to
[`APINAJA/inverse-fx-phase1-effect-type`](https://huggingface.co/datasets/APINAJA/inverse-fx-phase1-effect-type).

```
source (1-Bass, 1-Drums, 1-Mixture, ...)          450 sources (150 songs x 3 categories), 448 usable
   └─ 10 s non-silent clip, mono, -20 LUFS        -> data/<split>/dry/1-Bass.wav
        └─ EQ -> Compressor -> Reverb (7 combos)  -> data/<split>/wet/1-Bass-EQ-Compressor.wav
                                                   = 3136 dry/wet pairs
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
train 105 songs (70 %), validation 23 (15 %), test 22 (15 %) → 2191 / 483 / 462 pairs.
Two bass stems (`101-Bass`, `124-Bass`) have no 10 s window that is at least 50 % non-silent and are
skipped (listed in `metadata/skipped.csv`).
All 7 versions of a clip stay in the same split as their dry audio, so no song leaks.

**Reproducible** — `generation.seed: 42` controls the clip window and every effect
parameter. Running again gives byte-identical audio and metadata.

---

## Files

| file | purpose |
|---|---|
| `config.yaml` | all settings: repos, categories, audio, preprocessing, seed, parameter ranges |
| `config_order.yaml` | 15-chain effect-order dataset: only the overrides of `config.yaml` (`extends: config.yaml`) |
| `run_batches.py` | batch build: download → generate → check → upload → delete, batch by batch (effect-order dataset) |
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

Expected last line: `OK: 450 source samples available` (~18 GB). Re-running only downloads missing files.

If disk space is short, skip this step and point `paths.source_dir` at an existing local copy with the
same layout, e.g. `../Dataset_Push/build/inverse-fx-source-audio-cleaned` (the folder that was pushed).

### 3. Generate the Phase 1 dataset

Optional quick test (first 3 sources, 21 pairs):

```bat
python generate_phase1.py --limit 3 --overwrite
```

Full dataset (about 3 min, ~6 GB of WAV; ~13 GB after the Parquet export):

```bat
python generate_phase1.py --overwrite
```

Expected summary:

```
Pairs generated : 3136 / 3136 expected (2 sources skipped, 0 failed)
  train      songs=105 (70%)  dry=313  pairs=2191
  validation songs=23  (15%)  dry=69   pairs=483
  test       songs=22  (15%)  dry=66   pairs=462
Labels          : {'EQ': 1792, 'Compressor': 1792, 'Reverb': 1792}
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

## Effect-order dataset (15 chains, batch build)

[`APINAJA/inverse-fx-phase1-effect-order`](https://huggingface.co/datasets/APINAJA/inverse-fx-phase1-effect-order)
renders **every order of every non-empty subset** of EQ, Compressor and Reverb: 3 singles + 6 ordered
pairs + 6 ordered triples = **15 chains** per dry clip, each applied in its own order. The presence
label `[1,0,1]` cannot tell `EQ → Reverb` from `Reverb → EQ`, so `label` here is the **position** of
each effect: `[pos_EQ, pos_Comp, pos_Rev]`, `0` = absent, `1`/`2`/`3` = position in the chain.

| chain_class | chain | `label` (position) | `presence_label` | `slot_label` |
|---|---|---|---|---|
| 0 | EQ | `[1,0,0]` | `[1,0,0]` | `[1,0,0]` |
| 1 | Compressor | `[0,1,0]` | `[0,1,0]` | `[2,0,0]` |
| 2 | Reverb | `[0,0,1]` | `[0,0,1]` | `[3,0,0]` |
| 3 | EQ → Compressor | `[1,2,0]` | `[1,1,0]` | `[1,2,0]` |
| 4 | Compressor → EQ | `[2,1,0]` | `[1,1,0]` | `[2,1,0]` |
| 5 | EQ → Reverb | `[1,0,2]` | `[1,0,1]` | `[1,3,0]` |
| 6 | Reverb → EQ | `[2,0,1]` | `[1,0,1]` | `[3,1,0]` |
| 7 | Compressor → Reverb | `[0,1,2]` | `[0,1,1]` | `[2,3,0]` |
| 8 | Reverb → Compressor | `[0,2,1]` | `[0,1,1]` | `[3,2,0]` |
| 9 | EQ → Compressor → Reverb | `[1,2,3]` | `[1,1,1]` | `[1,2,3]` |
| 10 | EQ → Reverb → Compressor | `[1,3,2]` | `[1,1,1]` | `[1,3,2]` |
| 11 | Compressor → EQ → Reverb | `[2,1,3]` | `[1,1,1]` | `[2,1,3]` |
| 12 | Compressor → Reverb → EQ | `[3,1,2]` | `[1,1,1]` | `[2,3,1]` |
| 13 | Reverb → EQ → Compressor | `[2,3,1]` | `[1,1,1]` | `[3,1,2]` |
| 14 | Reverb → Compressor → EQ | `[3,2,1]` | `[1,1,1]` | `[3,2,1]` |

Extra columns (on top of the Phase 1 columns): `eq_position`, `compressor_position`, `reverb_position`
(the `label` as separate ints), `presence_label` (the old `[EQ, Comp, Rev]` presence vector;
`eq_label` / `compressor_label` / `reverb_label` stay presence), `slot_label` (effect at each slot,
1 = EQ, 2 = Compressor, 3 = Reverb, 0 = empty), `chain_class` (0–14, for a single 15-way classifier),
`num_effects`, and `parameter_set_id`. Sample ids keep the order: `1-Bass-Reverb-EQ`.

**Shared parameters** (`generation.share_parameters: true`): for one dry clip, every order of the same
effect set uses identical parameters — sampled once for the canonical order (EQ → Comp → Reverb) with the
Phase 1 seed — so order is the only difference between e.g. `EQ → Reverb` and `Reverb → EQ`. As a result
the 7 canonical-order wet files are byte-identical to the 7-chain Phase 1 dataset.

### Batch build

The full dataset is ~37 GB (≈13 GB WAV + ≈24 GB Parquet), so `run_batches.py` keeps only one batch of
audio on disk (15 songs, ≈5.5 GB peak):

```bat
python run_batches.py --config config_order.yaml                              :: all batches
python run_batches.py --config config_order.yaml --max-batches 1              :: one batch
python run_batches.py --config config_order.yaml --max-batches 1 --no-upload  :: test, keep files
```

Per batch: download its source stems from the Hub → render 15 chains per clip → delete the stems →
check metadata (all batches) + audio (this batch, incl. re-rendering) → write `data/<split>-bNN-*.parquet`
and the combined metadata/card → upload one commit (earlier batches stay) → delete the batch's audio.

- **Resume:** progress is in `output/phase1_effect_order/batches/bNN.json`. Re-running the same
  command skips uploaded batches and finishes an interrupted one (results are identical — every sample
  has its own seed). Keep this folder until the whole build is done.
- **Time:** each batch downloads ≈1.8 GB and uploads ≈3.7 GB; at ~0.7 MB/s that is ~2–2.5 h per batch.
- **Final check:** `python validate_phase1.py --config config_order.yaml --metadata-only`.
- The repo is created **private** (`huggingface.private: true`); switch it in the Hub settings if needed.

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
