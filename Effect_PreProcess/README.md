# Effect_PreProcess — dry/wet effect datasets for the Inverse FX Chain Predictor

This folder turns the clean source stems into paired **dry / wet** audio with **Pedalboard** (EQ, Compressor,
Reverb) and publishes the datasets that the Phase 1 models are trained on. Every effect parameter used for every
wet file is stored with it, so the same datasets serve effect-type, effect-order and effect-parameter
training.

## Status (up to effect-parameter training)

```
source stems (150 songs x Bass/Drums/Mixture)                  APINAJA/inverse-fx-source-audio-cleaned
   │  10 s non-silent clip, mono, 44.1 kHz, -20 LUFS  (448 dry clips)
   ├─ 7 chains  (fixed order EQ → Comp → Reverb)   3,136 pairs  APINAJA/inverse-fx-phase1-effect-type
   │     └─ model 1  effect TYPE                    (3 sigmoid outputs)
   └─ 15 chains (every order of every subset)       6,720 pairs  APINAJA/inverse-fx-phase1-effect-order
         ├─ model 2  presence + ORDER               (two heads)
         └─ model 3  presence + ORDER + PARAMETERS  (three heads, fine-tuned from model 2)
```

| stage | data / model on the Hub | notebook (`../NoteBooks/`) | state |
|---|---|---|---|
| Source dataset | [`inverse-fx-source-audio-cleaned`](https://huggingface.co/datasets/APINAJA/inverse-fx-source-audio-cleaned) (built by `../Dataset_Push`) | — | done: 150 songs, 450 stems |
| Effect-type dataset | [`inverse-fx-phase1-effect-type`](https://huggingface.co/datasets/APINAJA/inverse-fx-phase1-effect-type) | — | done: 3,136 pairs |
| Effect-order dataset | [`inverse-fx-phase1-effect-order`](https://huggingface.co/datasets/APINAJA/inverse-fx-phase1-effect-order) | — | done: 6,720 pairs |
| Model 1 — effect type | [`inverse-fx-phase1-logmel-transformer-v2`](https://huggingface.co/APINAJA/inverse-fx-phase1-logmel-transformer-v2) | `phase-1_inverse-fx-chain-predictor.ipynb` (demo: `phase-1_fx_predictor_gradio_demo.ipynb`) | trained |
| Model 2 — presence + order | [`inverse-fx-order-logmel-transformer`](https://huggingface.co/APINAJA/inverse-fx-order-logmel-transformer) | `phase1_inverse-fx-chain-predictor-effect-order.ipynb` (demo: `phase1_gradio-test-app-effect-presence-order.ipynb`) | trained |
| Model 3 — presence + order + parameters | [`inverse-fx-order-params-logmel-transformer`](https://huggingface.co/APINAJA/inverse-fx-order-params-logmel-transformer) | demo: `phase1_inverse-fx-predictor-gradio-demo-3-heads.ipynb` (the training code `fxp3/` and results are in the model repo) | trained |

`phase1_logmel_transformer.ipynb` and `../Training_notebooks/` hold the first version of model 1
(`APINAJA/inverse-fx-phase1-logmel-transformer`).

---

## Datasets

All datasets share the same dry clips, splits and seed:

- **Dry clip:** 10 s window with at least 50 % non-silent frames, mono, 44.1 kHz, LUFS-normalised to −20 LUFS
  (peak ≤ −1 dBFS), 32-bit float WAV. The wet audio is never normalised.
- **Splits:** inherited from the source dataset, split by **song** — train 105 songs (70 %), validation 23 (15 %),
  test 22 (15 %). Every wet version of a clip is in the same split as its dry clip, so no song leaks.
- **Skipped:** `101-Bass` and `124-Bass` have no 10 s window that is at least 50 % non-silent
  (`metadata/skipped.csv`), so 450 stems give 448 dry clips.
- **Reproducible:** `generation.seed: 42` controls every clip window and every effect parameter; regenerating
  gives byte-identical audio and metadata.

### Effect-type dataset (7 chains)

[`APINAJA/inverse-fx-phase1-effect-type`](https://huggingface.co/datasets/APINAJA/inverse-fx-phase1-effect-type) —
`config.yaml` (`generation.chain_mode: subset`). The 7 non-empty subsets, always processed in the fixed order
EQ → Compressor → Reverb. Label = **presence** `[EQ, Comp, Rev]`.

| # | effect_chain | label `[EQ, Comp, Rev]` |
|---|---|---|
| 1 | EQ | `[1,0,0]` |
| 2 | Compressor | `[0,1,0]` |
| 3 | Reverb | `[0,0,1]` |
| 4 | EQ + Compressor | `[1,1,0]` |
| 5 | EQ + Reverb | `[1,0,1]` |
| 6 | Compressor + Reverb | `[0,1,1]` |
| 7 | EQ + Compressor + Reverb | `[1,1,1]` |

448 clips × 7 = **3,136 pairs**: train 2,191 / validation 483 / test 462.

### Effect-order dataset (15 chains)

[`APINAJA/inverse-fx-phase1-effect-order`](https://huggingface.co/datasets/APINAJA/inverse-fx-phase1-effect-order) —
`config_order.yaml` (`generation.chain_mode: ordered`). **Every order of every non-empty subset**: 3 singles +
6 ordered pairs + 6 ordered triples = 15 chains per clip, each applied in its own order. The presence label cannot
tell `EQ → Reverb` from `Reverb → EQ`, so `label` is the **position** of each effect,
`[pos_EQ, pos_Comp, pos_Rev]` (`0` = absent, `1`/`2`/`3` = position in the chain).

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

448 clips × 15 = **6,720 pairs**: train 4,695 / validation 1,035 / test 990 (448 pairs per chain). 26 GB on the Hub.

- **Shared parameters** (`generation.share_parameters: true`): for one dry clip, every order of the same effect set
  uses identical parameters, sampled once for the canonical order with the Phase 1 seed. Order is therefore the
  only difference between e.g. `EQ → Reverb` and `Reverb → EQ`, and the 7 canonical-order wet files are
  byte-identical to the effect-type dataset. `parameter_set_id` names the rows that share parameters.
- **Only 12 of the 15 chains sound different.** EQ (a peaking filter) and Reverb are both linear, so they commute:
  `EQ → Reverb` = `Reverb → EQ`, `EQ → Reverb → Comp` = `Reverb → EQ → Comp` and
  `Comp → EQ → Reverb` = `Comp → Reverb → EQ` (difference ≈ −89 dB, checked in the model-2 notebook). The labels
  still record the true processing order; models 2 and 3 predict the **12 audibly distinct chains**, and a perfect
  model can reach at most 80 % on the 15-chain view. The only audible order questions are *where the Compressor
  sits*: "EQ before Compressor?" and "Reverb before Compressor?".

Extra columns compared with the effect-type dataset: `eq_position`, `compressor_position`, `reverb_position`
(the `label` as ints), `presence_label` (`eq_label` / `compressor_label` / `reverb_label` stay presence),
`slot_label` (effect at each slot: 1 = EQ, 2 = Compressor, 3 = Reverb, 0 = empty), `chain_class` (0–14),
`num_effects` and `parameter_set_id`. Sample ids keep the order: `1-Bass-Reverb-EQ`.

### Effect parameters (targets for parameter training)

Each effect samples its parameters per pair from the ranges in `config.yaml` (log scale where marked). An effect is
re-sampled (up to 10 times) if it changes the signal by less than −30 dB, so every effect is audible. Values are
stored physically (`eq_params`, `compressor_params`, `reverb_params`, `parameters`) and scaled to 0–1
(`parameters_normalized`, log-aware); the normalised values are the regression targets of model 3.

| effect | parameter | range | scale | Pedalboard argument |
|---|---|---|---|---|
| EQ | `frequency` | 100 – 8,000 Hz | log | `PeakFilter(cutoff_frequency_hz)` |
| EQ | `gain` | ±3 – 12 dB (random sign; normalised over −12 – 12) | linear | `PeakFilter(gain_db)` |
| EQ | `q` | 0.5 – 4 | log | `PeakFilter(q)` |
| Compressor | `threshold` | −40 – −15 dB | linear | `Compressor(threshold_db)` |
| Compressor | `ratio` | 2 – 10 | log | `Compressor(ratio)` |
| Compressor | `attack` | 1 – 50 ms | log | `Compressor(attack_ms)` |
| Compressor | `release` | 50 – 500 ms | log | `Compressor(release_ms)` |
| Compressor | `makeup_gain` | 0 – 6 dB | linear | `Gain(gain_db)` after the compressor |
| Reverb | `room_size` | 0.3 – 0.9 | linear | `Reverb(room_size)` |
| Reverb | `damping` | 0.2 – 0.8 | linear | `Reverb(damping)` |
| Reverb | `wet_mix` | 0.15 – 0.5 | linear | `Reverb(wet_level)` |
| Reverb | `dry_mix` | 0.6 – 0.9 | linear | `Reverb(dry_level)` |
| Reverb | `width` | fixed 1.0 | — | `Reverb(width)` (not a training target) |

---

## Models trained so far (test split, 22 unseen songs)

All three use the same backbone: dry and wet Log-Mel (n_fft 2048, hop 512, 128 mels, 20 Hz – 20 kHz, 862 frames
per 10 s) → fusion → Linear → 256 → `[CLS]` + 4-layer Transformer encoder (4 heads, FF 512), about 2.2–2.3 M
parameters. They were trained on Kaggle with 2 × T4 GPUs (DDP), and each pins the dataset revision it used.

**Model 1 — effect type** (`fusion concat`, 3 sigmoid outputs, BCE; dataset revision `b95597a`, best epoch 6):

| metric | test |
|---|---|
| presence macro F1 | 0.850 |
| exact match (= 7-combination accuracy) | 0.474 |
| F1 EQ / Compressor / Reverb | 0.709 / 0.963 / 0.876 |

**Model 2 — presence + order** (`fusion concat_diff` = dry ‖ wet ‖ wet − dry; presence head + order head
"EQ before Comp", "Reverb before Comp"; joint decoding over the 12 audible chains; dataset revision `a5d0eef`,
best epoch 27):

| metric | test | trivial baseline |
|---|---|---|
| 12-chain accuracy | 0.472 | 0.133 (majority class) |
| 15-chain accuracy | 0.366 | ceiling 0.800 |
| order accuracy (EQ < Comp / Rev < Comp) | 0.646 (0.557 / 0.735) | 0.5 (coin flip) |
| presence macro F1 | 0.914 | 0.846 (always "present") |

**Model 3 — presence + order + parameters** (model 2 plus a new head of 12 sigmoid outputs = the
`parameters_normalized` values; L1 loss only on effects that are present; fine-tuned from model 2; best epoch 21 by
validation combined score):

| metric | test | trivial baseline |
|---|---|---|
| 12-chain accuracy | 0.539 | 0.133 |
| 15-chain accuracy | 0.420 | ceiling 0.800 |
| order accuracy (EQ < Comp / Rev < Comp) | 0.663 (0.568 / 0.758) | 0.5 |
| presence macro F1 | 0.932 | 0.846 |
| parameter MAE (0–1, mean over 12) | 0.213 | 0.254 (predict the training mean) |
| parameter skill (1 − MAE / baseline MAE) | 0.160 | 0 |
| parameters within tolerance | 41.2 % | — |

Model 3 parameters on test (726 pairs per parameter, effects that are really present):

| parameter | typical error | within tolerance (model / baseline) | R² |
|---|---|---|---|
| EQ frequency | 0.95 octave | 38.7 % / 16.1 % (±0.5 oct) | 0.44 |
| EQ gain | 4.25 dB | 24.1 % / 0.0 % (±1.5 dB) | 0.50 |
| EQ Q | ×1.50 | 59.8 % / 38.3 % (×1.5) | 0.23 |
| Comp threshold | 3.50 dB | 51.2 % / 26.9 % (±3 dB) | 0.57 |
| Comp ratio | ×1.43 | 60.5 % / 51.0 % (×1.5) | 0.15 |
| Comp attack | ×2.43 | 44.5 % / 38.7 % (×2) | 0.05 |
| Comp release | ×1.81 | 35.1 % / 31.8 % (×1.5) | −0.05 |
| Comp makeup gain | 1.38 dB | 37.6 % / 34.3 % (±1 dB) | 0.11 |
| Rev room size | 0.13 | 39.1 % / 35.0 % (±0.1) | 0.05 |
| Rev damping | 0.16 | 29.9 % / 28.7 % (±0.1) | −0.10 |
| Rev wet mix | 0.08 | 36.9 % / 30.9 % (±0.05) | 0.12 |
| Rev dry mix | 0.07 | 37.3 % / 32.8 % (±0.05) | 0.02 |

What the results show so far:

- **Presence is learned well**, especially Compressor (F1 0.98 in model 3); EQ is the hardest effect.
- **Order is harder:** "Reverb before Compressor" reaches 0.76, but "EQ before Compressor" is close to chance (0.57).
  The triples `EQ → Comp → Reverb` and `Reverb → Comp → EQ` are almost never predicted correctly.
- **Parameters:** EQ frequency / gain and Compressor threshold are clearly learned (R² 0.44–0.57). Compressor
  time constants and all Reverb parameters are close to the mean-value baseline.

The full metrics, plots and test predictions are in each model repo (`results/`).

---

## Files

| file | purpose |
|---|---|
| `config.yaml` | all settings: repos, categories, audio, preprocessing, seed, parameter ranges; `chain_mode: subset` (7 chains) |
| `config_order.yaml` | effect-order dataset: only the overrides of `config.yaml` (`extends: config.yaml`, `chain_mode: ordered`) |
| `download_source.py` | download the source dataset to `data/source/` (read only) |
| `generate_phase1.py` | build dry clips + every chain + metadata (7 or 15 chains, by `chain_mode`) |
| `validate_phase1.py` | check labels, ids, splits, shared parameters, audio, and re-render pairs from stored parameters |
| `export_parquet.py` | write Parquet shards with `dry_audio` / `wet_audio` as `Audio` (playable in the Hub viewer) |
| `push_phase1.py` | upload a fully generated dataset folder to the Hub |
| `run_batches.py` | batch build: download → generate → check → upload → delete, one batch of songs at a time |
| `fx.py` | Pedalboard effects, parameter sampling, chains, labels |
| `audio_utils.py` | audio I/O and dry-clip preprocessing |
| `common.py` | config loading (incl. `extends`), token, CSV helpers |

The Hugging Face token is read from `../Dataset/.env` (`HF_TOKEN=...`).

---

## Guide: effect-type dataset, 7 chains (Windows Command Prompt)

### 0. Open cmd in this folder

```bat
cd /d "D:\Y4S1(jul-Dec-2026)\Research-Project\Effect_PreProcess"
```

### 1. Virtual environment (one time only)

A `.venv` (Python 3.12) with all libraries is already created here. To recreate it on another machine:

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
If disk space is short, point `paths.source_dir` at an existing local copy with the same layout, e.g.
`../Dataset_Push/build/inverse-fx-source-audio-cleaned`, or use the batch build below.

### 3. Generate

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

Writes `data/<split>-*.parquet`, where `dry_audio` and `wet_audio` are Hugging Face `Audio` columns (WAV bytes +
original path) and all other columns are unchanged, and points the dataset card's `configs` at them. Every value
is read back and checked against the CSV/WAV files. Without this step the viewer shows the two columns as plain
`string` paths.

### 6. Push to Hugging Face

```bat
python push_phase1.py
```

Creates the dataset repo `APINAJA/inverse-fx-phase1-effect-type` if needed (private on creation) and uploads
everything; re-running after a regeneration replaces the old files on the Hub. To use a different name:

```bat
python push_phase1.py --repo-id APINAJA/your-repo-name
```

---

## Guide: effect-order dataset, 15 chains (batch build)

The full dataset is ~26 GB (WAV + Parquet), so `run_batches.py` keeps only one batch of audio on disk
(15 songs per batch, 10 batches, ≈5.5 GB peak):

```bat
python run_batches.py --config config_order.yaml                              :: all batches
python run_batches.py --config config_order.yaml --max-batches 1              :: one batch
python run_batches.py --config config_order.yaml --max-batches 1 --no-upload  :: test, keep files
```

Per batch: download its source stems from the Hub → render 15 chains per clip → delete the stems →
check the metadata of all batches + the audio of this batch (incl. re-rendering) → write
`data/<split>-bNN-*.parquet` and the combined metadata / dataset card → upload one commit (earlier batches stay)
→ delete the batch's audio.

- **Resume:** progress is kept in `output/phase1_effect_order/batches/bNN.json`. Re-running the same command
  skips uploaded batches and finishes an interrupted one; results are identical because every sample has its own
  seed. Keep this folder until the whole build is done.
- **Time:** each batch downloads ≈1.8 GB and uploads ≈2.5 GB, about 1 h 10 min at ~0.8 MB/s upload.
- **Final check:** `python validate_phase1.py --config config_order.yaml --metadata-only`.
- New repos are created private (`huggingface.private`); visibility can be changed in the Hub settings.

---

## Output layout

```
output/phase1_effect_type/                 (the effect-order build has the same layout, shards named <split>-bNN-*)
├── README.md                              dataset card (shown on the Hub)
├── data/
│   ├── train-0000N-of-0000M.parquet       viewer / load_dataset (Audio columns)
│   ├── validation-…parquet, test-…parquet
│   ├── train/dry/2-Bass.wav               dry clip (shared by all its wet versions)
│   ├── train/wet/2-Bass-EQ.wav ... 2-Bass-EQ-Compressor-Reverb.wav
│   ├── validation/...
│   └── test/...
└── metadata/
    ├── train.csv / validation.csv / test.csv   one row per dry/wet pair
    ├── sources.csv                             how each dry clip was cut and normalised
    ├── skipped.csv / failures.csv
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
| `label` | `[1,1,0]` (presence; position in the effect-order dataset) |
| `eq_label`, `compressor_label`, `reverb_label` | `1`, `1`, `0` |
| `eq_params` | `{"frequency":1188.0914,"gain":-6.3806,"q":1.6983}` |
| `compressor_params` | `{"threshold":-37.53,"ratio":4.95,"attack":1.26,"release":180.13,"makeup_gain":0.24}` |
| `reverb_params` | empty (effect not in the chain) |
| `parameters` / `parameters_normalized` | all used parameters, physical and scaled to [0, 1] |
| `effect_change_db` | how much each effect changed the signal (audibility check) |
| `sample_seed` | per-sample seed derived from `generation.seed` |

---

## Common changes (edit `config.yaml`, then generate, validate, export and push again)

| want | change |
|---|---|
| only Bass + Drums | `source.categories: [Bass, Drums]` |
| longer clips | `audio.duration: 20.0` |
| stereo | `audio.channels: 2` |
| same parameters for every sample | `generation.parameter_mode: fixed` |
| different random parameters | `generation.seed: 123` |
| every order of every subset (15 chains) | use `config_order.yaml` (`generation.chain_mode: ordered`) |
| independent parameters per order | `generation.share_parameters: false` in `config_order.yaml` |
| public repo | `huggingface.private: false` (only applies when the repo is first created) |

Changing any of these produces a different dataset; train on a new repo or a new pinned revision, so existing
models stay reproducible.

## Troubleshooting

- **`HF_TOKEN not found`** — check `../Dataset/.env` contains `HF_TOKEN=hf_...`, or run `set HF_TOKEN=hf_...` first.
- **`already contains data`** — add `--overwrite` to regenerate.
- **`Source dataset not found`** — run `python download_source.py` first.
- **`holds batches built with different settings`** (`run_batches.py`) — the config changed after a batch was
  built; restore the settings or use a new `output_dir`.
- **`pyarrow` fails with "An Application Control policy has blocked this file"** — newer pyarrow builds are
  blocked on this machine; keep the pinned `pyarrow==21.0.0` from `requirements.txt`.
- **`import pedalboard` fails with a DLL error** — Windows Smart App Control / antivirus may block the native
  module; allow it, or recreate the venv with `py -3.12`.
