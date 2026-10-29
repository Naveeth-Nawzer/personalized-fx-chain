# personalized-fx-chain — Inverse FX Chain Predictor

Given a **dry** recording and its **processed (wet)** version, predict the audio-effects chain that produced the
wet audio: **which effects** were applied (EQ, Compressor, Reverb), **in which order**, and **with which
parameters**. This branch (`Member1/Inverse_Fx_Chain`) contains the data pipeline, training notebooks and demos
for that component, up to and including effect-parameter prediction.

```
Dry audio + Wet audio ──> Inverse FX Chain Predictor ──> effects  +  order  +  parameters
                                                          [EQ, Comp, Rev]   Comp position   12 settings
```

## Pipeline so far

```
MUSDB18 stems (150 songs)
   │  Dataset_Push/        clean, number songs, song-level 70/15/15 split
   ▼
APINAJA/inverse-fx-source-audio-cleaned          150 songs × Bass / Drums / Mixture = 450 stems
   │  Effect_PreProcess/   10 s dry clips (448) + Pedalboard EQ / Compressor / Reverb
   ├─> APINAJA/inverse-fx-phase1-effect-type     7 chains,  3,136 dry/wet pairs  ─> model 1: effect type
   └─> APINAJA/inverse-fx-phase1-effect-order    15 chains, 6,720 dry/wet pairs  ─> model 2: presence + order
                                                                                 ─> model 3: presence + order + parameters
```

| stage | Hugging Face | built / trained with |
|---|---|---|
| Source audio | [dataset `inverse-fx-source-audio-cleaned`](https://huggingface.co/datasets/APINAJA/inverse-fx-source-audio-cleaned) | `Dataset_Push/` |
| Effect-type data (7 chains, presence label) | [dataset `inverse-fx-phase1-effect-type`](https://huggingface.co/datasets/APINAJA/inverse-fx-phase1-effect-type) | `Effect_PreProcess/` (`config.yaml`) |
| Effect-order data (15 chains, position label) | [dataset `inverse-fx-phase1-effect-order`](https://huggingface.co/datasets/APINAJA/inverse-fx-phase1-effect-order) | `Effect_PreProcess/` (`config_order.yaml`, `run_batches.py`) |
| Model 1 — effect type | [model `inverse-fx-phase1-logmel-transformer-v2`](https://huggingface.co/APINAJA/inverse-fx-phase1-logmel-transformer-v2) | `NoteBooks/phase-1_inverse-fx-chain-predictor.ipynb` |
| Model 2 — presence + order | [model `inverse-fx-order-logmel-transformer`](https://huggingface.co/APINAJA/inverse-fx-order-logmel-transformer) | `NoteBooks/phase1_inverse-fx-chain-predictor-effect-order.ipynb` |
| Model 3 — presence + order + parameters | [model `inverse-fx-order-params-logmel-transformer`](https://huggingface.co/APINAJA/inverse-fx-order-params-logmel-transformer) | fine-tuned from model 2 (training code `fxp3/` and results in the model repo) |

## Results on the test split (22 songs never seen in training)

| model | main result | trivial baseline |
|---|---|---|
| 1 — effect type | presence macro F1 **0.850**, exact match 0.474 | — |
| 2 — presence + order | 12-chain accuracy **0.472**, order 0.646, presence macro F1 0.914 | 0.133 / 0.5 / 0.846 |
| 3 — + parameters | 12-chain accuracy **0.539**, order 0.663, presence macro F1 0.932, parameter MAE **0.213** (41 % within tolerance) | 0.133 / 0.5 / 0.846 / 0.254 |

- **12 audible chains, not 15:** EQ and Reverb are both linear and commute, so `EQ → Reverb` and `Reverb → EQ` sound
  the same. Models 2 and 3 predict the 12 audibly distinct chains, and the only order that matters is where the
  Compressor sits.
- **What works best:** effect presence, especially the Compressor, plus EQ frequency / gain and Compressor threshold.
- **What is hard:** whether EQ comes before or after the Compressor (close to chance), and the Compressor time
  constants and Reverb parameters, which stay close to the mean-value baseline.

Full metrics, label tables and parameter ranges are in [`Effect_PreProcess/README.md`](Effect_PreProcess/README.md)
and in each model repo (`results/`).

## Repository layout

| folder | contents |
|---|---|
| [`Dataset_Push/`](Dataset_Push) | builds the cleaned source dataset (audiofolder layout, song-level splits) from `Dataset/` + `Dataset-02/` and pushes it to the Hub; `locked_sources.csv` keeps published ids and splits stable |
| [`Effect_PreProcess/`](Effect_PreProcess) | generates, validates, exports (Parquet) and pushes the dry/wet effect datasets with Pedalboard; batch build for the 15-chain dataset — **see its [README](Effect_PreProcess/README.md) for the full guide** |
| [`NoteBooks/`](NoteBooks) | Kaggle notebooks: model training (effect type, presence + order) and Gradio test apps for models 1–3 |
| [`Training_notebooks/`](Training_notebooks) | first versions of the effect-type Log-Mel Transformer notebook |
| [`Project_Code/RP_progres_Repo/`](Project_Code/RP_progres_Repo) | earlier project code: Phase 1 config, dataset scripts, `src/` package and tests |

Local data folders (`Dataset/`, `Dataset-02/`, `Dataset_Push/build/`, `Effect_PreProcess/data/`,
`Effect_PreProcess/output/`) are git-ignored; the data lives on the Hugging Face Hub.

## Notebooks

| notebook | purpose | model |
|---|---|---|
| `phase-1_inverse-fx-chain-predictor.ipynb` | train the effect-type model (3 sigmoid outputs) | 1 |
| `phase1_logmel_transformer.ipynb` | first version of the effect-type training notebook | `inverse-fx-phase1-logmel-transformer` |
| `phase1_inverse-fx-chain-predictor-effect-order.ipynb` | train the two-head presence + order model | 2 |
| `phase-1_fx_predictor_gradio_demo.ipynb` | Gradio test app: effect type | 1 |
| `phase1_gradio-test-app-effect-presence-order.ipynb` | Gradio test app: presence + order | 2 |
| `phase1_inverse-fx-predictor-gradio-demo-3-heads.ipynb` | Gradio test app: presence + order + parameters | 3 |

All notebooks run on Kaggle: training uses **GPU T4 × 2**, the demos run on CPU. They need the Kaggle secret
`HF_TOKEN` and **Internet ON**. The training notebooks resume automatically from the newest checkpoint after an
interruption. Each demo prints a temporary public `gradio.live` link.

## Quick start (data pipeline, Windows)

```bat
cd /d "D:\Y4S1(jul-Dec-2026)\Research-Project\Effect_PreProcess"
.venv\Scripts\activate

:: 7-chain effect-type dataset
python download_source.py
python generate_phase1.py --overwrite
python validate_phase1.py
python export_parquet.py
python push_phase1.py

:: 15-chain effect-order dataset (batch by batch: download -> generate -> upload -> delete)
python run_batches.py --config config_order.yaml
```

The Hugging Face token is read from `Dataset/.env` (`HF_TOKEN=hf_...`); this file is git-ignored.

## License

The audio is derived from [MUSDB18](https://sigsep.github.io/datasets/musdb.html) and may be used for
**non-commercial research only**, under the original dataset terms.
