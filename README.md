# DAPN: Differentiable Audio Processing Network

> **Project Context:**  
> DAPN is the differentiable DSP audio processing environment component of the undergraduate research project:  
> *"Collaborative Audio Effects Chain Synthesis via Differentiable Reinforcement Learning and Human-in-the-Loop Personalisation"*

---

> [!IMPORTANT]  
> **Scope Notice (MVP Stage):**  
> This repository contains only the **DAPN MVP** (differentiable audio processing network, canonical FX schema, parameter mapping, and unit tests).  
> The **GRRL-Agent**, **Inverse FX Chain Predictor**, **PPO/GAE Reinforcement Learning**, **PTA / preference learning / XAI**, and **Transformer models** are future integration stages and are intentionally **NOT** implemented in this repository.

---

## 1. Architecture Overview

DAPN provides a clean, PyTorch-native, end-to-end differentiable audio effects execution environment.

```
                    ┌─────────────────────────┐
                    │ Canonical FX-Chain Dict │
                    └───────────┬─────────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │   FXChainValidator   │ (checks schema, ranges, lengths)
                     └──────────┬───────────┘
                                │ FXChainSpec
                                ▼
                     ┌──────────────────────┐
      Audio Tensor ─►│       FXChain        │
      (B, C, T)      └──────────┬───────────┘
                                │ Sequential Execution
                     ┌──────────┴───────────┐
                     │  1. EQ               │  (biquad peaking filter via TIR)
                     │  2. Compressor       │  (soft-knee RMS feedforward)
                     │  3. Delay            │  (fractional-delay multi-tap)
                     │  4. Reverb           │  (Schroeder comb + all-pass)
                     └──────────┬───────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │ Processed Audio Tensor  │ ──► Loss ──► loss.backward()
                    │ (same shape, float32)   │              (Autograd preserved)
                    └─────────────────────────┘
```

### Module Structure

```
├── config/
│   ├── system_config.yaml      # Sample rate, channels, max duration, chain limits
│   └── effects_config.yaml     # 8 effect specs: ranges, units, mapping types
├── common/
│   ├── audio_types.py          # Canonical types: AudioMeta, EffectParameterSpec
│   ├── config_loader.py        # Centralized YAML loader with LRU caching
│   ├── fx_schema.py            # FXChainValidator and FXChainSpec dataclasses
│   ├── parameter_mapper.py     # Differentiable normalized [0, 1] <-> physical DSP mapper
│   └── validation.py           # Audio tensor runtime validation (shape, finiteness)
├── dapn/
│   ├── base_effect.py          # Abstract BaseEffect class (nn.Module)
│   ├── fx_chain.py             # Sequential FXChain executor with param_override support
│   ├── dapn.py                 # Top-level DAPN facade module
│   └── effects/
│       ├── eq.py               # Differentiable Parametric Equalizer
│       ├── compressor.py       # Differentiable Soft-Knee Compressor
│       ├── delay.py            # Differentiable Tape Delay
│       └── reverb.py           # Differentiable Schroeder Reverb
├── audio_io.py                 # Standard WAV I/O (pure Python wave module, outside autograd)
├── examples/
│   └── run_basic_dapn.py       # Runnable end-to-end experiment with gradient verification
└── tests/                      # 78 unit and integration tests (100% pass)
```

---

## 2. Configuration Explanation

All numeric constants, sample rates, durations, and physical ranges are driven by configuration files. **Nothing is hard-coded in the Python implementation.**

### `config/system_config.yaml`
```yaml
audio:
  sample_rate: 44100       # Hz
  channels: 1              # Mono
  audio_format: WAV
  dtype: float32
  max_duration_seconds: 10 # Pipeline rejects files longer than this

chain:
  min_chain_length: 1
  max_chain_length: 8
  allow_repeated_effects: false
```

### `config/effects_config.yaml`
Defines 8 effects: `EQ`, `Compressor`, `Reverb`, `Distortion`, `Delay`, `Chorus`, `Flanger`, and `STOP`.

For each parameter, it specifies:
- `normalized_range`: `[0.0, 1.0]` (the ML/agent facing interface)
- `physical_range`: the DSP physical unit range (e.g. `[-12.0, 12.0]` dB for EQ gain)
- `unit`: physical unit string (`Hz`, `dB`, `ms`, `s`, `dimensionless`)
- `mapping_type`: `linear` (with architectural support for `log`)
- `required`: boolean flag

> **Note on STOP**: `STOP` is defined in the config as a reserved terminal action for the future GRRL-Agent. It has no audio parameters and DAPN rejects it if included in an audio processing chain.

---

## 3. Canonical FX-Chain Schema

Chains are specified as JSON/dict objects adhering to the following contract:

```json
{
  "chain": [
    {
      "effect": "EQ",
      "parameters": {
        "frequency": 0.35,
        "gain": 0.75,
        "q": 0.40
      }
    },
    {
      "effect": "Compressor",
      "parameters": {
        "threshold": 0.50,
        "ratio": 0.40,
        "attack": 0.20,
        "release": 0.30,
        "makeup_gain": 0.35
      }
    },
    {
      "effect": "Delay",
      "parameters": {
        "delay_time": 0.05,
        "feedback": 0.35,
        "wet_mix": 0.60,
        "dry_mix": 0.40
      }
    }
  ]
}
```

### Validation Rules (enforced by `FXChainValidator`):
1. **Unknown effects** are rejected with an explicit exception.
2. **Unknown parameters** are rejected with an explicit exception.
3. **Parameters outside [0, 1]** are rejected.
4. **Missing required parameters** are rejected.
5. **Chain length** outside `[min_chain_length, max_chain_length]` is rejected.
6. **Repeated effects** are rejected when `allow_repeated_effects: false`.
7. **STOP** is rejected when passed to DAPN for audio processing.

---

## 4. Parameter Normalization & Mapping

The `ParameterMapper` (`common/parameter_mapper.py`) is the **single source of truth** for all parameter conversions:

$$\text{physical} = \text{physical\_min} + \text{clamped} \times (\text{physical\_max} - \text{physical\_min})$$

$$\text{normalized} = \frac{\text{physical} - \text{physical\_min}}{\text{physical\_max} - \text{physical\_min}}$$

- Uses `torch.clamp` in the differentiable path (soft-clamp preserves autograd gradients in $[0, 1]$ and produces $0$ gradient outside).
- Zero NumPy inside the forward path.
- Individual effects **never** duplicate this mapping logic; they call `self.map_to_physical(name, tensor)` provided by `BaseEffect`.

---

## 5. Differentiability and Autograd Design

DAPN maintains an unbroken computational graph from loss all the way back to the effect parameters:

1. **EQ (`dapn/effects/eq.py`)**:
   - Calculates Audio EQ Cookbook peaking biquad coefficients using PyTorch math.
   - Computes a truncated impulse response (TIR) via Direct-Form II recurrence.
   - Convolves audio via `F.conv1d`. Gradients flow back through the TIR to the frequency, gain, and Q parameters.
2. **Compressor (`dapn/effects/compressor.py`)**:
   - Replaces the non-differentiable hard-knee `min(0, ...)` with a smooth soft-knee `F.softplus` hinge and sigmoid interpolation.
   - Differentiable across `threshold`, `ratio`, `attack`, `release`, and `makeup_gain`.
3. **Delay (`dapn/effects/delay.py`)**:
   - Implements fractional-sample delay using linear interpolation between adjacent samples.
   - Differentiable w.r.t. `delay_time`, `feedback`, `wet_mix`, and `dry_mix`.
4. **Reverb (`dapn/effects/reverb.py`)**:
   - Differentiable Schroeder reverberator with 8 parallel comb filters and 2 all-pass diffusers.
   - Differentiable w.r.t. `room_size`, `damping`, `decay`, `wet_mix`, and `dry_mix`.

---

## 6. How to Run

### Installation
```bash
pip install torch torchaudio pyyaml pytest
```

### Running the Basic DAPN Experiment
```bash
python examples/run_basic_dapn.py
```

Expected output:
```
=================================================================
 DAPN MVP: Differentiable Audio Processing Network Demonstration
=================================================================

[Step 1] Preparing dry audio...
  Saved dry signal to: .../audio/dry.wav

[Step 2] Loading dry audio via audio_io...
  Loaded audio: shape=(1, 22050), sample_rate=44100 Hz
  Dry audio successfully loaded: OK

[Step 3] Defining canonical FX chain (EQ -> Compressor -> Delay)...
  FX chain defined: EQ -> Compressor -> Delay

[Step 4] Creating learnable parameter tensors (requires_grad=True)...

[Step 5] Validating FX chain and running DAPN forward pass...
  Processed audio: shape=(1, 22050), finite=True

[Step 6] Saving wet audio...
  Saved wet audio to: .../audio/wet.wav

[Step 7] Computing loss (MSE against half-amplitude dry target)...
  Loss value: 0.029014

[Step 8] Executing backward pass: loss.backward()...
  Backward pass completed successfully.

[Step 9] Gradient Validation Results:
-------------------------------------------------------
  EQ frequency              : PASSED (non-zero: -2.777543e-04)
  EQ gain                   : PASSED (non-zero: +3.409702e-04)
  EQ Q                      : PASSED (non-zero: -2.007866e-04)
  Compressor threshold      : PASSED (non-zero: +4.757722e-02)
  Compressor ratio          : PASSED (non-zero: -4.910794e-03)
  Compressor makeup         : PASSED (non-zero: +1.069851e-02)
  Delay delay_time          : PASSED (non-zero: -2.994070e+00)
  Delay feedback            : PASSED (non-zero: +6.985292e-03)
  Delay wet_mix             : PASSED (non-zero: +9.375209e-03)
  Delay dry_mix             : PASSED (non-zero: +5.296795e-03)
-------------------------------------------------------
  ✓ All gradient checks PASSED. Graph is fully differentiable.
=================================================================
 DAPN MVP execution completed successfully!
=================================================================
```

### Running the Test Suite
```bash
python -m pytest -v
```
All 78 unit and integration tests will execute and pass.

---

## 7. Current Limitations

1. **Mono Processing**: Default configuration is 1-channel mono. Stereo expansion is architecturally supported via the `channels: 2` config flag and `(B, C, T)` shape contract, but effect internals currently operate on single or grouped channels.
2. **FIR Truncation for IIR EQ**: Biquad IIR filters are approximated via 2048-sample truncated impulse responses (TIR) to preserve autograd efficiency. Very low frequency boosts (<30 Hz) have limited resolution with 2048 samples at 44.1 kHz.
3. **Delay Tap Count**: The delay feedback loop is truncated to $N=8$ taps as a geometric series for differentiable execution without unbounded stateful loops.
4. **Reverb Diffusion**: Schroeder reverb uses fixed all-pass feedback ($g=0.5$) with 8 fixed-ratio comb delay lengths scaled by `room_size`.
5. **Linear Mapping Only**: `mapping_type: linear` is active for MVP. The schema and mapper include an architectural hook for `log` frequency mapping to be enabled in Phase 2 of research.

---

## 8. Future Integration Roadmap

```
Phase A: DAPN MVP (CURRENT - COMPLETED)
   └── Validated differentiable DSP core, schema, and gradient propagation.

Phase B: Inverse FX Chain Predictor (FUTURE)
   └── Deep acoustic feature extractor predicting structured FXChainSpec from audio targets.

Phase C: GRRL-Agent (FUTURE)
   └── RL agent (PPO/GAE) selecting discrete effect ordering + continuous parameters with STOP action.

Phase D: Human-in-the-Loop Personalisation & PTA (FUTURE)
   └── Preference learning, pairwise feedback, and explainability (XAI).
```
