# Interface Contract: Personalized Taste Adapter (PTA) ↔ GRRL Agent (M3)

## Status: Provisional Production Contract
**Version:** 1.0  
**Date:** October 2026  
**Components Involved:** 
- Member 4: Personalized Taste Adapter (PTA)
- Member 3: Goal-Driven Reinforcement Learning Agent (GRRL)
- Member 2: Differentiable Audio Processing Network (DAPN)
- Member 1: Inverse Acoustic Predictor (Feature Extractor)

---

## 1. System Architecture & Nomenclature

```
[Dry Audio + Reference] ──► [M1: Inverse Predictor] ──► Warm-start Chain / Features
                                                              │
                                                              ▼
[PTA Guidance (z_taste, bounds)] ──────────────────────► [M3: GRRL Agent]
                                                              │
                                                              ▼ Action / Chain Steps
                                                         [M2: DAPN Renderer]
                                                              │
                                                              ▼ Audio Output
[User Feedback] ◄─────────────────────────────────────────────┘
      │
      ▼
[PTA Engine] ──► Update Preference ──► Build Next Guidance ──► M3
```

- **M1:** Inverse Acoustic Predictor (Warm-start effect & parameter estimator)
- **M2:** Differentiable Audio Processing Network (DAPN audio rendering engine)
- **M3:** Goal-Driven Reinforcement Learning Agent (GRRL policy & chain constructor)
- **M4 / PTA:** Personalized Taste Adapter (Online preference learning, guidance adapter, XAI)

---

## 2. Preference Representation (7-Dimensional Vector)

The preference vector represents relative user taste shifts with respect to a neutral baseline (`0.0`). It does **not** specify raw DSP parameter values.

### Canonical Dimension Order
The vector order is immutable across all systems:
1. `brightness` (index 0) — Spectral brightness vs. darkness
2. `compression` (index 1) — Dynamic range compression strength
3. `reverb` (index 2) — Reverberant space and decay
4. `delay` (index 3) — Echo repeats and delay wetness
5. `saturation` (index 4) — Non-linear harmonic drive/distortion
6. `width` (index 5) — Perceived stereo width (see Mono Pipeline note)
7. `warmth` (index 6) — Low-mid tonal richness

### Value Domain
- Each element $P_d \in [-1.0, 1.0]$
- $0.0$: Neutral / No preference
- $+1.0$: Strong preference for more
- $-1.0$: Strong preference for less
- Values must be finite floating-point numbers (`NaN` and `Inf` are rejected).

---

## 3. Seven-to-Sixteen Taste Projection ($z_{taste} \in \mathbb{R}^{16}$)

The GRRL agent observation space accepts `pta_taste_dim = 16`. PTA projects the 7-dimensional preference vector into $\mathbb{R}^{16}$ using a canonical formulation:

$$z_{taste} = \left[ P_{0..6}, \, I_{0..5}, \, A_{0..2} \right]$$

### 1. Raw Preferences ($z[0..6]$)
- $z[0] = P_{\text{brightness}}$
- $z[1] = P_{\text{compression}}$
- $z[2] = P_{\text{reverb}}$
- $z[3] = P_{\text{delay}}$
- $z[4] = P_{\text{saturation}}$
- $z[5] = P_{\text{width}}$
- $z[6] = P_{\text{warmth}}$

### 2. Pairwise Interaction Terms ($z[7..12]$)
Captures key acoustic interdependencies between primary processor dimensions:
- $z[7] = P_{\text{compression}} \times P_{\text{reverb}}$ (Dynamic density vs. acoustic space)
- $z[8] = P_{\text{reverb}} \times P_{\text{delay}}$ (Time-based spatial interaction)
- $z[9] = P_{\text{brightness}} \times P_{\text{warmth}}$ (Spectral tilt balance)
- $z[10] = P_{\text{saturation}} \times P_{\text{warmth}}$ (Harmonic coloring vs. low-mid body)
- $z[11] = P_{\text{compression}} \times P_{\text{saturation}}$ (Dynamic density vs. waveshaping)
- $z[12] = P_{\text{width}} \times P_{\text{reverb}}$ (Spatial width vs. reverberant diffuse field)

### 3. Statistical Aggregate Norms ($z[13..15]$)
- $z[13] = \frac{1}{7} \sum_{i=0}^6 |P_i|$ (Mean L1 preference intensity)
- $z[14] = \frac{\|P\|_2}{\sqrt{7}}$ (Normalized L2 Euclidean norm)
- $z[15] = \max_{i} |P_i|$ (Peak active preference magnitude)

All values in $z_{taste}$ are strictly clamped to $[-1.0, 1.0]$.

---

## 4. Soft Preferences vs. Hard Constraints

PTA cleanly separates three operational concepts:

| Category | Definition | GRRL Handling |
| :--- | :--- | :--- |
| **Soft Preference** | Continuous vector $z_{taste}$ indicating desirable acoustic balance | Injected into GRRL policy observation to bias action/parameter selection |
| **Hard Effect Constraint** | `prohibited_effects` list (e.g. `["Distortion"]`) | Masked out from GRRL policy action logits prior to sampling; cannot be selected |
| **Parameter Bound Constraint** | `parameter_bounds` dict (e.g. `{"Reverb": {"wet_mix": [0.4, 0.9]}}`) | Bounded interval in normalized $[0.0, 1.0]$ space enforced on continuous actions |
| **Terminal Feedback** | `SATISFIED` command | Concludes feedback loop; skips generation; preserves preference vector |

### Mono Pipeline & Width Limitation
`config/system_config.yaml` defines `audio.channels = 1` (mono). Because a mono pipeline cannot render stereo spatial width, `width` preferences are recorded in the user taste profile and $z_{taste}[5]$, but generate **zero synthetic parameter constraints**. The XAI engine explicitly reports stereo width as unsupported by the current renderer.

---

## 5. Adapter Request / Response Specifications

### Generation Request (`GRRLAdapter.generate`)
```python
response = adapter.generate(
    dry_audio="path/to/dry.wav",          # Audio path or tensor reference
    target_audio="path/to/target.wav",    # Audio target reference
    preferences={                         # 7-dimensional preference dict
        "brightness": 0.0,
        "compression": -0.3,
        "reverb": 0.51,
        "delay": 0.0,
        "saturation": 0.0,
        "width": 0.3,
        "warmth": 0.0
    },
    constraints={
        "prohibited_effects": [],
        "parameter_bounds": {
            "Reverb": {"wet_mix": [0.4, 0.9]},
            "Compressor": {"threshold": [0.1, 0.3]}
        }
    },
    session_id="S001"
)
```

### Generation Response
```json
{
  "status": "success",
  "session_id": "S001",
  "generation_id": "gen_a8f9c102",
  "sample_id": "sample_7b3e9a14",
  "chain": [
    {
      "effect": "Compressor",
      "parameters": {"threshold": 0.25, "ratio": 0.35, "makeup_gain": 0.20}
    },
    {
      "effect": "Reverb",
      "parameters": {"wet_mix": 0.65, "room_size": 0.55, "decay": 0.40}
    }
  ],
  "metrics": {
    "steps": 2,
    "stopped": true,
    "acoustic_loss": 0.142
  },
  "audio_ref": "outputs/audio/gen_a8f9c102.wav"
}
```

### Response Status Enum
- `success`: Episode completed, effect chain generated, audio rendered.
- `failed`: Environment or DAPN failure (details in `error_message`).
- `session_closed`: User indicated `SATISFIED`; session terminated without generation.
- `no_regeneration`: Non-generative guidance acknowledgment.

---

## 6. Feedback & Session Lifecycle Rule

1. **Initial Generation:** Created with neutral preferences $P = \vec{0}$.
2. **User Feedback:** One of 11 recognized commands submitted.
3. **Selective Dimension Smoothing:**
   $$P_{\text{new}}[d] = (1 - \alpha) P_{\text{old}}[d] + \alpha F_d \quad (\alpha = 0.30)$$
   *Untouched dimensions do not decay.*
4. **Terminal Action:** When `SATISFIED` is received:
   - Mark session satisfied.
   - Do NOT update vector.
   - Do NOT call GRRL regeneration.
   - Persist session summary and merge into long-term profile.
