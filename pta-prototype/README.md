# Preference Tracking Agent (PTA) Prototype

## Overview
We have built a deterministic **Preference Tracking Agent (PTA)** prototype. The purpose of this prototype is to prove that user feedback (e.g., "MORE REVERB") can be mathematically accumulated into a preference profile for a music production session. 

This prototype is a "JSON-first" implementation. Before introducing complex databases like MongoDB or deploying to a web server, this local version guarantees that the core math, storage interfaces, and explainability algorithms function correctly.

## What We Have Built
1. **Feedback Encoder**: Converts natural language buttons (like "MORE REVERB") into one-hot mathematical directional vectors.
2. **Preference Engine**: Uses an Exponential Smoothing algorithm (`P = (1-α)P + αF`) to track and update the user's preferences over time. It intelligently updates only the active feedback dimensions to prevent unchanged preferences from decaying.
3. **Explainable AI (XAI) Engine**: Automatically translates vector math updates into human-readable sentences for debugging and user trust.
4. **JSON Storage Layer**: Persists the preference vectors and the raw feedback session history to local JSON and JSONL files.
5. **Testing Suite**: 15 unit tests proving the mathematical constraints, contradictory feedback handling, and edge cases.

## Project Structure
```text
pta-prototype/
├── data/                        # Locally generated simulation data
│   ├── feedback/                # Session histories (e.g., S001.jsonl)
│   ├── sessions/                # Session summaries
│   └── users/                   # Final user profiles (e.g., U001__electric_guitar__global.json)
├── demo/
│   ├── simulate_user_session.py # Multi-session lifecycle demo
│   └── simulate_m3_guidance.py  # Three-mode M3 guidance demo
├── evidence/                    # Verification evidence artifacts
│   ├── E16_guidance_demo.txt
│   └── sample_guidance_message.json
├── pta/                         # Core application code
│   ├── __init__.py
│   ├── core/                    # Core engines
│   │   ├── __init__.py
│   │   ├── explanation_engine.py
│   │   ├── feedback_encoder.py
│   │   ├── preference_engine.py
│   │   ├── session_manager.py
│   │   └── guidance_builder.py
│   ├── integration/             # Component adapters
│   │   ├── __init__.py
│   │   └── m3_adapter.py
│   ├── models/                  # Data schemas
│   │   ├── __init__.py
│   │   ├── schemas.py
│   │   └── guidance.py
│   └── storage/                 # Data persistence layer
│       ├── __init__.py
│       ├── base.py
│       └── json_store.py
├── tests/                       # Unit testing suite (38 tests)
│   ├── test_feedback_encoder.py
│   ├── test_guidance_builder.py
│   ├── test_json_store.py
│   ├── test_preference_engine.py
│   └── test_session_manager.py
├── venv/                        # Python virtual environment
└── README.md                    # This overview file
```

## What's in Each File?

### `pta/models/schemas.py`
Defines the `FeedbackEvent`, `Session`, `SessionPreference`, and `PreferenceProfile` dataclasses. These are 1:1 representations of what will eventually become MongoDB documents. It also defines the 7-dimensional bounds of the preference vector (brightness, compression, reverb, delay, saturation, width, warmth).

### `pta/models/guidance.py`
Defines the M3-contract guidance message schema:
- **Mode 1**: Scalar reward guidance (`r_pta` in `[-1, 1]`)
- **Mode 2**: Conditioning vector (`z_taste` in `R^16` in `[-1, 1]`)
- **Mode 3**: Constraint masking (`prohibited_effects` and normalized `parameter_bounds`)
- Action commands (`REGENERATE` vs `NONE`) and `sample_id` tracking.

### `pta/core/feedback_encoder.py`
Contains the `FEEDBACK_CONTROLS` taxonomy. It exports a function that translates a user command (like `"LESS_COMPRESSION"`) into a mathematical array.

### `pta/core/preference_engine.py`
The core math engine. It stores a user's current vector and updates it when feedback is received using exponential smoothing with selective dimension decay.

### `pta/core/session_manager.py`
Manages the lifecycle of an interactive session: accumulates feedback, treats `SATISFIED` as terminal, logs per-event timestamps, and merges session preferences into the persistent profile without decaying untouched dimensions.

### `pta/core/guidance_builder.py`
Transforms user feedback and preference vectors into the three guidance modes accepted by M3. Computes `z_taste` projection (R^16), scalar `r_pta`, and parameter constraint bounds.

### `pta/integration/m3_adapter.py`
Interface adapter and mock receiver simulating M3 receiving and acknowledging three-mode guidance payloads.

### `pta/core/explanation_engine.py`
Grounded explanations mapping preference shifts into human-readable production concepts.

### `pta/storage/base.py` & `pta/storage/json_store.py`
`base.py` creates an abstract contract (`PreferenceStore`). `json_store.py` persists profiles, session summaries, and raw feedback to the `data/` directory.

### `demo/simulate_user_session.py`
Multi-session demo verifying neutral session initialization and long-term profile accumulation with `--fresh` reset option.

### `demo/simulate_m3_guidance.py`
Simulates live guidance message generation and transmission to M3, saving sample JSON outputs to `evidence/`.

### `tests/test_*.py`
A comprehensive suite of 59 automated tests using `pytest` covering all encoders, math engines, session lifecycle, guidance builders, and M3 adapter contracts.

### ⚠️ Experimental components (pending contract confirmation)

- `compute_reward_guidance()` — default neutral placeholder ($r_{pta} = 0.0$, confidence = 0.0) pending M3's reward semantics and $\lambda_{pta}$ confirmation. Supports pluggable `reward_fn`.
- `build_constraints()` — placeholder parameter bounds `[0.4, 0.9]` / `[0.1, 0.3]`. Not yet validated against M2's physical ranges.
- `WIDTH_UNSUPPORTED` — width preference is recorded and tracked in $z_{taste}$, but produces no constraints because M3 does not expose a direct stereo width processor.
- `project_taste_vector()` — handcrafted 7→16 projection. Awaiting confirmation of M3's expected $z_{taste}$ dimension ($D=16$ or $D=32$) and whether a learned projection is expected.

These are functional for the mock demo and adapter tests, but **must not be interpreted as validated guidance** for the real RL policy until M3 confirms the contract.

## How to Run

**1. Run the test suite:**
```bash
python -m pytest tests/ -v
```

**2. Run the multi-session demo:**
```bash
python demo/simulate_user_session.py --fresh
```

**3. Run the M3 guidance demo:**
```bash
python demo/simulate_m3_guidance.py
```
