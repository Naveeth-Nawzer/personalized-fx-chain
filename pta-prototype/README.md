# Preference Tracking Agent (PTA) Prototype

## Overview
We have built a deterministic **Preference Tracking Agent (PTA)** prototype. The purpose of this prototype is to prove that user feedback (e.g., "MORE REVERB") can be mathematically accumulated into a preference profile for a music production session. 

This prototype is a "JSON-first" implementation. Before introducing complex databases like MongoDB or deploying to a web server, this local version guarantees that the core math, storage interfaces, explainability algorithms, and M3 guidance contracts function correctly.

## What We Have Built
1. **Feedback Encoder**: Converts natural language buttons (like "MORE REVERB") into one-hot mathematical directional vectors.
2. **Preference Engine**: Uses an Exponential Smoothing algorithm (`P = (1-α)P + αF`) to track and update the user's preferences over time. It intelligently updates only the active feedback dimensions to prevent unchanged preferences from decaying.
3. **Session Lifecycle Manager**: Tracks session-scoped preference vectors independently from long-term profiles, supports terminal `SATISFIED` actions, and selectively merges session preferences into long-term user storage.
4. **Three-Mode Guidance Interface**: Formats preference updates into the canonical M3 guidance contract (scalar reward $r_{pta}$, 16-dim conditioning vector $z_{taste}$, and normalized parameter constraint masks).
5. **Contract-First Mock Integration**: Complete mock ecosystem (`MockM3Generator`, `MockM1Features`, `M3Adapter`) simulating end-to-end audio regeneration loops without requiring live external ML dependencies.
6. **Explainable AI (XAI) Engine**: Automatically translates vector math updates into human-readable sentences, grounded in active DSP constraints and chain configurations, while truthfully surfacing system limitations (e.g., mono-audio pipeline width constraints).
7. **JSON Storage Layer**: Persists the preference vectors, session summaries, and raw feedback session history to local JSON and JSONL files.
8. **GRRL Integration Adapter**: Production adapter connecting PTA to Member 3's GRRL pipeline, projecting 7D preferences into canonical 16D conditioning vectors with pairwise interactions and statistical metrics, enforcing effect prohibition masks and parameter intervals, and executing via dependency injection.
9. **Formal Contract Specification**: Documented in `contracts/pta_grrl_contract.md`, defining exact projection math, DSP intent mapping, mono rendering limitations, and schema boundaries.
10. **Testing Suite**: 111 unit & contract tests across 11 test modules proving mathematical constraints, contradictory feedback handling, edge cases, adapter validation, mock generation behaviors, loop rule invariants, and multi-turn feedback loops in ~0.3s.

## Project Structure
```text
pta-prototype/
├── contracts/                   # Source-of-truth & provisional JSON schemas
│   ├── README.md
│   ├── m1_features.schema.json  # M1 acoustic features contract
│   ├── m2_chain.schema.json     # M2/M3 FX-chain canonical contract
│   ├── m3_guidance.schema.json  # PTA → M3 three-mode guidance contract
│   ├── m3_guidance.schema.provisional.json # Provisional schema (16D/32D z_taste)
│   ├── m3_response.schema.json  # M3 acknowledgment response contract
│   └── pta_grrl_contract.md     # Production PTA-GRRL interface contract specification
├── data/                        # Locally generated simulation data
│   ├── feedback/                # Session histories (e.g., S001.jsonl)
│   ├── sessions/                # Session summaries
│   └── users/                   # Final user profiles (e.g., U001__electric_guitar__global.json)
├── demo/
│   ├── simulate_user_session.py    # Multi-session lifecycle demo
│   ├── simulate_m3_guidance.py     # Three-mode M3 guidance demo (stateful)
│   └── simulate_full_mocked_loop.py# Full E2E loop with all components mocked
├── evidence/                    # Verification evidence artifacts
│   ├── E16_guidance_demo.txt
│   ├── E20_mocked_loop.txt
│   ├── E29_task9_validation.txt
│   ├── E30_grrl_adapter_contract.txt # Verification record of 16D projection & contract execution
│   ├── sample_guidance_message.json
│   ├── mocked_loop_sessions.json
│   └── task9_checklist.md
├── pta/                         # Core application code
│   ├── __init__.py
│   ├── core/                    # Core engines
│   │   ├── __init__.py
│   │   ├── explanation_engine.py   # Decision-grounded XAI
│   │   ├── feedback_encoder.py     # 11-intent directional encoder
│   │   ├── preference_engine.py    # Exponential smoothing
│   │   ├── preference_mapping.py   # Intent catalog & DSP mappings
│   │   ├── session_manager.py      # Session lifecycle & merging
│   │   └── guidance_builder.py     # 3-mode M3 guidance formatter
│   ├── integration/             # Component adapters & mocks
│   │   ├── __init__.py
│   │   ├── grrl_adapter.py         # Production PTA-GRRL integration adapter
│   │   ├── m3_adapter.py           # Contract validation & mock receiver
│   │   ├── mock_loop_rules.py      # Terminal guard rules & counter validation
│   │   ├── mock_m3_generator.py    # Mock FX chain generator
│   │   └── mock_m1_features.py     # Mock 11D acoustic feature extractor
│   ├── models/                  # Data schemas
│   │   ├── __init__.py
│   │   ├── schemas.py              # User, session, and feedback data models
│   │   └── guidance.py             # 3-mode guidance message models
│   └── storage/                 # Data persistence layer
│       ├── __init__.py
│       ├── base.py                 # Abstract storage contract
│       └── json_store.py           # Atomic JSON/JSONL store
├── tests/                       # Unit & contract test suite (111 tests)
│   ├── test_explanation_engine.py
│   ├── test_feedback_encoder.py
│   ├── test_grrl_adapter.py
│   ├── test_guidance_builder.py
│   ├── test_json_store.py
│   ├── test_m3_adapter.py
│   ├── test_mock_loop_rules.py
│   ├── test_mock_m3_generator.py
│   ├── test_preference_engine.py
│   ├── test_real_loop_contract.py
│   └── test_session_manager.py
├── venv/                        # Python virtual environment
├── PTA_EVIDENCE_TRACKER.md      # Detailed evidence catalog
└── README.md                    # This overview file
```

## What's in Each File?

### `contracts/pta_grrl_contract.md`
Defines the formal interface contract between PTA and the GRRL agent (Member 3). Formally documents the 7-dimensional perceptual ordering, canonical 16D projection equations, soft preferences vs. hard constraints, mono rendering constraints, and complete JSON request/response schemas.

### `pta/models/schemas.py`
Defines `FeedbackEvent`, `Session`, `SessionPreference`, and `PreferenceProfile` dataclasses. These are 1:1 representations of what will eventually become MongoDB documents. It also defines the 7-dimensional bounds of the preference vector (brightness, compression, reverb, delay, saturation, width, warmth).

### `pta/models/guidance.py`
Defines the M3-contract guidance message schema:
- **Mode 1**: Scalar reward guidance (`r_pta` in `[-1, 1]`)
- **Mode 2**: Conditioning vector (`z_taste` in `R^16` in `[-1, 1]`)
- **Mode 3**: Constraint masking (`prohibited_effects` and normalized `parameter_bounds`)
- Action commands (`REGENERATE` vs `NONE`) and `sample_id` tracking.

### `pta/core/feedback_encoder.py`
Contains the `FEEDBACK_CONTROLS` taxonomy. It exports a function that translates a user command (like `"LESS_COMPRESSION"`) into a mathematical array.

### `pta/core/preference_mapping.py`
Perceptual-to-DSP guidance intent catalog (`FeedbackIntent`). Maps all 11 user feedback commands to preference deltas, target processors, parameters, and operational categories (soft preference vs. hard constraint vs. terminal action).

### `pta/core/preference_engine.py`
The core math engine. It stores a user's current vector and updates it when feedback is received using exponential smoothing with selective dimension decay.

### `pta/core/session_manager.py`
Manages the lifecycle of an interactive session: accumulates feedback, treats `SATISFIED` as terminal, logs per-event timestamps, and merges session preferences into the persistent profile without decaying untouched dimensions.

### `pta/core/guidance_builder.py`
Transforms user feedback and preference vectors into the three guidance modes accepted by M3. Computes `z_taste` projection (R^16), scalar `r_pta`, and parameter constraint bounds.

### `pta/integration/grrl_adapter.py`
Production integration adapter for the GRRL agent (Member 3). Validates 7D preferences, projects into canonical 16D conditioning vectors with pairwise interactions and statistical metrics, enforces effect prohibition masks and parameter intervals, and executes via dependency-injected gym environments (`env_factory`) without live GPU/PyTorch requirements.

### `pta/integration/m3_adapter.py`
Interface adapter and mock receiver simulating M3 receiving and acknowledging three-mode guidance payloads.

### `pta/integration/mock_m3_generator.py`
Simulates M3 generating new FX chains in response to PTA guidance messages. Respects parameter bounds, provides `generate_initial()` for clean neutral state start, generates new `sample_id`s, and handles session closure.

### `pta/integration/mock_loop_rules.py`
Encapsulates terminal guard rules (`should_regenerate`) and generation counter validation (`record_regeneration_count`) ensuring that terminal satisfaction messages never trigger regeneration.

### `pta/integration/mock_m1_features.py`
Deterministic mock feature extractor returning 11 acoustic features matching M1's documented schema.

### `pta/core/explanation_engine.py`
Decision-grounded explanations mapping preference shifts into human-readable production concepts, explaining active constraints and generated chains while truthfully noting system limitations (e.g. mono pipeline width constraints).

### `pta/storage/base.py` & `pta/storage/json_store.py`
`base.py` creates an abstract contract (`PreferenceStore`). `json_store.py` persists profiles, session summaries, and raw feedback to the `data/` directory.

### `demo/simulate_user_session.py`
Multi-session demo verifying neutral session initialization and long-term profile accumulation with `--fresh` reset option.

### `demo/simulate_m3_guidance.py`
Simulates live stateful guidance message generation and transmission to M3.

### `demo/simulate_full_mocked_loop.py`
End-to-end loop simulation demonstrating dry audio → M1 features → M3 initial chain → user feedback → PTA update → guidance → M3 regeneration → repeat → SATISFIED.

### `tests/test_*.py`
A comprehensive suite of 111 automated unit and contract tests across 11 test modules using `pytest` covering all encoders, math engines, session lifecycle, guidance builders, M3 adapter contracts, loop rules, mock generator behavior, GRRL integration adapters, multi-turn feedback contract loops, and decision-grounded explainability.

## Mocking Strategy for PP1 (Contract-First Integration)
For the preliminary project gate (PP1), PTA uses **mock-driven contract validation**:
- The interfaces between M1, M2, M3, and PTA are locked via canonical JSON schemas in `contracts/` and `contracts/pta_grrl_contract.md`.
- Mocks (`MockM3Generator`, `MockM1Features`, `MockM3Receiver`) and test doubles are thin (≤100 lines), deterministic, and validate incoming/outgoing messages against the agreed contracts.
- This allows the PTA component to be demonstrated running full feedback loops end-to-end without waiting for live PPO training loops or real audio rendering engines.
- When live components arrive, the adapter executes against real gym environments (`rl/grrl_env.py`) without changing internal PTA algorithm code.

### ⚠️ Experimental components (pending contract confirmation)

- `compute_reward_guidance()` — default neutral placeholder ($r_{pta} = 0.0$, confidence = 0.0) pending M3's reward semantics and $\lambda_{pta}$ confirmation. Supports pluggable `reward_fn`.
- `build_constraints()` — placeholder parameter bounds `[0.4, 0.9]` / `[0.1, 0.3]`. Not yet validated against M2's physical ranges.
- `WIDTH_UNSUPPORTED` — width preference is recorded and tracked in $z_{taste}$, but produces no constraints because M3 does not expose a direct stereo width processor.
- `project_taste_vector()` — handcrafted 7→16 projection. Awaiting confirmation of M3's expected $z_{taste}$ dimension ($D=16$ or $D=32$) and whether a learned projection is expected.

These are functional for the mock demo and adapter tests, but **must not be interpreted as validated guidance** for the real RL policy until M3 confirms the contract.

## How to Run

**1. Run the full test suite (111 tests):**
```bash
python -m pytest tests/ -v
```

**2. Run the multi-session demo:**
```bash
python demo/simulate_user_session.py --fresh
```

**3. Run the stateful M3 guidance demo:**
```bash
python demo/simulate_m3_guidance.py
```

**4. Run the full mocked end-to-end loop:**
```bash
python demo/simulate_full_mocked_loop.py
```
