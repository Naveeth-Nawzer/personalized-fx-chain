# Evidence Tracker — PTA Component (Task #5 & PP1)

## Session Preference Model Evidence (E9 – E16)

### E9 — Session Lifecycle Architecture Diagram
```
                     [ Start New Session ]
                               │
                               ▼
               SessionManager(user_id, instrument)
               (session_vector starts neutral at 0.0)
                               │
                               ▼
              ┌───► submit_feedback(command)
              │                │
              │  [Non-Terminal Feedback Command]
              │                ├─► PreferenceVectorEngine.update(command)
              │                ├─► generate_explanation(command)  [Grounded XAI wording]
              │                ├─► store.log_feedback(event with per-event UTC ISO ts)
              │                └─► iterations += 1
              │
              └────────────────┘
                               │
                      [Command == "SATISFIED"]
                               │
                               ▼
                     _handle_satisfied()
                               ├─► satisfied = True, set completed_at
                               ├─► No vector change, no iteration increment
                               └─► store.save_session(session.to_preference())
                                       (Saved to data/sessions/U001__<uuid>.json)
                               │
                               ▼
                   merge_into_long_term(weight)
                               ├─► Validate 0.0 <= weight <= 1.0
                               ├─► Check duplicate merge protection (_merged guard)
                               ├─► Load existing long-term profile (or zero vector)
                               ├─► Active dimension selection: only dimensions with session
                               │   feedback are updated; untouched dimensions DO NOT decay.
                               ├─► P_long[active] = (1 - w)*P_long[active] + w*P_session[active]
                               └─► store.save_profile()
                                       (Saved to data/users/U001__<inst>__<genre>.json)
```

---

### E10 — Generated Session Summary JSON (Clean Run)
From `data/sessions/U001__ab94bf15-6859-43a5-831f-2c534deff5dd.json`:
```json
{
  "session_id": "ab94bf15-6859-43a5-831f-2c534deff5dd",
  "user_id": "U001",
  "instrument": "electric_guitar",
  "genre": "global",
  "vector": [
    0.0,
    -0.3,
    0.51,
    0.0,
    0.0,
    0.0,
    0.0
  ],
  "iterations": 3,
  "satisfied": true,
  "started_at": "2026-10-12T16:32:57.865425+00:00",
  "ended_at": "2026-10-12T16:32:57.866408+00:00",
  "feedback_commands": [
    "MORE_REVERB",
    "MORE_REVERB",
    "LESS_COMPRESSION"
  ]
}
```

---

### E11 & E16 — Automated Test Verification (26 Passed, 0 Warnings)
Output from `python -m pytest tests/ -v`:
```
collected 26 items

tests/test_feedback_encoder.py::test_more_reverb PASSED                  [  3%]
tests/test_feedback_encoder.py::test_less_compression PASSED             [  7%]
tests/test_feedback_encoder.py::test_satisfied_is_terminal PASSED        [ 11%]
tests/test_feedback_encoder.py::test_unknown_command_raises PASSED       [ 15%]
tests/test_feedback_encoder.py::test_all_controls_work PASSED            [ 19%]
tests/test_json_store.py::test_save_and_load PASSED                      [ 23%]
tests/test_json_store.py::test_unknown_user_returns_zeros PASSED         [ 26%]
tests/test_json_store.py::test_feedback_logging PASSED                   [ 30%]
tests/test_preference_engine.py::test_starts_neutral PASSED              [ 34%]
tests/test_preference_engine.py::test_single_update PASSED               [ 38%]
tests/test_preference_engine.py::test_repeated_update_converges PASSED   [ 42%]
tests/test_preference_engine.py::test_contradictory_feedback PASSED      [ 46%]
tests/test_preference_engine.py::test_vector_is_clipped PASSED           [ 50%]
tests/test_preference_engine.py::test_reset PASSED                       [ 53%]
tests/test_preference_engine.py::test_independent_dimensions PASSED      [ 57%]
tests/test_session_manager.py::test_neutral_session PASSED               [ 61%]
tests/test_session_manager.py::test_repeated_reverb_converges PASSED     [ 65%]
tests/test_session_manager.py::test_satisfied_is_terminal_no_vector_change PASSED [ 69%]
tests/test_session_manager.py::test_post_satisfaction_feedback_raises PASSED [ 73%]
tests/test_session_manager.py::test_session_persisted PASSED             [ 76%]
tests/test_session_manager.py::test_merge_when_no_profile_exists PASSED  [ 80%]
tests/test_session_manager.py::test_two_sessions_accumulate PASSED       [ 84%]
tests/test_session_manager.py::test_duplicate_merge_rejected PASSED      [ 88%]
tests/test_session_manager.py::test_invalid_merge_weight_rejected PASSED [ 92%]
tests/test_session_manager.py::test_user_isolation PASSED                [ 96%]
tests/test_session_manager.py::test_untouched_dimensions_do_not_decay PASSED [100%]

============================= 26 passed in 0.34s ==============================
```

---

### E12 — Clean Multi-Session Execution Output (`--fresh` run)
Output from `python demo/simulate_user_session.py --fresh`:
```
[FRESH MODE] Cleared previous data in 'data'.

=================================================================
      PTA v0.2 — VERIFIED MULTI-SESSION DEMONSTRATION
=================================================================

--- SESSION 1 (Initial Track Setup: Electric Guitar) ---
Session 1 initialized neutral: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
[1] MORE_REVERB        -> Preference for more reverb recorded. The ambience preference strength has increased and is queued to guide the next FX-chain generation.
[2] MORE_REVERB        -> Preference for more reverb recorded. The ambience preference strength has increased and is queued to guide the next FX-chain generation.
[3] LESS_COMPRESSION   -> Preference for less compression recorded. Target dynamics expanded toward more natural dynamics.
[Terminal] Clicked SATISFIED -> Session marked satisfied. No further adjustments applied.

Session 1 final session vector : [0.0, -0.3, 0.51, 0.0, 0.0, 0.0, 0.0]
Long-term profile after S1     : [0.0, -0.3, 0.51, 0.0, 0.0, 0.0, 0.0]

--- SESSION 2 (Subsequent Session: Fresh Neutral State) ---
Session 2 starts neutral       : [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
[1] MORE_DELAY         -> Preference updated.
[2] WIDER              -> Preference for wider stereo image recorded. Side-channel weighting target increased.
[Terminal] Clicked SATISFIED -> Session marked satisfied. No further adjustments applied.

Session 2 final session vector : [0.0, 0.0, 0.0, 0.3, 0.0, 0.3, 0.0]
Long-term profile after S2     : [0.0, -0.3, 0.51, 0.15, 0.0, 0.15, 0.0]

Summary of Dimension Evolution:
  brightness  : After S1=+0.000  ->  After S2=+0.000
  compression : After S1=-0.300  ->  After S2=-0.300 (Preserved!)
  reverb      : After S1=+0.510  ->  After S2=+0.510 (Preserved!)
  delay       : After S1=+0.000  ->  After S2=+0.150 (Updated!)
  saturation  : After S1=+0.000  ->  After S2=+0.000
  width       : After S1=+0.000  ->  After S2=+0.150 (Updated!)
  warmth      : After S1=+0.000  ->  After S2=+0.000
=================================================================
```

---

### E13 — Long-Term Profile Persistence JSON
Inspect `data/users/U001__electric_guitar__global.json`:
```json
{
  "user_id": "U001",
  "instrument": "electric_guitar",
  "genre": "global",
  "vector": [
    0.0,
    -0.3,
    0.51,
    0.15,
    0.0,
    0.15,
    0.0
  ]
}
```

---

### E14 — Session Directory Listing
`data/sessions/` contains:
- `U001__ab94bf15-6859-43a5-831f-2c534deff5dd.json` (Session 1)
- `U001__2196d0d2-c113-4a54-9189-1436dd927134.json` (Session 2)

---

### E15 — Dedicated Test for Non-Decaying Untouched Dimensions
Covered and passing in:
`tests/test_session_manager.py::test_untouched_dimensions_do_not_decay`
Verifies that when Session 2 introduces delay and width, the persistent reverb value of `0.30` remains strictly identical down to `1e-6`.

---

## Preference Guidance Interface Evidence (Task #8 & M3 Contract)

### E16 — Sample Canonical Guidance Message JSON
Saved in `evidence/sample_guidance_message.json`:
```json
{
  "schema_version": "1.0",
  "user_id": "U001",
  "session_id": "sess_20261017_001",
  "sample_id": "sample_00484",
  "event_type": "PREFERENCE_UPDATED",
  "feedback_command": "WIDER",
  "changed_dimensions": [
    "width"
  ],
  "reward": {
    "r_pta": 0.0,
    "confidence": 0.0,
    "explanation": "Placeholder reward for WIDER; awaiting M3 contract."
  },
  "z_taste": [
    0.0,
    -0.3,
    0.3,
    0.0,
    0.0,
    0.3,
    0.0,
    -0.09,
    0.0,
    0.0,
    0.0,
    -0.0,
    0.09,
    0.1285714285714286,
    0.19639610121239345,
    0.3
  ],
  "constraints": {
    "prohibited_effects": [],
    "parameter_bounds": {
      "Compressor": {
        "threshold": [0.1, 0.3],
        "ratio": [0.1, 0.3],
        "makeup_gain": [0.1, 0.3]
      },
      "Reverb": {
        "wet_mix": [0.4, 0.9],
        "dry_mix": [0.4, 0.9],
        "room_size": [0.4, 0.9]
      }
    }
  },
  "action": "REGENERATE"
}
```

---

### E17 — Stateful Guidance Demo Output
Output from `python demo/simulate_m3_guidance.py` (captured in `evidence/E16_guidance_demo.txt`):
```
======================================================================
  TASK #8 DEMO -- PTA -> M3 THREE-MODE GUIDANCE (STATEFUL)
======================================================================

[MORE_REVERB] sample_id=sample_00482
  vector after update : [0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0]
  changed_dimensions  : ['reverb']
  r_pta               : 0.000 (confidence: 0.0)
  z_taste dim         : 16
  constraints         : ['Reverb']
  response            : ACKNOWLEDGED

[LESS_COMPRESSION] sample_id=sample_00483
  vector after update : [0.0, -0.3, 0.3, 0.0, 0.0, 0.0, 0.0]
  changed_dimensions  : ['compression']
  r_pta               : 0.000 (confidence: 0.0)
  z_taste dim         : 16
  constraints         : ['Compressor', 'Reverb']
  response            : ACKNOWLEDGED

[WIDER] sample_id=sample_00484
  vector after update : [0.0, -0.3, 0.3, 0.0, 0.0, 0.3, 0.0]
  changed_dimensions  : ['width']
  r_pta               : 0.000 (confidence: 0.0)
  z_taste dim         : 16
  constraints         : ['Compressor', 'Reverb']
  response            : ACKNOWLEDGED

[SATISFIED] sample_id=sample_00485
  vector after update : [0.0, -0.3, 0.3, 0.0, 0.0, 0.3, 0.0]
  changed_dimensions  : []
  r_pta               : 1.000 (confidence: 1.0)
  z_taste dim         : 16
  constraints         : []
  response            : SESSION_CLOSED

======================================================================
Total messages received: 4
Reward history: [0.0, 0.0, 0.0]

Example message saved to evidence/sample_guidance_message.json
======================================================================
```

---

### E18 — Full Test Suite (59 Passed, 0 Warnings)
Running `python -m pytest tests/ -v`:
- 5 tests: `test_feedback_encoder.py`
- 3 tests: `test_json_store.py`
- 7 tests: `test_preference_engine.py`
- 11 tests: `test_session_manager.py`
- 14 tests: `test_guidance_builder.py`
- 19 tests: `test_m3_adapter.py`
**Total: 59 passed in 0.23s**.

---

### E19 — M3 Adapter Integration Tests
Verified in `tests/test_m3_adapter.py` (19 tests):
- Acknowledges preference update & closes session on satisfied
- Preserves `sample_id` across communication
- Enforces non-empty `user_id` and `session_id`
- Rejects NaN/Inf in vectors and out-of-range confidence/rewards
- Verifies command-to-dimension mapping consistency and roundtrip JSON serialization

---

### E20 — Explicit Handling of Width Dimension (`WIDTH_UNSUPPORTED`)
- `DIM_TO_EFFECT["width"] = None`
- Documented in `WIDTH_UNSUPPORTED` mapping
- Tested in `test_width_produces_no_constraint`: preferences for "WIDER" update `z_taste[5]` and `changed_dimensions` but generate zero artificial constraints until confirmed by M2/M3.

---

### E21 — Experimental Labels & Transparency
Tagged in docstrings and `README.md`:
- `compute_reward_guidance()`: Neutral placeholder ($r_{pta} = 0.0$, confidence = 0.0) pending confirmation of $\lambda_{pta}$ scaling and reward semantics. Supports pluggable `reward_fn`.
- `build_constraints()`: Normalized parameter ranges `[0.4, 0.9]` and `[0.1, 0.3]` marked as unvalidated placeholders pending Sarumathy's confirmation of physical parameter bounds.

---

## Mock-Driven Integration Evidence (Contract-First PP1 Readiness)

### E22 — Full Mocked Loop Execution Transcript
Output from `python demo/simulate_full_mocked_loop.py` (captured in `evidence/E20_mocked_loop.txt`):
```
========================================================================
  FULL PTA LOOP -- ALL COMPONENTS MOCKED
  (No real audio | No RL policy | Contract-validated demo)
========================================================================

[M1 MOCK] Dry features:
  spectral_centroid_hz: 3200.0
  crest_factor_db:      14.7
  rms_dbFS:             -18.2

[M3 MOCK] Initial chain: [{'effect': 'EQ', 'parameters': {'gain': 0.5}}]
[M3 MOCK] sample_id:     sample_2430f198

[USER] MORE_REVERB
  PTA vector        : [0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0]
  changed_dimensions: ['reverb']
  r_pta             : 0.000 (confidence: 0.0)
  constraints       : ['Reverb']
  M3 response       : ACKNOWLEDGED
  new sample_id     : sample_c62e9b1d
  new chain         : [{'effect': 'Reverb', 'parameters': {'wet_mix': 0.65, 'dry_mix': 0.65, 'room_size': 0.65}}]

[USER] MORE_REVERB
  PTA vector        : [0.0, 0.0, 0.51, 0.0, 0.0, 0.0, 0.0]
  changed_dimensions: ['reverb']
  r_pta             : 0.000 (confidence: 0.0)
  constraints       : ['Reverb']
  M3 response       : ACKNOWLEDGED
  new sample_id     : sample_3864007f
  new chain         : [{'effect': 'Reverb', 'parameters': {'wet_mix': 0.65, 'dry_mix': 0.65, 'room_size': 0.65}}]

[USER] LESS_COMPRESSION
  PTA vector        : [0.0, -0.3, 0.51, 0.0, 0.0, 0.0, 0.0]
  changed_dimensions: ['compression']
  r_pta             : 0.000 (confidence: 0.0)
  constraints       : ['Compressor', 'Reverb']
  M3 response       : ACKNOWLEDGED
  new sample_id     : sample_b59958be
  new chain         : [{'effect': 'Compressor', 'parameters': {'threshold': 0.2, 'ratio': 0.2, 'makeup_gain': 0.2}}, {'effect': 'Reverb', 'parameters': {'wet_mix': 0.65, 'dry_mix': 0.65, 'room_size': 0.65}}]

[USER] WIDER
  PTA vector        : [0.0, -0.3, 0.51, 0.0, 0.0, 0.3, 0.0]
  changed_dimensions: ['width']
  r_pta             : 0.000 (confidence: 0.0)
  constraints       : ['Compressor', 'Reverb']
  M3 response       : ACKNOWLEDGED
  new sample_id     : sample_e11cbb80
  new chain         : [{'effect': 'Compressor', 'parameters': {'threshold': 0.2, 'ratio': 0.2, 'makeup_gain': 0.2}}, {'effect': 'Reverb', 'parameters': {'wet_mix': 0.65, 'dry_mix': 0.65, 'room_size': 0.65}}]

[USER] SATISFIED
  PTA vector        : [0.0, -0.3, 0.51, 0.0, 0.0, 0.3, 0.0]
  changed_dimensions: []
  r_pta             : 1.000 (confidence: 1.0)
  constraints       : []
  M3 response       : SESSION_CLOSED

========================================================================
LOOP COMPLETE
  Total feedback events  : 5
  Total M3 regenerations : 5
  Session satisfied      : True
  Final PTA vector       : [0.0, -0.3, 0.51, 0.0, 0.0, 0.3, 0.0]

  Evidence saved to evidence/mocked_loop_sessions.json
========================================================================
```

---

### E23 — Mocked Loop Regenerated Session Artifact
Persisted in `evidence/mocked_loop_sessions.json`:
Contains sample history mapping each `sample_id` to its metadata, synthesized FX chain, and deterministic checksum.

---

### E24 — Mock M3 Generator Test Suite (8 Tests)
Verified in `tests/test_mock_m3_generator.py`:
- Generates unique prefixed `sample_id`s on regeneration
- Halts regeneration on `SESSION_SATISFIED`
- Synthesizes chains respecting active guidance constraints
- Enforces deterministic chain synthesis across identical guidance inputs
- Enforces presence of required metadata fields and confirms absence of raw audio tensors

---

### E25 — Source-of-Truth & Provisional Interface Contracts
Cataloged in `contracts/`:
- `m1_features.schema.json`: 11 acoustic features + metadata
- `m2_chain.schema.json`: Canonical DAPN 7-effect normalized module schema
- `m3_guidance.schema.json`: PTA → M3 three-mode guidance message schema
- `m3_response.schema.json`: M3 acknowledgment response schema
- `m3_guidance.schema.provisional.json`: Provisional schema allowing D=16 or D=32 z_taste vector until M3 confirms

---

### E26 — Mock Loop Rules & Invariant Enforcement
Implemented in `pta/integration/mock_loop_rules.py`:
- `should_regenerate()`: Returns `False` for terminal `SATISFIED` commands / `SESSION_SATISFIED` events, and honors action `REGENERATE`.
- `record_regeneration_count()`: Increments generation count only when a new sample is actually produced, raising `ValueError` on terminal messages.
- Full loop invariant verified in `demo/simulate_full_mocked_loop.py`:
  - Initial generation: 1
  - Total feedback events: 5
  - Feedback-triggered regenerations: 4
  - Terminal closures: 1
  - Total samples stored: 5

---

### E27 — Mock Loop Rules Test Suite (10 Tests)
Verified in `tests/test_mock_loop_rules.py`:
- Feedback with `REGENERATE` action permits regeneration
- `SATISFIED` command or event never regenerates
- Terminal messages cannot be counted as generated
- Negative count rejected
- Object/dataclass messages supported
- Regression: Vector is confirmed updated before guidance message is built
- Regression: Each feedback message preserves intended `sample_id`

---

### E28 — Complete Test Suite Verification (87 Passed, 0 Warnings)
Output from `python -m pytest tests/ -v`:
- `test_feedback_encoder.py`: 5 passed
- `test_guidance_builder.py`: 21 passed (includes 8 validation regression tests)
- `test_json_store.py`: 3 passed
- `test_m3_adapter.py`: 20 passed
- `test_mock_loop_rules.py`: 10 passed
- `test_mock_m3_generator.py`: 10 passed (includes 2 initial-generation tests)
- `test_preference_engine.py`: 7 passed
- `test_session_manager.py`: 11 passed
**Total: 87 passed in 0.40s**.

---

### E29 — Task 9 Validation & Hardening Artifact
Persisted in `evidence/E29_task9_validation.txt`:
Documents full test suite results (87 passing), loop invariant breakdown (1 initial + 4 feedback-triggered + 1 terminal closure), hardened guidance validation rules, and remaining provisional integration items.

---

### E30 — GRRL Production Adapter & Real-Loop Contract Verification
Persisted in `evidence/E30_grrl_adapter_contract.txt`:
- Implemented `pta/integration/grrl_adapter.py` with dependency-injected `env_factory`.
- Implemented `pta/core/preference_mapping.py` mapping feedback to structured DSP guidance intents.
- Documented agreed production contract in `contracts/pta_grrl_contract.md`.
- Contract tests in `tests/test_grrl_adapter.py` and `tests/test_real_loop_contract.py`:
  - 7D preference validation (finiteness, bounds $[-1, 1]$, missing keys).
  - 16D taste projection ($7 \text{ raw} + 6 \text{ pairwise} + 3 \text{ aggregate norms}$).
  - Action masking enforcing prohibited effects cannot be selected by GRRL.
  - End-to-end loop: guidance -> generation -> chain & metrics -> feedback -> terminal `SATISFIED`.

---

### E31 — Truthful XAI Explanations & Full Test Suite (111 Passed, 0 Warnings)
Verified in `tests/test_explanation_engine.py`:
- `WIDER` explicitly reports mono pipeline limitation (no fake stereo widening claims).
- Explanations grounded in active constraints and generated chain.
Output from `python -m pytest tests/ -v`:
- `test_feedback_encoder.py`: 5 passed
- `test_explanation_engine.py`: 5 passed
- `test_grrl_adapter.py`: 17 passed
- `test_guidance_builder.py`: 21 passed
- `test_json_store.py`: 3 passed
- `test_m3_adapter.py`: 20 passed
- `test_mock_loop_rules.py`: 10 passed
- `test_mock_m3_generator.py`: 10 passed
- `test_preference_engine.py`: 7 passed
- `test_real_loop_contract.py`: 2 passed
- `test_session_manager.py`: 11 passed
**Total: 111 passed in 0.51s**.

---

### E32 — Three-Level Decision-Trace XAI & Fidelity Verification (126 Passed, 0 Warnings)
Implemented and verified full evidence-grounded XAI pipeline connecting user feedback, preference updates, GRRL decisions, and DAPN outcomes:
- Implemented `pta/core/decision_trace.py` defining `DecisionTrace` with generation-level traceability (`session_id`, `generation_id`, `feedback_event_id`, `selected_chain`, `previous_chain`, `rendering_status`, `audio_ref`, `audio_metrics`, `unsupported_requests`).
- Extended `pta/core/explanation_engine.py` with `explain_preference_update()` and `explain_generation()`, enforcing strict evidence-based constraints (no perceptual claims without signal metrics, failed renders never claim audio outcome, prohibited effects verified).
- Updated `pta/core/preference_engine.py` to expose active dimension metadata, event IDs, and update method.
- Updated `pta/core/session_manager.py` to associate feedback events with active `generation_id`s.
- Created `docs/xai_design.md` detailing the 5-stage trace pipeline, 3 levels of explanation, evidence-grounding matrix, and academic evaluation methodology.
- Created `tests/test_explanation_fidelity.py` (15 tests passing) verifying trace fidelity, unrendered fallbacks, parameter diffs, mono truthfulness, and hard constraint enforcement.
- Output from `python -m pytest tests/ -v`:
  - `test_explanation_engine.py`: 5 passed
  - `test_explanation_fidelity.py`: 15 passed
  - `test_feedback_encoder.py`: 5 passed
  - `test_grrl_adapter.py`: 17 passed
  - `test_guidance_builder.py`: 21 passed
  - `test_json_store.py`: 3 passed
  - `test_m3_adapter.py`: 20 passed
  - `test_mock_loop_rules.py`: 10 passed
  - `test_mock_m3_generator.py`: 10 passed
  - `test_preference_engine.py`: 7 passed
  - `test_real_loop_contract.py`: 2 passed
  - `test_session_manager.py`: 11 passed
**Total: 126 passed in 0.29s**.

