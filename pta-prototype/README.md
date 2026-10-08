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
├── data/                       # Locally generated simulation data
│   ├── feedback/               # Session histories (e.g., S001.jsonl)
│   └── users/                  # Final user profiles (e.g., U001__electric_guitar__global.json)
├── demo/
│   └── simulate_user_session.py # End-to-end simulation script
├── pta/                        # Core application code
│   ├── __init__.py
│   ├── core/                   # The brain of the agent
│   │   ├── __init__.py
│   │   ├── explanation_engine.py
│   │   ├── feedback_encoder.py
│   │   └── preference_engine.py
│   ├── models/                 # Data schemas
│   │   ├── __init__.py
│   │   └── schemas.py
│   └── storage/                # Data persistence layer
│       ├── __init__.py
│       ├── base.py
│       └── json_store.py
├── tests/                      # Unit testing suite
│   ├── test_feedback_encoder.py
│   ├── test_json_store.py
│   └── test_preference_engine.py
├── venv/                       # Python virtual environment
└── README.md                   # This overview file
```

## What's in Each File?

### `pta/models/schemas.py`
Defines the `FeedbackEvent`, `Session`, and `PreferenceProfile` dataclasses. These are 1:1 representations of what will eventually become MongoDB documents. It also defines the 7-dimensional bounds of the preference vector (brightness, compression, reverb, delay, saturation, width, warmth).

### `pta/core/feedback_encoder.py`
Contains the `FEEDBACK_CONTROLS` taxonomy. It exports a function that translates a user command (like `"LESS_COMPRESSION"`) into a mathematical array. 

### `pta/core/preference_engine.py`
The core math engine. It stores a user's current vector and updates it when feedback is received. It ensures values are clipped between `-1.0` and `+1.0`.

### `pta/core/explanation_engine.py`
Contains the `TEMPLATES` mapping and logic to generate a human-readable explanation of why the preference vector shifted, providing transparency to the user.

### `pta/storage/base.py` & `pta/storage/json_store.py`
`base.py` creates an abstract contract (`PreferenceStore`) that dictates how saving/loading must behave. `json_store.py` is the prototype implementation that writes this data to the `data/` folder instead of a real database.

### `demo/simulate_user_session.py`
An executable script that simulates a user named `U001` editing an `electric_guitar` track. It feeds 4 feedback clicks into the engine, logs the explanations, saves the history, and writes the final JSON profile.

### `tests/test_*.py`
A comprehensive suite of 15 automated tests using `pytest` that rigorously stress-test the math and storage logic.

## How to Run

**1. Run the test suite:**
```bash
python -m pytest tests/ -v
```

**2. Run the simulation demo:**
```bash
python demo/simulate_user_session.py
```
