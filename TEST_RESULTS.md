# DAPN Test Execution Report

- **Date**: 2026-10-02
- **Test Framework**: Pytest 9.1.1
- **Python Version**: 3.14.3
- **Overall Result**: **78 / 78 PASSED (100%)**
- **Execution Time**: **1.47s**
- **Machine readable results**: [`test_results.xml`](./test_results.xml)

---

## Summary by Module

| Test Module | Component Tested | Total Tests | Status |
| :--- | :--- | :---: | :---: |
| `tests/test_audio_pipeline.py` | Audio file loading, saving, tensor validation | 3 | **PASSED** |
| `tests/test_config.py` | YAML configuration loading & schema checking | 8 | **PASSED** |
| `tests/test_parameter_mapper.py` | Normalized $[0, 1] \leftrightarrow$ physical DSP mapping & clamping | 12 | **PASSED** |
| `tests/test_fx_schema.py` | FX chain JSON validation & constraint enforcement | 13 | **PASSED** |
| `tests/test_eq.py` | Biquad peaking EQ filtering, shapes, and gradients | 8 | **PASSED** |
| `tests/test_compressor.py` | Soft-knee dynamic compression & parameter gradients | 8 | **PASSED** |
| `tests/test_delay.py` | Fractional delay line, feedback series, and gradients | 8 | **PASSED** |
| `tests/test_reverb.py` | Algorithmic Schroeder reverberation, decay, and diffusion | 8 | **PASSED** |
| `tests/test_fx_chain.py` | Dynamic multi-effect chaining & non-commutativity order checks | 6 | **PASSED** |
| `tests/test_gradients.py` | End-to-end backpropagation through multi-effect chains | 4 | **PASSED** |
| **Total** | | **78** | **ALL PASSED** |

---

## Verification Criteria Checklist

- [x] **Effect behavior**: Output shapes (2D and 3D) preserved; EQ, Compressor, Delay, and Reverb alter audio according to expected DSP equations.
- [x] **Parameter responsiveness**: Dynamic range, gain, delay time, and reverb decay change smoothly with parameter updates.
- [x] **Chain non-commutativity**: `EQ -> Compressor` produces mathematically distinct audio compared to `Compressor -> EQ`.
- [x] **Autograd gradients**: Gradients successfully propagate backward through the full graph with zero vanishing/exploding issues.
- [x] **Numerical stability**: All output tensors and gradients are guaranteed to be finite (no NaNs or Infs).
- [x] **Processing speed**: Entire test suite executes in under 1.5 seconds on CPU.
