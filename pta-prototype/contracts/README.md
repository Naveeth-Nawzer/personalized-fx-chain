# Interface Contracts

⚠️ **These are derived from the published integration PDF (Oct 2026).
They are the source of truth for mock validation. When real M1/M2/M3
components arrive, their outputs must validate against these schemas.**

If a real component's output does not match the schema, either:
1. The schema is wrong → update schema + all mocks + tests
2. The component is buggy → fix the component

## Files

- `m1_features.schema.json` — M1 audio feature output (11 features + metadata)
- `m2_chain.schema.json` — M2/M3 FX-chain output (canonical DAPN contract)
- `m3_guidance.schema.json` — PTA → M3 guidance (3 modes)
- `m3_response.schema.json` — M3 → PTA acknowledgment

## Validation rule

Every mock and every real component MUST validate against its schema.
Contract tests run in CI. Swap mock → real only when schema validation passes.
