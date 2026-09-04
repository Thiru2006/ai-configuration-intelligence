# Verification Engine Fixes

## log_processor.py
- Replaced broad key/value extraction with line-aware parsing so timestamps/severity prefixes do not become CSV columns.
- Added canonical aliases for seed/config/environment/telemetry fields.
- Added context-aware handling for overloaded `mem` (memory mode vs memory MB) and runtime/coverage aliases.
- Explicit final RESULT/OUTCOME/STATUS has precedence over ERROR/WARNING text.
- Benign `UVM_ERROR count=0` / `0 errors` evidence is not treated as a failure.
- Ambiguous/truncated logs remain `Unknown` with an outcome confidence/source instead of being guessed.
- Added duplicate canonical-field conflict detection (`parse_conflict`, `conflict_fields`, `conflict_details`).
- Added `is_truncated` and `parse_warning` flags.
- Added canonical failure types and separate failure signatures.
- Unknown/unrecognized key-value data goes into `extra_fields` JSON instead of creating hundreds of accidental CSV columns.
- Replaced file-order fallback config IDs with deterministic hashes of actual pre-execution configuration.

## app.py
- Added normalization for legacy CSV aliases before analytics.
- Removes legacy timestamp-fragment columns.
- Preserves canonical values when aliases conflict and surfaces a warning.
- `Unknown` outcome is no longer converted to FAIL.
- ML target is now 1=FAIL and Unknown outcomes are excluded from supervised training.
- Expanded numeric coercion and pre-execution feature set while keeping post-execution telemetry/seeds out of ML prediction.
- Fallback config IDs are derived from actual configuration signatures rather than execution order.
- Surfaces parser conflict and unknown-outcome warnings in the dashboard.
- ML metric labels now explicitly report FAIL precision/recall.

## V2 regression result
- 1000 logs parsed.
- PASS/FAIL predictions made for 923 logs: 923/923 correct on this adversarial dataset.
- 77 logs left Unknown because evidence was insufficient/ambiguous rather than guessed.
- 0 false-positive failures among decided outcomes.
- Failure type exact match on ground-truth failures: 92.9%.
- Conflict flags: 17, corresponding to deliberately conflicting scheduler duplicates.
- Output schema reduced to ~33 meaningful columns instead of hundreds of timestamp/token fragments.
