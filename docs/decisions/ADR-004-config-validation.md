# ADR-004 — Configuration: stdlib dataclasses + explicit validation

Increment: config

## Problem

Every experimental parameter lives in a configuration file, so that a run
is described by a file rather than by a command line nobody recorded.
Something must turn that YAML into typed values the components can trust —
and decide how strict to be about mistakes (typos, wrong types,
contradictory values).

## Options considered

1. **Frozen stdlib dataclasses + hand-written validation** with one
   `ConfigError` whose messages name the exact YAML path.
2. pydantic models (declarative validation, coercion, good errors).
3. Raw dict pass-through: components read `config["retrieval"]["k"]`.

## Decision

Option 1: `config.py` with one small frozen dataclass per section, a
single `load_config(path)` entry point, unknown keys/sections rejected,
no type coercion, value-dependent shape checks (chunking strategy,
retrieval mode).

## Reason

- The dependency policy is stdlib-first with a capped list; pydantic is
  outside it, and the config surface (7 sections, ~20 keys) is far too
  small to justify widening the cap.
- pydantic's convenience — coercion — is here a liability: `k: "10"`
  becoming the integer 10 silently repairs what is more likely a
  generated-config bug. In an experimental pipeline a typo must never
  change an experiment; we want errors, not repairs.
- Raw dicts are stringly-typed: every consumer re-validates, failures
  surface mid-run instead of at load time, and there is no single place
  where "what is configurable" can be read.

## Additional decisions folded in (same seam, same date)

- **`minimum_tokens` is required for `recursive`**, although a shorter
  configuration could default it (the two example forms in circulation
  disagree — one lists it, the other omits it, so the point is
  internally inconsistent). Resolution: stricter side; an experiment must
  state its minimum chunk size explicitly rather than inherit an implicit
  one. The rejection of the short form is pinned by a dedicated test.
- **Duplicate YAML keys are rejected** (custom `SafeLoader` subclass)
  instead of PyYAML's silent last-wins — a stray second `k:` line must
  not change an experiment without a trace.
- **Optional keys are expressed by omission**: an explicit `key: null` is
  rejected like any wrong type, keeping "absent" representable in exactly
  one way.

## Consequences

- ~390 explicit lines we own and unit-test (45 tests), instead of a
  dependency. Adding a config field means touching `config.py` and its
  tests — deliberate friction: every new experimental knob is reviewed.
- Validation boundaries are fixed and documented in the module docstring:
  config checks structure and topology; component factories check that
  names (`loader: owi`, `type: chroma`) are registered; components check
  filesystem facts at run time. So configs parse on machines that do not
  hold the data, and adding an adapter does not touch `config.py` unless
  it introduces a new shape.
- Closed sets are validated here only where the *shape* depends on the
  value: `chunking.strategy` (each strategy has different required keys)
  and `retrieval.mode` (topology). Component names stay free-form strings
  resolved later.
