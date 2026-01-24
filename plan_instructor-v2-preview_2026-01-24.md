# Instructor V2 Preview - Implementation Plan (Checklist)

## Goals
- Make `instructor.from_provider()` a hard switch to v2 for all providers that have v2 implementations.
- Normalize provider-specific legacy modes inside v2, with deprecation warnings.
- Add deprecation warnings to v1 provider factories to steer to v2.
- Update docs/examples to v2-first usage (generic modes + v2 provider guidance).

## Assumptions
- We will keep v1 code paths intact for providers without v2 support (azure_openai, ollama, litellm) until v2 parity exists.
- We will accept provider-specific legacy modes in v2 by mapping them to generic modes and emitting `DeprecationWarning`.
- We will prefer v2 docs across the site, even if historical blog posts need edits.

## Non-Goals (for this pass)
- No performance optimization claims or benchmarks unless we add real measurements.
- No removal of v1 code; only deprecation warnings and routing changes.

## Detailed Checklist (Spec)

### 1) Routing: `from_provider()` hard-switch to v2
- [ ] Inventory all providers in `instructor/auto_client.py` and map to v2 equivalents.
- [ ] Update OpenAI/derivatives (openai, databricks, anyscale, together, deepseek) to import from `instructor.v2` and use v2 `from_openai`/`from_*` helpers.
- [ ] Update Anthropic, Cohere, xAI (already v2) to keep v2 import paths consistent.
- [ ] Update Google GenAI (`provider == "google"`) to use `instructor.v2.from_genai` and default to generic `Mode.TOOLS` or `Mode.JSON` as appropriate.
- [ ] Update VertexAI to use `instructor.v2.from_vertexai`.
- [ ] Update Perplexity, OpenRouter to use `instructor.v2.from_perplexity` / `from_openrouter`.
- [ ] Update remaining v2-capable providers: mistral, groq, fireworks, cerebras, writer, bedrock.
- [ ] Add explicit fallback behavior for providers without v2 (azure_openai, ollama, litellm) and document it (warn or keep v1).
- [ ] Acceptance criteria:
  - `from_provider("<provider>/...")` uses v2 for all providers with v2 implementations.
  - `from_provider` behavior for non-v2 providers is explicit and documented.

### 2) v2 Mode normalization with warnings
- [ ] Implement `instructor.v2.core.registry.normalize_mode()` to map legacy provider-specific modes to generic modes.
- [ ] Reuse `instructor.utils.providers.normalize_mode_for_provider()` or `DEPRECATED_TO_CORE` mapping; ensure `Mode.warn_deprecated_mode()` is called.
- [ ] Ensure `normalize_mode()` keeps generic modes unchanged and emits warnings only for legacy modes.
- [ ] Update v2 provider client factories to rely on normalization (no provider-specific special casing).
- [ ] Acceptance criteria:
  - Passing legacy modes (e.g., `Mode.ANTHROPIC_TOOLS`) into v2 returns the generic mode and emits a `DeprecationWarning`.
  - Existing generic modes behave identically (no warnings, no changes).

### 3) Deprecation warnings for v1 factories
- [ ] Add `DeprecationWarning` in each v1 provider factory (`instructor/providers/*/client.py`) pointing to:
  - `instructor.from_provider()` (preferred)
  - `instructor.v2.providers.<provider>.from_<provider>()`
  - Generic modes (`Mode.TOOLS`, `Mode.JSON`, `Mode.JSON_SCHEMA`, `Mode.MD_JSON`)
- [ ] Ensure warning text is consistent and actionable.
- [ ] Update or unskip tests that validate deprecation warnings for v1 providers.
- [ ] Acceptance criteria:
  - Calling v1 `from_*` emits a single `DeprecationWarning` with v2 guidance.

### 4) Tests (must be included)
- [ ] Update `tests/v2/test_mode_normalization.py` to expect normalization + warnings in v2.
- [ ] Add/adjust `tests/v2/test_routing.py` to cover from_provider → v2 routing for all v2-capable providers (skip if SDK missing).
- [ ] Add/adjust tests for v1 factory deprecation warnings (skip if SDK missing).
- [ ] Run targeted test set:
  - `uv run pytest tests/v2/test_mode_normalization.py`
  - `uv run pytest tests/v2/test_routing.py`
  - `uv run pytest tests/v2/test_client_unified.py`

### 5) Docs & examples: v2-first
- [ ] Update `docs/concepts/mode-migration.md` and `docs/modes-comparison.md` to emphasize generic modes + v2 routing.
- [ ] Update `docs/concepts/patching.md`, `docs/concepts/from_provider.md`, `docs/concepts/parallel.md` to v2 patterns.
- [ ] Update provider integration docs (`docs/integrations/*.md`) to use:
  - `instructor.from_provider(...)` (v2 default)
  - Generic modes (TOOLS/JSON/JSON_SCHEMA/MD_JSON)
  - v2 provider imports when shown explicitly
- [ ] Update examples in `examples/` to use generic modes and from_provider.
- [ ] Decide how to handle blog posts:
  - Option A: update all to v2 usage
  - Option B: add a top note: "This post uses v1 API; see v2 docs" (not preferred per request)
- [ ] Acceptance criteria:
  - No new docs recommend provider-specific legacy modes.
  - `from_provider` is presented as the standard path.

### 6) Release/preview notes
- [ ] Add a short release note in `CHANGELOG.md` describing the v2 preview behavior, deprecations, and routing changes.
- [ ] Note that v2 is primarily organizational (no guaranteed performance gains).

## Risks
- SDK differences may break v2 routing or warning tests if import behavior differs from v1.
- Some docs or examples might be intentionally historical; updating them may be contentious.
- Users relying on provider-specific modes may see warnings and behavior changes.

## Open Questions
- Should non-v2 providers hard error in `from_provider`, or keep v1 fallback with a warning?
- Do we want a feature flag to control v2 routing for staged rollout, or always-on in preview?
