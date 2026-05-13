# apps/api Working Notes

## Scope
- FastAPI contract design for `schemas.py`, `routers`, and `services`.
- This folder should expose stable HTTP contracts even when RL, SHAP, or RAG modules are not fully wired.

## Current Structure
- `main.py`
  - App assembly and router registration.
- `schemas.py`
  - Pydantic response/request contracts.
  - Main mismatch area for Sprint 3 `/backtest` ANOVA payload and `window` support.
- `routers/`
  - Thin endpoint layer.
  - Should own request validation, endpoint path/query definitions, and response models only.
- `services.py`
  - Currently mixes fallback builders, readiness logic, and some direct module integration.
  - Should be reorganized conceptually into:
    - readiness checks
    - fallback payload builders
    - runtime adapters for RL/RAG/SHAP modules
- `config.py`
  - Environment-backed settings only.

## Design Decisions In Progress
- Keep `router -> service -> upstream module` dependency direction.
- Keep fallback support as first-class behavior, not an error path.
- Prefer explicit typed models for nested `/backtest` payloads over `dict[str, float]` where structure is known.
- Preserve backward-compatible fields only if dashboard or docs actively depend on them.

## Contract Risks To Resolve
- `/backtest` ANOVA payload shape
- `window` query typing and default
- health readiness semantics for `rl`, `shap`, and `backtest`
- benchmark naming consistency (`SPY` vs `KOSPI` display assumptions)
