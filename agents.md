# Local Working Notes

## Purpose
- This file is a local working guide for API design work on branch `feature-j-api-contract`.
- Highest-priority requirement source is `docs/과제명세서.md`.
- Shared interfaces should be aligned against `docs/labels_and_interfaces.md` and `docs/api_spec.md`.
- After each meaningful task, update `CLAUDE.md` so the current project state stays discoverable.

## Current Branch
- `feature-j-api-contract`
- Goal: organize API contract design for `schemas.py`, `routers`, and `services` before implementation changes.

## Folder Notes
- `docs/`
  - Source of truth for assignment requirements and shared contracts.
  - `docs/과제명세서.md` overrides any looser or older local assumptions.
  - `docs/api_spec.md` contains the team-facing API contract draft, but some sections are ahead of code.
- `apps/api/`
  - Owned by 박지민.
  - Current code exposes fallback-oriented endpoints and partial real integrations.
  - Main design task is to align schema shape with dashboard expectations and RL/RAG module boundaries.
- `apps/dashboard/`
  - Consumes FastAPI only.
  - Contains mock payloads that reveal expected response shapes, especially for `/backtest`.
- `src/agent/`
  - LangGraph and risk-tag integration surface for `/research`.
  - API should depend on stable output keys only, not internal graph node details.
- `src/rl/`
  - Upstream provider for optimize/explain/backtest runtime data.
  - API should treat RL outputs as adapters into HTTP response schemas.
- `src/data/`
  - Provides parquet inputs used by API fallback and readiness checks.

## Current API Contract Gaps
- `AnovaResult.post_hoc` is still `str` in code, but shared docs and dashboard expect structured rows.
- `interaction` and `strategy_effect` exist in docs/mock payloads but not in current schema.
- `/backtest` `window` query support is specified but not implemented.
- Dashboard mock names and API benchmark labels need one canonical contract.

## Working Rule
- For this branch, do not jump to implementation first.
- First produce a reviewed design spec, then update shared docs and code from that spec.
