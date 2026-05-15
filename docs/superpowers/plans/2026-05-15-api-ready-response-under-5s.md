# API Ready Response Under 5s Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the five required FastAPI endpoints from `docs/과제명세서.md` return real usable results within 5 seconds per request, with fallback used only for missing dependencies or runtime failures.

**Architecture:** Treat API serving as an online lookup/inference layer, not a training/analysis runner. Expensive offline computations such as SHAP explanations, walk-forward backtests, ANOVA, and RAG indexing should be precomputed or cached. `/research` must keep returning a `ResearchResponse`; streaming remains an optional dashboard UX path, not the definition of API success.

**Tech Stack:** FastAPI, Pydantic, Stable-Baselines3 PPO, pandas/parquet artifacts, LangGraph, ChromaDB, Streamlit, pytest, Docker Compose.

---

## Cross-Check Against `docs/과제명세서.md`

The spec says:

- `POST /research : 투자 질문에 대한 에이전틱 리서치 결과 반환`
- `추론 시간은 요청 1건당 5초 이내여야 한다.`
- RAG must use LangGraph + vector DB, Self-Correction, source citation, and dashboard reasoning logs.
- Dashboard must communicate with FastAPI only.

Therefore this plan does **not** redefine `/research` as “accepted job only” and does **not** count fallback as the normal successful path. The API can still expose optional `/research/stream` or polling helpers, but the required endpoint must normally return a real report fast enough.

## Current Root Causes

1. `/optimize` can be slow on cold start because data/model loading may happen during the request.
2. `/explain` can be slow because SHAP is currently computed on demand. That is the wrong serving shape for a dashboard API.
3. `/backtest` can be slow because walk-forward and ANOVA can be recomputed during the request. Those should be artifacts.
4. `/research` can be slow because full LangGraph can do query refinement, vector retrieval, grading, retry, and final LLM generation in one request.
5. RAG can claim readiness from API key presence even when Chroma has zero documents. That makes “ready” misleading.

## Target Behavior

- `/health`: full response under 5 seconds.
- `/optimize`: real PPO-backed `ready` response under 5 seconds when model/data are present.
- `/explain`: real SHAP explanation from precomputed artifact or bounded cached explanation under 5 seconds when artifact/data are present.
- `/backtest`: real backtest/ANOVA result from cached/precomputed artifact under 5 seconds when artifact/data are present.
- `/research`: real RAG report under 5 seconds for normal dashboard questions by limiting online work; fallback only when API key/vectorstore/LLM fails.
- `/research/stream`: optional live trace; first event under 5 seconds, but not used to weaken the required `/research` contract.

## Task 1: Keep Observability, Remove Fallback-as-Success Tests

**Files:**
- Modify: `tests/test_api.py`
- Modify: `tests/test_api_performance.py`

- [ ] Ensure timing metadata tests only check observability.
- [ ] Add tests that explicitly distinguish normal `ready` paths from fallback paths with monkeypatched fast real functions.
- [ ] Do not assert “fallback under 5s” as a success criterion except in failure-mode tests.

## Task 2: Cache Data and Model Hot Paths

**Files:**
- Modify: `apps/api/services.py`
- Test: `tests/test_api_performance.py`

- [ ] Add `_load_returns()` and `_load_features()` with `@lru_cache(maxsize=1)`.
- [ ] Use cached data in PPO, SHAP fallback charting, backtest, health checks, and fallback payload builders.
- [ ] Keep `_load_ppo_model()` cached.
- [ ] Add performance smoke tests for all five required endpoints.
- [ ] Add a cache identity test for `_load_returns()` and `_load_features()`.

## Task 3: Serve SHAP From Artifact First

**Files:**
- Modify: `apps/api/services.py`
- Optionally create/read: `data/processed/shap_explanations.parquet` or `data/processed/shap_explanations.json`
- Test: `tests/test_api.py`

- [ ] Add `_load_shap_artifact()` that reads precomputed explanations if present.
- [ ] Make `build_explanation_response()` prefer artifact lookup by date/top_k.
- [ ] Use live SHAP only as a bounded fallback when artifact is missing.
- [ ] If neither artifact nor live SHAP succeeds, return fallback and mark message clearly.

Rationale: The spec requires dashboard SHAP 조회 and report plots; it does not require recomputing Kernel SHAP on every HTTP request.

## Task 4: Serve Backtest and ANOVA From Cached Result

**Files:**
- Modify: `apps/api/services.py`
- Optionally create/read: `data/processed/backtest_results.json`
- Test: `tests/test_api.py`, `tests/test_api_performance.py`

- [ ] Cache `_build_ready_backtest_response(window)` with `@lru_cache(maxsize=4)`.
- [ ] If a persisted backtest artifact exists, read it before recomputing.
- [ ] Keep recomputation as a fallback for local development.
- [ ] Verify repeated `/backtest?window=final` requests do not recompute expensive ANOVA.

Rationale: Backtest and ANOVA are validation outputs, not per-click online inference.

## Task 5: Make `/research` Actually Fast Instead of Accepted/Fallback

**Files:**
- Modify: `src/agent/nodes.py`
- Modify: `src/agent/graph.py`
- Modify: `apps/api/services.py`
- Test: `tests/test_api.py`, `tests/test_api_performance.py`

- [ ] Add a serving mode for API research with strict online budget:
  - retrieval top_k capped, e.g. 3
  - self-correction max retry capped to 1 for serving
  - prompt context capped by characters/tokens
  - final LLM max output capped
  - no network news collection during request
- [ ] Add an in-process cache for identical research questions.
- [ ] Add a local no-LLM extractive report path if Chroma has relevant documents but LLM times out.
- [ ] Keep `build_research_response()` returning `ResearchResponse`, not job status.
- [ ] Treat fallback as failure/degraded mode, not expected success.

Rationale: If a full agent cannot return within 5 seconds, the online graph is too heavy for the API serving requirement. Long-form report generation can remain offline or streaming, but the required endpoint needs a bounded serving graph.

## Task 6: Correct RAG Readiness

**Files:**
- Modify: `apps/api/services.py`
- Inspect: `src/agent/retriever.py`, vectorstore setup files
- Test: `tests/test_api.py`

- [ ] Add `_rag_has_documents()` by checking the Chroma collection used by the retriever.
- [ ] Make `/health.modules.rag` return `ready` only when API key exists and vectorstore count > 0.
- [ ] If Chroma is empty, `/research` should report degraded/fallback honestly.
- [ ] Decide whether Docker should include a small seeded Chroma index or run an indexing step before E2E tests.

## Task 7: Keep Streaming as UX, Not Contract Escape

**Files:**
- Modify: `apps/api/services.py`
- Modify: `apps/dashboard/app.py`
- Test: `tests/test_api.py`, `tests/test_dashboard_api_client.py`

- [ ] Keep `/research/stream` for dashboard reasoning logs.
- [ ] Compact stream events with an allowlist and small text payload.
- [ ] Use `st.write_stream()` for visible logs.
- [ ] Do not rely on stream TTFB to satisfy the required `/research` endpoint.

## Task 8: Docker E2E Verification

**Files:**
- Test-only unless failures reveal implementation gaps.

- [ ] Build containers: `docker compose build api dashboard`.
- [ ] Start containers: `docker compose up -d api dashboard`.
- [ ] Run a timing script against:
  - `GET /health`
  - `POST /optimize`
  - `POST /explain`
  - `POST /research`
  - `GET /backtest?window=final`
- [ ] For normal configured Docker, expect required endpoints to return full bodies under 5 seconds.
- [ ] Record whether each endpoint returns `ready` or degraded/fallback.
- [ ] If `/research` is fallback because Chroma is empty, treat that as a remaining E2E quality blocker, not success.

## Commit Plan

1. `docs(api): require ready responses within five seconds`
2. `perf(api): cache hot endpoint data`
3. `perf(api): serve backtest results from cache`
4. `perf(api): prefer precomputed shap explanations`
5. `perf(research): bound serving graph work`
6. `fix(api): report rag readiness from vectorstore`
7. `perf(api): compact research stream events`
8. `test(api): verify ready endpoint response budgets`

## Reverted / Kept Decisions

- Revert obsolete plan that treated accepted job or TTFB as sufficient for the required `/research` contract.
- Keep timing metadata if it remains useful for verification; it does not replace performance tests.
- Keep fallback behavior for resilience, but tests and docs should not frame fallback as the target success path.
