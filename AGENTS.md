# AGENTS.md

Three unrelated projects share this repo. Do not mix their deps or commands.

- **Root `litserve-mcp-starter`**: runnable scripts in `examples/`, no package, no automated tests.
- **`jev-call-lab/`**: self-contained Python lab (own `requirements.txt`, pytest suite).
- **`flysaver/`**: separate Rust screensaver (`Cargo.toml`). See `flysaver/README.md`.

## Root (LitServe + MCP)

- Pattern: each server subclasses `ls.LitAPI` (`setup()` + `predict()`), launched via `ls.LitServer(...).run(port=...)`. Minimal example: `examples/server_minimal.py`.
- Commands: `make run-min` (port 8000), `make run-rag` (port 8001, needs `OPENAI_API_KEY`), `make run-agent`, `make run-ui-test` (port 8002). Or `python examples/server_<name>.py` directly.
- MCP bridge (`examples/mcp_bridge.py`, FastMCP over stdio) expects LitServe already running on port 8000 and forwards via HTTP (`httpx`).
- CI (`.github/workflows/ci.yml`) only runs `python -m py_compile examples/*.py` + `docker build`. Verify manually:
  `curl -s -X POST localhost:8000/predict -H "content-type: application/json" -d '{"input": 4}'` → `{"output": 16.0}`
- Docker (`Dockerfile`) ships only `examples/`, serves `server_minimal.py` on 8000.

## jev-call-lab/

Run everything from inside `jev-call-lab/` (imports assume repo root on `sys.path` via `tests/conftest.py`).

- Setup: `pip install -r requirements.txt` (only `requests`, `pytest`, `streamlit`, `altair`, `pandas`). UI: `streamlit run app.py`.
- Offline check (never touches network): `python -m pytest -q tests`, then `python run_evals.py` → `evals/REPORT.md` (E1–E7).
- `JEV_MODE`: `auto` = `live` if `OPENROUTER_API_KEY` is set else `mock`. Also `record` (live + save to `fixtures/`), `replay` (fixtures only, free). Tests force `mock` via `tests/conftest.py` autouse fixture, which also wipes the key and points `JEV_FIXTURES` at `tmp_path`.
- Live smoke before long runs: `python jev_client.py --smoke`. On 404, use `JEV_URL=https://openrouter.ai/api/v1/systemone JEV_MODEL=jev-1.13`. Defaults: `JEV_URL=https://openrouter.ai/api/alpha/decisions`, `JEV_MODEL=typesafe/jev-1.13`.
- Key hygiene (hard rule, also enforced in `app.py`/`jev_client.py`): the OpenRouter key lives only in env (`OPENROUTER_API_KEY`) or Streamlit session memory. Never write it to disk, fixtures, results, or logs; `Config.api_key` is `repr=False`.
- Gotchas: `call_id` is deliberately excluded from `state` sent to Jev (seed ids leak labels); `usage.cost` is absent from the API so cost falls back to `input_tokens × $0.042/1M`; forecast points are after every `agent`/`tool` event (`simulate.forecast_points`), plus `pre_outcome` and `attempted-booking` subsets in `evaluate.py`.
- Data: `data/synthetic_calls.json` (153) + `data/sgd_{train,dev,test}.json` regenerate via `python generate_synthetic.py` and `python reduce_sgd.py --fetch`. `results/` is gitignored; commit `fixtures/` to make a `record` run reproducible.
- Replay risk: distinct calls can share byte-identical early prefix states (47 such keys in `sgd_test`), so one fixture key can cover several turns. With `record` + parallel workers + the stochastic live endpoint, fixture writes are last-write-wins and replay serves one value to all sharers (seen: Δp ≤ 0.03). Record with `--workers 1` for strict reproducibility. Latency provenance: `record` saves each live call's `latency_ms` in its fixture and `replay` returns it (`latency_source=recorded`); mock answers and replays of fixtures recorded before this field (all fixtures committed so far) are `mock`/`replay_local` and are excluded from latency stats and the Benchmark "fastest" verdict. The workbench **Audit** tab (`audit.py`, covered in `tests/test_audit.py`) reports these shared keys per run.
- GLM scorer (`llm_client.py`, default `z-ai/glm-5.3` via OpenRouter chat completions, same `OPENROUTER_API_KEY`): renders the typed questions into a prompt, repairs bad JSON with up to 2 fixups, returns the Jev wire shape so all evaluators work unchanged. Fixtures are `glm_`-prefixed (never collide with Jev's); `LLM_MODE` mirrors `JEV_MODE` (mock routes to `mock_jev`, offline). Reasoning is always on — output bills ~$4.40/1M, so keep benchmark N small. `simulate.py --scorer glm`; results carry a `scorer` tag the workbench **Benchmark** tab filters on.
- Workbench v2 (`app.py`, spec `specs/jev-workbench-v2.md`): six-step workflow + Summary view. Engine: `benchmark.py` (pairing validation, held-out rules, resumable runner with cost cap, latency probe, scorecard, verdict rule, report), `stats.py` (paired cluster bootstrap over calls), `sampling.py` (frozen stratified samples; never "first N": sgd_test's first 30 calls are all unbooked), `workspace.py` (`workspace/<project>/`, gitignored, `$JEV_WORKSPACE`; manifests are whitelisted fields so the key can't be written). Tests set `JEV_WORKSPACE` to `tmp_path`. UI tests: `tests/test_app.py` (Streamlit AppTest). Streamlit drops state of widgets not rendered on the current step; `app.py` re-assigns persistent keys at the top of each run, so keep new cross-step settings in that list. Escape `$` in Markdown prose (`md()`), or Streamlit renders it as math.

