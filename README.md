
# litserve-mcp-starter

Deploy **any ML model, RAG, or Agent** as an **MCP server** with **LitServe** — in ~10 lines of core code.

See `TUTORIAL.md` for a full-length guide. Runnable examples live in `examples/`.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Minimal API
python examples/server_minimal.py
curl -s -X POST localhost:8000/predict -H "content-type: application/json" -d '{"input": 4}'
```
