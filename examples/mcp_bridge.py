import json, httpx
from typing import Any, Dict
from mcp.server.fastmcp import FastMCP
LITSERVE_URL = "http://127.0.0.1:8000"
mcp = FastMCP("litserve-bridge")
@mcp.tool()
def call_predict(payload: Dict[str, Any]) -> str:
    r = httpx.post(f"{LITSERVE_URL}/predict", json=payload, timeout=30.0)
    r.raise_for_status()
    return json.dumps(r.json())
if __name__ == "__main__":
    mcp.run(transport="stdio")
