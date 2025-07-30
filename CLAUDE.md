# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **litserve-mcp-starter** repository that enables deploying ML models, RAG systems, or AI agents as MCP (Model Context Protocol) servers using LitServe. The core architecture is designed around minimal code (~10 lines) to create powerful integrations.

## Development Commands

### Environment Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Running Examples
```bash
# Minimal API server
make run-min
# or: python examples/server_minimal.py

# RAG server (requires OPENAI_API_KEY environment variable)
make run-rag
# or: python examples/server_rag.py

# Agent server
make run-agent
# or: python examples/server_agent.py

# UI Test server (for testing/development)
make run-ui-test
# or: python examples/server_ui_test.py

# Docker deployment
make docker
# or: docker build -t litserve-mcp . && docker run --rm -p 8000:8000 litserve-mcp
```

### Quick Start Scripts
```bash
# Build and run with Docker
./scripts/build_and_run_docker.sh

# Run locally
./scripts/run_local.sh
```

### Testing Endpoints
```bash
# Test minimal server (expects squared output)
curl -s -X POST localhost:8000/predict -H "content-type: application/json" -d '{"input": 4}'
# Expected: {"output": 16.0}

# Test RAG server (runs on port 8001, requires OPENAI_API_KEY)
curl -s -X POST localhost:8001/predict -H "content-type: application/json" -d '{"query": "your question", "top_k": 3}'
# Expected: {"answer": "...", "sources": [...]}

# Test MCP bridge (requires LitServe running on port 8000)
python examples/mcp_bridge.py
```

### UI Test Harness
For interactive testing with a web interface:
```bash
# Start servers in separate terminals
make run-min        # Terminal 1 (port 8000)
make run-ui-test    # Terminal 2 (port 8002)
make run-rag        # Terminal 3 (port 8001, optional)

# Open test UI
make test-ui        # Shows instructions to open test_ui.html
```

The UI test harness (`test_ui.html`) provides:
- Server status monitoring
- Quick test buttons for common operations
- Custom request builder
- Sample data and expected responses

### Test Cases
Since no automated tests exist, use these manual verification steps:

1. **Minimal Server Test**:
   ```bash
   python examples/server_minimal.py &
   curl -X POST localhost:8000/predict -H "content-type: application/json" -d '{"input": 5}' | grep "25.0"
   kill %1
   ```

2. **RAG Server Test** (requires OPENAI_API_KEY):
   ```bash
   export OPENAI_API_KEY="your-key"
   python examples/server_rag.py &
   curl -X POST localhost:8001/predict -H "content-type: application/json" -d '{"query": "test", "top_k": 1}'
   kill %1
   ```

3. **MCP Bridge Test**:
   ```bash
   # Terminal 1: Start LitServe
   python examples/server_minimal.py
   
   # Terminal 2: Test MCP bridge
   echo '{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "call_predict", "arguments": {"payload": {"input": 9}}}}' | python examples/mcp_bridge.py
   ```

## Architecture

### Core Components

1. **LitServe Integration**: All examples inherit from `ls.LitAPI` base class
   - `predict()` method handles request/response logic
   - `setup()` method initializes models and resources

2. **MCP Bridge Pattern**: `examples/mcp_bridge.py` demonstrates how to expose LitServe endpoints as MCP tools
   - Uses FastMCP framework
   - Bridges HTTP requests to MCP protocol
   - Runs on stdio transport

3. **Example Patterns**:
   - **Minimal**: Simple mathematical operations (`examples/server_minimal.py`)
   - **RAG**: Vector search + OpenAI completion (`examples/server_rag.py`)
   - **Agent**: Multi-step reasoning workflows (`examples/server_agent.py`)
   - **Pipeline**: Multi-stage processing (`examples/server_pipeline.py`)
   - **OpenAI Compatible**: Drop-in replacement API (`examples/server_openai_compat.py`)

### Key Dependencies
- `litserve`: Core serving framework
- `fastmcp`: MCP server implementation
- `langchain`: For RAG and embeddings
- `chromadb`: Vector database
- `openai`: LLM completions
- `httpx`: HTTP client for bridge pattern

### Environment Variables
- `OPENAI_API_KEY`: Required for RAG and agent examples using OpenAI models

## File Structure
- `examples/`: All runnable server implementations
- `scripts/`: Utility scripts for building and running
- `TUTORIAL.md`: Comprehensive guide with implementation details
- `claude_desktop_config.example.json`: Claude Desktop MCP configuration example