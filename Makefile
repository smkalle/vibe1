
.PHONY: install run-min run-rag run-agent run-ui-test docker test-ui
install:
\tpython -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
run-min:
\tpython examples/server_minimal.py
run-rag:
\tpython examples/server_rag.py
run-agent:
\tpython examples/server_agent.py
run-ui-test:
\tpython examples/server_ui_test.py
docker:
\tdocker build -t litserve-mcp . && docker run --rm -p 8000:8000 litserve-mcp
test-ui:
\t@echo "Opening test UI harness..."
\t@echo "Make sure servers are running first:"
\t@echo "  make run-min (port 8000)"
\t@echo "  make run-rag (port 8001)"  
\t@echo "  make run-ui-test (port 8002)"
\t@echo "Then open: file://$(PWD)/test_ui.html"
