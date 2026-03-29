
.PHONY: install run-min run-rag run-agent run-ui-test docker test-ui bookmarks
install:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
run-min:
	python examples/server_minimal.py
run-rag:
	python examples/server_rag.py
run-agent:
	python examples/server_agent.py
run-ui-test:
	python examples/server_ui_test.py
docker:
	docker build -t litserve-mcp . && docker run --rm -p 8000:8000 litserve-mcp
test-ui:
	@echo "Opening test UI harness..."
	@echo "Make sure servers are running first:"
	@echo "  make run-min (port 8000)"
	@echo "  make run-rag (port 8001)"
	@echo "  make run-ui-test (port 8002)"
	@echo "Then open: file://$(PWD)/test_ui.html"
bookmarks:
	@echo "Twitter Bookmarks Semantic Graph"
	@echo ""
	@echo "Quick start:"
	@echo "  1. bookmarks auth          -- authenticate with Twitter"
	@echo "  2. bookmarks fetch         -- pull your bookmarks"
	@echo "  3. bookmarks build         -- build graph + categories"
	@echo "  4. bookmarks categories    -- browse topics"
	@echo "  5. bookmarks search 'rag'  -- semantic search"
	@echo ""
	@echo "Run any command:"
	@echo "  python -m examples.bookmarks_graph.cli --help"
