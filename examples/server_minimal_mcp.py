import litserve as ls
class Minimal(ls.LitAPI):
    def predict(self, request):
        return {"output": float(request["input"]) ** 2}
if __name__ == "__main__":
    server = ls.LitServer(Minimal())
    # Enable MCP endpoint if supported in your LitServe version:
    server.run(port=8000, enable_mcp=True)
