import litserve as ls

class Minimal(ls.LitAPI):
    def predict(self, request):
        x = request["input"]
        return {"output": float(x) ** 2}

if __name__ == "__main__":
    ls.LitServer(Minimal()).run(port=8000)
