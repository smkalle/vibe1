import litserve as ls

class InferencePipeline(ls.LitAPI):
    def setup(self, device):
        self.square = lambda v: v**2
        self.cube = lambda v: v**3
    def predict(self, request):
        v = float(request["input"])
        return {"sum_sq_cu": self.square(v) + self.cube(v)}
if __name__ == "__main__":
    ls.LitServer(InferencePipeline(), accelerator="auto").run(port=8000)
