import litserve as ls
class EchoAPI(ls.LitAPI):
    def predict(self, req):
        msg = req["messages"][-1]["content"]
        return {"choices":[{"message":{"content":f"You said: {msg}"}}]}
if __name__ == "__main__":
    ls.LitServer(EchoAPI(), spec=ls.OpenAISpec()).run(port=8003)
