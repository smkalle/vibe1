import os, re, requests, litserve as ls
from openai import OpenAI

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
class NewsAgent(ls.LitAPI):
    def setup(self, device):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
    def predict(self, request):
        url = request.get("url", "https://text.npr.org/")
        html = requests.get(url, timeout=10).text
        text = re.sub(r"<[^>]+>", " ", html)[:4000]
        resp = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
              {"role": "system", "content": "Summarize the main stories and notable trends."},
              {"role": "user", "content": text}
            ],
            max_tokens=400, temperature=0.3,
        )
        return {"summary": resp.choices[0].message.content.strip(), "source": url}
if __name__ == "__main__":
    ls.LitServer(NewsAgent()).run(port=8002)
