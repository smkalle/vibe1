FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY examples/ examples/
COPY README.md TUTORIAL.md ./
EXPOSE 8000
CMD ["python", "examples/server_minimal.py"]
