import litserve as ls
from typing import Dict, Any
import json

class UITestAPI(ls.LitAPI):
    def setup(self, device):
        self.test_cases = {
            "math": {"input": 5, "expected": 25.0, "description": "Square a number"},
            "text": {"input": "hello", "expected": "HELLO", "description": "Convert to uppercase"},
            "json": {"input": {"key": "value"}, "expected": {"key": "VALUE"}, "description": "Process JSON object"}
        }
    
    def predict(self, request: Dict[str, Any]):
        operation = request.get("operation", "math")
        data = request.get("data")
        
        if operation == "math":
            return {"output": float(data) ** 2, "type": "math"}
        elif operation == "text":
            return {"output": str(data).upper(), "type": "text"}
        elif operation == "json":
            if isinstance(data, dict):
                processed = {k: str(v).upper() for k, v in data.items()}
                return {"output": processed, "type": "json"}
            return {"error": "Expected dict for json operation"}
        elif operation == "test_cases":
            return {"test_cases": self.test_cases}
        else:
            return {"error": f"Unknown operation: {operation}"}

if __name__ == "__main__":
    print("Starting UI Test Harness Server...")
    print("Available endpoints:")
    print("  POST /predict - Main prediction endpoint")
    print("  GET /health - Health check")
    print("\nSample requests:")
    print("  Math: {'operation': 'math', 'data': 5}")
    print("  Text: {'operation': 'text', 'data': 'hello'}")
    print("  JSON: {'operation': 'json', 'data': {'key': 'value'}}")
    print("  Test Cases: {'operation': 'test_cases'}")
    print("\nServer running on http://localhost:8002")
    
    ls.LitServer(UITestAPI()).run(port=8002)