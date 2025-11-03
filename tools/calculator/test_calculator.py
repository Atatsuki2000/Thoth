import requests

url = "http://127.0.0.1:8001/mcp/calculate"
payload = {
    "mcp_version": "1.0",
    "tool": "calculator",
    "input": {
        "expression": "2 + 2 * 3"
    },
    "metadata": {"request_id": "calc-test-1", "agent_id": "test"}
}

response = requests.post(url, json=payload)
print("Calculator Response:")
print(response.json())
