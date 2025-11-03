import requests

url = "http://127.0.0.1:8000/mcp/plot"
payload = {
    "mcp_version": "1.0",
    "tool": "plot-service",
    "input": {
        "instructions": "Plot a histogram of the numeric column 'value' grouped by 'category'",
        "data_reference": {
            "type": "inline",
            "payload": {
                "columns": ["category", "value"],
                "rows": [["A", 10], ["B", 3], ["A", 2]]
            }
        },
        "options": {"bins": 10, "title": "Value by Category"}
    },
    "metadata": {"request_id": "abc123", "agent_id": "research_agent_v1"}
}

response = requests.post(url, json=payload)
print(response.json())
