import requests
import json

def chat_ollama(model: str, system: str, user: str, url: str = "http://localhost:11434/api/chat") -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "stream": False,
        # Encourage strict JSON
        "options": {"temperature": 0}
    }
    resp = requests.post(url, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", "")
