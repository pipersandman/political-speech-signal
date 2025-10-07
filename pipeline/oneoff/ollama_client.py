import requests
from typing import Dict, Any, Optional

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"

def chat_ollama(
    model: str,
    system: str,
    user: str,
    url: str = DEFAULT_OLLAMA_URL,
    options: Optional[Dict[str, Any]] = None,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": options or {"temperature": 0},
    }
    resp = requests.post(url, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", "")
