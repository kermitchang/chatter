import requests
import json
from typing import Generator, List, Dict, Any
from models.config import OllamaConfig
from models.message import Message


class OllamaService:
    def __init__(self, config: OllamaConfig):
        self.config = config
    
    def chat_stream(self, messages: List[Dict[str, Any]]) -> Generator[Dict[str, Any], None, None]:
        payload = {
            "model": self.config.model,
            "messages": messages
        }
        
        response = requests.post(
            self.config.api_url, 
            json=payload, 
            stream=True
        )
        
        for line in response.iter_lines():
            if line:
                data = line.decode("utf-8")
                try:
                    json_data = json.loads(data)
                    yield json_data
                except json.JSONDecodeError:
                    continue