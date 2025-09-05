from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum


class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user" 
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]


@dataclass
class Message:
    role: MessageRole
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "role": self.role.value,
            "content": self.content
        }
        if self.tool_calls:
            result["tool_calls"] = [
                {"name": tc.name, "arguments": tc.arguments} 
                for tc in self.tool_calls
            ]
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        role = MessageRole(data["role"])
        content = data["content"]
        tool_calls = None
        
        if "tool_calls" in data:
            tool_calls = [
                ToolCall(tc["name"], tc.get("arguments", {}))
                for tc in data["tool_calls"]
            ]
        
        return cls(role=role, content=content, tool_calls=tool_calls)