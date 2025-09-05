from dataclasses import dataclass, field
from typing import List
from .message import Message, MessageRole


@dataclass
class ChatSession:
    messages: List[Message] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.messages:
            self.messages = [
                Message(
                    role=MessageRole.SYSTEM,
                    content="You are a helpful AI assistant. You can use tools: get_today_date, get_current_time, simple_calculator, get_weather"
                )
            ]
    
    def add_message(self, message: Message):
        self.messages.append(message)
    
    def add_user_message(self, content: str):
        self.add_message(Message(role=MessageRole.USER, content=content))
    
    def add_assistant_message(self, content: str):
        self.add_message(Message(role=MessageRole.ASSISTANT, content=content))
    
    def add_tool_message(self, content: str):
        self.add_message(Message(role=MessageRole.TOOL, content=content))
    
    def get_messages_as_dict(self) -> List[dict]:
        return [msg.to_dict() for msg in self.messages]
    
    def clear(self):
        self.__post_init__()