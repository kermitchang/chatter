from typing import Generator
from models.chat_session import ChatSession
from models.message import Message, MessageRole, ToolCall
from models.config import AppConfig
from services.ollama_service import OllamaService
from services.speech_service import SpeechService
from tool_box import ToolService


class ChatViewModel:
    """
    ViewModel for chat application implementing MVVM pattern.
    Coordinates between speech service, LLaMA service, and chat session management.
    """
    
    def __init__(self, config: AppConfig):
        """Initialize ViewModel with all required services."""
        self.config = config
        self.chat_session = ChatSession()
        self.ollama_service = OllamaService(config.ollama)
        self.speech_service = SpeechService(config.speech)
        self.tool_service = ToolService()
    
    def listen_for_trigger(self) -> str:
        """Listen for English trigger word through speech service."""
        return self.speech_service.listen_for_trigger()
    
    def listen_for_speech_input(self) -> str:
        """Use VAD for Chinese speech input through speech service."""
        return self.speech_service.listen_for_speech_input()
    
    def is_trigger_detected(self, text: str) -> bool:
        """Check if trigger word was detected in the text."""
        return self.speech_service.is_trigger_detected(text)
    
    def is_exit_command(self, text: str) -> bool:
        """Check if exit command was detected in the text."""
        return self.speech_service.is_exit_command(text)
    
    def add_user_message(self, content: str):
        """Add user message to the chat session."""
        self.chat_session.add_user_message(content)
    
    def generate_response(self) -> Generator[str, None, None]:
        """
        Generate AI response using LLaMA model.
        
        Yields:
            str: Individual characters of the AI response for streaming display
        """
        messages = self.chat_session.get_messages_as_dict()
        ai_content = ""
        
        # Stream response from LLaMA model
        for response_data in self.ollama_service.chat_stream(messages):
            # Handle any tool calls in the response
            if "message" in response_data and "tool_calls" in response_data["message"]:
                self._handle_tool_calls(response_data["message"]["tool_calls"])
            
            # Yield each character for streaming display
            content = response_data.get("message", {}).get("content", "")
            for char in content:
                ai_content += char
                yield char
            
            if response_data.get("done"):
                break
        
        # Save complete AI response to chat session
        if ai_content:
            self.chat_session.add_assistant_message(ai_content)
    
    def _handle_tool_calls(self, tool_calls):
        """
        Handle tool calls from AI response.
        
        Args:
            tool_calls: List of tool calls from AI response
        """
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            args = tool_call.get("arguments", {})
            
            # Execute the tool and save result to chat session
            result = self.tool_service.execute_tool(tool_name, args)
            self.chat_session.add_tool_message(f"{tool_name} 回傳: {result}")
            
            yield ("tool_usage", tool_name, args)
    
    def clear_session(self):
        """Clear the current chat session."""
        self.chat_session.clear()
    
    def get_trigger_word(self) -> str:
        """Get the configured trigger word."""
        return self.config.speech.trigger_word