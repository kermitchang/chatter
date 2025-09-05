from dataclasses import dataclass


@dataclass
class OllamaConfig:
    """Configuration for LLaMA/Ollama service."""
    api_url: str = "http://localhost:11434/api/chat"  # Ollama API endpoint
    model: str = "llama3"  # LLaMA model name
    
    
@dataclass
class SpeechConfig:
    """Configuration for speech recognition services."""
    trigger_word: str = "hello"  # English trigger word to activate AI
    trigger_language: str = "en-US"  # Language for trigger word recognition
    input_language: str = "zh-TW"   # Language for user question recognition
    
    @property
    def language(self):
        """Backward compatibility property - returns trigger language."""
        return self.trigger_language
    
    
@dataclass
class AppConfig:
    """Main application configuration combining all service configs."""
    ollama: OllamaConfig = None
    speech: SpeechConfig = None
    
    def __post_init__(self):
        """Initialize default configurations if not provided."""
        if self.ollama is None:
            self.ollama = OllamaConfig()
        if self.speech is None:
            self.speech = SpeechConfig()