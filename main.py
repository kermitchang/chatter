from models.config import AppConfig
from views.console_view import ConsoleView
from viewmodels.chat_viewmodel import ChatViewModel


class ChatApp:
    """
    Main chat application class implementing MVVM architecture.
    Integrates VAD (Voice Activity Detection) for speech input after trigger word detection.
    """
    
    def __init__(self):
        """Initialize the chat application with MVVM components."""
        self.config = AppConfig()
        self.view = ConsoleView()
        self.viewmodel = ChatViewModel(self.config)
    
    def run(self):
        """
        Main application loop:
        1. Listen for trigger word (English "holiday")
        2. Once triggered, use VAD for speech input (Chinese)
        3. Send speech-to-text result to LLaMA3 model
        4. Display AI response character by character
        """
        self.view.display_welcome_message(trigger_word = self.config.speech.trigger_word)
        
        while True:
            # Listen for trigger word using English speech recognition
            # trigger = self.viewmodel.listen_for_trigger()
            trigger = self.config.speech.trigger_word  # For testing without microphone
            print(f"Trigger detected: {trigger}")
            
            # Check if user wants to exit
            if self.viewmodel.is_exit_command(trigger):
                self.view.display_goodbye_message()
                break
            
            # Verify trigger word was detected
            if not self.viewmodel.is_trigger_detected(trigger):
                self.view.display_not_triggered_message()
                continue
            
            # Use VAD for speech input after trigger detection
            # VAD automatically detects when user starts/stops speaking
            query = self.viewmodel.listen_for_speech_input()
            if not query:
                self.view.display_no_question_message()
                continue
            
            # Add user's speech-to-text query to chat session
            self.viewmodel.add_user_message(query)
            
            # Generate and display AI response character by character
            self.view.display_ai_response_start()
            for char in self.viewmodel.generate_response():
                self.view.display_ai_character(char)
            self.view.display_ai_response_end()


if __name__ == "__main__":
    app = ChatApp()
    app.run()
