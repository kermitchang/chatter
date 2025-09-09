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
        1. Check input mode configuration (voice or text)
        2. For voice mode: Listen for trigger word, then use VAD for speech input
        3. For text mode: Skip trigger detection, use keyboard input
        4. Send input to LLaMA3 model
        5. Display AI response character by character
        """
        input_mode = self.viewmodel.get_input_mode()
        
        if input_mode.lower() == 'text':
            self.view.display_welcome_message_text()
        else:
            self.view.display_welcome_message(trigger_word = self.config.speech.trigger_word)
        
        while True:
            if input_mode.lower() == 'text':
                # Text input mode: get input directly from keyboard
                query = self.viewmodel.get_user_input()
                
                # Check if user wants to exit
                if self.viewmodel.is_exit_command(query):
                    self.view.display_goodbye_message()
                    break
                
                if not query:
                    self.view.display_no_question_message()
                    continue
            else:
                # Voice input mode: listen for trigger word first
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
                query = self.viewmodel.get_user_input()
                if not query:
                    self.view.display_no_question_message()
                    continue
            
            # Add user's query to chat session
            self.viewmodel.add_user_message(query)
            
            # Generate and display AI response character by character
            self.view.display_ai_response_start()
            for char in self.viewmodel.generate_response():
                self.view.display_ai_character(char)
            self.view.display_ai_response_end()

if __name__ == "__main__":
    app = ChatApp()
    app.run()
