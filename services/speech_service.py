import os
import threading
import speech_recognition as sr
from models.config import SpeechConfig
from models.silero_vad_audio_recorder import SileroVadAudioRecorder

class SpeechService:
    """
    Speech service handling both trigger word detection and VAD-based speech input.
    Uses different languages for trigger detection (English) and speech input (Chinese).
    """
    
    def __init__(self, config: SpeechConfig):
        """Initialize speech service with configuration."""
        self.config = config
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.vad_recorder = None
    
    def listen_for_trigger(self) -> str:
        """
        Listen for English trigger word using traditional speech recognition.
        
        Returns:
            str: Recognized text in lowercase, empty string if recognition fails
        """
        with self.microphone as source:
            print(f"🎙️ 說 '{self.config.trigger_word}' 來喚醒 AI")
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source)
        
        try:
            text = self.recognizer.recognize_google(
                audio, 
                language=self.config.trigger_language
            ).lower()
            print(f"🗣️ 偵測到: {text}")
            return text
        except sr.UnknownValueError:
            print("😅 沒聽清楚，請再試一次。")
            return ""
        except sr.RequestError:
            print("⚠️ 語音辨識服務錯誤")
            return ""
    
    def listen_for_speech_input(self) -> str:
        """
        Use VAD for speech input and convert to text using callback mechanism.
        
        Returns:
            str: Recognized Chinese text, empty string if recognition fails
        """
        speech_result = {"text": "", "file": None}
        
        def on_speech_end_callback(speech_file):
            """Callback function called when speech ends."""
            speech_result["file"] = speech_file
            try:
                # Convert audio file to text using Chinese speech recognition
                with sr.AudioFile(speech_file) as source:
                    audio = self.recognizer.record(source)
                
                # Use Chinese language recognition for user input
                text = self.recognizer.recognize_google(
                    audio, 
                    language=self.config.input_language
                )
                
                speech_result["text"] = text
                print(f"📝 識別文字: {text}")
                
            except sr.UnknownValueError:
                print("❌ 無法識別語音內容")
                speech_result["text"] = ""
            except sr.RequestError as e:
                print(f"⚠️ 語音辨識服務錯誤: {e}")
                speech_result["text"] = ""
        
        # Create VAD recorder with callback
        self.vad_recorder = SileroVadAudioRecorder(on_speech_end=on_speech_end_callback)
        
        # Start VAD recording in separate thread
        recording_thread = threading.Thread(target=self.vad_recorder.start_recording)
        recording_thread.start()
        
        # Wait for speech to end (VAD will auto-stop)
        recording_thread.join()
        
        # Clean up temporary audio file and resources
        try:
            if speech_result["file"] and os.path.exists(speech_result["file"]):
                os.remove(speech_result["file"])
        except:
            pass
        
        if self.vad_recorder:
            self.vad_recorder.cleanup()
        
        return speech_result["text"]
    
    def is_trigger_detected(self, text: str) -> bool:
        """Check if trigger word is present in recognized text."""
        return self.config.trigger_word.lower() in text.lower()
    
    def is_exit_command(self, text: str) -> bool:
        """Check if exit command is present in recognized text."""
        return "exit" in text.lower()