import os
import threading
import configparser
import speech_recognition as sr
from models.config import SpeechConfig
from models.silero_vad_audio_recorder import SileroVadAudioRecorder
from models.webrtc_vad_audio_recorder import WebrtcVadAudioRecorder
from models.ten_vad_audio_recorder import TenVadAudioRecorder

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
        self.vad_config = self._load_vad_config()
    
    def _load_vad_config(self):
        """Load VAD configuration from config.ini file."""
        config = configparser.ConfigParser()
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'vad_config.ini')
        
        # Default values
        default_config = {
            'input_mode': 'voice',
            'vad_type': 'silero',
            'webrtc_aggressiveness': 3,
            'webrtc_frame_size': 320,
            'tenvad_min_silence_duration': 0.5,
            'tenvad_min_speech_duration': 0.25,
            'tenvad_frame_size': 512,
            'sample_rate': 16000,
            'threshold': 0.5,
            'no_speech_timeout': 8.0
        }
        
        try:
            config.read(config_path)
            result_config = default_config.copy()
            
            # Load INPUT section
            if 'INPUT' in config:
                input_section = config['INPUT']
                result_config['input_mode'] = input_section.get('input_mode', default_config['input_mode'])
            
            # Load VAD section
            if 'VAD' in config:
                vad_section = config['VAD']
                result_config.update({
                    'vad_type': vad_section.get('vad_type', default_config['vad_type']),
                    'webrtc_aggressiveness': vad_section.getint('webrtc_aggressiveness', default_config['webrtc_aggressiveness']),
                    'webrtc_frame_size': vad_section.getint('webrtc_frame_size', default_config['webrtc_frame_size']),
                    'tenvad_min_silence_duration': vad_section.getfloat('tenvad_min_silence_duration', default_config['tenvad_min_silence_duration']),
                    'tenvad_min_speech_duration': vad_section.getfloat('tenvad_min_speech_duration', default_config['tenvad_min_speech_duration']),
                    'tenvad_frame_size': vad_section.getint('tenvad_frame_size', default_config['tenvad_frame_size']),
                    'sample_rate': vad_section.getint('sample_rate', default_config['sample_rate']),
                    'threshold': vad_section.getfloat('threshold', default_config['threshold']),
                    'no_speech_timeout': vad_section.getfloat('no_speech_timeout', default_config['no_speech_timeout'])
                })
            
            return result_config
        except Exception as e:
            print(f"⚠️ 無法讀取配置檔案，使用預設值: {e}")
        
        return default_config
    
    def _create_vad_recorder(self, on_speech_end_callback):
        """Create VAD recorder based on configuration."""
        vad_type = self.vad_config['vad_type'].lower()
        
        if vad_type == 'webrtc':
            print(f"🔧 使用 WebRTC VAD (敏感度: {self.vad_config['webrtc_aggressiveness']})")
            return WebrtcVadAudioRecorder(
                sample_rate=self.vad_config['sample_rate'],
                frame_size=self.vad_config['webrtc_frame_size'],
                threshold=self.vad_config['threshold'],
                on_speech_end=on_speech_end_callback,
                aggressiveness=self.vad_config['webrtc_aggressiveness']
            )
        elif vad_type == 'tenvad':
            print("🔧 使用 TEN-VAD")
            return TenVadAudioRecorder(
                sample_rate=self.vad_config['sample_rate'],
                frame_size=self.vad_config['tenvad_frame_size'],
                threshold=self.vad_config['threshold'],
                on_speech_end=on_speech_end_callback,
                min_silence_duration=self.vad_config['tenvad_min_silence_duration'],
                min_speech_duration=self.vad_config['tenvad_min_speech_duration']
            )
        else:  # Default to silero
            print("🔧 使用 Silero VAD")
            return SileroVadAudioRecorder(
                sample_rate=self.vad_config['sample_rate'],
                frame_size=512,  # Silero uses its own frame size
                threshold=self.vad_config['threshold'],
                on_speech_end=on_speech_end_callback
            )
    
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
        
        # Create VAD recorder with callback based on configuration
        self.vad_recorder = self._create_vad_recorder(on_speech_end_callback)
        
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
    
    def get_text_input(self) -> str:
        """
        Get text input directly from user keyboard input.
        
        Returns:
            str: User input text, empty string if no input
        """
        try:
            user_input = input("💭 請輸入您的問題: ").strip()
            if user_input:
                print(f"📝 您輸入的問題: {user_input}")
                return user_input
            return ""
        except (EOFError, KeyboardInterrupt):
            print("\n👋 輸入已取消")
            return ""
    
    def get_input_mode(self) -> str:
        """Get the configured input mode (voice or text)."""
        return self.vad_config.get('input_mode', 'voice')
    
    def is_trigger_detected(self, text: str) -> bool:
        """Check if trigger word is present in recognized text."""
        return self.config.trigger_word.lower() in text.lower()
    
    def is_exit_command(self, text: str) -> bool:
        """Check if exit command is present in recognized text."""
        return "exit" in text.lower()