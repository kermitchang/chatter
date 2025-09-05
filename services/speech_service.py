import speech_recognition as sr
import torch
import torchaudio
import pyaudio
import threading
import time
import os
import wave
import numpy as np
from models.config import SpeechConfig


# Global VAD model instance to avoid re-downloading
_vad_model = None
_vad_utils = None

def get_vad_model():
    """Get or load the Silero VAD model (singleton pattern)."""
    global _vad_model, _vad_utils
    if _vad_model is None:
        print("🔄 正在初始化 Silero VAD 模型...")
        _vad_model, _vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,  # Don't force reload to use cached version
            onnx=False
        )
        print("✅ Silero VAD 模型初始化完成")
    return _vad_model, _vad_utils


class SileroVADAudioRecorder:
    """
    Voice Activity Detection (VAD) audio recorder using Silero VAD.
    Uses real-time frame-by-frame detection with callback mechanism.
    """
    
    def __init__(self, sample_rate=16000, frame_size=512, threshold=0.5, on_speech_end=None):
        """
        Initialize Silero VAD audio recorder.
        
        Args:
            sample_rate: Audio sample rate in Hz (default: 16000)
            frame_size: Audio frame size for processing (default: 512)
            threshold: VAD detection threshold (default: 0.5)
            on_speech_end: Callback function called when speech ends
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.threshold = threshold
        self.on_speech_end = on_speech_end
        
        # Get shared VAD model (singleton)
        self.vad_model, utils = get_vad_model()
        
        self.audio = pyaudio.PyAudio()
        self.is_recording = False
        self.is_speaking = False
        
        self.speech_frames = []
        self.speech_file = None
        self.no_speech_timeout = 8.0    # Auto-end if no speech detected for this long
        self.recording_start_time = None
        
    def start_recording(self):
        """
        Start real-time VAD recording using frame-by-frame detection.
        """
        self.is_recording = True
        self.is_speaking = False
        self.speech_frames = []
        self.recording_start_time = time.time()
        
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.frame_size
        )
        
        print("🎙️ 請開始說話...")
        
        try:
            while self.is_recording:
                # Read audio frame
                frame_bytes = stream.read(self.frame_size, exception_on_overflow=False)
                
                # Convert to numpy array and normalize to [-1, 1]
                frame_np = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                frame_tensor = torch.from_numpy(frame_np)
                
                # Use Silero VAD for speech detection
                with torch.no_grad():
                    speech_prob = self.vad_model(frame_tensor, self.sample_rate).item()
                    is_speech = speech_prob > self.threshold
                
                # State management - exactly like vad_text.py
                if is_speech and not self.is_speaking:
                    self.is_speaking = True
                    print(f"🗣️ 偵測到語音開始 (置信度: {speech_prob:.3f})...")
                    self.speech_frames = []  # Start fresh recording
                
                if self.is_speaking:
                    self.speech_frames.append(frame_bytes)
                    
                if not is_speech and self.is_speaking:
                    # Speech ended - save and notify
                    self.is_speaking = False
                    print(f"✅ 語音結束 (置信度: {speech_prob:.3f})")
                    self._save_speech_and_callback()
                    return  # Exit recording loop
                
                # Auto-timeout if no speech detected for too long
                if not self.is_speaking and (time.time() - self.recording_start_time) > self.no_speech_timeout:
                    print("⏱️ 未檢測到語音，自動結束")
                    return
                
        finally:
            stream.stop_stream()
            stream.close()
            self.is_recording = False
    
    def _save_speech_and_callback(self):
        """Save speech segment and trigger callback."""
        if not self.speech_frames:
            return
        
        # Save speech to file using wave module
        timestamp = int(time.time())
        filename = f"temp_speech_{timestamp}.wav"
        
        # Save using wave module (like vad_text.py)
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(b''.join(self.speech_frames))
        
        self.speech_file = filename
        print(f"💾 語音已保存: {filename}")
        
        # Trigger callback if provided
        if self.on_speech_end:
            self.on_speech_end(filename)
    
    
    def stop_recording(self):
        """Stop recording session."""
        self.is_recording = False
    
    def get_speech_file(self):
        """Get path to recorded speech file."""
        return self.speech_file
    
    def cleanup(self):
        """Clean up temporary files and audio resources."""
        if self.speech_file and os.path.exists(self.speech_file):
            os.remove(self.speech_file)
        if hasattr(self, 'audio'):
            self.audio.terminate()


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
        self.vad_recorder = SileroVADAudioRecorder(on_speech_end=on_speech_end_callback)
        
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