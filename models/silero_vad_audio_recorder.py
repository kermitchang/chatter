import os
import time
import numpy as np
import torch
import pyaudio
import wave

from .base_vad_audio_recorder import BaseVadAudioRecorder

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

class SileroVadAudioRecorder(BaseVadAudioRecorder):
    def __init__(self, sample_rate=16000, frame_size=512, threshold=0.5, on_speech_end=None):
        super().__init__(sample_rate, frame_size, threshold, on_speech_end)
        self.vad_model, utils = get_vad_model()

        self.audio = pyaudio.PyAudio()
        self.is_recording = False
        self.is_speaking = False
        
        self.speech_frames = []
        self.speech_file = None
        self.no_speech_timeout = 8.0    # Auto-end if no speech detected for this long
        self.recording_start_time = None
    
    def _save_speech_and_callback(self):
        super()._save_speech_and_callback()
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
    
    def cleanup(self):
        super().cleanup()
        """Clean up temporary files and audio resources."""
        if self.speech_file and os.path.exists(self.speech_file):
            os.remove(self.speech_file)
        if hasattr(self, 'audio'):
            self.audio.terminate()
    
    def get_speech_file(self):
        super().get_speech_file()
        """Get path to recorded speech file."""
        return self.speech_file
    
    def start_recording(self):
        super().start_recording()
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
    
    def stop_recording(self):
        super().stop_recording()
        """Stop recording session."""
        self.is_recording = False