
import os
import time
import numpy as np
import pyaudio
import wave
import webrtcvad

from .base_vad_audio_recorder import BaseVadAudioRecorder

class WebrtcVadAudioRecorder(BaseVadAudioRecorder):
    def __init__(self, sample_rate=16000, frame_size=320, threshold=0.5, on_speech_end=None, aggressiveness=3):
        # WebRTC VAD requires specific frame sizes: 160, 320, or 480 samples for 16kHz
        # Adjust frame_size if needed
        valid_frame_sizes = [160, 320, 480]
        if frame_size not in valid_frame_sizes:
            frame_size = 320  # Default to 320 samples (20ms at 16kHz)
            
        super().__init__(sample_rate, frame_size, threshold, on_speech_end)
        
        # WebRTC VAD initialization
        self.vad = webrtcvad.Vad(aggressiveness)  # 0-3, higher = more aggressive
        self.aggressiveness = aggressiveness
        
        self.audio = pyaudio.PyAudio()
        self.is_recording = False
        self.is_speaking = False
        
        self.speech_frames = []
        self.speech_file = None
        self.no_speech_timeout = 8.0
        self.recording_start_time = None
        
        # WebRTC VAD works with 10ms, 20ms, or 30ms frames at 8kHz, 16kHz, 32kHz, or 48kHz
        # For smoothing, we'll use a simple majority vote over recent frames
        self.frame_history_size = 5
        self.speech_history = []
        
    def _is_speech_detected(self, frame_bytes):
        """Use WebRTC VAD to detect speech in audio frame."""
        try:
            # WebRTC VAD requires PCM16 format
            is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)
            
            # Add to history for smoothing
            self.speech_history.append(is_speech)
            if len(self.speech_history) > self.frame_history_size:
                self.speech_history.pop(0)
            
            # Use majority vote for smoothing
            speech_votes = sum(self.speech_history)
            smoothed_is_speech = speech_votes > len(self.speech_history) / 2
            
            return smoothed_is_speech
            
        except Exception as e:
            print(f"WebRTC VAD 錯誤: {e}")
            return False
    
    def _save_speech_and_callback(self):
        """Save speech segment and trigger callback."""
        if not self.speech_frames:
            return
        
        # Save speech to file using wave module
        timestamp = int(time.time())
        filename = f"temp_speech_webrtc_{timestamp}.wav"
        
        # Save using wave module
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(b''.join(self.speech_frames))
        
        self.speech_file = filename
        print(f"💾 語音已保存 (WebRTC VAD): {filename}")
        
        # Trigger callback if provided
        if self.on_speech_end:
            self.on_speech_end(filename)
    
    def cleanup(self):
        """Clean up temporary files and audio resources."""
        if self.speech_file and os.path.exists(self.speech_file):
            os.remove(self.speech_file)
        if hasattr(self, 'audio'):
            self.audio.terminate()
    
    def get_speech_file(self):
        """Get path to recorded speech file."""
        return self.speech_file
    
    def start_recording(self):
        """Start real-time VAD recording using WebRTC VAD."""
        self.is_recording = True
        self.is_speaking = False
        self.speech_frames = []
        self.speech_history = []
        self.recording_start_time = time.time()
        
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.frame_size
        )
        
        print(f"🎙️ 請開始說話... (WebRTC VAD, 敏感度: {self.aggressiveness})")
        
        try:
            while self.is_recording:
                # Read audio frame
                frame_bytes = stream.read(self.frame_size, exception_on_overflow=False)
                
                # Use WebRTC VAD for speech detection
                is_speech = self._is_speech_detected(frame_bytes)
                
                # State management - same logic as Silero implementation
                if is_speech and not self.is_speaking:
                    self.is_speaking = True
                    print(f"🗣️ WebRTC VAD 偵測到語音開始...")
                    self.speech_frames = []  # Start fresh recording
                
                if self.is_speaking:
                    self.speech_frames.append(frame_bytes)
                    
                if not is_speech and self.is_speaking:
                    # Speech ended - save and notify
                    self.is_speaking = False
                    print(f"✅ WebRTC VAD 語音結束")
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
        """Stop recording session."""
        self.is_recording = False