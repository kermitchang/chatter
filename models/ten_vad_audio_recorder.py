import os
import time
import numpy as np
import pyaudio
import wave

from .base_vad_audio_recorder import BaseVadAudioRecorder

try:
    from ten_vad import TenVad
    TEN_VAD_AVAILABLE = True
except ImportError:
    TEN_VAD_AVAILABLE = False
    print("⚠️  TEN-VAD 未安裝，請運行: pip install git+https://huggingface.co/TEN-framework/ten-vad")

class TenVadAudioRecorder(BaseVadAudioRecorder):
    def __init__(self, sample_rate=16000, frame_size=512, threshold=0.5, on_speech_end=None, 
                 min_silence_duration=0.5, min_speech_duration=0.25):
        super().__init__(sample_rate, frame_size, threshold, on_speech_end)
        
        if not TEN_VAD_AVAILABLE:
            raise ImportError("TEN-VAD 未安裝，請執行: pip install git+https://huggingface.co/TEN-framework/ten-vad")
        
        # TEN-VAD 初始化
        self.vad = TenVad()
        
        # 音頻設定
        self.audio = pyaudio.PyAudio()
        self.is_recording = False
        self.is_speaking = False
        
        # 語音片段儲存
        self.speech_frames = []
        self.speech_file = None
        self.no_speech_timeout = 8.0
        self.recording_start_time = None
        
        # 語音檢測平滑參數
        self.min_silence_duration = min_silence_duration  # 最小靜默時間（秒）
        self.min_speech_duration = min_speech_duration    # 最小語音時間（秒）
        
        # 狀態追蹤
        self.last_speech_time = 0
        self.speech_start_time = 0
        self.silence_start_time = 0
        
        # 音頻緩衝
        self.audio_buffer = np.array([], dtype=np.float32)
        
    def _is_speech_detected(self, audio_data):
        """使用 TEN-VAD 檢測語音"""
        try:
            # 確保音頻數據為 16kHz, float32 格式
            if isinstance(audio_data, bytes):
                # 從 bytes 轉換為 numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                audio_array = audio_data
            
            # TEN-VAD 需要足夠的音頻數據進行檢測
            # 添加到緩衝區
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_array])
            
            # 確保有足夠的數據進行檢測（至少 512 樣本）
            min_samples = 512
            if len(self.audio_buffer) < min_samples:
                return False
            
            # 使用 TEN-VAD 進行檢測
            speech_probs = self.vad(self.audio_buffer)
            
            # 計算平均語音概率
            avg_prob = np.mean(speech_probs) if len(speech_probs) > 0 else 0.0
            
            # 清理緩衝區（保留最新的數據）
            if len(self.audio_buffer) > min_samples * 2:
                self.audio_buffer = self.audio_buffer[-min_samples:]
            
            # 根據閾值判斷是否為語音
            is_speech = avg_prob > self.threshold
            
            return is_speech
            
        except Exception as e:
            print(f"TEN-VAD 檢測錯誤: {e}")
            return False
    
    def _save_speech_and_callback(self):
        """保存語音片段並觸發回調"""
        if not self.speech_frames:
            return
        
        # 保存語音到檔案
        timestamp = int(time.time())
        filename = f"temp_speech_tenvad_{timestamp}.wav"
        
        # 使用 wave 模組保存
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(b''.join(self.speech_frames))
        
        self.speech_file = filename
        print(f"💾 語音已保存 (TEN-VAD): {filename}")
        
        # 觸發回調
        if self.on_speech_end:
            self.on_speech_end(filename)
    
    def cleanup(self):
        """清理臨時檔案和音頻資源"""
        if self.speech_file and os.path.exists(self.speech_file):
            os.remove(self.speech_file)
        if hasattr(self, 'audio'):
            self.audio.terminate()
    
    def get_speech_file(self):
        """取得錄製的語音檔案路徑"""
        return self.speech_file
    
    def start_recording(self):
        """開始即時 VAD 錄音使用 TEN-VAD"""
        self.is_recording = True
        self.is_speaking = False
        self.speech_frames = []
        self.audio_buffer = np.array([], dtype=np.float32)
        self.recording_start_time = time.time()
        
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.frame_size
        )
        
        print(f"🎙️ 請開始說話... (TEN-VAD)")
        
        try:
            current_time = time.time()
            
            while self.is_recording:
                # 讀取音頻幀
                frame_bytes = stream.read(self.frame_size, exception_on_overflow=False)
                current_time = time.time()
                
                # 使用 TEN-VAD 進行語音檢測
                is_speech = self._is_speech_detected(frame_bytes)
                
                # 狀態管理
                if is_speech:
                    if not self.is_speaking:
                        # 語音開始
                        self.speech_start_time = current_time
                        
                        # 檢查是否滿足最小語音持續時間
                        if (current_time - self.speech_start_time) >= 0:  # 立即開始
                            self.is_speaking = True
                            print(f"🗣️ TEN-VAD 偵測到語音開始...")
                            self.speech_frames = []
                    
                    self.last_speech_time = current_time
                    self.silence_start_time = 0  # 重置靜默計時
                else:
                    if self.is_speaking and self.silence_start_time == 0:
                        # 開始靜默計時
                        self.silence_start_time = current_time
                
                # 收集語音數據
                if self.is_speaking:
                    self.speech_frames.append(frame_bytes)
                    
                    # 檢查是否應該結束語音（靜默時間足夠長）
                    if (self.silence_start_time > 0 and 
                        (current_time - self.silence_start_time) > self.min_silence_duration and
                        (current_time - self.speech_start_time) > self.min_speech_duration):
                        
                        # 語音結束
                        self.is_speaking = False
                        print(f"✅ TEN-VAD 語音結束")
                        self._save_speech_and_callback()
                        return
                
                # 自動超時檢查
                if not self.is_speaking and (current_time - self.recording_start_time) > self.no_speech_timeout:
                    print("⏱️ 未檢測到語音，自動結束")
                    return
                    
        finally:
            stream.stop_stream()
            stream.close()
            self.is_recording = False
    
    def stop_recording(self):
        """停止錄音會話"""
        self.is_recording = False