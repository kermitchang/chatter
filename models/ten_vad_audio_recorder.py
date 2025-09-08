import os
import sys
import time
import numpy as np
import pyaudio
import wave

from .base_vad_audio_recorder import BaseVadAudioRecorder

# 添加 ten-vad 本地模組路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../sample/ten-vad/include")))

try:
    from ten_vad import TenVad
    TEN_VAD_AVAILABLE = True
except ImportError:
    TEN_VAD_AVAILABLE = False
    print("⚠️  找不到 ten_vad.py，請確認 sample/ten-vad/include/ 目錄存在")

class TenVadAudioRecorder(BaseVadAudioRecorder):
    def __init__(self, sample_rate=16000, frame_size=512, threshold=0.5, on_speech_end=None, 
                 min_silence_duration=0.5, min_speech_duration=0.25):
        super().__init__(sample_rate, frame_size, threshold, on_speech_end)
        
        if not TEN_VAD_AVAILABLE:
            raise ImportError("找不到 ten_vad.py，請確認 sample/ten-vad/include/ 目錄存在")
        
        # TEN-VAD 初始化 - 參考範例的方式
        self.hop_size = 256  # 16 ms per frame at 16kHz
        self.vad = TenVad(self.hop_size, threshold)
        
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
        
        # 音頻緩衝 - 改用 int16 格式配合 TEN-VAD
        self.audio_buffer_int16 = np.array([], dtype=np.int16)
        
    def _is_speech_detected(self, audio_data):
        """使用 TEN-VAD 檢測語音"""
        try:
            # 確保音頻數據為正確格式
            if isinstance(audio_data, bytes):
                # 從 bytes 轉換為 numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
            else:
                audio_array = audio_data
            
            # 添加到緩衝區
            if not hasattr(self, 'audio_buffer_int16'):
                self.audio_buffer_int16 = np.array([], dtype=np.int16)
            
            self.audio_buffer_int16 = np.concatenate([self.audio_buffer_int16, audio_array])
            
            # 確保有足夠的數據進行檢測（至少 hop_size 樣本）
            if len(self.audio_buffer_int16) < self.hop_size:
                return False
            
            # 使用 TEN-VAD 的 process 方法進行檢測 - 參考範例
            audio_chunk = self.audio_buffer_int16[:self.hop_size]
            out_probability, out_flag = self.vad.process(audio_chunk)
            
            # 移除已處理的數據
            self.audio_buffer_int16 = self.audio_buffer_int16[self.hop_size:]
            
            # 根據 flag 直接判斷是否為語音（範例中的做法）
            is_speech = bool(out_flag)
            
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
        self.audio_buffer_int16 = np.array([], dtype=np.int16)
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