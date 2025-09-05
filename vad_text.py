import torch
import pyaudio
import wave
import threading
import time
from collections import deque
import numpy as np

class ContinuousVAD:
    def __init__(self, sample_rate=16000, frame_duration=512, threshold=0.5):
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration  # samples (changed from ms to samples)
        self.frame_size = frame_duration  # samples
        self.threshold = threshold
        
        # Initialize SileroVAD
        self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                              model='silero_vad',
                                              force_reload=True,
                                              onnx=False)
        (self.get_speech_timestamps, 
         self.save_audio, 
         self.read_audio,
         self.VADIterator,
         self.collect_chunks) = utils
        
        # Audio recording setup
        self.audio = pyaudio.PyAudio()
        self.is_recording = False
        self.is_speaking = False
        
        # Audio buffer for continuous recording
        self.audio_buffer = deque(maxlen=100)  # Keep last 100 frames
        self.speech_frames = []
        
    def start_recording(self):
        """開始連續錄音"""
        self.is_recording = True
        
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.frame_size
        )
        
        print("開始連續收音...")
        
        try:
            while self.is_recording:
                # 讀取音頻數據
                frame_bytes = stream.read(self.frame_size)
                
                # 轉換為numpy array並標準化到[-1, 1]
                frame_np = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                frame_tensor = torch.from_numpy(frame_np)
                
                # 使用SileroVAD檢測語音活動
                with torch.no_grad():
                    speech_prob = self.vad_model(frame_tensor, self.sample_rate).item()
                    is_speech = speech_prob > self.threshold
                
                # 狀態管理
                if is_speech and not self.is_speaking:
                    self.is_speaking = True
                    print(f"偵測到語音開始 (置信度: {speech_prob:.3f})...")
                    # 包含之前的一些幀以避免截斷
                    self.speech_frames = list(self.audio_buffer)
                
                if self.is_speaking:
                    self.speech_frames.append(frame_bytes)
                    
                if not is_speech and self.is_speaking:
                    # 語音結束，保存音頻
                    self.is_speaking = False
                    print(f"語音結束 (置信度: {speech_prob:.3f})，保存音頻...")
                    self.save_speech_segment()
                    self.speech_frames = []
                
                # 更新緩衝區
                self.audio_buffer.append(frame_bytes)
                
        except KeyboardInterrupt:
            print("\n停止錄音...")
        finally:
            stream.stop_stream()
            stream.close()
            self.is_recording = False
    
    def save_speech_segment(self):
        """保存語音片段到文件"""
        if not self.speech_frames:
            return
            
        timestamp = int(time.time())
        filename = f"speech_{timestamp}.wav"
        
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(b''.join(self.speech_frames))
        
        print(f"語音片段已保存到: {filename}")
        
        # 這裡可以添加語音轉文字或其他處理
        return filename
    
    def stop_recording(self):
        """停止錄音"""
        self.is_recording = False
    
    def __del__(self):
        """清理資源"""
        if hasattr(self, 'audio'):
            self.audio.terminate()

def main():
    # 創建連續VAD實例
    print("正在初始化 SileroVAD 模型...")
    continuous_vad = ContinuousVAD(threshold=0.5)  # 可調整閾值，0.5是默認值
    print("模型初始化完成！")
    
    try:
        # 在新線程中開始錄音
        recording_thread = threading.Thread(target=continuous_vad.start_recording)
        recording_thread.start()
        
        # 主線程等待用戶輸入
        input("按Enter鍵停止錄音...\n")
        
        # 停止錄音
        continuous_vad.stop_recording()
        recording_thread.join()
        
    except Exception as e:
        print(f"錯誤: {e}")
    finally:
        print("程序結束")

if __name__ == "__main__":
    main()