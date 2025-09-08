from .base_vad_audio_recorder import BaseVadAudioRecorder

class TenVadAudioRecorder(BaseVadAudioRecorder):
    def __init__(self, sample_rate=16000, frame_size=512, threshold=0.5, on_speech_end=None):
        super().__init__(sample_rate, frame_size, threshold, on_speech_end)
    
    def _save_speech_and_callback(self):
        return super()._save_speech_and_callback()
    
    def cleanup(self):
        return super().cleanup()
    
    def get_speech_file(self):
        return super().get_speech_file()
    
    def start_recording(self):
        return super().start_recording()
    
    def stop_recording(self):
        return super().stop_recording()