class BaseVadAudioRecorder:
    def __init__(self, sample_rate=16000, frame_size=512, threshold=0.5, on_speech_end=None):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.threshold = threshold
        self.on_speech_end = on_speech_end

    def _save_speech_and_callback(self): 
        pass

    def cleanup(self):
        pass

    def get_speech_file(self):
        pass
    
    def start_recording(self):
        pass

    def stop_recording(self):
        pass
    


    