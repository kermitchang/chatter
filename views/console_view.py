import time
from abc import ABC, abstractmethod


class BaseView(ABC):
    @abstractmethod
    def display_message(self, message: str):
        pass
    
    @abstractmethod
    def get_user_input(self, prompt: str) -> str:
        pass
    
    @abstractmethod
    def display_welcome_message(self, trigger_word: str):
        pass
    
    @abstractmethod
    def display_goodbye_message(self):
        pass


class ConsoleView(BaseView):
    def display_message(self, message: str):
        print(message)
    
    def get_user_input(self, prompt: str) -> str:
        return input(prompt).strip()
    
    def display_welcome_message(self, trigger_word: str = "holiday"):
        print(f"🦙 說 '{trigger_word}' 來喚醒 AI，然後使用語音輸入問題，說 'exit' 離開")
    
    def display_goodbye_message(self):
        print("👋 再見！")
    
    def display_listening_message(self, trigger_word: str):
        print(f"🎙️ 說 '{trigger_word}' 來喚醒 AI")
    
    def display_detection_message(self, text: str):
        print(f"🗣️ 偵測到: {text}")
    
    def display_not_triggered_message(self):
        print("🤖 （未觸發）")
    
    def display_no_question_message(self):
        print("🤖 沒有檢測到語音問題，請重新嘗試")
    
    def display_tool_usage(self, tool_name: str, args: dict):
        print(f"\n🔧 使用工具: {tool_name}({args})")
    
    def display_ai_response_start(self):
        print("AI:", end=" ", flush=True)
    
    def display_ai_character(self, char: str):
        print(char, end="", flush=True)
        time.sleep(0.01)
    
    def display_ai_response_end(self):
        print()
    
    def display_speech_error(self, error_type: str):
        if error_type == "unknown_value":
            print("😅 沒聽清楚，請再試一次。")
        elif error_type == "request_error":
            print("⚠️ 語音辨識服務錯誤")