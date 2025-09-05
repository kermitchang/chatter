import requests
import datetime
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable


class ToolService:
    def __init__(self):
        self.tools = {
            "get_today_date": ToolBox.get_today_date,
            "get_current_time": ToolBox.get_current_time,
            "simple_calculator": ToolBox.simple_calculator,
            "get_weather": ToolBox.get_weather,
        }
    
    def execute_tool(self, tool_name: str, args: Dict[str, Any] = None) -> str:
        if tool_name not in self.tools:
            return f"未知工具: {tool_name}"
        
        try:
            if args:
                result = self.tools[tool_name](**args)
            else:
                result = self.tools[tool_name]()
            return str(result)
        except Exception as e:
            return f"工具錯誤: {str(e)}"
    
    def get_available_tools(self) -> Dict[str, Callable]:
        return self.tools.copy()


class ToolBox:
    @staticmethod
    def get_today_date():
        return datetime.datetime.now().strftime("%Y-%m-%d")

    @staticmethod
    def get_current_time():
        return datetime.datetime.now().strftime("%H:%M:%S")

    @staticmethod
    def simple_calculator(expression):
        try:
            return str(eval(expression))
        except Exception as e:
            return f"計算錯誤: {e}"

    @staticmethod
    def get_weather(city="Taipei"):
        """
        使用 wttr.in 取得即時天氣
        """
        try:
            url = f"https://wttr.in/{city}?format=3"
            resp = requests.get(url)
            if resp.status_code == 200:
                return resp.text
            else:
                return f"無法取得天氣 (HTTP {resp.status_code})"
        except Exception as e:
            return f"取得天氣時發生錯誤: {str(e)}"
