import operator
from datetime import datetime
import json
from typing import Any, List, Dict
from colorama import Fore, Back, Style, init

# Initialize colorama for Windows compatibility
init()

class FlowTracer:
    """Enhanced visual tracer for LangGraph execution flow"""
    def __init__(self):
        self.step_count = 0
        self.indent_level = 0
        self.start_time = datetime.now()
        
    def log_step(self, step_type: str, description: str, data: Any = None):
        """Log a step with visual formatting"""
        self.step_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        indent = "  " * self.indent_level
        
        colors = {
            "USER": Fore.CYAN,
            "AGENT": Fore.GREEN, 
            "TOOL": Fore.YELLOW,
            "DECISION": Fore.MAGENTA,
            "ERROR": Fore.RED,
            "INFO": Fore.BLUE,
            "RAG": Fore.LIGHTMAGENTA_EX
        }
        
        color = colors.get(step_type, Fore.WHITE)
        
        print(f"\n{color}{'='*60}")
        print(f"{color}[{timestamp}] STEP {self.step_count}: {step_type}")
        print(f"{color}{'='*60}")
        print(f"{color}{indent}📝 {description}")
        
        if data:
            print(f"{color}{indent}📊 Data:")
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, str) and len(value) > 100:
                        value = value[:100] + "..."
                    print(f"{color}{indent}   {key}: {value}")
            elif isinstance(data, str):
                preview = data[:200] + "..." if len(data) > 200 else data
                print(f"{color}{indent}   {preview}")
            else:
                print(f"{color}{indent}   {str(data)}")
        
        print(f"{color}{'='*60}{Style.RESET_ALL}")
    
    def log_flow_transition(self, from_node: str, to_node: str, condition: str = None):
        print(f"\n{Fore.BLUE}🔄 FLOW: {from_node} → {to_node}")
        if condition:
            print(f"{Fore.BLUE}   Condition: {condition}")
        print(f"{Fore.BLUE}{'─'*40}")
    
    def indent(self):
        self.indent_level += 1
    
    def dedent(self):
        self.indent_level = max(0, self.indent_level - 1)
    
    def log_summary(self, messages: List[Dict]):
        """Log a summary of the conversation"""
        print(f"\n{Back.BLUE}{Fore.WHITE}📋 CONVERSATION SUMMARY{Style.RESET_ALL}")
        print(f"{Fore.BLUE}{'='*50}")
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])
            if role == "user":
                print(f"{Fore.CYAN}👤 User: {content}")
            elif role == "assistant":
                if tool_calls:
                    print(f"{Fore.GREEN}🤖 Assistant: [Tool calls made]")
                    for tc in tool_calls:
                        print(f"{Fore.GREEN}   🔧 {tc.get('name', 'unknown')}({tc.get('args', {})})")
                else:
                    preview = content[:100] + "..." if len(content) > 100 else content
                    print(f"{Fore.GREEN}🤖 Assistant: {preview}")
            elif role == "tool":
                preview = content[:100] + "..." if len(content) > 100 else content
                print(f"{Fore.YELLOW}🔧 Tool Result: {preview}")
        print(f"{Fore.BLUE}{'='*50}{Style.RESET_ALL}")

tracer = FlowTracer()
  