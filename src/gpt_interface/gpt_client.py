"""
GPT client interface
"""

from openai import OpenAI
import time
import random
import os
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from utils.config_loader import ConfigLoader
except ImportError:
    print("Warning: Could not import ConfigLoader")
    ConfigLoader = None

def load_api_key():
    """
    Load OpenAI API key from multiple sources in order of priority:
    1. Direct config file setting (api_key)
    2. Environment variable (api_key_env)
    3. API key file (api_key_file)
    """
    try:
        if ConfigLoader:
            # 加載系統配置
            system_config = ConfigLoader.load('system')
            openai_config = system_config.get('api', {}).get('openai', {})
            
            # 方法1：直接從配置文件讀取
            api_key = openai_config.get('api_key', '').strip()
            if api_key:
                print("Using API key from config file")
                return api_key
            
            # 方法2：從環境變數讀取
            env_var = openai_config.get('api_key_env', 'OPENAI_API_KEY')
            api_key = os.environ.get(env_var, '').strip()
            if api_key:
                print(f"Using API key from environment variable: {env_var}")
                return api_key
            
            # 方法3：從文件讀取
            key_file = openai_config.get('api_key_file', '').strip()
            if key_file and Path(key_file).exists():
                try:
                    with open(key_file, 'r', encoding='utf-8') as f:
                        api_key = f.read().strip()
                    if api_key:
                        print(f"Using API key from file: {key_file}")
                        return api_key
                except Exception as e:
                    print(f"Error reading API key file: {e}")
        
        # 備用方案：直接從環境變數讀取
        api_key = os.environ.get('OPENAI_API_KEY', '').strip()
        if api_key:
            print("Using API key from OPENAI_API_KEY environment variable")
            return api_key
            
    except Exception as e:
        print(f"Error loading API key: {e}")
    
    print("No API key configured")
    return None

def get_openai_config():
    """Get OpenAI configuration from config file"""
    if ConfigLoader:
        system_config = ConfigLoader.load('system')
        return system_config.get('api', {}).get('openai', {})
    return {}

# Initialize OpenAI client
api_key = load_api_key()
client = None

if api_key:
    try:
        client = OpenAI(api_key=api_key)
        print("OpenAI client initialized successfully")
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        client = None
else:
    print("Warning: No OpenAI API key configured")

# Get configuration
openai_config = get_openai_config()
DEFAULT_MODEL = openai_config.get('default_model', 'gpt-3.5-turbo')
DEFAULT_MAX_TOKENS = openai_config.get('max_tokens', 1000)
DEFAULT_TEMPERATURE = openai_config.get('temperature', 0.7)

def chat(prompt: str, model: str = None, max_tokens: int = None, temperature: float = None) -> str:
    """
    Chat with OpenAI GPT model
    
    Args:
        prompt: The input prompt
        model: Model name (defaults to config setting)
        max_tokens: Maximum tokens (defaults to config setting)
        temperature: Temperature for randomness (defaults to config setting)
    """
    if not client:
        print("Warning: No OpenAI client available, using fallback response")
        return "I understand your point. Let me think about this issue from a different perspective."
    
    # Use provided values or defaults from config
    model = model or DEFAULT_MODEL
    max_tokens = max_tokens or DEFAULT_MAX_TOKENS
    temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Add some variance
        time.sleep(random.uniform(0.5, 1.5))
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"GPT API failed: {e}")
        # Simple fallback response
        return "I understand your point. Let me think about this issue from a different perspective."


