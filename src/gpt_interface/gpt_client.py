"""
GPT client interface
"""

import openai
import time
import random

try:
    from .config import OPENAI_API_KEY
    openai.api_key = OPENAI_API_KEY
except ImportError:
    print("No API key configured, using default")

def chat(prompt: str, model: str = None) -> str:
    if not model:
        model = "gpt-3.5-turbo"
    
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7
        )
        
        # Add some variance
        time.sleep(random.uniform(0.5, 1.5))
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"GPT API failed: {e}")
        # Simple fallback response
        return "I understand your point. Let me think about this issue from a different perspective."


