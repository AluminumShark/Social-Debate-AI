#!/usr/bin/env python
"""
Flask startup script for Social Debate AI
"""

import sys
import os
import traceback
from datetime import datetime

# Ensure app can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_banner():
    print("\n" + "=" * 50)
    print("Social Debate AI - Web Interface")
    print("=" * 50)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50 + "\n")

if __name__ == '__main__':
    try:
        print_banner()
        
        print("Loading Flask app...")
        from ui.app import app, init_system
        print("Flask app loaded")
        
        print("Initializing system...")
        init_result = init_system()
        
        if init_result:
            print("System initialized successfully")
            
            print("\nService info:")
            print("  Local: http://localhost:5000")
            print("  Network: http://0.0.0.0:5000")
            print("  Debug: enabled")
            
            print("\nPress Ctrl+C to stop")
            print("=" * 50 + "\n")
            
            app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
        else:
            print("System initialization failed")
            print("Check:")
            print("  - OPENAI_API_KEY environment variable")
            print("  - Required model files")
            print("  - Configuration files")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
        print("Thanks for using Social Debate AI!\n")
    except Exception as e:
        print(f"\nStartup failed: {e}")
        print("\nError details:")
        traceback.print_exc()
        sys.exit(1) 