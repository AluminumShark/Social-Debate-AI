#!/bin/bash

# Social Debate AI - Flask Startup Script
# Run this script to start the web interface

set -e

echo "Starting Social Debate AI..."
echo "================================"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 not found"
    exit 1
fi

# Check virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Using virtual environment: $VIRTUAL_ENV"
else
    echo "Warning: No virtual environment detected"
fi

# Install dependencies if needed
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found"
    exit 1
fi

echo "Installing dependencies..."
pip install -r requirements.txt

# Check environment variables
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY not set"
    echo "Please set your OpenAI API key:"
    echo "export OPENAI_API_KEY='your-api-key-here'"
fi

echo "Starting Flask server..."
echo "Access at: http://localhost:5000"
echo "Press Ctrl+C to stop"
echo ""

python3 run_flask.py 