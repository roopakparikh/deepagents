#!/usr/bin/env python3
"""
Test script to demonstrate the DeepAgents CLI functionality.
Run this to test the CLI without installing the package.
"""

import sys
import os

# Add the src directory to Python path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from deepagents.cli import main

if __name__ == "__main__":
    print("🧪 Testing DeepAgents CLI...")
    print("This will start the interactive CLI session.")
    print("Try typing '@' followed by TAB to test file path completion!")
    print()
    main()
