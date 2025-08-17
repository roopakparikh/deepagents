#!/usr/bin/env python3
"""
Standalone test for file path completion functionality.
This tests only the tab completion feature without requiring DeepAgents dependencies.
"""

import os
import sys
import readline
import glob
from typing import List, Optional


class FilePathCompleter:
    """Custom completer for file paths triggered by '@' symbol."""
    
    def __init__(self):
        self.current_dir = os.getcwd()
        self.completion_prefix = ""
        
    def complete_filepath(self, text: str, state: int) -> Optional[str]:
        """Complete file paths when '@' is detected."""
        if not text.startswith('@'):
            return None
            
        # Remove '@' prefix for path completion
        path_text = text[1:]
        
        if state == 0:
            # First call - generate all possible completions
            self.matches = []
            
            # Handle different path types
            if not path_text:
                # Just '@' typed - show current directory contents
                search_pattern = os.path.join(self.current_dir, '*')
            elif path_text.startswith('/'):
                # Absolute path
                search_pattern = path_text + '*'
            else:
                # Relative path
                if '/' in path_text:
                    # Path with directories
                    search_pattern = os.path.join(self.current_dir, path_text + '*')
                else:
                    # Simple filename in current directory
                    search_pattern = os.path.join(self.current_dir, path_text + '*')
            
            # Get all matching paths
            try:
                matches = glob.glob(search_pattern)
                for match in matches:
                    # Convert back to relative path if it was relative
                    if not path_text.startswith('/') and match.startswith(self.current_dir):
                        # Make relative to current directory
                        rel_path = os.path.relpath(match, self.current_dir)
                        if os.path.isdir(match):
                            self.matches.append('@' + rel_path + '/')
                        else:
                            self.matches.append('@' + rel_path)
                    else:
                        # Keep absolute path
                        if os.path.isdir(match):
                            self.matches.append('@' + match + '/')
                        else:
                            self.matches.append('@' + match)
                        
                # Sort matches for consistent ordering
                self.matches.sort()
            except Exception as e:
                # For debugging
                self.matches = [f'@ERROR: {str(e)}']
        
        # Return the next match
        try:
            return self.matches[state]
        except IndexError:
            return None


def setup_readline_completion():
    """Set up readline with custom file path completion."""
    completer = FilePathCompleter()
    
    def custom_completer(text: str, state: int) -> Optional[str]:
        """Custom completer that handles '@' for file paths."""
        try:
            line = readline.get_line_buffer()
            begin = readline.get_begidx()
            end = readline.get_endidx()
            
            # Debug info for testing
            if state == 0:
                print(f"\n[DEBUG] Completing: text='{text}', line='{line}', begin={begin}, end={end}")
            
            # Multiple strategies to detect '@' completion
            
            # Strategy 1: Check if text itself starts with '@'
            if text.startswith('@'):
                result = completer.complete_filepath(text, state)
                if state == 0 and result:
                    print(f"[DEBUG] Strategy 1 worked: {result}")
                return result
            
            # Strategy 2: Check the current word being completed
            current_word = line[begin:end]
            if current_word.startswith('@'):
                result = completer.complete_filepath(current_word, state)
                if state == 0 and result:
                    print(f"[DEBUG] Strategy 2 worked: {result}")
                return result
            
            # Strategy 3: Look for '@' anywhere in the line and check if we're completing after it
            if '@' in line:
                # Find the last '@' before the cursor position
                at_pos = line.rfind('@', 0, end)
                if at_pos != -1:
                    # Extract the part after '@' up to cursor
                    completion_text = '@' + line[at_pos + 1:end]
                    result = completer.complete_filepath(completion_text, state)
                    if state == 0 and result:
                        print(f"[DEBUG] Strategy 3 worked: {result}")
                    return result
            
            return None
            
        except Exception as e:
            if state == 0:
                print(f"[DEBUG] Completion error: {e}")
            return None
    
    # Configure readline with multiple settings for better compatibility
    readline.set_completer(custom_completer)
    
    # Try different readline configurations
    try:
        readline.parse_and_bind("tab: complete")
        readline.parse_and_bind("set completion-ignore-case on")
        readline.parse_and_bind("set show-all-if-ambiguous on")
        readline.parse_and_bind("set completion-query-items 100")
    except Exception as e:
        print(f"Warning: Some readline settings failed: {e}")
    
    # Set word delimiters - be more aggressive about keeping '@' as part of words
    try:
        readline.set_completer_delims(' \t\n`!#$%^&*()=+[{]}\\|;:\'",<>?')
    except Exception as e:
        print(f"Warning: Could not set completer delims: {e}")


def test_completion_programmatically():
    """Test the file path completion functionality programmatically."""
    print("\n🧪 Testing File Path Completion Programmatically")
    print("-" * 50)
    
    completer = FilePathCompleter()
    
    # Test cases
    test_cases = ['@', '@src', '@/Users', '@.', '@..', '@test']
    
    for test_case in test_cases:
        print(f"\nTesting: '{test_case}'")
        matches = []
        state = 0
        while True:
            match = completer.complete_filepath(test_case, state)
            if match is None:
                break
            matches.append(match)
            state += 1
            if state > 10:  # Prevent infinite loop
                break
        
        if matches:
            print(f"  Matches: {matches[:5]}")  # Show first 5 matches
            if len(matches) > 5:
                print(f"  ... and {len(matches) - 5} more")
        else:
            print("  No matches found")
    
    print(f"\nCurrent directory: {os.getcwd()}")


def interactive_test():
    """Interactive test session for tab completion."""
    print("\n🧠 Tab Completion Test Session")
    print("=" * 40)
    print("Type '@' followed by TAB to test file path completion.")
    print("Type 'test' to run programmatic tests.")
    print("Type 'help' for more info.")
    print("Type 'exit' or 'quit' to end the session.")
    print()
    
    # Set up completion
    setup_readline_completion()
    
    while True:
        try:
            user_input = input("🔍 > ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye! 👋")
                break
                
            if user_input.lower() == 'help':
                print("""
Available Commands:
  help  - Show this help message
  test  - Run programmatic completion tests
  exit  - Exit the test session
  quit  - Exit the test session

Tab Completion Test:
  Type '@' followed by a path and press TAB to autocomplete file paths.
  Examples:
    @<TAB>              - Show current directory contents
    @src<TAB>           - Complete paths starting with 'src'
    @/Users<TAB>        - Complete absolute paths
    @..<TAB>            - Complete parent directory paths
                """)
                continue
                
            if user_input.lower() == 'test':
                test_completion_programmatically()
                continue
                
            # Echo back what was typed (to see if @ completion worked)
            print(f"You typed: {user_input}")
            if '@' in user_input:
                print("✅ Great! The '@' symbol was detected in your input.")
                print("If tab completion worked, you should see completed file paths above.")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except EOFError:
            print("\nGoodbye! 👋")
            break


def main():
    """Main entry point."""
    print("🧪 DeepAgents Tab Completion Test")
    print("This tests the '@' triggered file path completion feature.")
    
    # First run programmatic tests
    test_completion_programmatically()
    
    # Then start interactive session
    interactive_test()


if __name__ == "__main__":
    main()
