#!/usr/bin/env python3
"""
CLI interface for DeepAgents with file path autocompletion.
Provides interactive command-line interface with '@' triggered file path completion.
This version uses prompt_toolkit for robust, cross-platform tab completion.
"""

import os
import sys
import argparse
import glob
import json
from typing import Optional
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.model import get_default_model
import asyncio

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from langchain_mcp_adapters.client import MultiServerMCPClient



def _get_prog_name() -> str:
    """Return the invoked program name (supports symlink renaming).

    Uses sys.argv[0] basename; if empty, falls back to 'deepagents-cli'.
    """
    try:
        base = os.path.basename(sys.argv[0]) or "deepagents-cli"
        return base
    except Exception:
        return "deepagents-cli"


class AtPathCompleter(Completer):
    """prompt_toolkit completer that completes filesystem paths when the
    token being completed starts with '@'. The inserted text keeps the '@' prefix.
    """

    def __init__(self, cwd: Optional[str] = None):
        self.cwd = cwd or os.getcwd()

    def _list_matches(self, path_text: str) -> list[str]:
        # Expand user home (~) and environment vars
        expanded = os.path.expandvars(os.path.expanduser(path_text))
        # Build search pattern
        if expanded == "":
            pattern = os.path.join(self.cwd, "*")
        elif os.path.isabs(expanded):
            pattern = expanded + "*"
        else:
            pattern = os.path.join(self.cwd, expanded + "*")

        matches = []
        for match in glob.glob(pattern):
            if not os.path.isabs(expanded) and match.startswith(self.cwd + os.sep):
                rel = os.path.relpath(match, self.cwd)
            else:
                rel = match
            if os.path.isdir(match):
                matches.append(rel + "/")
            else:
                matches.append(rel)
        matches.sort()
        return matches

    def get_completions(self, document: Document, complete_event):
        text = document.text_before_cursor
        # Find last token boundary (space-separated for simplicity)
        last_space = text.rfind(" ")
        token_start = last_space + 1
        token = text[token_start:]
        if not token.startswith("@"):
            return
        path_text = token[1:]

        for match in self._list_matches(path_text):
            # Insert with '@' prefix, replace the whole token
            display = match
            insert_text = "@" + match
            yield Completion(
                insert_text,
                start_position=-(len(token)),
                display=display,
            )


def _build_session(root: Optional[str] = None) -> PromptSession:
    """Create a PromptSession with our '@' path completer.

    If 'root' is provided, it will be used as the base directory for
    path completion; otherwise the current working directory is used.
    """
    return PromptSession(completer=AtPathCompleter(cwd=root))


def _load_config(config_path: Optional[str]) -> dict:
    """Load CLI config from a JSON file.

    Supports either a full object with fields like {"servers": [...], "model": {...}}
    or, for backward compatibility, a direct array that is treated as servers.
    """
    default = {"servers": [], "model": {}}
    if not config_path:
        return default
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return {"servers": data, "model": {}}
        if not isinstance(data, dict):
            print("⚠️ Invalid config format: expected object or array. Ignoring and continuing with defaults.")
            return default
        servers = data.get("servers", [])
        if not isinstance(servers, list):
            print("⚠️ Invalid 'servers' value in config: expected an array. Ignoring servers.")
            servers = []
        model_cfg = data.get("model", {})
        if not isinstance(model_cfg, dict):
            print("⚠️ Invalid 'model' value in config: expected an object. Ignoring model settings.")
            model_cfg = {}
        return {"servers": servers, "model": model_cfg}
    except FileNotFoundError:
        print(f"⚠️ Config file not found: {config_path}. Continuing with defaults.")
        return default
    except Exception as e:
        print(f"⚠️ Failed to load config '{config_path}': {e}. Continuing with defaults.")
        return default


def _build_model_from_config(model_cfg: dict):
    """Build a model instance using config values by temporarily applying
    environment overrides and delegating to get_default_model()."""
    if not model_cfg:
        return None
    overrides = {}
    provider = model_cfg.get("provider")
    name = model_cfg.get("name") or model_cfg.get("model")
    max_tokens = model_cfg.get("max_tokens")
    base_url = model_cfg.get("base_url")

    if provider:
        overrides["DEEPAGENTS_MODEL_PROVIDER"] = str(provider)
    if name:
        overrides["DEEPAGENTS_MODEL_NAME"] = str(name)
    if max_tokens is not None:
        overrides["DEEPAGENTS_MAX_TOKENS"] = str(max_tokens)
    if base_url:
        overrides["DEEPAGENTS_OLLAMA_BASE_URL"] = str(base_url)

    # Save originals and apply overrides
    originals = {k: os.environ.get(k) for k in overrides}
    try:
        os.environ.update(overrides)
        return get_default_model()
    finally:
        # Restore originals (unset keys that didn't exist)
        for k, v in originals.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def create_interactive_session(config_path: Optional[str] = None, root: Optional[str] = None):
    """Create an interactive CLI session with the deep agent."""
    prog = _get_prog_name()
    print(f"🧠🤖 {prog} Interactive CLI")
    print("=" * 40)
    print("Type your requests below. Use '@' followed by TAB to autocomplete file paths.")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'help' for more commands.")
    print("Type 'test-completion' to test the tab completion feature.")
    print()
    
    # Resolve optional root for path completion
    resolved_root = None
    if root:
        try:
            candidate = os.path.abspath(os.path.expanduser(os.path.expandvars(root)))
            if os.path.isdir(candidate):
                resolved_root = candidate
            else:
                print(f"⚠️ Provided --root is not a directory: {candidate}. Using current working directory.")
        except Exception as e:
            print(f"⚠️ Failed to resolve --root '{root}': {e}. Using current working directory.")

    # Set up prompt_toolkit session with completion
    session = _build_session(root=resolved_root)
    # Collect MCP tools (handle async API gracefully)
    mcp_tools = []
    try:
        cfg = _load_config(config_path)
        servers = cfg.get("servers", [])
        # If --root provided, inject it into any filesystem MCP server env as ROOT
        if servers and resolved_root:
            for srv in servers:
                try:
                    if isinstance(srv, dict) and srv.get("name") == "filesystem":
                        # Ensure env ROOT matches --root
                        env = srv.get("env") or {}
                        prev = env.get("ROOT")
                        env["ROOT"] = resolved_root
                        srv["env"] = env
                        if prev and prev != resolved_root:
                            print(
                                f"ℹ️ Overriding MCP filesystem ROOT env from '{prev}' to '--root' value '{resolved_root}'."
                            )

                        # Also pass the root as a positional argument for servers that expect it
                        args = srv.get("args")
                        if not isinstance(args, list):
                            args = [] if args is None else [str(args)]

                        # If the last argument looks like an absolute path and differs, warn and replace
                        if args and isinstance(args[-1], str) and args[-1].startswith("/") and args[-1] != resolved_root:
                            print(
                                f"ℹ️ Overriding MCP filesystem positional root arg from '{args[-1]}' to '{resolved_root}'."
                            )
                            args[-1] = resolved_root
                        elif resolved_root not in args:
                            args.append(resolved_root)

                        srv["args"] = args
                except Exception:
                    # Best-effort injection; ignore malformed entries
                    pass
        # Normalize server configs for broad compatibility (e.g., transport may be string or object)
        def _normalize_server_cfg(srv: dict) -> dict:
            try:
                cfg = dict(srv)
                # Normalize transport: some versions expect a plain string like 'stdio'
                tr = cfg.get("transport")
                if isinstance(tr, dict) and "type" in tr:
                    cfg["transport"] = tr.get("type")
                return cfg
            except Exception:
                return srv

        servers = [_normalize_server_cfg(s) if isinstance(s, dict) else s for s in servers]

        # Initialize MultiServerMCPClient with maximum compatibility across versions
        mcp_client = None
        servers_input = servers
        if servers:
            try:
                # Newer API may accept keyword
                mcp_client = MultiServerMCPClient(servers=servers_input)
            except TypeError:
                try:
                    # Some versions accept positional only
                    mcp_client = MultiServerMCPClient(servers_input)
                except Exception:
                    # Fallback to no-arg
                    mcp_client = MultiServerMCPClient()
        else:
            mcp_client = MultiServerMCPClient()

        def _get_tools(client):
            res = client.get_tools()
            if asyncio.iscoroutine(res):
                return asyncio.run(res)
            return res

        try:
            mcp_tools = _get_tools(mcp_client)
        except AttributeError as e:
            # Some versions expect a dict mapping name->server_config and internally call .values()
            if "has no attribute 'values'" in str(e) and isinstance(servers, list):
                try:
                    servers_dict = {}
                    for srv in servers:
                        if not isinstance(srv, dict) or 'name' not in srv:
                            continue
                        name = srv['name']
                        cfg_clean = {k: v for k, v in srv.items() if k != 'name'}
                        cfg_clean = _normalize_server_cfg(cfg_clean)
                        servers_dict[name] = cfg_clean
                    # Retry client construction with dict form
                    try:
                        mcp_client = MultiServerMCPClient(servers=servers_dict)
                    except TypeError:
                        mcp_client = MultiServerMCPClient(servers_dict)
                    mcp_tools = _get_tools(mcp_client)
                except Exception as inner:
                    print(f"⚠️ MCP tools unavailable: {inner}. Continuing without MCP tools.")
                    mcp_tools = []
            else:
                raise
    except Exception as e:
        print(f"⚠️ MCP tools unavailable: {e}. Continuing without MCP tools.")
    
    # Create a basic deep agent (user can customize this)
    model = None
    try:
        model = _build_model_from_config(cfg.get("model", {}))
    except Exception as e:
        print(f"⚠️ Failed to build model from config: {e}. Falling back to default model.")
        model = None
    agent = create_deep_agent(
        tools=mcp_tools,
        instructions="You are a helpful assistant with file system access and task management capabilities.",
        model=model,
    )
    
    # Initialize agent state
    state = {
        "messages": [],
        "files": {},
        "todos": []
    }
    
    while True:
        try:
            # Get user input with prompt_toolkit (supports tab completion)
            user_input = session.prompt("🤖 > ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye! 👋")
                break
                
            if user_input.lower() == 'help':
                print_help()
                continue
                
            if user_input.lower() == 'status':
                print_status(state)
                continue
                
            if user_input.lower() == 'clear':
                os.system('clear' if os.name == 'posix' else 'cls')
                continue
                
            if user_input.lower() == 'test-completion':
                test_completion()
                continue
            
            # Process the request with the agent
            print("\n🔄 Processing request...")
            
            # Add user message to state
            state["messages"] = [{"role": "user", "content": user_input}]
            
            # Invoke the agent
            try:
                result = agent.invoke(state)
                
                # Update state with result
                state.update(result)
                
                # Display the response
                if result.get("messages"):
                    last_message = result["messages"][-1]
                    if hasattr(last_message, 'content'):
                        print(f"\n✅ {last_message.content}\n")
                    else:
                        print(f"\n✅ {last_message}\n")
                else:
                    print("\n✅ Task completed.\n")
                    
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except EOFError:
            print("\nGoodbye! 👋")
            break


def test_completion():
    """Test the file path completion functionality."""
    print("\n🧪 Testing File Path Completion")
    print("-" * 30)
    
    completer = FilePathCompleter()
    
    # Test cases
    test_cases = ['@', '@src', '@/Users', '@.', '@..']
    
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
    print("Try typing '@' followed by TAB in the CLI to test interactively!")
    print()


def print_help():
    """Print help information."""
    help_text = """
Available Commands:
  help            - Show this help message
  status          - Show current agent state (todos, files)
  clear           - Clear the terminal screen
  test-completion - Test file path completion functionality
  exit            - Exit the CLI
  quit            - Exit the CLI

File Path Completion:
  Type '@' followed by a path and press TAB to autocomplete file paths.
  Examples:
    @/home/user/doc<TAB>     - Complete absolute paths
    @src/deep<TAB>           - Complete relative paths
    @../parent<TAB>          - Complete parent directory paths

Agent Capabilities:
  - Task management with todos
  - File system operations (read, write, edit files)
  - Complex multi-step task planning
  - Sub-agent spawning for specialized tasks
    """
    print(help_text)


def print_status(state):
    """Print current agent state."""
    print("\n📊 Agent Status:")
    print("-" * 20)
    
    # Show todos
    todos = state.get("todos", [])
    if todos:
        print(f"📝 Todos ({len(todos)}):")
        for i, todo in enumerate(todos, 1):
            status_emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}.get(todo["status"], "❓")
            print(f"  {i}. {status_emoji} {todo['content']} ({todo['status']})")
    else:
        print("📝 No todos")
    
    # Show files
    files = state.get("files", {})
    if files:
        print(f"\n📁 Files ({len(files)}):")
        for filepath in sorted(files.keys()):
            file_size = len(files[filepath])
            print(f"  📄 {filepath} ({file_size} chars)")
    else:
        print("\n📁 No files in agent filesystem")
    
    print()


def main():
    """Main CLI entry point."""
    prog = _get_prog_name()
    parser = argparse.ArgumentParser(
        prog=prog,
        description=f"{prog} with file path autocompletion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f"""
Examples:
  {prog}                    # Start interactive session
  {prog} --help            # Show this help
            """
        ),
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="DeepAgents CLI 0.1.0"
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        type=str,
        help=(
            "Path to JSON config. Supports {servers: [...]} and optional {model: {provider, name, max_tokens, base_url}}. "
            "Also supports passing an array directly as servers (backward compatible)."
        ),
    )
    parser.add_argument(
        "--root",
        dest="root",
        type=str,
        help="Root directory used for '@' file path autocompletion.",
    )
    
    args = parser.parse_args()
    
    # Start interactive session
    create_interactive_session(config_path=args.config_path, root=args.root)


if __name__ == "__main__":
    main()
