"""
Debug utilities for DeepAgents tools and components.
"""
import sys
import time
import json
import inspect
import functools
import traceback
from typing import Any, Callable, Dict, Optional, Union, TypeVar, cast

# Type variables for better type hints
F = TypeVar('F', bound=Callable[..., Any])
T = TypeVar('T')

# Global debug flag - can be set via CLI args
DEBUG_ENABLED = False

def set_debug(enabled: bool = True) -> None:
    """Enable or disable debug logging globally."""
    global DEBUG_ENABLED
    DEBUG_ENABLED = enabled

def is_debug_enabled() -> bool:
    """Check if debug logging is enabled."""
    return DEBUG_ENABLED

def truncate(obj: Any, max_length: int = 500) -> str:
    """Safely truncate any object for debug output."""
    try:
        if obj is None:
            return "None"
        
        # Convert to string representation
        if isinstance(obj, (dict, list)):
            s = json.dumps(obj, default=str)
        else:
            s = str(obj)
        
        # Truncate if needed
        if len(s) > max_length:
            return s[:max_length] + f"... [truncated, total length: {len(s)}]"
        return s
    except Exception as e:
        return f"[Error serializing object: {e}]"

def log_debug(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log a debug message to stderr if debug is enabled."""
    if not DEBUG_ENABLED:
        return
    
    try:
        # Format args and kwargs if provided
        formatted_args = ""
        if args:
            formatted_args = " ".join(truncate(arg) for arg in args)
        
        formatted_kwargs = ""
        if kwargs:
            formatted_kwargs = " ".join(f"{k}={truncate(v)}" for k, v in kwargs.items())
        
        # Combine all parts
        full_msg = f"[TOOL DEBUG] {msg}"
        if formatted_args:
            full_msg += f" {formatted_args}"
        if formatted_kwargs:
            full_msg += f" {formatted_kwargs}"
        
        # Write to stderr
        sys.stderr.write(f"{full_msg}\n")
        sys.stderr.flush()
    except Exception as e:
        sys.stderr.write(f"[TOOL DEBUG] Error in debug logging: {e}\n")
        sys.stderr.flush()

def debug_tool(func: F) -> F:
    """
    Decorator to add debug logging to any tool function.
    
    Logs:
    - Tool name and arguments when called
    - Return value or exception
    - Execution time
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not DEBUG_ENABLED:
            return func(*args, **kwargs)
        
        # Get tool name
        tool_name = getattr(func, "__name__", str(func))
        
        # Log call with args
        arg_str = ", ".join([truncate(a) for a in args[1:]] if len(args) > 1 else [])
        kwarg_str = ", ".join(f"{k}={truncate(v)}" for k, v in kwargs.items())
        params = []
        if arg_str:
            params.append(arg_str)
        if kwarg_str:
            params.append(kwarg_str)
        
        log_debug(f"→ Tool call: {tool_name}({', '.join(params)})")
        
        # Track execution time
        start_time = time.time()
        
        try:
            # Call the original function
            result = func(*args, **kwargs)
            
            # Log result
            elapsed = time.time() - start_time
            log_debug(f"← Tool result: {tool_name} ({elapsed:.3f}s): {truncate(result)}")
            
            return result
        except Exception as e:
            # Log exception
            elapsed = time.time() - start_time
            log_debug(f"✗ Tool error: {tool_name} ({elapsed:.3f}s): {e}")
            log_debug(f"Stack trace: {traceback.format_exc()}")
            
            # Re-raise the exception
            raise
    
    return cast(F, wrapper)

def debug_mcp_tool(func: F) -> F:
    """
    Decorator specifically for MCP tools.
    Similar to debug_tool but with MCP-specific formatting.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not DEBUG_ENABLED:
            return func(*args, **kwargs)
        
        # Get tool name
        tool_name = getattr(func, "__name__", str(func))
        
        # Format args for logging (skip self if it's a method)
        params = {}
        for k, v in kwargs.items():
            params[k] = v
        
        # Log call
        log_debug(f"→ MCP Tool call: {tool_name}({truncate(params)})")
        
        # Track execution time
        start_time = time.time()
        
        try:
            # Call the original function
            result = func(*args, **kwargs)
            
            # Log result
            elapsed = time.time() - start_time
            log_debug(f"← MCP Tool result: {tool_name} ({elapsed:.3f}s): {truncate(result)}")
            
            return result
        except Exception as e:
            # Log exception
            elapsed = time.time() - start_time
            log_debug(f"✗ MCP Tool error: {tool_name} ({elapsed:.3f}s): {e}")
            log_debug(f"Stack trace: {traceback.format_exc()}")
            
            # Re-raise the exception
            raise
    
    return cast(F, wrapper)

def wrap_all_tools(module: Any) -> None:
    """
    Wrap all tool functions in a module with debug_tool decorator.
    """
    if not DEBUG_ENABLED:
        return
    
    count = 0
    for name, obj in inspect.getmembers(module):
        # Skip private methods and non-callables
        if name.startswith('_') or not callable(obj):
            continue
        
        # Skip already wrapped functions
        if hasattr(obj, '__wrapped__'):
            continue
        
        # Check if it's a tool (has a name attribute or is decorated with @tool)
        if hasattr(obj, 'name') or getattr(obj, '__module__', '').startswith('langchain'):
            setattr(module, name, debug_tool(obj))
            count += 1
    
    if count > 0:
        log_debug(f"Wrapped {count} tools in module {module.__name__}")
