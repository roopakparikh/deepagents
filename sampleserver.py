#!/usr/bin/env python3
# FastMCP server: get_weather with op-based API
# Requires: fastmcp, requests

from __future__ import annotations

import os
import sys
import time
import json
import uuid
import threading
import traceback
from typing import Optional, Dict, Any, List, Callable, TypeVar, cast
import functools

import requests

try:
    from fastmcp import FastMCP
except ImportError:
    raise SystemExit(
        "fastmcp is not installed. Install it via:\n\n  uv add fastmcp requests\n  # or\n  pip install fastmcp requests"
    )

# Type variable for function annotations
F = TypeVar('F', bound=Callable[..., Any])

# Debug flag - can be enabled via environment variable
DEBUG_ENABLED = os.environ.get("DEEPAGENTS_DEBUG", "0").lower() in ("1", "true", "yes")

mcp = FastMCP("weather-server")

# Debug utilities
def set_debug(enabled: bool = True) -> None:
    """Enable or disable debug logging."""
    global DEBUG_ENABLED
    DEBUG_ENABLED = enabled

def log_debug(msg: str, *args: Any, **kwargs: Any) -> None:
    """Log a debug message to stderr if debug is enabled."""
    if not DEBUG_ENABLED:
        return
    
    try:
        # Format args and kwargs if provided
        formatted_args = ""
        if args:
            formatted_args = " ".join(str(arg)[:500] for arg in args)
        
        formatted_kwargs = ""
        if kwargs:
            formatted_kwargs = " ".join(f"{k}={str(v)[:500]}" for k, v in kwargs.items())
        
        # Combine all parts
        full_msg = f"[MCP DEBUG] {msg}"
        if formatted_args:
            full_msg += f" {formatted_args}"
        if formatted_kwargs:
            full_msg += f" {formatted_kwargs}"
        
        # Write to stderr
        sys.stderr.write(f"{full_msg}\n")
        sys.stderr.flush()
    except Exception as e:
        sys.stderr.write(f"[MCP DEBUG] Error in debug logging: {e}\n")
        sys.stderr.flush()

def debug_tool(func: F) -> F:
    """Decorator to add debug logging to any tool function."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not DEBUG_ENABLED:
            return func(*args, **kwargs)
        
        # Get tool name
        tool_name = getattr(func, "__name__", str(func))
        
        # Format args for logging (skip self if it's a method)
        params = {}
        for k, v in kwargs.items():
            # Truncate large values for readability
            if isinstance(v, str) and len(v) > 100:
                params[k] = f"{v[:100]}... [truncated]"
            else:
                params[k] = v
        
        # Log call
        log_debug(f"→ Tool call: {tool_name}({params})")
        
        # Track execution time
        start_time = time.time()
        
        try:
            # Call the original function
            result = func(*args, **kwargs)
            
            # Log result (truncated for large results)
            elapsed = time.time() - start_time
            result_str = str(result)
            if len(result_str) > 500:
                result_str = f"{result_str[:500]}... [truncated, total length: {len(result_str)}]"
            log_debug(f"← Tool result: {tool_name} ({elapsed:.3f}s): {result_str}")
            
            return result
        except Exception as e:
            # Log exception
            elapsed = time.time() - start_time
            log_debug(f"✗ Tool error: {tool_name} ({elapsed:.3f}s): {e}")
            log_debug(f"Stack trace: {traceback.format_exc()}")
            
            # Re-raise the exception
            raise
    
    return cast(F, wrapper)

# Enable debug if environment variable is set
if os.environ.get("DEEPAGENTS_DEBUG", "0").lower() in ("1", "true", "yes"):
    log_debug("Debug logging enabled for MCP tools")

# In-memory op storage
# In a production system, this would be a database
OPS = {}

# Op statuses
STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


def generate_op_id() -> str:
    """Generate a unique op ID"""
    return str(uuid.uuid4())


def process_weather_op(op_id: str, city: Optional[str], zip_code: Optional[str]) -> None:
    """Process a weather op in the background"""
    try:
        if DEBUG_ENABLED:
            log_debug(f"Processing op {op_id} for city={city}, zip={zip_code}")
        
        # Update op status to processing
        OPS[op_id]["status"] = STATUS_PROCESSING
        
        # Simulate processing delay (1-3 minutes)
        delay = 1 + (hash(op_id) % 3*60)
        if DEBUG_ENABLED:
            log_debug(f"Op {op_id} will take {delay} seconds to complete")
        time.sleep(delay)
        
        # Generate weather data
        lat = lon = "1.0"
        resolved_label = f"Location for {city or zip_code}"
        
        weather = {
            "temperature_c": "22.5",
            "windspeed_kmh": "3",
            "winddirection_deg": "12",
            "weathercode": "22",
            "units": {
                "temperature": "°C",
                "windspeed": "km/h",
                "winddirection": "deg",
            },
            "provider": "open-meteo",
        }
        
        # Update op with results
        OPS[op_id].update({
            "status": STATUS_COMPLETED,
            "result": {
                "query": {"city": city, "zip": zip_code},
                "resolved": {"label": resolved_label, "latitude": lat, "longitude": lon},
                "current_weather": weather,
                "timestamp": int(time.time()),
            },
            "completed_at": int(time.time())
        })
        
        if DEBUG_ENABLED:
            log_debug(f"Op {op_id} completed successfully")
        
    except Exception as e:
        # Update op with error
        OPS[op_id].update({
            "status": STATUS_FAILED,
            "error": str(e),
            "completed_at": int(time.time())
        })
        
        if DEBUG_ENABLED:
            log_debug(f"Op {op_id} failed with error: {e}")
            log_debug(f"Stack trace: {traceback.format_exc()}")


@mcp.tool()
@debug_tool
def get_weather(city: Optional[str] = None, zip: Optional[str] = None) -> Dict[str, Any]:
    """
    Request weather data for a city and/or US ZIP code.
    Returns a op_id that can be used with get_op_status to retrieve results. Sleep for 10-30 seconds
    before checking the op status.
    Provide at least one of: city or zip.

    Args:
      city: City name, e.g., "San Francisco" or "Paris, FR"
      zip: US ZIP code, e.g., "94103"

    Returns:
      Dict with op_id and initial status information.
    """
    # Validate input
    if not city and not zip:
        log_debug("Missing required parameters: both city and zip are None")
        return {
            "error": "Provide at least one of: 'city' or 'zip'",
            "status": "failed"
        }
    
    # Create a new op
    op_id = generate_op_id()
    created_at = int(time.time())
    log_debug(f"Created new weather op {op_id} for city={city}, zip={zip}")
    
    # Store op in memory
    OPS[op_id] = {
        "op_id": op_id,
        "status": STATUS_PENDING,
        "query": {"city": city, "zip": zip},
        "created_at": created_at,
    }
    
    # Start background processing
    thread = threading.Thread(
        target=process_weather_op,
        args=(op_id, city, zip),
        daemon=True
    )
    thread.start()
    log_debug(f"Started background processing thread for op {op_id}")
    
    # Return op information
    return {
        "op_id": op_id,
        "status": STATUS_PENDING,
        "created_at": created_at,
        "message": "Weather request accepted. Use get_op_status with this op_id to check results."
    }


@mcp.tool()
@debug_tool
def get_op_status(op_id: str) -> Dict[str, Any]:
    """
    Get the status and results of a previously submitted weather op.
    Sleep for 10-30 seconds before checking the op status again.
    
    Args:
      op_id: The op ID returned by get_weather
      
    Returns:
      Dict with op status and results if completed
    """
    # Check if op exists
    if op_id not in OPS:
        log_debug(f"Op ID '{op_id}' not found in OPS dictionary")
        return {
            "error": f"Op ID '{op_id}' not found",
            "status": "not_found"
        }
    
    # Return current op state
    op = OPS[op_id]
    log_debug(f"Retrieved op {op_id} with status {op['status']}")
    
    # Include result if completed
    if op["status"] == STATUS_COMPLETED:
        log_debug(f"Returning completed op {op_id} results")
        return {
            "op_id": op_id,
            "status": op["status"],
            "created_at": op["created_at"],
            "completed_at": op.get("completed_at"),
            "result": op["result"]
        }
    elif op["status"] == STATUS_FAILED:
        log_debug(f"Returning failed op {op_id} with error: {op.get('error', 'Unknown error')}")
        return {
            "op_id": op_id,
            "status": op["status"],
            "created_at": op["created_at"],
            "completed_at": op.get("completed_at"),
            "error": op.get("error", "Unknown error")
        }
    else:
        # Still pending or processing
        log_debug(f"Op {op_id} still in progress with status {op['status']}")
        return {
            "op_id": op_id,
            "status": op["status"],
            "created_at": op["created_at"],
            "message": "Op is still being processed. Check again later."
        }


@mcp.tool()
@debug_tool
def list_ops(limit: int = 10) -> Dict[str, Any]:
    """
    List recent weather ops and their statuses.
    
    Args:
      limit: Maximum number of ops to return (default: 10)
      
    Returns:
      Dict with list of recent ops
    """
    log_debug(f"Listing ops with limit={limit}, total ops: {len(OPS)}")
    
    # Get recent ops, sorted by creation time (newest first)
    recent_ops = sorted(
        [{
            "op_id": op_id,
            "status": op["status"],
            "query": op["query"],
            "created_at": op["created_at"],
            "completed_at": op.get("completed_at")
        } for op_id, op in OPS.items()],
        key=lambda t: t["created_at"],
        reverse=True
    )[:limit]
    
    log_debug(f"Returning {len(recent_ops)} recent ops")
    return {
        "ops": recent_ops,
        "count": len(recent_ops),
        "total": len(OPS)
    }


if __name__ == "__main__":
    # Check for debug flag in environment
    if DEBUG_ENABLED:
        log_debug("Starting MCP server with debug logging enabled")
    
    # Run over stdio for MCP
    try:
        mcp.run()
    except Exception as e:
        if DEBUG_ENABLED:
            log_debug(f"MCP server error: {e}")
            log_debug(f"Stack trace: {traceback.format_exc()}")
        raise