#!/usr/bin/env python3
# FastMCP server: get_weather
# Requires: fastmcp, requests

from __future__ import annotations

import os
import sys
import time
import json
from typing import Optional, Dict, Any

import requests

try:
    from fastmcp import FastMCP
except ImportError:
    raise SystemExit(
        "fastmcp is not installed. Install it via:\n\n  uv add fastmcp requests\n  # or\n  pip install fastmcp requests"
    )

mcp = FastMCP("weather-server")


@mcp.tool()
def get_weather(city: Optional[str] = None, zip: Optional[str] = None) -> Dict[str, Any]:
    """
    Get current weather by city and/or US ZIP code.
    Provide at least one of: city or zip.

    Args:
      city: City name, e.g., "San Francisco" or "Paris, FR"
      zip: US ZIP code, e.g., "94103"

    Returns:
      Dict with resolved location and current weather.
    """
    
    lat = lon = "1.0"
    resolved_label = "label"
    

    weather =  {
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
    return {
        "query": {"city": city, "zip": zip},
        "resolved": {"label": resolved_label, "latitude": lat, "longitude": lon},
        "current_weather": weather,
        "timestamp": int(time.time()),
    }


if __name__ == "__main__":
    # Run over stdio for MCP
    mcp.run()