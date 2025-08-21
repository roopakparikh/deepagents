from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from typing import Annotated, Any, Dict, List, Optional, Union, cast
from langgraph.prebuilt import InjectedState
import functools
import traceback
import time
import sys

from deepagents.prompts import (
    WRITE_TODOS_DESCRIPTION,
    EDIT_DESCRIPTION,
    TOOL_DESCRIPTION,
)
from deepagents.state import Todo, DeepAgentState
from deepagents.debug_utils import debug_tool, log_debug, is_debug_enabled


@tool(description="Sleep for specified seconds")
@debug_tool
def sleep_tool(seconds: int) -> Command:
    time.sleep(seconds)
    return Command(
        update={
            "messages": [
                ToolMessage(f"Slept for {seconds} seconds", tool_call_id=tool_call_id)
            ],
        }
    )

@tool(description=WRITE_TODOS_DESCRIPTION)
@debug_tool
def write_todos(
    todos: list[Todo], tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    if is_debug_enabled():
        log_debug(f"Updating todos list with {len(todos)} items")
    
    return Command(
        update={
            "todos": todos,
            "messages": [
                ToolMessage(f"Updated todo list to {todos}", tool_call_id=tool_call_id)
            ],
        }
    )


@debug_tool
def ls(state: Annotated[DeepAgentState, InjectedState]) -> list[str]:
    """List all files"""
    files = list(state.get("files", {}).keys())
    if is_debug_enabled():
        log_debug(f"Listed {len(files)} files")
    return files


@tool(description=TOOL_DESCRIPTION)
@debug_tool
def read_file(
    file_path: str,
    state: Annotated[DeepAgentState, InjectedState],
    offset: int = 0,
    limit: int = 2000,
) -> str:
    """Read file."""
    if is_debug_enabled():
        log_debug(f"Reading file '{file_path}' with offset={offset}, limit={limit}")
    
    mock_filesystem = state.get("files", {})
    if file_path not in mock_filesystem:
        if is_debug_enabled():
            log_debug(f"File not found: '{file_path}'")
        return f"Error: File '{file_path}' not found"

    # Get file content
    content = mock_filesystem[file_path]

    # Handle empty file
    if not content or content.strip() == "":
        if is_debug_enabled():
            log_debug(f"File '{file_path}' exists but is empty")
        return "System reminder: File exists but has empty contents"

    # Split content into lines
    lines = content.splitlines()
    if is_debug_enabled():
        log_debug(f"File '{file_path}' has {len(lines)} lines")

    # Apply line offset and limit
    start_idx = offset
    end_idx = min(start_idx + limit, len(lines))

    # Handle case where offset is beyond file length
    if start_idx >= len(lines):
        if is_debug_enabled():
            log_debug(f"Offset {offset} exceeds file length ({len(lines)} lines)")
        return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"

    # Format output with line numbers (cat -n format)
    result_lines = []
    for i in range(start_idx, end_idx):
        line_content = lines[i]

        # Truncate lines longer than 2000 characters
        if len(line_content) > 2000:
            line_content = line_content[:2000]
            if is_debug_enabled():
                log_debug(f"Truncated line {i+1} (length: {len(lines[i])})")

        # Line numbers start at 1, so add 1 to the index
        line_number = i + 1
        result_lines.append(f"{line_number:6d}\t{line_content}")

    if is_debug_enabled():
        log_debug(f"Returning {len(result_lines)} lines from file '{file_path}'")
    return "\n".join(result_lines)


@debug_tool
def write_file(
    file_path: str,
    content: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Write to a file."""
    if is_debug_enabled():
        content_preview = content[:100] + "..." if len(content) > 100 else content
        log_debug(f"Writing to file '{file_path}', content length: {len(content)}")
        log_debug(f"Content preview: {content_preview}")
    
    files = state.get("files", {})
    # Check if file already exists
    is_new_file = file_path not in files
    files[file_path] = content
    
    if is_debug_enabled():
        action = "Created" if is_new_file else "Updated"
        log_debug(f"{action} file '{file_path}'")
    
    return Command(
        update={
            "files": files,
            "messages": [
                ToolMessage(f"Updated file {file_path}", tool_call_id=tool_call_id)
            ],
        }
    )


@tool(description=EDIT_DESCRIPTION)
@debug_tool
def edit_file(
    file_path: str,
    old_string: str,
    new_string: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    replace_all: bool = False,
) -> str:
    """Write to a file."""
    if is_debug_enabled():
        old_preview = old_string[:50] + "..." if len(old_string) > 50 else old_string
        new_preview = new_string[:50] + "..." if len(new_string) > 50 else new_string
        log_debug(f"Editing file '{file_path}', replace_all={replace_all}")
        log_debug(f"Replacing: '{old_preview}' with '{new_preview}'")
    
    mock_filesystem = state.get("files", {})
    # Check if file exists in mock filesystem
    if file_path not in mock_filesystem:
        if is_debug_enabled():
            log_debug(f"File not found: '{file_path}'")
        return f"Error: File '{file_path}' not found"

    # Get current file content
    content = mock_filesystem[file_path]

    # Check if old_string exists in the file
    if old_string not in content:
        if is_debug_enabled():
            log_debug(f"String not found in file: '{old_preview}'")
        return f"Error: String not found in file: '{old_string}'"

    # If not replace_all, check for uniqueness
    if not replace_all:
        occurrences = content.count(old_string)
        if is_debug_enabled():
            log_debug(f"Found {occurrences} occurrences of the string in file")
            
        if occurrences > 1:
            if is_debug_enabled():
                log_debug(f"Multiple occurrences found but replace_all=False")
            return f"Error: String '{old_string}' appears {occurrences} times in file. Use replace_all=True to replace all instances, or provide a more specific string with surrounding context."
        elif occurrences == 0:
            if is_debug_enabled():
                log_debug(f"String not found in file: '{old_preview}'")
            return f"Error: String not found in file: '{old_string}'"

    # Perform the replacement
    if replace_all:
        new_content = content.replace(old_string, new_string)
        replacement_count = content.count(old_string)
        result_msg = f"Successfully replaced {replacement_count} instance(s) of the string in '{file_path}'"
        if is_debug_enabled():
            log_debug(f"Replaced {replacement_count} instances in file '{file_path}'")
    else:
        new_content = content.replace(
            old_string, new_string, 1
        )  # Replace only first occurrence
        result_msg = f"Successfully replaced string in '{file_path}'"
        if is_debug_enabled():
            log_debug(f"Replaced 1 instance in file '{file_path}'")

    # Update the mock filesystem
    mock_filesystem[file_path] = new_content
    return Command(
        update={
            "files": mock_filesystem,
            "messages": [
                ToolMessage(f"Updated file {file_path}", tool_call_id=tool_call_id)
            ],
        }
    )
