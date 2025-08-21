from deepagents.prompts import TASK_DESCRIPTION_PREFIX, TASK_DESCRIPTION_SUFFIX
from deepagents.state import DeepAgentState
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import BaseTool
from typing import TypedDict, Any, Dict, List, Optional, Union, cast
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from typing import Annotated, NotRequired
from langgraph.types import Command
import time
import traceback
import sys

from langgraph.prebuilt import InjectedState
from deepagents.debug_utils import debug_tool, log_debug, is_debug_enabled, set_debug


class SubAgent(TypedDict):
    name: str
    description: str
    prompt: str
    tools: NotRequired[list[str]]


def _create_task_tool(tools, instructions, subagents: list[SubAgent], model, state_schema):
    if is_debug_enabled():
        log_debug("Creating task tool with subagents: " + ", ".join([a.get("name", "unnamed") for a in subagents]))
    
    agents = {
        "general-purpose": create_react_agent(model, prompt=instructions, tools=tools)
    }
    
    if is_debug_enabled():
        log_debug(f"Created general-purpose agent with {len(tools)} tools")
    
    tools_by_name = {}
    for tool_ in tools:
        if not isinstance(tool_, BaseTool):
            tool_ = tool(tool_)
        tools_by_name[tool_.name] = tool_
    
    for _agent in subagents:
        agent_name = _agent.get("name", "unnamed")
        if is_debug_enabled():
            log_debug(f"Creating subagent '{agent_name}'")
            
        if "tools" in _agent:
            _tools = [tools_by_name[t] for t in _agent["tools"]]
            if is_debug_enabled():
                log_debug(f"Subagent '{agent_name}' using {len(_tools)} specific tools: {', '.join([t.name for t in _tools])}")
        else:
            _tools = tools
            if is_debug_enabled():
                log_debug(f"Subagent '{agent_name}' using all {len(_tools)} tools")
                
        agents[_agent["name"]] = create_react_agent(
            model, prompt=_agent["prompt"], tools=_tools, state_schema=state_schema
        )
        if is_debug_enabled():
            log_debug(f"Subagent '{agent_name}' created successfully")

    other_agents_string = [
        f"- {_agent['name']}: {_agent['description']}" for _agent in subagents
    ]
    
    if is_debug_enabled():
        log_debug(f"Task tool will have access to {len(agents)} agents")

    @tool(
        description=TASK_DESCRIPTION_PREFIX.format(other_agents=other_agents_string)
        + TASK_DESCRIPTION_SUFFIX
    )
    @debug_tool
    def task(
        description: str,
        subagent_type: str,
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        if is_debug_enabled():
            desc_preview = description[:100] + "..." if len(description) > 100 else description
            log_debug(f"Task tool called with subagent_type='{subagent_type}', description='{desc_preview}'")
            log_debug(f"Available agent types: {list(agents.keys())}")
        
        if subagent_type not in agents:
            error_msg = f"Error: invoked agent of type {subagent_type}, the only allowed types are {[f'`{k}`' for k in agents]}"
            if is_debug_enabled():
                log_debug(f"Invalid subagent type: '{subagent_type}'")
            return error_msg
        
        if is_debug_enabled():
            log_debug(f"Invoking subagent of type '{subagent_type}'")
        
        sub_agent = agents[subagent_type]
        state["messages"] = [{"role": "user", "content": description}]
        
        start_time = time.time()
        try:
            if is_debug_enabled():
                log_debug(f"Starting subagent '{subagent_type}' execution")
            
            result = sub_agent.invoke(state)
            
            elapsed = time.time() - start_time
            if is_debug_enabled():
                log_debug(f"Subagent '{subagent_type}' completed in {elapsed:.2f}s")
                log_debug(f"Subagent result has {len(result.get('files', {}))} files and {len(result.get('messages', []))} messages")
            
            return Command(
                update={
                    "files": result.get("files", {}),
                    "messages": [
                        ToolMessage(
                            result["messages"][-1].content, tool_call_id=tool_call_id
                        )
                    ],
                }
            )
        except Exception as e:
            elapsed = time.time() - start_time
            if is_debug_enabled():
                log_debug(f"Subagent '{subagent_type}' failed after {elapsed:.2f}s with error: {e}")
                log_debug(f"Stack trace: {traceback.format_exc()}")
            raise

    return task
