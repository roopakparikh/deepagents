# Meta Agentic AI Framework with LangGraph and MCP Support
# Complete implementation with memory, conversation history, and long-running task support

import json
import sqlite3
import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
import os
from pathlib import Path
import yaml
import pickle
import hashlib
from abc import ABC, abstractmethod

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

# MCP related imports (simulated - in real implementation use official MCP SDK)
import subprocess
import requests
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data Models
@dataclass
class MemoryEntry:
    id: str
    key: str
    value: Any
    tags: List[str]
    created_at: datetime
    expires_at: Optional[datetime] = None
    importance: float = 1.0
    access_count: int = 0

@dataclass
class ConversationTurn:
    id: str
    session_id: str
    user_message: str
    assistant_message: str
    timestamp: datetime
    metadata: Dict[str, Any]
    tools_used: List[str]

@dataclass
class MCPServer:
    name: str
    uri: str
    tools: List[Dict[str, Any]]
    status: str = "disconnected"
    last_ping: Optional[datetime] = None

@dataclass
class LongRunningTask:
    task_id: str
    server_name: str
    tool_name: str
    status: str  # pending, running, completed, failed
    created_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None

class AgentState(BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)
    memory_context: Dict[str, Any] = Field(default_factory=dict)
    active_tasks: List[str] = Field(default_factory=list)
    user_instructions: str = ""
    session_id: str = ""
    iteration_count: int = 0

# Memory System
class MemorySystem:
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS memory (
                id TEXT PRIMARY KEY,
                key TEXT,
                value BLOB,
                tags TEXT,
                created_at TEXT,
                expires_at TEXT,
                importance REAL,
                access_count INTEGER
            )
        ''')
        conn.commit()
        conn.close()
    
    def store(self, key: str, value: Any, tags: List[str] = None, 
              expires_in: Optional[timedelta] = None, importance: float = 1.0) -> str:
        entry_id = str(uuid.uuid4())
        tags = tags or []
        expires_at = datetime.now() + expires_in if expires_in else None
        
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT INTO memory (id, key, value, tags, created_at, expires_at, importance, access_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            entry_id, key, pickle.dumps(value), json.dumps(tags),
            datetime.now().isoformat(), 
            expires_at.isoformat() if expires_at else None,
            importance, 0
        ))
        conn.commit()
        conn.close()
        
        logger.info(f"Stored memory: {key} with ID: {entry_id}")
        return entry_id
    
    def retrieve(self, key: str = None, tags: List[str] = None, limit: int = 10) -> List[MemoryEntry]:
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM memory WHERE (expires_at IS NULL OR expires_at > ?)"
        params = [datetime.now().isoformat()]
        
        if key:
            query += " AND key LIKE ?"
            params.append(f"%{key}%")
        
        if tags:
            for tag in tags:
                query += " AND tags LIKE ?"
                params.append(f"%{tag}%")
        
        query += " ORDER BY importance DESC, access_count DESC LIMIT ?"
        params.append(limit)
        
        cursor = conn.execute(query, params)
        results = []
        
        for row in cursor.fetchall():
            # Update access count
            conn.execute("UPDATE memory SET access_count = access_count + 1 WHERE id = ?", (row[0],))
            
            entry = MemoryEntry(
                id=row[0],
                key=row[1],
                value=pickle.loads(row[2]),
                tags=json.loads(row[3]),
                created_at=datetime.fromisoformat(row[4]),
                expires_at=datetime.fromisoformat(row[5]) if row[5] else None,
                importance=row[6],
                access_count=row[7]
            )
            results.append(entry)
        
        conn.commit()
        conn.close()
        return results
    
    def forget(self, key: str = None, entry_id: str = None):
        conn = sqlite3.connect(self.db_path)
        if entry_id:
            conn.execute("DELETE FROM memory WHERE id = ?", (entry_id,))
        elif key:
            conn.execute("DELETE FROM memory WHERE key = ?", (key,))
        conn.commit()
        conn.close()

# Conversation History System
class ConversationHistory:
    def __init__(self, db_path: str = "conversations.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                user_message TEXT,
                assistant_message TEXT,
                timestamp TEXT,
                metadata TEXT,
                tools_used TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def add_turn(self, session_id: str, user_msg: str, assistant_msg: str, 
                 metadata: Dict[str, Any] = None, tools_used: List[str] = None) -> str:
        turn_id = str(uuid.uuid4())
        metadata = metadata or {}
        tools_used = tools_used or []
        
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT INTO conversations (id, session_id, user_message, assistant_message, timestamp, metadata, tools_used)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            turn_id, session_id, user_msg, assistant_msg,
            datetime.now().isoformat(), json.dumps(metadata), json.dumps(tools_used)
        ))
        conn.commit()
        conn.close()
        
        return turn_id
    
    def get_session_history(self, session_id: str, limit: int = 50) -> List[ConversationTurn]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('''
            SELECT * FROM conversations WHERE session_id = ? 
            ORDER BY timestamp DESC LIMIT ?
        ''', (session_id, limit))
        
        results = []
        for row in cursor.fetchall():
            turn = ConversationTurn(
                id=row[0],
                session_id=row[1],
                user_message=row[2],
                assistant_message=row[3],
                timestamp=datetime.fromisoformat(row[4]),
                metadata=json.loads(row[5]),
                tools_used=json.loads(row[6])
            )
            results.append(turn)
        
        conn.close()
        return list(reversed(results))  # Return in chronological order

# MCP Server Management
class MCPServerManager:
    def __init__(self):
        self.servers: Dict[str, MCPServer] = {}
        self.tasks: Dict[str, LongRunningTask] = {}
        self.task_db_path = "tasks.db"
        self._init_task_db()
    
    def _init_task_db(self):
        conn = sqlite3.connect(self.task_db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                server_name TEXT,
                tool_name TEXT,
                status TEXT,
                created_at TEXT,
                completed_at TEXT,
                result BLOB,
                error TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def register_server(self, name: str, uri: str, tools_config: List[Dict[str, Any]]):
        """Register an MCP server with its available tools"""
        server = MCPServer(name=name, uri=uri, tools=tools_config)
        self.servers[name] = server
        
        # Test connection
        if self._test_connection(server):
            server.status = "connected"
            server.last_ping = datetime.now()
            logger.info(f"Successfully registered MCP server: {name}")
        else:
            server.status = "failed"
            logger.error(f"Failed to connect to MCP server: {name}")
    
    def _test_connection(self, server: MCPServer) -> bool:
        """Test connection to MCP server"""
        try:
            # Simulate MCP server ping - in real implementation use MCP SDK
            if server.uri.startswith("http"):
                response = requests.get(f"{server.uri}/health", timeout=5)
                return response.status_code == 200
            elif server.uri.startswith("file://"):
                return Path(server.uri[7:]).exists()
            else:
                # For other protocols, assume success for demo
                return True
        except Exception as e:
            logger.error(f"Connection test failed for {server.name}: {e}")
            return False
    
    async def call_tool(self, server_name: str, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on an MCP server"""
        if server_name not in self.servers:
            raise ValueError(f"Server {server_name} not registered")
        
        server = self.servers[server_name]
        
        # Find the tool
        tool_config = None
        for tool in server.tools:
            if tool["name"] == tool_name:
                tool_config = tool
                break
        
        if not tool_config:
            raise ValueError(f"Tool {tool_name} not found on server {server_name}")
        
        # Simulate MCP tool call
        try:
            result = await self._execute_mcp_tool(server, tool_name, parameters)
            
            # Check if this is a long-running task
            if isinstance(result, dict) and "task_id" in result:
                task_id = result["task_id"]
                task = LongRunningTask(
                    task_id=task_id,
                    server_name=server_name,
                    tool_name=tool_name,
                    status="pending",
                    created_at=datetime.now()
                )
                self.tasks[task_id] = task
                self._save_task(task)
                
            return result
            
        except Exception as e:
            logger.error(f"Error calling tool {tool_name} on {server_name}: {e}")
            raise
    
    async def _execute_mcp_tool(self, server: MCPServer, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute MCP tool - simulation for demo purposes"""
        # In real implementation, use MCP SDK to make the actual call
        
        if server.uri.startswith("http"):
            # HTTP-based MCP server
            payload = {
                "tool": tool_name,
                "parameters": parameters
            }
            async with asyncio.timeout(30):
                # Simulate HTTP call
                await asyncio.sleep(0.1)  # Simulate network delay
                
                # Simulate different response types
                if "long_running" in tool_name:
                    return {"task_id": str(uuid.uuid4()), "status": "started"}
                else:
                    return {"result": f"Executed {tool_name} with {parameters}", "status": "completed"}
        
        elif server.uri.startswith("file://"):
            # File-based or local MCP server
            config_path = server.uri[7:]
            # Load and execute local tool
            return {"result": f"Local execution of {tool_name}", "status": "completed"}
        
        else:
            # Other protocol handling
            return {"result": f"Executed {tool_name} via {server.uri}", "status": "completed"}
    
    def get_task_status(self, task_id: str) -> Optional[LongRunningTask]:
        """Get status of a long-running task"""
        if task_id in self.tasks:
            return self.tasks[task_id]
        
        # Load from database
        conn = sqlite3.connect(self.task_db_path)
        cursor = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            task = LongRunningTask(
                task_id=row[0],
                server_name=row[1],
                tool_name=row[2],
                status=row[3],
                created_at=datetime.fromisoformat(row[4]),
                completed_at=datetime.fromisoformat(row[5]) if row[5] else None,
                result=pickle.loads(row[6]) if row[6] else None,
                error=row[7]
            )
            self.tasks[task_id] = task
            return task
        
        return None
    
    def _save_task(self, task: LongRunningTask):
        """Save task to database"""
        conn = sqlite3.connect(self.task_db_path)
        conn.execute('''
            INSERT OR REPLACE INTO tasks (task_id, server_name, tool_name, status, created_at, completed_at, result, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            task.task_id, task.server_name, task.tool_name, task.status,
            task.created_at.isoformat(),
            task.completed_at.isoformat() if task.completed_at else None,
            pickle.dumps(task.result) if task.result else None,
            task.error
        ))
        conn.commit()
        conn.close()
    
    async def check_task_completion(self, task_id: str) -> bool:
        """Check if a long-running task has completed"""
        task = self.get_task_status(task_id)
        if not task:
            return False
        
        if task.status in ["completed", "failed"]:
            return True
        
        # Poll the server for updates
        try:
            server = self.servers[task.server_name]
            # Simulate checking task status on server
            # In real implementation, call the server's status endpoint
            
            # For demo, randomly complete some tasks
            import random
            if random.random() < 0.3:  # 30% chance of completion
                task.status = "completed"
                task.completed_at = datetime.now()
                task.result = f"Task {task_id} completed successfully"
                self._save_task(task)
                return True
                
        except Exception as e:
            logger.error(f"Error checking task {task_id}: {e}")
            task.status = "failed"
            task.error = str(e)
            self._save_task(task)
        
        return False

# Built-in Tools
class MemoryTool(BaseTool):
    name: str = "memory_store"
    description: str = "Store information in long-term memory for future reference"
    # Declare as pydantic field
    memory_system: MemorySystem

    def __init__(self, memory_system: MemorySystem):
        # Pass fields through BaseTool (pydantic) initializer
        super().__init__(memory_system=memory_system)

    def _run(self, key: str, value: str, tags: str = "", importance: float = 1.0) -> str:
        tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
        entry_id = self.memory_system.store(key, value, tag_list, importance=importance)
        return f"Stored in memory with ID: {entry_id}"

class MemoryRetrieveTool(BaseTool):
    name: str = "memory_retrieve"
    description: str = "Retrieve information from long-term memory"
    memory_system: MemorySystem

    def __init__(self, memory_system: MemorySystem):
        super().__init__(memory_system=memory_system)

    def _run(self, query: str, tags: str = "", limit: int = 5) -> str:
        tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else None
        entries = self.memory_system.retrieve(key=query, tags=tag_list, limit=limit)
        
        if not entries:
            return "No relevant memories found."
        
        result = "Retrieved memories:\n"
        for entry in entries:
            result += f"- {entry.key}: {entry.value} (importance: {entry.importance})\n"
        
        return result

class TaskStatusTool(BaseTool):
    name: str = "check_task_status"
    description: str = "Check the status of a long-running task by task ID"
    mcp_manager: MCPServerManager

    def __init__(self, mcp_manager: MCPServerManager):
        super().__init__(mcp_manager=mcp_manager)

    def _run(self, task_id: str) -> str:
        task = self.mcp_manager.get_task_status(task_id)
        if not task:
            return f"Task {task_id} not found"
        
        return f"Task {task_id}: {task.status}"

# User Instruction System
class InstructionEngine:
    def __init__(self, instructions_path: str = "user_instructions.yaml"):
        self.instructions_path = instructions_path
        self.base_instructions = ""
        self.user_instructions = {}
        self.load_instructions()
    
    def load_instructions(self):
        """Load user instructions from file"""
        try:
            if Path(self.instructions_path).exists():
                with open(self.instructions_path, 'r') as f:
                    data = yaml.safe_load(f)
                    self.base_instructions = data.get('base_instructions', '')
                    self.user_instructions = data.get('user_instructions', {})
                    logger.info("Loaded user instructions")
            else:
                # Create default instructions file
                self._create_default_instructions()
        except Exception as e:
            logger.error(f"Error loading instructions: {e}")
            self._create_default_instructions()
    
    def _create_default_instructions(self):
        """Create default instructions file"""
        default_config = {
            'base_instructions': 'You are a helpful AI assistant with memory and task management capabilities.',
            'user_instructions': {
                'personality': 'Be friendly and professional',
                'preferences': 'Prefer detailed explanations',
                'constraints': 'Always cite sources when using memory'
            }
        }
        
        with open(self.instructions_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        self.base_instructions = default_config['base_instructions']
        self.user_instructions = default_config['user_instructions']
    
    def get_system_prompt(self, context: Dict[str, Any] = None) -> str:
        """Build system prompt from base and user instructions"""
        prompt_parts = [self.base_instructions]
        
        if self.user_instructions:
            prompt_parts.append("\nUser Instructions:")
            for category, instruction in self.user_instructions.items():
                prompt_parts.append(f"- {category.title()}: {instruction}")
        
        if context and context.get('memory_context'):
            prompt_parts.append(f"\nRelevant memories: {context['memory_context']}")
        
        if context and context.get('active_tasks'):
            prompt_parts.append(f"\nActive tasks: {context['active_tasks']}")
        
        return "\n".join(prompt_parts)

# Main Meta Agentic Framework
class MetaAgenticFramework:
    def __init__(self, 
                 model_provider: str = "anthropic",
                 model_name: str = "claude-sonnet-4-20250514",
                 api_key: str = None):
        
        # Initialize components
        self.memory_system = MemorySystem()
        self.conversation_history = ConversationHistory()
        self.mcp_manager = MCPServerManager()
        self.instruction_engine = InstructionEngine()
        
        # Initialize LLM (use DummyLLM if API key is missing or placeholder)
        class _DummyLLM:
            def invoke(self, messages):
                # Return a simple AIMessage echoing last human content
                last = None
                for m in reversed(messages):
                    if isinstance(m, HumanMessage):
                        last = m.content
                        break
                return AIMessage(content=f"[Dummy response] {last or 'Hello'}")

        def _is_placeholder_key(k: str) -> bool:
            if not k:
                return True
            low = str(k).lower()
            return low.startswith("your-") or "placeholder" in low or low == "none"

        if model_provider == "anthropic":
            self.llm = ChatAnthropic(model=model_name, api_key=api_key) if not _is_placeholder_key(api_key) else _DummyLLM()
        elif model_provider == "openai":
            self.llm = ChatOpenAI(model=model_name, api_key=api_key) if not _is_placeholder_key(api_key) else _DummyLLM()
        else:
            raise ValueError(f"Unsupported model provider: {model_provider}")
        
        # Initialize tools
        self.tools = [
            MemoryTool(self.memory_system),
            MemoryRetrieveTool(self.memory_system),
            TaskStatusTool(self.mcp_manager),
        ]
        
        # Initialize in-memory checkpointer for conversation persistence (compatible with async)
        self.checkpointer = MemorySaver()

        # Create LangGraph (uses self.checkpointer in _create_graph)
        self.graph = self._create_graph()
        
        logger.info("Meta Agentic Framework initialized")
    
    def register_mcp_server(self, name: str, uri: str, tools_config: List[Dict[str, Any]]):
        """Register an MCP server"""
        self.mcp_manager.register_server(name, uri, tools_config)
        
        # Add MCP tools to available tools
        for tool_config in tools_config:
            mcp_tool = self._create_mcp_tool(name, tool_config)
            self.tools.append(mcp_tool)
    
    def _create_mcp_tool(self, server_name: str, tool_config: Dict[str, Any]) -> BaseTool:
        """Create a BaseTool wrapper for MCP server tool"""
        
        class MCPTool(BaseTool):
            # Declare fields; values provided at init time to avoid NameError at class definition
            name: str
            description: str
            mcp_manager: MCPServerManager
            server_name: str
            tool_name: str

            def __init__(self, mcp_manager: MCPServerManager, server_name: str, tool_name: str):
                tool_full_name = f"{server_name}_{tool_config['name']}"
                tool_desc = tool_config.get('description', f"Tool {tool_config['name']} from {server_name}")
                # Pass name/description as pydantic fields via BaseTool initializer
                super().__init__(
                    name=tool_full_name,
                    description=tool_desc,
                    mcp_manager=mcp_manager,
                    server_name=server_name,
                    tool_name=tool_name,
                )

            def _run(self, **kwargs) -> str:
                try:
                    # Run async call synchronously
                    import asyncio
                    loop = asyncio.get_event_loop()
                    result = loop.run_until_complete(
                        self.mcp_manager.call_tool(self.server_name, self.tool_name, kwargs)
                    )
                    return json.dumps(result)
                except Exception as e:
                    return f"Error calling {self.tool_name}: {str(e)}"
        
        return MCPTool(self.mcp_manager, server_name, tool_config['name'])
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph state graph"""
        
        # Define the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("planner", self._planning_node)
        workflow.add_node("memory_retrieval", self._memory_retrieval_node)
        workflow.add_node("tool_execution", self._tool_execution_node)
        workflow.add_node("task_monitoring", self._task_monitoring_node)
        workflow.add_node("response_generation", self._response_generation_node)
        
        # Add edges
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "memory_retrieval")
        workflow.add_edge("memory_retrieval", "tool_execution")
        workflow.add_edge("tool_execution", "task_monitoring")
        workflow.add_edge("task_monitoring", "response_generation")
        # Dynamically decide to continue looping or end
        workflow.add_conditional_edges(
            "response_generation",
            self._should_continue,
            {
                "continue": "planner",
                "end": END,
            },
        )
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    def _planning_node(self, state: AgentState) -> AgentState:
        """Planning node - analyze the request and decide on approach"""
        latest_message = state.messages[-1] if state.messages else None
        
        if latest_message and isinstance(latest_message, HumanMessage):
            # Extract potential memory queries and tool needs
            content = latest_message.content.lower()
            
            # Simple keyword-based planning (in production, use LLM for planning)
            if "remember" in content or "recall" in content:
                state.memory_context["needs_memory"] = True
            
            if "task" in content and "status" in content:
                state.memory_context["needs_task_check"] = True
        
        state.iteration_count += 1
        return state
    
    def _memory_retrieval_node(self, state: AgentState) -> AgentState:
        """Memory retrieval node - fetch relevant memories"""
        if state.messages:
            latest_message = state.messages[-1]
            if isinstance(latest_message, HumanMessage):
                # Retrieve relevant memories
                memories = self.memory_system.retrieve(
                    key=latest_message.content[:100],  # Use first 100 chars as query
                    limit=3
                )
                
                if memories:
                    memory_context = []
                    for memory in memories:
                        memory_context.append(f"{memory.key}: {memory.value}")
                    state.memory_context["retrieved_memories"] = memory_context
        
        return state
    
    def _should_continue(self, state: AgentState) -> str:
        """Decide whether to continue traversing the graph or end.
        Heuristics:
        - End if an explicit objective marker is present in memory_context.
        - Continue if there are active tasks pending.
        - End after a small number of iterations to avoid loops.
        """
        # Prevent runaway loops
        if state.iteration_count >= 3:
            return "end"
        
        # If we explicitly marked the objective as met, end
        if state.memory_context.get("objective_met"):
            return "end"
        
        # If there are pending tasks, keep going
        if state.active_tasks:
            return "continue"
        
        # Default: end
        return "end"
    
    def _tool_execution_node(self, state: AgentState) -> AgentState:
        """Tool execution node - execute necessary tools"""
        # This is where tool calling logic would go
        # For demo purposes, we'll simulate tool execution
        
        latest_message = state.messages[-1] if state.messages else None
        if latest_message and isinstance(latest_message, HumanMessage):
            content = latest_message.content.lower()
            
            # Check if we need to execute any tools
            if "store" in content and "memory" in content:
                # Simulate memory storage
                key = "user_request"
                value = latest_message.content
                self.memory_system.store(key, value, ["user_interaction"])
                state.memory_context["stored_memory"] = f"Stored: {key}"
                # Mark objective as met for this simple goal
                state.memory_context["objective_met"] = True
        
        return state
    
    def _task_monitoring_node(self, state: AgentState) -> AgentState:
        """Task monitoring node - check status of active tasks"""
        if state.active_tasks:
            completed_tasks = []
            for task_id in state.active_tasks:
                task = self.mcp_manager.get_task_status(task_id)
                if task and task.status in ["completed", "failed"]:
                    completed_tasks.append(task_id)
            
            # Remove completed tasks
            for task_id in completed_tasks:
                state.active_tasks.remove(task_id)
        
        return state
    
    def _response_generation_node(self, state: AgentState) -> AgentState:
        """Response generation node - generate final response"""
        # Build system prompt with context
        system_prompt = self.instruction_engine.get_system_prompt(state.memory_context)
        
        # Prepare messages for LLM
        messages = [SystemMessage(content=system_prompt)]
        
        # Add conversation history context
        if state.session_id:
            recent_history = self.conversation_history.get_session_history(state.session_id, limit=5)
            for turn in recent_history:
                messages.append(HumanMessage(content=turn.user_message))
                messages.append(AIMessage(content=turn.assistant_message))
        
        # Add current messages
        messages.extend(state.messages)
        
        # Generate response
        try:
            response = self.llm.invoke(messages)
            state.messages.append(response)
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            state.messages.append(AIMessage(content=error_msg))
            logger.error(error_msg)
        
        return state
    
    async def process_message(self, 
                             message: str, 
                             session_id: str = None,
                             config: Dict[str, Any] = None) -> str:
        """Process a user message through the framework"""
        
        session_id = session_id or str(uuid.uuid4())
        
        # Create initial state
        initial_state = AgentState(
            messages=[HumanMessage(content=message)],
            session_id=session_id,
            user_instructions=self.instruction_engine.get_system_prompt()
        )
        
        # Execute the graph
        config = config or {"configurable": {"thread_id": session_id}}
        
        try:
            result = await self.graph.ainvoke(initial_state, config=config)

            # Normalize result to a dict state
            state_out = result if isinstance(result, dict) else (
                result.dict() if hasattr(result, "dict") else {}
            )

            messages_out = state_out.get("messages", [])
            # Extract response
            if messages_out:
                response_message = messages_out[-1]
                response = response_message.content if isinstance(response_message, AIMessage) else str(response_message)

                # Store conversation turn
                self.conversation_history.add_turn(
                    session_id=session_id,
                    user_msg=message,
                    assistant_msg=response,
                    metadata={"iteration_count": state_out.get("iteration_count", 0)},
                    tools_used=[]  # TODO: track actual tools used
                )

                return response
            
            return "I apologize, but I couldn't generate a response."
            
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def load_config(self, config_path: str):
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
                # Load MCP servers
                if 'mcp_servers' in config:
                    for server_config in config['mcp_servers']:
                        self.register_mcp_server(
                            name=server_config['name'],
                            uri=server_config['uri'],
                            tools_config=server_config.get('tools', [])
                        )
                
                # Update instruction engine
                if 'instructions' in config:
                    self.instruction_engine.user_instructions.update(config['instructions'])
                
                logger.info(f"Loaded configuration from {config_path}")
                
        except Exception as e:
            logger.error(f"Error loading config: {e}")

# Usage Example and Testing
async def main():
    """Example usage of the Meta Agentic Framework"""
    
    # Initialize the framework
    framework = MetaAgenticFramework(
        model_provider="anthropic",
        model_name="claude-sonnet-4-20250514",
        api_key="your-anthropic-api-key-here"  # Replace with actual API key
    )
    
    # Register some example MCP servers
    example_mcp_servers = [
        {
            "name": "file_operations",
            "uri": "http://localhost:8001",
            "tools": [
                {
                    "name": "read_file",
                    "description": "Read contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "Path to the file"}
                        },
                        "required": ["file_path"]
                    }
                },
                {
                    "name": "long_running_analysis",
                    "description": "Perform long-running data analysis",
                    "parameters": {
                        "type": "object", 
                        "properties": {
                            "dataset_path": {"type": "string", "description": "Path to dataset"},
                            "analysis_type": {"type": "string", "description": "Type of analysis"}
                        },
                        "required": ["dataset_path", "analysis_type"]
                    }
                }
            ]
        },
        {
            "name": "web_scraper",
            "uri": "file:///path/to/local/scraper.py",
            "tools": [
                {
                    "name": "scrape_website",
                    "description": "Scrape content from a website",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "URL to scrape"},
                            "selector": {"type": "string", "description": "CSS selector"}
                        },
                        "required": ["url"]
                    }
                }
            ]
        }
    ]
    
    # Register the MCP servers
    for server_config in example_mcp_servers:
        framework.register_mcp_server(
            name=server_config["name"],
            uri=server_config["uri"],
            tools_config=server_config["tools"]
        )
    
    # Example conversation session
    session_id = "demo_session_001"
    
    prog = get_prog_name()
    print(f"🤖 {prog} Demo")
    print("=" * 50)
    
    # Test 1: Memory storage and retrieval
    print("\n📝 Test 1: Memory Storage")
    response1 = await framework.process_message(
        "Remember that my favorite programming language is Python and I prefer using FastAPI for web development.",
        session_id=session_id
    )
    print(f"Assistant: {response1}")
    
    # Test 2: Memory retrieval
    print("\n🧠 Test 2: Memory Retrieval")
    response2 = await framework.process_message(
        "What do you remember about my programming preferences?",
        session_id=session_id
    )
    print(f"Assistant: {response2}")
    
    # Test 3: MCP Tool usage
    print("\n🔧 Test 3: MCP Tool Usage")
    response3 = await framework.process_message(
        "Use the file operations tool to read a configuration file at /etc/config.json",
        session_id=session_id
    )
    print(f"Assistant: {response3}")
    
    # Test 4: Long-running task
    print("\n⏳ Test 4: Long-running Task")
    response4 = await framework.process_message(
        "Start a long-running analysis on the dataset at /data/sales.csv using statistical analysis",
        session_id=session_id
    )
    print(f"Assistant: {response4}")
    
    # Test 5: Task status check
    print("\n📊 Test 5: Task Status Check")
    # Simulate checking task status
    task_id = "demo_task_123"  # This would come from the actual response
    framework.mcp_manager.tasks[task_id] = LongRunningTask(
        task_id=task_id,
        server_name="file_operations",
        tool_name="long_running_analysis",
        status="running",
        created_at=datetime.now()
    )
    
    response5 = await framework.process_message(
        f"Check the status of task {task_id}",
        session_id=session_id
    )
    print(f"Assistant: {response5}")
    
    # Test 6: Conversation history
    print("\n💬 Test 6: Conversation Context")
    response6 = await framework.process_message(
        "What was the first thing I told you in this conversation?",
        session_id=session_id
    )
    print(f"Assistant: {response6}")
    
    print("\n" + "=" * 50)
    print("Demo completed!")
    
    # Display some statistics
    print(f"\n📈 Session Statistics:")
    print(f"Memory entries: {len(framework.memory_system.retrieve(limit=100))}")
    print(f"Conversation turns: {len(framework.conversation_history.get_session_history(session_id))}")
    print(f"Registered MCP servers: {len(framework.mcp_manager.servers)}")
    print(f"Active tasks: {len(framework.mcp_manager.tasks)}")

# Configuration Management
class ConfigManager:
    """Utility class for managing framework configuration"""
    
    @staticmethod
    def create_sample_config(output_path: str = None):
        """Create a sample configuration file"""
        if not output_path:
            output_path = f"{get_prog_name()}_config.yaml"
        sample_config = {
            'model': {
                'provider': 'anthropic',
                'name': 'claude-sonnet-4-20250514',
                'api_key': 'your-anthropic-api-key-here'
            },
            'instructions': {
                'personality': 'Be helpful, accurate, and concise',
                'preferences': 'Always explain your reasoning',
                'constraints': 'Always cite sources when using memory or external tools'
            },
            'mcp_servers': [
                {
                    'name': 'local_tools',
                    'uri': 'file:///path/to/local/tools.py',
                    'tools': [
                        {
                            'name': 'calculate',
                            'description': 'Perform mathematical calculations',
                            'parameters': {
                                'type': 'object',
                                'properties': {
                                    'expression': {
                                        'type': 'string',
                                        'description': 'Mathematical expression to evaluate'
                                    }
                                },
                                'required': ['expression']
                            }
                        }
                    ]
                },
                {
                    'name': 'api_tools',
                    'uri': 'http://localhost:8080',
                    'tools': [
                        {
                            'name': 'weather_forecast',
                            'description': 'Get weather forecast for a location',
                            'parameters': {
                                'type': 'object',
                                'properties': {
                                    'location': {
                                        'type': 'string',
                                        'description': 'Location for weather forecast'
                                    },
                                    'days': {
                                        'type': 'integer',
                                        'description': 'Number of days to forecast',
                                        'default': 3
                                    }
                                },
                                'required': ['location']
                            }
                        }
                    ]
                }
            ],
            'memory': {
                'max_entries': 10000,
                'cleanup_interval_hours': 24,
                'importance_threshold': 0.1
            },
            'tasks': {
                'max_concurrent': 10,
                'timeout_minutes': 60,
                'retry_attempts': 3
            }
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(sample_config, f, default_flow_style=False, indent=2)
        
        print(f"Sample configuration created at: {output_path}")

# Advanced Features
class AdvancedFeatures:
    """Additional advanced features for the framework"""
    
    @staticmethod
    def setup_monitoring(framework: MetaAgenticFramework):
        """Setup monitoring and metrics collection"""
        # This could integrate with monitoring systems like Prometheus
        pass
    
    @staticmethod
    def setup_security(framework: MetaAgenticFramework):
        """Setup security features like input validation and sandboxing"""
        # Implement security measures for MCP server execution
        pass
    
    @staticmethod
    def setup_scaling(framework: MetaAgenticFramework):
        """Setup horizontal scaling capabilities"""
        # This could include load balancing and distributed execution
        pass

# CLI Interface
import argparse

def get_prog_name() -> str:
    """Return the program name as invoked (symlink-friendly), sans .py suffix."""
    try:
        import sys
        name = Path(sys.argv[0]).name or "metaagent"
        return name[:-3] if name.endswith(".py") else name
    except Exception:
        return "metaagent"

def create_cli():
    """Create command-line interface for the framework"""
    prog = get_prog_name()
    # Default config in user's home: ~/.<prog>/config.yaml
    default_config = os.path.expanduser(f"~/.{prog}/config.yaml")
    parser = argparse.ArgumentParser(prog=prog, description="Meta Agentic AI Framework")
    parser.add_argument("--config", default=default_config, 
                       help="Path to configuration file")
    parser.add_argument("--create-config", action="store_true",
                       help="Create a sample configuration file")
    parser.add_argument("--interactive", action="store_true",
                       help="Start interactive mode")
    parser.add_argument("--message", type=str,
                       help="Process a single message")
    parser.add_argument("--session-id", type=str,
                       help="Session ID for conversation continuity")
    
    return parser

async def cli_main():
    """Main CLI entry point"""
    parser = create_cli()
    args = parser.parse_args()
    
    if args.create_config:
        ConfigManager.create_sample_config()
        return
    
    # Initialize framework
    try:
        framework = MetaAgenticFramework()
        
        if Path(args.config).exists():
            framework.load_config(args.config)
        else:
            print(f"Warning: Config file {args.config} not found. Using defaults.")
        
        if args.interactive:
            await interactive_mode(framework)
        elif args.message:
            session_id = args.session_id or str(uuid.uuid4())
            response = await framework.process_message(args.message, session_id)
            print(response)
        else:
            await main()  # Run demo
            
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"CLI error: {e}")

async def interactive_mode(framework: MetaAgenticFramework):
    """Interactive chat mode"""
    print(f"🤖 {get_prog_name()} - Interactive Mode")
    print("Type 'quit' to exit, 'help' for commands")
    print("-" * 50)
    
    session_id = str(uuid.uuid4())
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'help':
                print("""
Available commands:
- quit: Exit the program
- help: Show this help message
- status: Show framework status
- memory: List stored memories
- tasks: List active tasks
- clear: Clear conversation history
                """)
                continue
            elif user_input.lower() == 'status':
                print(f"Session ID: {session_id}")
                print(f"Memory entries: {len(framework.memory_system.retrieve(limit=100))}")
                print(f"MCP servers: {len(framework.mcp_manager.servers)}")
                print(f"Active tasks: {len(framework.mcp_manager.tasks)}")
                continue
            elif user_input.lower() == 'memory':
                memories = framework.memory_system.retrieve(limit=10)
                if memories:
                    for memory in memories:
                        print(f"  - {memory.key}: {memory.value}")
                else:
                    print("  No memories stored")
                continue
            elif user_input.lower() == 'tasks':
                tasks = list(framework.mcp_manager.tasks.values())
                if tasks:
                    for task in tasks:
                        print(f"  - {task.task_id}: {task.status}")
                else:
                    print("  No active tasks")
                continue
            elif user_input.lower() == 'clear':
                session_id = str(uuid.uuid4())
                print("Conversation history cleared")
                continue
            
            if user_input:
                response = await framework.process_message(user_input, session_id)
                print(f"\nAssistant: {response}")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # Run the CLI
    import sys
    if len(sys.argv) > 1:
        asyncio.run(cli_main())
    else:
        asyncio.run(main())