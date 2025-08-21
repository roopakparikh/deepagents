#!/usr/bin/env python3
"""
Interactive LangGraph Agent with Memory and Dynamic Tool Calling

This example demonstrates:
1. Persistent memory across conversations
2. Dynamic tool calling based on previous tool results
3. Interactive session management
4. Conditional routing based on tool outcomes

Requirements:
pip install langgraph langchain-anthropic python-dotenv

Set your Anthropic API key in environment or .env file:
ANTHROPIC_API_KEY=your_key_here
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import TypedDict, Annotated, List, Dict, Any
import operator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("dotenv not installed. Make sure to set ANTHROPIC_API_KEY environment variable.")

# Initialize LLM
llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0.7,
    api_key=os.getenv("ANTHROPIC_API_KEY", "your-anthropic-api-key-here")
)

# Define sample tools that can chain together
@tool
def get_user_profile(user_id: str) -> Dict[str, Any]:
    """Get user profile information"""
    # Simulated user data
    users = {
        "user123": {
            "name": "Alice Johnson",
            "age": 28,
            "preferences": ["technology", "music", "travel"],
            "location": "San Francisco",
            "subscription": "premium"
        },
        "user456": {
            "name": "Bob Smith", 
            "age": 35,
            "preferences": ["sports", "cooking", "books"],
            "location": "New York",
            "subscription": "basic"
        }
    }
    
    profile = users.get(user_id, {"error": "User not found"})
    print(f"🔍 Retrieved profile for {user_id}: {profile.get('name', 'Unknown')}")
    return profile

@tool
def get_recommendations(preferences: List[str], location: str) -> Dict[str, Any]:
    """Get recommendations based on user preferences and location"""
    recommendations = {
        "technology": ["Latest AI conference", "Tech meetup downtown"],
        "music": ["Jazz festival this weekend", "Local concert hall events"],
        "travel": ["Weekend getaway packages", "Flight deals to Europe"],
        "sports": ["Local gym membership deals", "Basketball game tickets"],
        "cooking": ["Cooking classes nearby", "Farmers market locations"],
        "books": ["Book club meetings", "Author reading events"]
    }
    
    user_recs = []
    for pref in preferences:
        if pref in recommendations:
            user_recs.extend(recommendations[pref])
    
    result = {
        "location": location,
        "recommendations": user_recs[:4],  # Limit to top 4
        "total_found": len(user_recs)
    }
    
    print(f"🎯 Generated {len(user_recs)} recommendations for {location}")
    return result

# Minimal local ToolExecutor replacement (since langgraph.prebuilt.ToolExecutor isn't available)
class LocalToolExecutor:
    """Lightweight executor to call LangChain tools by name.

    Expects tool_call dicts of shape {"name": str, "args": dict, "id": str?}.
    """

    def __init__(self, tools: List[Any]):
        self.tools_by_name = {}
        for t in tools:
            # LangChain tools created with @tool have a .name
            name = getattr(t, "name", None) or getattr(t, "__name__", None)
            if not name:
                raise ValueError("Tool missing a name attribute")
            self.tools_by_name[name] = t

    def invoke(self, tool_call: Dict[str, Any]) -> Any:
        name = tool_call.get("name")
        if name not in self.tools_by_name:
            raise KeyError(f"Unknown tool: {name}")
        args = tool_call.get("args") or {}
        tool = self.tools_by_name[name]
        # LangChain BaseTool supports .invoke with a dict input
        if hasattr(tool, "invoke"):
            return tool.invoke(args)
        # Fallback to direct callable
        return tool(**args)

@tool
def check_availability(item: str, location: str) -> Dict[str, Any]:
    """Check availability of recommended items"""
    # Simulated availability check
    availability = {
        "available": True if hash(item + location) % 2 == 0 else False,
        "item": item,
        "location": location,
        "price": f"${hash(item) % 100 + 20}",
        "next_available": "Tomorrow" if hash(item) % 3 == 0 else "This weekend"
    }
    
    status = "✅ Available" if availability["available"] else "❌ Not available"
    print(f"📅 {status}: {item} in {location}")
    return availability

@tool
def book_item(item: str, user_id: str) -> Dict[str, Any]:
    """Book an available item for the user"""
    booking_id = f"booking_{hash(item + user_id) % 10000}"
    result = {
        "booking_id": booking_id,
        "item": item,
        "user_id": user_id,
        "status": "confirmed",
        "booking_time": datetime.now().isoformat()
    }
    
    print(f"🎫 Booked: {item} for user {user_id} (ID: {booking_id})")
    return result

# Define the agent state
class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    tool_results: Dict[str, Any]
    conversation_memory: Dict[str, Any]
    next_action: str
    user_id: str

class ConversationMemory:
    """Manages conversation memory and context"""
    
    def __init__(self):
        self.short_term = {}  # Current session context
        self.long_term = {}   # Persistent across sessions
        self.tool_history = []  # Track tool call patterns
    
    def update_from_tool_results(self, results: Dict[str, Any]):
        """Update memory based on tool outcomes"""
        for tool_name, result in results.items():
            self.tool_history.append({
                "tool": tool_name,
                "result": result,
                "timestamp": datetime.now().isoformat()
            })
            
            # Extract important info for future use
            if tool_name == "get_user_profile" and "error" not in result:
                self.long_term["user_profile"] = result
                self.long_term["user_preferences"] = result.get("preferences", [])
                self.long_term["user_location"] = result.get("location", "")
            
            elif tool_name == "get_recommendations":
                self.short_term["last_recommendations"] = result
            
            elif tool_name == "book_item":
                if "bookings" not in self.long_term:
                    self.long_term["bookings"] = []
                self.long_term["bookings"].append(result)
    
    def get_context_for_next_action(self) -> Dict[str, Any]:
        """Provide relevant context for the next decision"""
        return {
            "recent_tools": self.tool_history[-3:],
            "user_context": self.long_term,
            "current_state": self.short_term
        }

def create_interactive_agent():
    """Create the LangGraph agent with memory and tool chaining"""
    
    # Initialize tools and memory
    tools = [get_user_profile, get_recommendations, check_availability, book_item]
    tool_executor = LocalToolExecutor(tools)
    
    # Create SQLite checkpointer for persistence
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    memory = SqliteSaver(conn)
    
    # Create workflow
    workflow = StateGraph(AgentState)
    
    def agent_node(state: AgentState) -> Dict[str, Any]:
        """Main agent decision node"""
        messages = state.get("messages", [])
        tool_results = state.get("tool_results", {})
        memory_context = state.get("conversation_memory", {})
        user_id = state.get("user_id", "")
        
        # Build context from previous interactions
        context_parts = []
        
        if tool_results:
            context_parts.append(f"Previous tool results: {json.dumps(tool_results, indent=2)}")
        
        if memory_context:
            context_parts.append(f"Memory context: {json.dumps(memory_context, indent=2)}")
        
        # Get the last human message
        last_human_msg = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_human_msg = msg.content
                break
        
        if not last_human_msg:
            last_human_msg = "Hello"
        
        # Create system prompt with dynamic context
        system_prompt = f"""You are a helpful assistant that can:
1. Get user profiles
2. Generate recommendations based on preferences
3. Check availability of items
4. Book items for users

Current user ID: {user_id}
Context from previous actions: {' '.join(context_parts)}

Based on the conversation and any previous tool results, decide what action to take.
If you have user profile data, use it to make recommendations.
If you have recommendations, you can check availability or help book items.

User message: {last_human_msg}

You should use tools when appropriate. Be conversational and helpful."""

        # Call LLM with tools
        response = llm.bind_tools(tools).invoke([
            HumanMessage(content=system_prompt + "\n\nUser: " + last_human_msg)
        ])
        
        # Determine next action based on response
        next_action = "end"
        if response.tool_calls:
            next_action = "tools"
        elif any(keyword in response.content.lower() for keyword in ["recommend", "book", "check", "profile"]):
            next_action = "continue"
        
        return {
            "messages": [response],
            "next_action": next_action
        }
    
    def tool_node(state: AgentState) -> Dict[str, Any]:
        """Execute tools and handle chaining logic"""
        last_message = state["messages"][-1]
        tool_results = {}
        
        # Execute requested tools
        for tool_call in last_message.tool_calls:
            try:
                result = tool_executor.invoke(tool_call)
                tool_results[tool_call["name"]] = result
                
                # Create tool message for the conversation
                tool_msg = ToolMessage(
                    content=json.dumps(result),
                    tool_call_id=tool_call["id"]
                )
                
                # Auto-trigger related tools based on results
                auto_tools = determine_next_tools(tool_call["name"], result, state)
                for auto_tool in auto_tools:
                    try:
                        auto_result = tool_executor.invoke(auto_tool)
                        tool_results[auto_tool["name"]] = auto_result
                        print(f"🔄 Auto-triggered: {auto_tool['name']}")
                    except Exception as e:
                        print(f"❌ Auto-tool error: {e}")
                        
            except Exception as e:
                tool_results[tool_call["name"]] = {"error": str(e)}
                print(f"❌ Tool error: {e}")
        
        return {
            "tool_results": tool_results,
            "messages": [ToolMessage(
                content=f"Tool results: {json.dumps(tool_results, indent=2)}",
                tool_call_id="summary"
            )]
        }
    
    def determine_next_tools(tool_name: str, result: Dict[str, Any], state: AgentState) -> List[Dict[str, Any]]:
        """Determine what tools to auto-trigger based on current tool result"""
        next_tools = []
        
        # Chain logic: profile -> recommendations
        if tool_name == "get_user_profile" and "error" not in result:
            preferences = result.get("preferences", [])
            location = result.get("location", "")
            if preferences and location:
                next_tools.append({
                    "name": "get_recommendations",
                    "args": {
                        "preferences": preferences,
                        "location": location
                    },
                    "id": f"auto_rec_{datetime.now().timestamp()}"
                })
        
        # Chain logic: recommendations -> availability check for first item
        elif tool_name == "get_recommendations" and "recommendations" in result:
            recommendations = result.get("recommendations", [])
            location = result.get("location", "")
            if recommendations and location:
                # Check availability for the first recommendation
                next_tools.append({
                    "name": "check_availability", 
                    "args": {
                        "item": recommendations[0],
                        "location": location
                    },
                    "id": f"auto_avail_{datetime.now().timestamp()}"
                })
        
        return next_tools
    
    def should_continue(state: AgentState) -> str:
        """Decide whether to continue, use tools, or end"""
        last_message = state["messages"][-1]
        
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        
        # Check if we should end based on conversation flow
        if "goodbye" in last_message.content.lower() or "bye" in last_message.content.lower():
            return "end"
        
        return "end"
    
    # Add nodes to workflow
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
    # Add edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    workflow.add_edge("tools", "agent")
    workflow.set_entry_point("agent")
    
    return workflow.compile(checkpointer=memory)

def run_interactive_session():
    """Run the interactive session"""
    print("🤖 Interactive LangGraph Agent with Memory & Tool Chaining")
    print("=" * 60)
    print("Commands:")
    print("  - Type 'quit' to exit")
    print("  - Type 'memory' to see conversation memory")
    print("  - Type 'user <id>' to switch user (e.g., 'user user123')")
    print("  - Ask about profiles, recommendations, or booking!")
    print("=" * 60)
    
    app = create_interactive_agent()
    conversation_memory = ConversationMemory()
    
    # Default user
    current_user = "user123"
    thread_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"💬 Starting session for {current_user}")
    print("Try: 'Get my profile' or 'I need recommendations'")
    print()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'memory':
                context = conversation_memory.get_context_for_next_action()
                print("🧠 Memory Context:")
                print(json.dumps(context, indent=2))
                continue
            
            if user_input.lower().startswith('user '):
                new_user = user_input.split(' ', 1)[1]
                current_user = new_user
                print(f"👤 Switched to user: {current_user}")
                continue
            
            if not user_input:
                continue
            
            # Prepare state
            config = {"configurable": {"thread_id": thread_id}}
            initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "tool_results": {},
                "conversation_memory": conversation_memory.get_context_for_next_action(),
                "next_action": "",
                "user_id": current_user
            }
            
            print("\n🤖 Agent:", end=" ")
            
            # Stream the response
            response_parts = []
            for chunk in app.stream(initial_state, config):
                if "agent" in chunk:
                    agent_msg = chunk["agent"]["messages"][-1]
                    if hasattr(agent_msg, 'content'):
                        response_parts.append(agent_msg.content)
                        print(agent_msg.content)
                
                if "tools" in chunk:
                    tool_results = chunk["tools"]["tool_results"]
                    conversation_memory.update_from_tool_results(tool_results)
            
            if not response_parts:
                print("I'm not sure how to help with that. Try asking about profiles or recommendations!")
            
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again or type 'quit' to exit.\n")

if __name__ == "__main__":
    # Check for Anthropic API key
    if not os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY") == "your-anthropic-api-key-here":
        print("⚠️  Please set your ANTHROPIC_API_KEY environment variable!")
        print("You can:")
        print("1. Set it in your environment: export ANTHROPIC_API_KEY=your_key")
        print("2. Create a .env file with: ANTHROPIC_API_KEY=your_key")
        print("3. Get your API key from: https://console.anthropic.com/")
        exit(1)
    
    run_interactive_session()