import sys
import os
import logging
from typing import Annotated, Dict, List, Sequence, TypedDict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq
from langchain_core.messages import ToolMessage
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('langgraph_agent.log')
    ]
)
logger = logging.getLogger(__name__)

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from scheme.user_scheme import UserScheme
    from tools.rag_process import rag_process
    from tools.search_query import search_query
    from tools.suggest_trips import suggest_trips
    from tools.summarize_trip import summarize_trip
    from config import get_settings
    from agents.tracer import FlowTracer
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

import operator
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

settings = get_settings()
tracer = FlowTracer()

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user: UserScheme
    context: str
    rag_context: str

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    max_retries=2,
    groq_api_key=settings.GROQ_API_KEY,
)

@tool
def rag_process_tool(query: str) -> str:
    """Process queries using RAG (Retrieval-Augmented Generation) to get relevant information from documents."""
    logger.info("rag_process_tool entered")

    result = rag_process(query)

    logger.info("rag_process_tool exited")
    return result

@tool
def search_query_tool(query: str) -> str:
    """Search for information using web search capabilities."""
    logger.info("search_query_tool entered")

    result = search_query(query)

    logger.info("search_query_tool exited")
    return result

@tool
def suggest_trips_tool(query: str) -> str:
    """Suggest travel itineraries based on user preferences and queries."""
    logger.info("suggest_trips_tool entered")

    result = suggest_trips(query)

    logger.info("suggest_trips_tool exited")
    return result

@tool
def summarize_trip_tool(query: str) -> str:
    """Summarize travel information and provide key highlights."""
    logger.info("summarize_trip_tool entered")

    result = summarize_trip(query)

    logger.info("summarize_trip_tool exited")
    return result

tools = [rag_process_tool, search_query_tool, suggest_trips_tool, summarize_trip_tool]
llm_with_tools = llm.bind_tools(tools)

def should_continue(state: AgentState) -> str:
    """Determine if the workflow should continue or end."""
    logger.info("should_continue function entered")

    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        logger.info("should_continue: returning 'tools'")
        return "tools"
    
    logger.info("should_continue: returning END")
    return END

def call_model(state: AgentState) -> AgentState:
    """Process the conversation and generate AI responses."""
    logger.info("call_model function entered")

    messages = state["messages"]
    rag_context = state.get("rag_context", "")
    user = state.get("user", UserScheme())

    # Check if we have recent tool results
    has_recent_tool_results = any(
        isinstance(msg, ToolMessage) for msg in messages[-3:] 
        if isinstance(msg, ToolMessage)
    )

    system_content = """You are a helpful AI travel assistant designed to help users plan trips, search for information, and get travel recommendations.

    Tools Available:
    - rag_process_tool: Get relevant information from travel documents and guides
    - search_query_tool: Search for current travel information, prices, and updates
    - suggest_trips_tool: Suggest travel itineraries based on preferences
    - summarize_trip_tool: Summarize travel plans and provide key highlights

    RESPONSE BEHAVIOR:
    - Execute requested actions immediately without asking for permission
    - Do not use phrases like "let me", "give me a moment" or "please wait"
    - Directly provide the information or execute the task requested
    - No acknowledgments about what you're about to do - just do it

    CRITICAL INSTRUCTION FOR TOOL RESULTS:
    When you receive tool results, you MUST:
    1. Analyze and interpret the raw data
    2. Structure it in a user-friendly format
    3. Add relevant context and explanations
    4. Provide actionable insights
    5. NEVER just repeat the raw tool output

    RESPONSE FORMATTING:
    - Use clear headings and bullet points
    - Keep paragraphs short and scannable
    - Always end with helpful next steps or questions
    - Maintain professional but friendly tone

    PRIMARY OBJECTIVE:
    Transform raw travel data into meaningful, actionable travel guidance that helps users make informed decisions about their trips."""

        # Explicit tool guidance so the model knows tool names and the expected call format.
    system_content += """

        TOOL USAGE GUIDELINES:
        - When the user asks for current information, prices, or external links (for example: flights, tickets, live prices), call the appropriate tool instead of replying directly.
        - Tool names you can call exactly as provided: `rag_process_tool`, `search_query_tool`, `suggest_trips_tool`, `summarize_trip_tool`.
        - When calling a tool, return a structured tool call with the tool name and a single `query` argument, for example (pseudocode):
            {"name": "search_query_tool", "args": {"query": "Flights from New York to Cairo roundtrip, departure next week, economy"}}
        - After receiving tool results, your assistant response must interpret and format the results for the user (do NOT simply echo raw tool output).
        """

    if rag_context:
        system_content += f"""
        
        RAG Context: {rag_context}
        Use this information to provide accurate, specific answers about travel destinations, requirements, and opportunities.
        """

    if has_recent_tool_results:
        system_content += """
        IMPORTANT: You have just received tool results. 
        Your task is to process this raw data and present it as a helpful, structured response to the user. 
        Do NOT simply repeat the tool output - interpret, organize, and enhance it for the user."""

    # Create system message (use SystemMessage so the model treats this as system-level instructions)
    system_message = SystemMessage(content=system_content)

    # Format messages for the LLM
    formatted_messages = [system_message]
    
    for msg in messages:
        if isinstance(msg, dict):
            if msg.get("role") == "user":
                formatted_messages.append(HumanMessage(content=msg["content"]))
            elif msg.get("role") == "assistant":
                if msg.get("tool_calls"):
                    ai_msg = AIMessage(content=msg.get("content", ""))
                    ai_msg.tool_calls = msg["tool_calls"]
                    formatted_messages.append(ai_msg)
                else:
                    formatted_messages.append(AIMessage(content=msg["content"]))
            elif msg.get("role") == "tool":
                formatted_messages.append(ToolMessage(
                    content=msg["content"],
                    tool_call_id=msg.get("tool_call_id", "")
                ))
        else:
            # If it's already a BaseMessage, add it directly
            formatted_messages.append(msg)

    # Get AI response
    response = llm_with_tools.invoke(formatted_messages)
    
    # Add the AI response to messages
    new_messages = messages + [response]
    
    logger.info("call_model function exited")

    return {
        "messages": new_messages,
        "user": user,
        "context": state.get("context", ""),
        "rag_context": rag_context
    }

def handle_tools(state: AgentState) -> AgentState:
    """Handle tool execution and return tool messages."""
    logger.info("handle_tools function entered")

    messages = state["messages"]
    user = state.get("user", UserScheme())
    context = state.get("context", "")
    rag_context = state.get("rag_context", "")

    # Get the last AI message with tool calls
    last_ai_message = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
            last_ai_message = msg
            break
    
    if not last_ai_message:
        return state

    # Execute tools
    tool_messages = []
    for tool_call in last_ai_message.tool_calls:
        try:
            # Find the tool by name
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("args", {})
            
            # Execute the appropriate tool
            if tool_name == "rag_process_tool":
                result = rag_process_tool.invoke(tool_args.get("query", ""))
            elif tool_name == "search_query_tool":
                result = search_query_tool.invoke(tool_args.get("query", ""))
            elif tool_name == "suggest_trips_tool":
                result = suggest_trips_tool.invoke(tool_args.get("query", ""))
            elif tool_name == "summarize_trip_tool":
                result = summarize_trip_tool.invoke(tool_args.get("query", ""))
            else:
                result = f"Unknown tool: {tool_name}"
            
            # Create tool message
            tool_message = ToolMessage(
                content=str(result),
                tool_call_id=tool_call.get("id", "")
            )
            tool_messages.append(tool_message)
            
        except Exception as e:
            error_message = ToolMessage(
                content=f"Error executing tool: {str(e)}",
                tool_call_id=tool_call.get("id", "")
            )
            tool_messages.append(error_message)

    # Add tool messages to state
    new_messages = messages + tool_messages
    
    logger.info("handle_tools function exited")
    return {
        "messages": new_messages,
        "user": user,
        "context": context,
        "rag_context": rag_context
    }

def rag_retrieval(state: AgentState) -> AgentState:
    """Handle RAG retrieval for context enhancement."""
    logger.info("rag_retrieval function entered")

    messages = state["messages"]
    user = state.get("user", UserScheme())
    context = state.get("context", "")
    
    # Get the last user message
    last_user_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_message = msg
            break
    
    if not last_user_message:
        return state
    
    try:
        # Use RAG to get relevant context
        rag_result = rag_process(last_user_message.content)
        
        logger.info("rag_retrieval function exited successfully")

        return {
            "messages": messages,
            "user": user,
            "context": context,
            "rag_context": rag_result
        }
    except Exception as e:
        # If RAG fails, continue without it
        logger.error(f"rag_retrieval function failed with error: {str(e)}")
        
        return {
            "messages": messages,
            "user": user,
            "context": context,
            "rag_context": ""
        }

def get_agent():
    """Create and return the compiled LangGraph agent."""
    logger.info("get_agent function entered")
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", handle_tools)
    workflow.add_node("rag_retrieval", rag_retrieval)

    # Set entry point
    workflow.set_entry_point("agent")

    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "rag_retrieval": "rag_retrieval",
            "agent": "agent",
            END: END
        }
    )

    # Add edges
    workflow.add_edge("tools", "agent")
    workflow.add_edge("rag_retrieval", "agent")
    
    memory = MemorySaver()

    logger.info("get_agent function exited")
    return workflow.compile(checkpointer=memory)

# Example usage function
def run_agent(user_input: str, user_preferences: UserScheme = None):
    """Run the agent with a user input."""
    logger.info("run_agent function entered")
    if user_preferences is None:
        user_preferences = UserScheme()
    
    # Initialize the agent
    agent = get_agent()
    
    config={"configurable": {"user_id": "1", "thread_id": "1"}}

    # Create initial state
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "user": user_preferences,
        "context": "",
        "rag_context": ""
    }
    # Run the agent
    result = agent.invoke(initial_state, config=config)
    
    # Extract tool message content - prefer the most recent tool result if present
    trip_itinerary = None
    for msg in result.get("messages", []):
        try:
            # Many message objects expose 'type' or class name
            if hasattr(msg, 'type') and getattr(msg, 'type') == "tool":
                trip_itinerary = getattr(msg, 'content', None)
                break
            # Fallback: inspect class name
            cls_name = msg.__class__.__name__ if hasattr(msg, '__class__') else ''
            if 'ToolMessage' in cls_name or 'Tool' in cls_name:
                trip_itinerary = getattr(msg, 'content', None)
                break
            # If it's a dict-shaped message
            if isinstance(msg, dict) and msg.get('role') == 'tool':
                trip_itinerary = msg.get('content')
                break
        except Exception:
            continue

    if not trip_itinerary:
        # If the LLM didn't call tools, try to extract assistant content as itinerary
        for msg in result.get("messages", []):
            if isinstance(msg, dict) and msg.get('role') == 'assistant':
                trip_itinerary = msg.get('content')
                break

    if not trip_itinerary:
        logger.info("run_agent function exited - no itinerary found")
        return "No itinerary found."

    # Clean up text
    cleaned_text = str(trip_itinerary).strip()                # remove leading/trailing spaces
    cleaned_text = re.sub(r"\n\s*\n", "\n\n", cleaned_text)  # normalize blank lines
    cleaned_text = re.sub(r"\t+", " ", cleaned_text)     # replace tabs with spaces
    cleaned_text = re.sub(r" +", " ", cleaned_text)      # collapse multiple spaces

    # Post-processing: if the user asked for flight links, call search_query
    additional_sections = []
    try:
        if getattr(user_preferences, 'want_flight_links', False):
            # Build a focused flight search query
            flight_query = f"Flights from {getattr(user_preferences, 'user_location', 'New York') or 'New York'} to {getattr(user_preferences, 'destination', 'Cairo') or 'Cairo'} roundtrip duration {getattr(user_preferences, 'duration', '') or '5 days'}"
            logger.info(f"Running fallback flight search: {flight_query}")
            try:
                flight_results = search_query(flight_query)
            except Exception as e:
                flight_results = f"Flight search failed: {type(e).__name__}: {e}"
            if flight_results:
                additional_sections.append("Flight search results:\n" + str(flight_results))

        # Always provide a short summary using summarize_trip to ensure the user gets a concise overview
        try:
            summary = summarize_trip(user_preferences if user_preferences is not None else UserScheme(), cleaned_text)
            if summary:
                additional_sections.append("Summary:\n" + str(summary))
        except Exception as e:
            logger.error(f"summarize_trip failed: {e}")

    except Exception as e:
        logger.error(f"Post-processing failed: {e}")

    final_output = cleaned_text
    if additional_sections:
        final_output += "\n\n" + "\n\n".join(additional_sections)

    logger.info("run_agent function exited")
    return final_output


if __name__ == "__main__":
    logger.info("Main execution started")
    
    user_input = input("Enter your query: ")
    user_preferences = UserScheme()
    user_preferences.user_location = input("Enter your user location: ")
    user_preferences.destination = input("Enter your destination: ")
    user_preferences.duration = input("Enter your duration: ")
    user_preferences.interests = input("Enter your interests: ")
    user_preferences.budget = input("Enter your budget: ")
    user_preferences.travel_style = input("Enter your travel style: ")
    user_preferences.want_flight_links = input("Enter if you want flight links (yes/no): ").strip().lower() == "yes"

    result = run_agent(user_input, user_preferences)
    print(result)
    
    logger.info("Main execution completed")