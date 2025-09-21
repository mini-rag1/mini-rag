import sys
import os
import logging
from typing import Annotated, Dict, List, Sequence, TypedDict, Any,Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq
from langchain_core.messages import ToolMessage
import re
import json

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
    """Process queries using RAG to get relevant information from documents."""
    logger.info("rag_process_tool entered")

    result = rag_process(query)

    logger.info("rag_process_tool exited")
    return result

@tool
def search_query_tool(query: str) -> str:
    """Search for information using web search for flights, pre-made trips, or anything you don't have information about."""
    logger.info("search_query_tool entered")

    result = search_query(query)

    logger.info("search_query_tool exited")
    return result

@tool
def suggest_trips_tool(query: str, user: Dict[str, Any]) -> str:
    """Suggest travel itineraries based on user preferences and queries."""
    logger.info(f"suggest_trips_tool entered with query: {query}, user: {user}")
    try:
        # Convert user dict to UserScheme
        user_scheme = UserScheme(**user)
        # Log user preferences for debugging
        logger.info(f"User preferences: {user_scheme.model_dump()}")

        result = suggest_trips(query, user_scheme)
        
        logger.info("suggest_trips_tool exited")
        return result
    except Exception as e:
        logger.error(f"Error in suggest_trips_tool: {str(e)}")
        return f"Error generating itinerary: {str(e)}"

tools = [rag_process_tool, search_query_tool, suggest_trips_tool]
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

    # Log user preferences
    logger.info(f"Processing with user preferences: {user.model_dump()}")

    has_recent_tool_results = any(
        isinstance(msg, ToolMessage) for msg in messages[-3:] 
        if isinstance(msg, ToolMessage)
    )

    system_content = """You are a helpful AI travel assistant designed to help users plan trips, search for information, and get travel recommendations.
    You are a travel assistant. Always try to extract structured fields (origin, destination, budget, interests, want_flight_links) from the user’s query when calling a tool. For example: 'I want to go from Dubai to Cape Town for hiking with a $2000 budget' → origin=Dubai, destination=Cape Town, interests=hiking, budget=$2000.

    Tools Available:
    - rag_process_tool: Get relevant information from travel documents and guides
    - search_query_tool: Search for current travel information, prices, and updates
    - suggest_trips_tool: Suggest travel itineraries based on preferences

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
    Transform raw travel data into meaningful, actionable travel guidance that helps users make informed decisions about their trips.

    TOOL USAGE GUIDELINES:
    - When the user asks for current information, prices, or external links (for example: flights, tickets, live prices), call the appropriate tool only if explicitly requested.
    - Tool names you can call exactly as provided: `rag_process_tool`, `search_query_tool`, `suggest_trips_tool`.
    - Use `rag_process_tool` for answering from static travel documents and guides, such as:
        - Flight schedules or pre-made itineraries or premade trips 
        - FAQs, travel restrictions, or rules contained in the RAG docs
        - Destination highlights already documented
    - When calling a tool, return a structured tool call with the tool name and arguments, for example (pseudocode):
        {"name": "search_query_tool", "args": {"query": "Flights from New York to Cairo roundtrip, departure next week, economy"}}
    - For suggest_trips_tool, include the user preferences as a dictionary in the args, for example:
        {"name": "suggest_trips_tool", "args": {"query": "Suggest a trip to Paris", "user": {...user preferences...}}}
    - After receiving tool results, your assistant response must interpret and format the results for the user (do NOT simply echo raw tool output).
    - Respect the user's preference for flight links (want_flight_links). If set to False, do not include flight information unless explicitly requested.
    """

    if rag_context:
        system_content += f"\nRAG Context: {rag_context}\nUse this information to provide accurate, specific answers about travel destinations, requirements, and opportunities."

    if has_recent_tool_results:
        system_content += "\nIMPORTANT: You have just received tool results. Your task is to process this raw data and present it as a helpful, structured response to the user. Do NOT simply repeat the tool output - interpret, organize, and enhance it for the user."

    system_message = SystemMessage(content=system_content)
    formatted_messages = [system_message]

    for msg in messages:
        if isinstance(msg, dict):
            if msg.get("role") == "user":
                formatted_messages.append(HumanMessage(content=msg["content"]))
            elif msg.get("role") == "assistant":
                if msg.get("tool_calls"):
                    ai_msg = AIMessage(content=msg.get("content", ""))
                    # Convert UserScheme to dict for tool calls
                    for tc in msg["tool_calls"]:
                        if tc["name"] == "suggest_trips_tool" and "user" in tc["args"]:
                            tc["args"]["user"] = user.model_dump() if hasattr(user, "model_dump") else tc["args"]["user"]
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
            formatted_messages.append(msg)

    response = llm_with_tools.invoke(formatted_messages)
    new_messages = messages + [response]
    
    logger.info("call_model function exited")
    return {
        "messages": new_messages,
        "user": user,
        "context": state.get("context", ""),
        "rag_context": rag_context
    }

# def handle_tools(state: AgentState) -> AgentState:
#     """Handle tool execution and return updated state with tool messages."""
#     logger.info("handle_tools function entered")

#     messages = state["messages"]
#     user = state.get("user")
#     context = state.get("context", "")
#     rag_context = state.get("rag_context", "")

#     last_message = messages[-1]
    
#     if not isinstance(last_message, AIMessage) or not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
#         logger.info("handle_tools: No tool calls found, returning state")
#         return {
#             "messages": messages,
#             "user": user,
#             "context": context,
#             "rag_context": rag_context
#         }

#     tool_messages = []
#     for tool_call in last_message.tool_calls:
#         tool_name = tool_call["name"]
#         tool_args = tool_call["args"]
#         tool_call_id = tool_call.get("id", "")

#         logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

#         tool = next((t for t in tools if t.name == tool_name), None)
#         if not tool:
#             logger.error(f"Tool {tool_name} not found")
#             tool_messages.append(ToolMessage(
#                 content=f"Error: Tool {tool_name} not found",
#                 tool_call_id=tool_call_id
#             ))
#             continue

#         try:
#             if tool_name == "suggest_trips_tool":
#                 tool_args["user"] = user.model_dump() if hasattr(user, "model_dump") else user
#             result = tool.invoke(tool_args)
#             tool_messages.append(ToolMessage(
#                 content=str(result),
#                 tool_call_id=tool_call_id
#             ))
#         except Exception as e:
#             logger.error(f"Error executing tool {tool_name}: {str(e)}")

#             tool_messages.append(ToolMessage(
#                 content=f"Error executing tool {tool_name}: {str(e)}",
#                 tool_call_id=tool_call_id
#             ))

#     logger.info("handle_tools function exited")
#     return {
#         "messages": messages + tool_messages,
#         "user": user,
#         "context": context,
#         "rag_context": rag_context
#     }

def route_to_tools(state: AgentState):
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

def rag_retrieval(state: AgentState) -> AgentState:
    """Handle RAG retrieval for context enhancement."""
    logger.info("rag_retrieval function entered")

    messages = state["messages"]
    user = state.get("user", UserScheme())
    context = state.get("context", "")
    
    last_user_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_message = msg
            break
    
    if not last_user_message:
        return state
    
    try:
        rag_result = rag_process(last_user_message.content)
        logger.info("rag_retrieval function exited successfully")
        return {
            "messages": messages,
            "user": user,
            "context": context,
            "rag_context": rag_result
        }
    except Exception as e:
        logger.error(f"rag_retrieval function failed with error: {str(e)}")
        return {
            "messages": messages,
            "user": user,
            "context": context,
            "rag_context": ""
        }

def custom_tool_node(tools):
    base_tool_node = ToolNode(tools)

    def _tool_node(state: AgentState) -> AgentState:
        result = base_tool_node.invoke(state)
        return {
            "messages": result["messages"],
            "user": state.get("user"),
            "context": state.get("context", ""),
            "rag_context": state.get("rag_context", "")
        }

    return _tool_node


def get_agent():
    """Create and return the compiled LangGraph agent."""
    logger.info("get_agent function entered")

    workflow = StateGraph(AgentState)

    workflow.add_node("call_model", call_model)
    workflow.add_node("tools", custom_tool_node(tools))
    workflow.add_node("rag_retrieval", rag_retrieval)

    workflow.set_entry_point("call_model")
    workflow.add_conditional_edges(
        "call_model",
        route_to_tools,
        {"tools": "tools", END: END}
    )
    workflow.add_edge("tools", "rag_retrieval")
    workflow.add_edge("rag_retrieval", "call_model")

    memory = MemorySaver()
    logger.info("get_agent function exited")
    return workflow.compile(checkpointer=memory)

def run_agent(user_input: str, user_preferences: Optional[UserScheme] = None):
    """Run the agent with a user input."""
    logger.info("run_agent function entered")

    if user_preferences is None:
        user_preferences = UserScheme()
    
    logger.info(f"User preferences in run_agent: {user_preferences.model_dump()}")
    agent = get_agent()
    config = {"configurable": {"user_id": "1", "thread_id": "1"}}

    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "user": user_preferences,
        "context": "",
        "rag_context": ""
    }
    result = agent.invoke(initial_state, config=config)
    
    trip_itinerary = None
    for msg in result.get("messages", []):
        try:
            content = None
            if isinstance(msg, ToolMessage):
                content = msg.content
                logger.info("Found ToolMessage (candidate) for itinerary")
            elif isinstance(msg, AIMessage):
                content = msg.content
                logger.info("Found AIMessage (candidate) for itinerary")
            elif isinstance(msg, dict) and msg.get('role') == 'tool':
                content = msg.get('content')
                logger.info("Found dict tool message (candidate) for itinerary")
            elif isinstance(msg, dict) and msg.get('role') == 'assistant':
                content = msg.get('content')
                logger.info("Found dict assistant message (candidate) for itinerary")

            if isinstance(content, str) and content.strip():
                trip_itinerary = content
                logger.info("Selected non-empty itinerary content")
                break
        except Exception as e:
            logger.error(f"Error processing message: {type(e).__name__}: {str(e)}")
            continue

    if not trip_itinerary:
        logger.info("run_agent function exited - no itinerary found")
        return "No itinerary found."

    cleaned_text = str(trip_itinerary).strip()
    cleaned_text = re.sub(r"\n\s*\n", "\n\n", cleaned_text)
    cleaned_text = re.sub(r"\t+", " ", cleaned_text)
    cleaned_text = re.sub(r" +", " ", cleaned_text)

    additional_sections = []
    if getattr(user_preferences, 'want_flight_links', False):
        try:
            user_location = getattr(user_preferences, 'user_location', None)
            destination = getattr(user_preferences, 'destination', None)
            duration = getattr(user_preferences, 'duration', None)
            if not user_location or not destination:
                logger.warning("Missing user_location or destination for flight search")
                additional_sections.append("Error: Please provide both a starting location and destination for flight search.")
            else:
                flight_query = f"Flights from {user_location} to {destination} roundtrip duration {duration}"
                logger.info(f"Running flight search: {flight_query}")
                flight_results = search_query(flight_query)
                if flight_results:
                    additional_sections.append("Flight search results:\n" + str(flight_results))
        except Exception as e:
            logger.error(f"Flight search failed: {type(e).__name__}: {e}")
            additional_sections.append(f"Flight search failed: {type(e).__name__}: {e}")

    final_output = cleaned_text
    if additional_sections:
        final_output += "\n\n" + "\n\n".join(additional_sections)

    logger.info("run_agent function exited")
    return final_output

if __name__ == "__main__":
    logger.info("Main execution started")
    
    user_input = input("Enter your query: ")
    have_preferences = input("Do you have preferences? (yes/no): ").strip().lower() == "yes"

    user_preferences = None

    if have_preferences:
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