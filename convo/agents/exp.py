import sys
import os
import logging
from typing import Annotated, Dict, List, Sequence, TypedDict, Any, Optional, Union, cast
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq
from langchain_core.messages import ToolMessage
import re
import json
import uuid
from langgraph.store.memory import InMemoryStore
from langchain_core.runnables.config import RunnableConfig
from langgraph.store.base import BaseStore
from trustcall import create_extractor
from langsmith import traceable

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
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
settings = get_settings()
tracer = FlowTracer()

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user: UserScheme
    context: str
    rag_context: str

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                             google_api_key=settings.GOOGLE_API_KEY)

# llm = ChatGroq(
#     model="llama-3.1-8b-instant",
#     temperature=0.0,
#     max_retries=2,
#     # other params...
# )

user_extractor = create_extractor(
    llm,
    tools = [
        {
            "type": "function", 
            "function": {
                "name": "UserScheme",
                "description": "Extract user preferences from conversation",
                "parameters": UserScheme.model_json_schema()
            }
        }
    ],
    tool_choice = "UserScheme",
    enable_inserts= True,
)

@tool
@traceable
def rag_process_tool(query: str) -> str:
    """Process queries using RAG to get relevant information from documents."""
    logger.info("rag_process_tool entered")

    result = rag_process(query)

    logger.info("rag_process_tool exited")
    return result

@tool
@traceable
def search_query_tool(query: str) -> str:
    """Search for information using web search for flights, pre-made trips, or anything you don't have information about."""
    logger.info("search_query_tool entered")

    result = search_query(query)

    logger.info("search_query_tool exited")
    return result

@tool
@traceable
def suggest_trips_tool(query: str, user: Dict[str, Any]) -> str:
    """Suggest travel itineraries based on user preferences and queries."""
    logger.info(f"suggest_trips_tool entered with query: {query}")
    try:
        # Convert to dictionary if string format
        if isinstance(user, str):
            try:
                user = json.loads(user.replace("'", '"'))
            except:
                user = {}
                
        # Use empty dict if user is None or empty
        user = user or {}
        
        # Create UserScheme instance
        try:
            user_scheme = UserScheme(**user)
        except:
            user_scheme = UserScheme()
            
        logger.info(f"User preferences: {user_scheme.model_dump()}")
        
        # Get trip suggestions
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

def extract_user_info(state: AgentState, config: RunnableConfig, store: BaseStore):
    """Extract user information from conversation and store it in memory."""
    logger.info("extract_user_info function entered")
    
    user_id = config["configurable"].get("user_id", "default_user")
    namespace = ("memory", user_id)
    
    # Don't override existing user preferences from direct input
    if state["user"] and any(v for v in state["user"].model_dump().values() if v):
        logger.info("Using provided user preferences instead of extracting from conversation")
        user_preferences = state["user"].model_dump()
        
        # Store user preferences
        key = "user_preferences"
        store.put(namespace, key, user_preferences)
        
        logger.info(f"Using existing user preferences: {user_preferences}")
    else:
        # Extract user profile
        system_msg = "Extract the user profile from the following conversation"
        result = user_extractor.invoke({"messages": [SystemMessage(content=system_msg)] + state["messages"]})
        scheme = result.get("responses", [])
        user_preferences = scheme[0].model_dump() if scheme else {}
        
        # Store user preferences
        key = "user_preferences"
        store.put(namespace, key, user_preferences)
        
        # Update the state with extracted user preferences
        if user_preferences:
            # Handle None values for boolean field
            if user_preferences.get('want_flight_links') is None:
                user_preferences['want_flight_links'] = False
            
            try:
                updated_user = UserScheme(**user_preferences)
                state["user"] = updated_user
                logger.info(f"Updated user preferences in state: {updated_user.model_dump()}")
            except Exception as e:
                logger.error(f"Error creating UserScheme: {e}")
                logger.warning("Keeping default user preferences")
        else:
            logger.warning("No user preferences extracted, keeping default")
    
    logger.info("extract_user_info function exited")
    return state

def call_model(state: AgentState, config: RunnableConfig, 
               store: BaseStore) -> AgentState:
    """Process the conversation and generate AI responses."""
    logger.info("call_model function entered")

    user_id = config["configurable"].get("user_id", "default_user")

    namespace = ("memory", user_id)
    key = "user_memory"
    existing_memory = store.get(namespace, key)

    if existing_memory:
        existing_memory_content = existing_memory.value.get("memory")
    else:
        existing_memory_content = "No existing memory."

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
    - Use `search_query_tool` for current information, live prices, and flight booking links, such as:
        - Current flight prices and booking links
        - Real-time travel information
        - Hotel booking links and current rates
        - Live transportation schedules
    - IMPORTANT: If the user wants flight links (want_flight_links is True), you MUST call BOTH tools:
        1. Call suggest_trips_tool first to create the detailed trip itinerary
        2. Call search_query_tool to get actual flight information and booking links
    - When calling a tool, return a structured tool call with the tool name and arguments, for example (pseudocode):
        {"name": "search_query_tool", "args": {"query": "Flights from New York to Cairo roundtrip, departure next week, economy"}}
    - For suggest_trips_tool, include the user preferences as a dictionary in the args, for example:
        {"name": "suggest_trips_tool", "args": {"query": "Suggest a trip to Paris", "user": {...user preferences...}}}
    - After receiving tool results, your assistant response must interpret and format the results for the user (do NOT simply echo raw tool output).
    - Respect the user's preference for flight links (want_flight_links). If set to False, do not include flight information unless explicitly requested.
    """

    # Add user preferences to system message
    if user.want_flight_links:
        system_content += f"\n\nCRITICAL INSTRUCTION: The user wants flight links included (want_flight_links=True). You MUST call search_query_tool with a query like \"Flights from {user.user_location or 'origin'} to {user.destination or 'destination'}\" to get current flight information and booking links for their trip. This is a REQUIRED step."
    
    system_content += f"\nUser Location: {user.user_location}, Destination: {user.destination}, Duration: {user.duration}, Interests: {user.interests}, Budget: {user.budget}, Travel Style: {user.travel_style}, Flight Links: {user.want_flight_links}"

    if rag_context:
        system_content += f"\nRAG Context: {rag_context}\nUse this information to provide accurate, specific answers about travel destinations, requirements, and opportunities."

    if has_recent_tool_results:
        system_content += "\nIMPORTANT: You have just received tool results. Your task is to process this raw data and present it as a helpful, structured response to the user. Do NOT simply repeat the tool output - interpret, organize, and enhance it for the user."

    system_message = SystemMessage(content=system_content)
    formatted_messages = [system_message]
    
    logger.info(f"System message length: {len(system_content)}")
    # Avoid logging unicode characters that might cause encoding issues
    try:
        logger.info(f"System message preview: {system_content[:200]}")
    except UnicodeEncodeError:
        logger.info("System message preview contains characters that cannot be encoded in console output")
    logger.info(f"User preferences in system: want_flight_links={user.want_flight_links}, location={user.user_location}, destination={user.destination}")

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

    logger.info(f"Model response type: {type(response)}")
    logger.info(f"Model response content: {repr(response.content if hasattr(response, 'content') else 'No content')}")
    if hasattr(response, 'tool_calls'):
        logger.info(f"Model response tool_calls: {response.tool_calls}")
    
    new_messages = messages + [response]
    
    logger.info("call_model function exited")
    return {
        "messages": new_messages,
        "user": user,
        "context": state.get("context", ""),
        "rag_context": rag_context
    }

def write_memory(state: AgentState, config: RunnableConfig, store: BaseStore) -> AgentState:
    """Reflect on the chat history and save a memory to the store."""
    logger.info("write_memory function entered")
    
    # Get the user ID from the config
    user_id = config["configurable"].get("user_id", "default_user")

    # Retrieve existing memory from the store
    namespace = ("memory", user_id)
    existing_memory = store.get(namespace, "user_memory")
        
    # Extract the memory
    if existing_memory:
        existing_memory_content = existing_memory.value.get('memory')
    else:
        existing_memory_content = "No existing memory found."

    # Create new memory from the chat history and any existing memory
    CREATE_MEMORY_INSTRUCTION = """You are collecting information about the user to personalize your responses.

    CURRENT USER INFORMATION:
    {memory}

    INSTRUCTIONS:
    1. Review the chat history below carefully
    2. Identify new information about the user, such as:
    - Personal details (name, location)
    - Preferences (likes, dislikes)
    - Interests and hobbies
    - Past experiences
    - Goals or future plans
    3. Merge any new information with existing memory
    4. Format the memory as a clear, bulleted list
    5. If new information conflicts with existing memory, keep the most recent version

    Remember: Only include factual information directly stated by the user. Do not make assumptions or inferences.

    Based on the chat history below, please update the user information:"""

    # Format the memory in the system prompt
    system_msg = CREATE_MEMORY_INSTRUCTION.format(memory = existing_memory_content)
    new_memory = llm.invoke([SystemMessage(content = system_msg)] + state['messages'])

    # Overwrite the existing memory in the store 
    key = "user_memory"

    # Write value as a dictionary with a memory key
    store.put(namespace, key, {"memory": new_memory.content})
    
    logger.info("write_memory function exited")
    return state

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
        # First, let's process any tool calls to fix JSON parsing issues
        messages = state["messages"]
        last_message = messages[-1]
        
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                if tool_call["name"] == "suggest_trips_tool" and "user" in tool_call["args"]:
                    # Check if user is a string that looks like a dict
                    user_arg = tool_call["args"]["user"]
                    if isinstance(user_arg, str):
                        try:
                            # Try to parse it as JSON
                            if user_arg.strip() in ["{}", "'{}'"]:
                                # Empty user dict, replace with actual user data from state
                                user_data = state.get("user")
                                if hasattr(user_data, "model_dump"):
                                    tool_call["args"]["user"] = user_data.model_dump()
                                else:
                                    # Create a basic empty dict if we can't get user data
                                    tool_call["args"]["user"] = {}
                            else:
                                # Try to parse as JSON if it's a non-empty string
                                parsed_user = json.loads(user_arg.replace("'", '"'))
                                tool_call["args"]["user"] = parsed_user
                        except (json.JSONDecodeError, TypeError):
                            # If parsing fails, create a basic empty dict
                            tool_call["args"]["user"] = {}
        
        # Now invoke the base tool node
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

    workflow.add_node("extract_user_info", extract_user_info)
    workflow.add_node("call_model", call_model)
    workflow.add_node("tools", custom_tool_node(tools))
    workflow.add_node("rag_retrieval", rag_retrieval)
    workflow.add_node("memory", write_memory)

    workflow.set_entry_point("extract_user_info")
    workflow.add_edge("extract_user_info", "call_model")
    workflow.add_conditional_edges(
        "call_model",
        route_to_tools,
        {"tools": "tools", END: "memory"}
    )
    # workflow.add_edge("tools", "rag_retrieval")
    # workflow.add_edge("rag_retrieval", "call_model")
    workflow.add_edge("tools", "call_model")
    workflow.add_edge("memory", END)

    # Create in-memory store for across thread memory
    across_thread_memory = InMemoryStore() 

    # Checkpointer for short-term (within-thread) memory
    within_thread_memory = MemorySaver()

    logger.info("get_agent function exited")
    return workflow.compile(checkpointer=within_thread_memory, store=across_thread_memory)

def run_agent(user_input: str, user_preferences: Optional[UserScheme] = None):
    """Process user query and return formatted travel itinerary with optional flight links."""
    logger.info("run_agent function entered")

    if user_preferences is None:
        user_preferences = UserScheme()

    logger.info(f"User preferences in run_agent: {user_preferences.model_dump()}")
    agent = get_agent()

    # Initialize agent invocation
    config = {"configurable": {"user_id": str(uuid.uuid4()), "thread_id": str(uuid.uuid4())}}
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "user": user_preferences,
        "context": "",
        "rag_context": ""
    }
    result = agent.invoke(initial_state, config=config)
    
    # Message categorization
    itinerary_message = None
    flight_links_message = None
    final_ai_message = None
    
    # Find different message types
    for msg in result.get("messages", []):
        if not hasattr(msg, 'content') or not msg.content.strip():
            continue
            
        content = msg.content.strip()
        
        # Check for itinerary (has Day 1, Day 2 format)
        if "Day 1" in content and "Day 2" in content:
            itinerary_message = msg
        # Check for search results (likely flight info)
        elif isinstance(msg, ToolMessage) and "URL:" in content and "flight" in content.lower():
            flight_links_message = msg
        # Keep track of latest AI message
        elif isinstance(msg, AIMessage):
            final_ai_message = msg
    
    # Build final output
    output_parts = []
    
    # 1. Add itinerary first (from either tool message or AI message)
    if itinerary_message:
        itinerary_content = itinerary_message.content
        
        # Remove any potential flight sections to avoid duplication
        if user_preferences.want_flight_links:
            # Simple pattern to find and remove flight sections
            itinerary_content = re.sub(r'(?i)(\n\n|^).*?FLIGHT.*?(\n\n|\Z)', '\n\n', itinerary_content)
            
        output_parts.append(itinerary_content.strip())
    
    # 2. Add summary content from final AI message if it's not duplicating the itinerary
    if final_ai_message and final_ai_message != itinerary_message:
        content = final_ai_message.content
        
        # Remove any flight sections to avoid duplication
        if user_preferences.want_flight_links:
            # Simple pattern to remove flight sections and other flight references
            content = re.sub(r'(?i)(\n\n|^).*?FLIGHT.*?(\n\n|\Z)', '\n\n', content)
            content = re.sub(r'(?i)I\'ll.*?flights?.*?(\n|\Z)', '', content)
        
        # Only add if not empty and adds value
        if content.strip() and not content.strip().isspace():
            output_parts.append(content.strip())
    
    # 3. Add flight links at the very end if user requested them
    if user_preferences.want_flight_links and flight_links_message:
        output_parts.append("\n\nFLIGHT INFORMATION:")
        output_parts.append(flight_links_message.content.strip())
    
    # If we have no content, return a fallback message
    if not output_parts:
        logger.info("No output generated.")
        return "No itinerary found."

    # Combine all parts and clean up formatting
    final_output = "\n\n".join(output_parts)
    final_output = re.sub(r'\n{3,}', '\n\n', final_output)  # Replace multiple newlines with just two
    final_output = re.sub(r' {2,}', ' ', final_output)      # Replace multiple spaces with just one
    
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