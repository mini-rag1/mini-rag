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

# Import for create_extractor - this might need to be adjusted based on your actual package
try:
    from trustcall import create_extractor
except ImportError:
    # If trustcall is not available, you may need to install it or use an alternative
    # You can modify this to use the correct package for your environment
    print("Warning: 'trustcall' package not found. Please install it if needed for create_extractor functionality.")
    
    # Placeholder function in case the import fails
    def create_extractor(llm, tools, tool_choice=None, enable_inserts=False):
        """Placeholder function when trustcall is not available."""
        print("Warning: Using placeholder create_extractor function")
        def invoke(inputs):
            return {"responses": []}
        return type('Extractor', (), {'invoke': invoke})

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


from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",google_api_key=settings.GOOGLE_API_KEY)

user_extractor = create_extractor(
    llm,
    tools = [{"type": "function", "function": {"name": "UserScheme", "description": "Extract user preferences from conversation", "parameters": UserScheme.model_json_schema()}}],
    tool_choice = "UserScheme",
    enable_inserts= True,
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
        # Handle case when user is empty or not properly formatted
        if not user or (isinstance(user, str) and user.strip() in ["{}", "'{}'"]):
            user = {}
        
        # Convert string to dict if needed
        if isinstance(user, str):
            try:
                user = json.loads(user.replace("'", '"'))
            except (json.JSONDecodeError, TypeError):
                user = {}
        
        # Ensure the user data has the correct types before creating UserScheme
        if "duration" in user and not isinstance(user["duration"], str):
            user["duration"] = str(user["duration"])
        
        if "interests" in user and not isinstance(user["interests"], str):
            if isinstance(user["interests"], list):
                user["interests"] = ", ".join(user["interests"]) if user["interests"] else ""
            else:
                user["interests"] = str(user["interests"])
        
        # Create a default UserScheme if user dict is empty
        if not user:
            user_scheme = UserScheme()
        else:
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

def extract_user_info(state: AgentState, config: RunnableConfig, store: BaseStore):
    """Extract user information from conversation and store it in memory."""
    logger.info("extract_user_info function entered")
    
    user_id = config["configurable"].get("user_id", "default_user")
    namespace = ("memory", user_id)
    
    # Extract user profile
    system_msg = "Extract the user profile from the following conversation"
    result = user_extractor.invoke({"messages": [SystemMessage(content=system_msg)] + state["messages"]})
    scheme = result.get("responses", [])
    user_preferences = scheme[0].model_dump() if scheme else {}
    
    # Store user preferences
    key = "user_preferences"
    store.put(namespace, key, user_preferences)
    
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

    MODEL_SYSTEM_MESSAGE = """You are a helpful assistant with memory that provides information about the user. 
    If you have memory for this user, use it to personalize your responses.
    Here is the memory (it may be empty): {memory}"""

    system_msg = MODEL_SYSTEM_MESSAGE.format(memory = existing_memory_content)

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
    system_msg = CREATE_MEMORY_INSTRUCTION.format(memory=existing_memory_content)
    new_memory = llm.invoke([SystemMessage(content=system_msg)] + state['messages'])

    # Overwrite the existing memory in the store 
    key = "user_memory"

    # Write value as a dictionary with a memory key
    store.put(namespace, key, {"memory": new_memory.content})
    
    logger.info("write_memory function exited")
    return state

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
    workflow.add_edge("tools", "rag_retrieval")
    workflow.add_edge("rag_retrieval", "call_model")
    workflow.add_edge("memory", END)

    # Create in-memory store for across thread memory
    across_thread_memory = InMemoryStore() 

    # Checkpointer for short-term (within-thread) memory
    within_thread_memory = MemorySaver()

    logger.info("get_agent function exited")
    return workflow.compile(checkpointer=within_thread_memory, store=across_thread_memory)

def run_agent(user_input: str, user_preferences: Optional[UserScheme] = None):
    """Run the agent with a user input."""
    logger.info("run_agent function entered")

    if user_preferences is None:
        user_preferences = UserScheme()
    
    logger.info(f"User preferences in run_agent: {user_preferences.model_dump()}")
    agent = get_agent()
    
    # Create a unique user_id and thread_id for this conversation
    user_id = str(uuid.uuid4())
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"user_id": user_id, "thread_id": thread_id}}

    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "user": user_preferences,
        "context": "",
        "rag_context": ""
    }
    
    # Get the complete result first
    result = agent.invoke(initial_state, config=config)
    
    # Optional: Stream the responses for real-time output
    # for chunk in agent.stream(initial_state, config, stream_mode="values"):
    #     if "messages" in chunk and chunk["messages"]:
    #         latest_msg = chunk["messages"][-1]
    #         if hasattr(latest_msg, "pretty_print"):
    #             latest_msg.pretty_print()
    
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