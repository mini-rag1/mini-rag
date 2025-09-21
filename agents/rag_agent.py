import sys
import os
import logging
from typing import Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import MessagesState
from langchain_google_genai import ChatGoogleGenerativeAI

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from config import get_settings
    from scheme.user_scheme import UserScheme
    from tools.rag_process import rag_process
    from tools.search_query import search_query
    from tools.suggest_trips import suggest_trips
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running the script from the project root directory and all dependencies are installed.")
    sys.exit(1)

#################################################
# Configure logging to display in terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

settings = get_settings()

class State(MessagesState):
    rag_context: Optional[str]

llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=settings.GOOGLE_API_KEY,
    )

# llm = ChatGroq(
#     model="openai/gpt-oss-120b",
#     temperature=0.3,
#     max_retries=2,
#     groq_api_key=settings.GROQ_API_KEY,
# )

@tool
def rag_process_tool(query: str) -> str:
    """Process queries using RAG to get relevant information from documents."""
    logger.info(f"RAG processing query: {query}")

    result = rag_process(query)
    return result

@tool
def suggest_trip_tool(query: str):
    """Suggest travel itineraries based on user preferences and queries."""
    logger.info(f"Suggesting trips for query: {query}")

    extraction_prompt = f"""
    Extract travel preferences from this user query: "{query}"
    
    Please identify and return the following information if available:
    - user_location: Where is the user starting from (current location)
    - destination: Where does the user want to go
    - duration: How long is the trip (e.g., "6 days", "1 week")
    - interests: What activities or interests (e.g., culture, adventure, beaches)
    - budget: Budget range if mentioned
    - travel_style: Travel style preference (luxury, budget, adventure, etc.)
    
    Return the information in this exact format:
    user_location: [value or "Not specified"]
    destination: [value or "Not specified"]  
    duration: [value or "Not specified"]
    interests: [value or "Not specified"]
    budget: [value or "Not specified"]
    travel_style: [value or "Not specified"]
    """
    
    extraction_response = llm.invoke(extraction_prompt)
    preferences_text = extraction_response.content
    
    # Parse the extracted preferences
    user = UserScheme()
    lines = preferences_text.split('\n')
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            if value != "Not specified" and value:
                if key == "user_location":
                    user.user_location = value
                elif key == "destination":
                    user.destination = value
                elif key == "duration":
                    user.duration = value
                elif key == "interests":
                    user.interests = value
                elif key == "budget":
                    user.budget = value
                elif key == "travel_style":
                    user.travel_style = value
    
    result = suggest_trips(query, user)
    return result

@tool
def search_tool(query: str):
    """Search for information related to the user's query including any flight links or real time information"""
    logger.info(f"Performing search query: {query}")

    result = search_query(query)
    return result

tools = [rag_process_tool, suggest_trip_tool, search_tool]
llm_with_tools = llm.bind_tools(tools)

def call_model(state: State):
    messages = state["messages"]
    logger.info(f"Calling model with {len(messages)} messages")
    
    system_prompt = """
    You are a helpful travel assistant. You can assist users in planning their trips, finding relevant information, and providing recommendations.
    Tools Available:
    - rag_process_tool: Get relevant information from documents and guides provided
    - suggest_trip_tool: Suggest travel itineraries based on user preferences
    - search_tool: Perform searches to find specific information including any flight or real time information
    """

    response = llm_with_tools.invoke(
        [SystemMessage(content=system_prompt)] + messages
    )
    
    if hasattr(response, 'tool_calls') and response.tool_calls:
        logger.info(f"Model wants to use tools: {[tc['name'] for tc in response.tool_calls]}")
    else:
        logger.info("Model generated direct response without using tools")
    
    return {"messages": [response]}

def route_to_tools(state: State):
    if state["messages"][-1].tool_calls:
        logger.info("Routing to tools execution")
        return "tools"
    logger.info("Routing to END - no tools needed")
    return END

def update_rag_context(state: State):
    if isinstance(state["messages"][-1], ToolMessage):
        state["rag_context"] = state["messages"][-1].content
    return state

def run_agent():
    workflow = StateGraph(State)

    workflow.add_node("call_model", call_model)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("update_rag_context", update_rag_context)

    workflow.set_entry_point("call_model")

    workflow.add_conditional_edges(
        "call_model",
        route_to_tools,
        {"tools": "tools", END: END}
    )
    workflow.add_edge("tools", "update_rag_context")
    workflow.add_edge("update_rag_context", "call_model")

    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    thread_id = "user_session_1"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break
        
        logger.info(f"Processing user input: {user_input}")
        state = State(messages=[HumanMessage(content=user_input)], rag_context=None)

        logger.info("Invoking agent graph...")
        response = graph.invoke(state, config=config)
        
        if response["messages"][-1].content:
            logger.info("Assistant response generated successfully")
            print(f"Assistant: {response['messages'][-1].content}")
        else:
            logger.warning("No response generated by assistant")
            print("Assistant: No response generated.")