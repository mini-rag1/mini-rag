import sys
import os
import logging
from typing import Annotated, Dict, List, Sequence, TypedDict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq
from langchain_core.messages import ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import re
from langgraph.prebuilt import create_react_agent

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


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('langgraph_agent.log')
    ]
)
logger = logging.getLogger(__name__)

settings = get_settings()

# Initialize the Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    max_retries=2,
    google_api_key=settings.GOOGLE_API_KEY,  # Ensure this is set in your config
)

# Define tools
@tool
def rag_process_tool(query: str) -> str:
    """Process queries using RAG to get relevant information from documents."""
    logger.info("rag_process_tool entered")
    try:
        result = rag_process(query)
        logger.info("rag_process_tool exited")
        return result
    except Exception as e:
        logger.error(f"Error in rag_process_tool: {str(e)}")
        return f"Error processing RAG query: {str(e)}"

@tool
def search_query_tool(query: str) -> str:
    """Search for information using web search capabilities."""
    logger.info("search_query_tool entered")
    try:
        result = search_query(query)
        logger.info("search_query_tool exited")
        return result
    except Exception as e:
        logger.error(f"Error in search_query_tool: {str(e)}")
        return f"Error searching query: {str(e)}"

@tool
def suggest_trips_tool(query: str) -> str:
    """Suggest travel itineraries based on user preferences and queries."""
    logger.info("suggest_trips_tool entered")
    try:
        result = suggest_trips(query)
        logger.info("suggest_trips_tool exited")
        return result
    except Exception as e:
        logger.error(f"Error in suggest_trips_tool: {str(e)}")
        return f"Error suggesting trips: {str(e)}"

@tool
def summarize_trip_tool(query: str) -> str:
    """Summarize travel information and provide key highlights."""
    logger.info("summarize_trip_tool entered")
    try:
        result = summarize_trip(query)
        logger.info("summarize_trip_tool exited")
        return result
    except Exception as e:
        logger.error(f"Error in summarize_trip_tool: {str(e)}")
        return f"Error summarizing trip: {str(e)}"

tools = [rag_process_tool, search_query_tool, suggest_trips_tool, summarize_trip_tool]

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# System prompt to guide the agent
system_prompt = """You are a travel planning assistant. For queries about trip planning, use suggest_trips_tool to create an itinerary. For flight searches, use search_query_tool to find prices and links. If the query involves summarizing travel details, use summarize_trip_tool. If tools fail, provide a general response based on available knowledge. Ensure responses are clear, concise, and include relevant links when requested."""

# Create the agent
agent = create_react_agent(
    model=llm_with_tools,
    tools=tools,
    prompt=system_prompt,
    checkpointer=MemorySaver()  # Enable memory for conversation context
)

config = {"configurable": {"thread_id": "1"}}

# Get user input and invoke the agent
user_input = input("Enter your query: ")
result = agent.invoke(
    {
        "messages": [
            HumanMessage(content=user_input)
        ]
    }, config=config
)

# Print the messages defensively
for m in result.get("messages", []):
    try:
        print(getattr(m, 'content', m))
    except Exception:
        print(m)