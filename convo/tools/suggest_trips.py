from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from config import get_settings
    from scheme.user_scheme import UserScheme
except ImportError:
    # Fallback for when running the file directly
    def get_settings():
        class MockSettings:
            GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
        return MockSettings()
    
    class UserScheme:
        def __init__(self):
            self.user_location = ""
            self.destination = ""
            self.duration = ""
            self.interests = ""
            self.budget = ""
            self.travel_style = ""

def suggest_trips(query: str,user : UserScheme) -> str:
    """
    Suggest travel itineraries based on user query and preferences.

    Args:
        query (str): User's travel request or preferences.

    Returns:
        str: Suggested travel itinerary with day-by-day plan.
    """
    logger.info(f"Starting trip suggestion for query: {query}")
    logger.info(f"User preferences - Location: {getattr(user, 'user_location', 'Not specified')}, "
                f"Destination: {getattr(user, 'destination', 'Not specified')}, "
                f"Duration: {getattr(user, 'duration', 'Not specified')}")
    
    settings = get_settings()

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=settings.GOOGLE_API_KEY,
    )

    prompt = f"""
        You are a travel assistant. Based on the following user query, suggest a detailed travel itinerary.

        User Query: {query}
        User Preferences:
        - Location: {user.user_location or 'Not specified'}
        - Destination: {user.destination or 'Not specified'}
        - Duration: {user.duration or 'Not specified'}
        - Interests: {user.interests or 'Not specified'}
        - Budget: {user.budget or 'Not specified'}
        - Travel Style: {user.travel_style or 'Not specified'}

        Use the user preferences combined with the query to create a tailored itinerary.

        Please provide a comprehensive day-by-day travel plan for the duration specified. 
        ⚠️ Do not exceed the duration given. If no duration is specified, provide a 7-day itinerary.

        For each day, include:
        - Morning activities
        - Afternoon activities  
        - Evening activities
        - Recommended restaurants or dining options
        - Transportation tips
        - Estimated costs for activities

        Format your response clearly with headings like:
        Day 1: ...
        Day 2: ...
        ... until the specified duration.

        DONT ADD UNNECESSARY CONTENT.
        ADD A HEADLINE FOR THE TRIP USING THE USER PREFERENCES GIVEN
        """

    logger.info("Sending prompt to LLM for trip suggestion")
    response = llm.invoke(prompt)
    logger.info(f"Received response from LLM, length: {len(response.content)} characters")
    return response.content


