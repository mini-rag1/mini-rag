from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from scheme import UserScheme
import sys
import os

# Add the project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from config import get_settings
except ImportError:
    # Fallback for when running the file directly
    def get_settings():
        class MockSettings:
            GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
        return MockSettings()

def summarize_trip(user: UserScheme, itinerary: str) -> str:
    """
    Summarize travel information and provide key highlights based on user query.
    
    Args:
        user (UserScheme): User preferences.
        itinerary (str): User's travel itinerary.

    Returns:
        str: Summarized travel information with key highlights.
    """
    settings = get_settings()

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=settings.GOOGLE_API_KEY,
    )

    # llm = ChatGroq(
    #         model="llama-3.1-8b-instant",
    #         temperature=0.0,
    #         max_retries=2,
    #         # other params...
    #     )

    prompt = f"""
    You are a travel assistant. Based on the following user query, provide a helpful summary 
    of travel information, tips, or recommendations.

    User Itinerary: {itinerary}

    Please provide:
    1. A concise summary of the requested information
    2. Key highlights or important points
    3. Practical tips or recommendations
    4. Next steps or additional resources if applicable
    
    Format your response in a clear, structured way that's easy to read and understand.
    Focus on being helpful and actionable.
    WE DONT HAVE TOKENS SO PLEASE BE CONCISE DONT PUT ANYTHING UNNECESSARY
    """

    response = llm.invoke(prompt)
    return response.content
