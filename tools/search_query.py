from langchain_tavily import TavilySearch
import sys
import os
import traceback
# Ensure parent directory is in sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import get_settings

def search_query(query: str) -> str:
    """
    Search for information using web search capabilities.
    
    Args:
        query (str): Search query string.
        
    Returns:
        str: Search results or error message.
    """
    try:
        if not isinstance(query, str) or not query.strip():
            return "Empty query provided. Please provide a non-empty search query."

        settings = get_settings()

        # Create Tavily search tool (use api_key kwarg expected by the library)
        search_tool = TavilySearch(
            max_results=5,
            tavily_api_key=settings.TAVILY_API_KEY,
        )

        # Execute search
        results = search_tool.invoke(query)
        
        # Normalize results into a list
        results_list = []
        if results is None:
            results_list = []
        elif isinstance(results, dict):
            # common patterns: {'results': [...]}
            if 'results' in results and isinstance(results['results'], (list, tuple)):
                results_list = list(results['results'])
            else:
                # treat dict as single result
                results_list = [results]
        elif isinstance(results, (list, tuple)):
            results_list = list(results)
        else:
            # string or other single value
            return str(results)

        # Format results
        if results_list:
            formatted_results = []
            for i, result in enumerate(results_list[:5], 1):
                if isinstance(result, dict):
                    title = result.get('title', 'No title')
                    url = result.get('url', 'No URL')
                    content = result.get('content', 'No content')
                else:
                    title = str(result)[:80]
                    url = 'No URL'
                    content = str(result)

                formatted_results.append(f"{i}. {title}\n   URL: {url}\n   {content[:200]}...")

            return "\n\n".join(formatted_results)
        else:
            return "No search results found for your query."

    except Exception as e:
        # Log traceback to stderr for debugging, but return a friendly message
        traceback.print_exc()
        return f"Search failed: {type(e).__name__}: {str(e)}. Please try rephrasing your query."

