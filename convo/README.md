# LangGraph Travel Agent

A sophisticated AI travel planning agent built with LangGraph, LangChain, and Groq. This agent can help users plan trips, search for travel information, and provide personalized travel recommendations.

## Features

- **Intelligent Trip Planning**: AI-powered itinerary suggestions based on user preferences
- **RAG Integration**: Document-based knowledge retrieval for enhanced responses
- **Web Search**: Real-time travel information and updates
- **Trip Summarization**: Condensed travel plans with key highlights
- **Modular Architecture**: Easy to extend and customize

## Architecture

The agent uses a LangGraph workflow with the following components:

- **Agent Node**: Processes user queries and generates responses
- **Tools Node**: Executes various tools (RAG, search, trip planning, summarization)
- **RAG Retrieval Node**: Enhances responses with document knowledge
- **State Management**: Maintains conversation context and user preferences

## Prerequisites

- Python 3.8+
- API keys for:
  - Groq (for LLM)
  - Cohere (for embeddings)
  - Tavily (for web search)
  - Google (optional, for additional features)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd convo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

4. (Optional) Place a PDF file in the `data/` directory for RAG functionality:
```bash
mkdir -p data
# Copy your travel guide PDF to data/rag.pdf
```

## Usage

### Basic Usage

```python
from agents.langgraph_agent import run_agent
from scheme.user_scheme import UserScheme

# Create user preferences
user = UserScheme()
user.user_location = "New York"
user.destination = "Paris"
user.duration = "7 days"
user.interests = "Art, food, culture"
user.budget = "$3000"
user.travel_style = "Cultural exploration"

# Run the agent
result = run_agent("Plan a trip to Paris", user)
print(result)
```

### Advanced Usage

```python
from agents.langgraph_agent import get_agent
from langchain_core.messages import HumanMessage

# Create the agent
agent = get_agent()

# Create initial state
initial_state = {
    "messages": [HumanMessage(content="I want to plan a trip to Tokyo")],
    "user": user,
    "context": "",
    "rag_context": ""
}

# Run the agent
result = agent.invoke(initial_state)
```

### Testing

Run the test suite to verify everything works:

```bash
python test_agent.py
```

### Examples

See the example usage:

```bash
python example_usage.py
```

## Tools

### 1. RAG Process Tool
- **Function**: `rag_process_tool(query: str) -> str`
- **Purpose**: Retrieves relevant information from documents
- **Use Case**: Answering specific questions about destinations, requirements, etc.

### 2. Search Query Tool
- **Function**: `search_query_tool(query: str) -> str`
- **Purpose**: Searches the web for current travel information
- **Use Case**: Getting up-to-date prices, reviews, and travel tips

### 3. Suggest Trips Tool
- **Function**: `suggest_trips_tool(query: str) -> str`
- **Purpose**: Generates detailed travel itineraries
- **Use Case**: Creating day-by-day travel plans

### 4. Summarize Trip Tool
- **Function**: `summarize_trip_tool(query: str) -> str`
- **Purpose**: Summarizes travel information and highlights
- **Use Case**: Condensing long itineraries into key points

## Configuration

### Environment Variables

- `GROQ_API_KEY`: Required for LLM functionality
- `COHERE_API_KEY`: Required for document embeddings
- `TAVILY_API_KEY`: Required for web search
- `GOOGLE_API_KEY`: Optional, for additional features

### Model Configuration

The agent uses Groq's `llama-3.1-8b-instant` model by default. You can modify this in `agents/langgraph_agent.py`:

```python
llm = ChatGroq(
    model="llama-3.1-8b-instant",  # Change this to your preferred model
    temperature=0.3,
    max_retries=2,
    groq_api_key=settings.GROQ_API_KEY,
)
```

## Customization

### Adding New Tools

1. Create a new tool function in the `tools/` directory
2. Decorate it with `@tool`
3. Add it to the `tools` list in `agents/langgraph_agent.py`
4. Update the `handle_tools` function to handle the new tool

### Modifying the Workflow

The agent workflow is defined in the `get_agent()` function. You can:
- Add new nodes
- Modify the routing logic
- Change the state structure
- Add new conditional edges

### Custom Prompts

Modify the system prompts in the `call_model` function to change the agent's behavior and personality.

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
2. **API Key Errors**: Check your `.env` file and ensure all required API keys are set
3. **RAG Failures**: Ensure the PDF file exists in `data/rag.pdf`
4. **Tool Execution Errors**: Check the tool implementations and API responses

### Debug Mode

Enable debug logging by modifying the tracer in `agents/tracer.py`:

```python
# Add debug prints
print(f"DEBUG: {variable}")
```

## Performance

- **Response Time**: Typically 2-5 seconds for simple queries
- **Memory Usage**: ~100-200MB for the agent
- **Scalability**: Can handle multiple concurrent users
- **Caching**: RAG results are cached in the vector store

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the code examples
3. Open an issue on GitHub
4. Check the LangGraph and LangChain documentation

## Roadmap

- [ ] Multi-language support
- [ ] Integration with booking APIs
- [ ] Real-time flight/hotel search
- [ ] Mobile app interface
- [ ] Voice interaction
- [ ] Group travel planning
- [ ] Budget optimization
- [ ] Weather integration

