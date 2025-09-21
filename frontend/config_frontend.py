import os
import streamlit as st
from pathlib import Path

def check_environment_setup():
    """Check if the environment is properly set up with required API keys"""
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"
    
    if not env_file.exists():
        return False, "Missing .env file in project root"
    
    required_keys = ["GOOGLE_API_KEY", "GROQ_API_KEY", "COHERE_API_KEY", "TAVILY_API_KEY"]
    missing_keys = []
    
    # Read .env file
    try:
        with open(env_file, 'r') as f:
            env_content = f.read()
            for key in required_keys:
                if key not in env_content or f"{key} =" not in env_content:
                    missing_keys.append(key)
    except Exception as e:
        return False, f"Error reading .env file: {str(e)}"
    
    if missing_keys:
        return False, f"Missing API keys: {', '.join(missing_keys)}"
    
    return True, "Environment setup is complete"

def display_setup_instructions():
    """Display setup instructions if environment is not properly configured"""
    st.error("🚨 **Environment Setup Required**")
    
    st.markdown("""
    ### 📋 Setup Instructions
    
    1. **Create a .env file** in the project root directory (`d:\\Enviroment\\convo\\.env`)
    2. **Add the following API keys** to your .env file:
    
    ```
    GOOGLE_API_KEY=your_google_api_key_here
    GROQ_API_KEY=your_groq_api_key_here
    COHERE_API_KEY=your_cohere_api_key_here
    TAVILY_API_KEY=your_tavily_api_key_here
    ```
    
    3. **Get your API keys**:
       - **Google AI**: Visit [Google AI Studio](https://ai.google.dev/)
       - **Groq**: Visit [Groq Console](https://console.groq.com/)
       - **Cohere**: Visit [Cohere Dashboard](https://dashboard.cohere.ai/)
       - **Tavily**: Visit [Tavily API](https://tavily.com/)
    
    4. **Restart the application** after adding the keys
    """)
    
    return False
