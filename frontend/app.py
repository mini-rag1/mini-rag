# app.py
import sys
import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import traceback
from datetime import datetime
import time

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Change working directory to project root to find .env file
os.chdir(project_root)

# Import configuration helper
from frontend.config_frontend import check_environment_setup, display_setup_instructions

try:
    from agents.rag_agent import run_agent  # import your workflow setup
    AGENT_AVAILABLE = True
except ImportError as e:
    AGENT_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Page configuration
st.set_page_config(
    page_title="Travel Agent", 
    page_icon="✈️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-container {
        background-color: transparent;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .user-message {
        background-color: rgba(33, 150, 243, 0.1);
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        border-left: 4px solid #2196f3;
    }
    
    .assistant-message {
        background-color: rgba(76, 175, 80, 0.1);
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        border-left: 4px solid #4caf50;
    }
    
    .error-message {
        background-color: rgba(244, 67, 54, 0.1);
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        border-left: 4px solid #f44336;
        color: #c62828;
    }
    
    .info-box {
        background-color: transparent;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #4caf50;
        margin: 1rem 0;
        border-style: dashed;
    }
    
    .feature-card {
        background-color: transparent;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-style: solid;
    }
    
    /* Remove default streamlit styling */
    .stMarkdown > div {
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1> Travel Agent</h1>
    <p>Your intelligent companion for travel planning and exploration</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with information and features
with st.sidebar:
    st.markdown("### 🎯 Features")
    st.markdown("""
    - **🔍 Smart Search**: Web-based travel information
    - **📚 Knowledge Base**: Access to travel documents
    - **🗺️ Trip Suggestions**: Personalized recommendations
    - **💬 Interactive Chat**: Natural conversation interface
    """)
    
    st.markdown("### 🚀 How to Use")
    st.markdown("""
    1. Type your travel question in the chat
    2. Ask about destinations, hotels, activities
    3. Request trip suggestions or itineraries
    4. Get real-time travel information
    """)
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    # System status
    st.markdown("### 🔧 System Status")
    env_ok, env_message = check_environment_setup()
    
    if env_ok:
        st.success("✅ Environment: Ready")
    else:
        st.error("❌ Environment: Not Ready")
    
    if AGENT_AVAILABLE:
        st.success("✅ AI Agent: Connected")
    else:
        st.error("❌ AI Agent: Not Available")

# Check environment setup first
if not env_ok:
    display_setup_instructions()
    st.info(f"🔧 **Status**: {env_message}")
    st.stop()

# Check if agent is available
if not AGENT_AVAILABLE:
    st.error(f"❌ **Agent Import Error**: {IMPORT_ERROR}")
    st.markdown("""
    <div class="error-message">
        <h3>🛠️ Troubleshooting Steps:</h3>
        <ol>
            <li>Check if all required packages are installed</li>
            <li>Verify API keys are set in the .env file</li>
            <li>Ensure the agents module is properly configured</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Initialize the agent
try:
    with st.spinner("🔄 Initializing Travel Agent..."):
        graph, config, State = run_agent()
    st.success("✅ Travel Agent is ready!")
except Exception as e:
    st.error(f"❌ **Initialization Error**: {str(e)}")
    st.markdown(f"""
    <div class="error-message">
        <h3>Error Details:</h3>
        <pre>{traceback.format_exc()}</pre>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "conversation_started" not in st.session_state:
    st.session_state.conversation_started = False

# Welcome message if no conversation started
if not st.session_state.conversation_started and not st.session_state.chat_history:
    st.markdown("""
    <div class="info-box">
        <h3>👋 Welcome to your Travel Agent!</h3>
        <p>Start by asking me anything about travel - destinations, hotels, activities, or trip planning!</p>
        <p><strong>Example questions:</strong></p>
        <ul>
            <li>"What are the best places to visit in Japan?"</li>
            <li>"Suggest a 7-day itinerary for Paris"</li>
            <li>"What's the weather like in Bali right now?"</li>
            <li>"Find budget hotels in Rome"</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick action buttons for common queries
    st.markdown("#### 🚀 Quick Start")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗾 Japan Travel", use_container_width=True):
            st.session_state.quick_query = "What are the best places to visit in Japan?"
        if st.button("🏛️ Paris Itinerary", use_container_width=True):
            st.session_state.quick_query = "Suggest a 7-day itinerary for Paris"
    
    with col2:
        if st.button("🏝️ Bali Weather", use_container_width=True):
            st.session_state.quick_query = "What's the weather like in Bali right now?"
        if st.button("🏨 Rome Hotels", use_container_width=True):
            st.session_state.quick_query = "Find budget hotels in Rome"

# Main chat interface
st.markdown("### 💬 Chat with your Agent")

# Display chat history
for i, msg in enumerate(st.session_state.chat_history):
    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="🧑"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg.content)

# Handle quick query buttons
if "quick_query" in st.session_state and st.session_state.quick_query:
    user_input = st.session_state.quick_query
    st.session_state.quick_query = None  # Clear the quick query
else:
    # Chat input
    user_input = st.chat_input("Type your travel query here... ✈️", key="travel_input")

if user_input:
    # Mark conversation as started
    st.session_state.conversation_started = True
    
    # Add user message to chat history
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    
    # Display user message immediately
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)
    
    # Generate assistant response
    with st.chat_message("assistant", avatar="🤖"):
        # Create a placeholder for the loading message
        response_placeholder = st.empty()
        response_placeholder.markdown("🔍 **Thinking...**")
        
        start_time = time.time()
        
        try:
            # Run the agent graph
            state = State(messages=st.session_state.chat_history, rag_context=None)
            response = graph.invoke(state, config=config)
            
            end_time = time.time()
            response_time = round(end_time - start_time, 2)
            
            assistant_reply = response["messages"][-1].content or "Sorry, I couldn't generate a response. Please try again."
            
            # Clear the loading message and display the response
            response_placeholder.empty()
            st.markdown(assistant_reply)
            st.caption(f"⏱️ Response time: {response_time}s")
            
            st.session_state.chat_history.append(AIMessage(content=assistant_reply))
            
        except Exception as e:
            end_time = time.time()
            response_time = round(end_time - start_time, 2)
            
            error_message = f"❌ **Error**: An error occurred while processing your request.\n\n**Details**: {str(e)}"
            
            # Clear the loading message and display the error
            response_placeholder.empty()
            st.error("Sorry, I encountered an error while processing your request.")
            
            with st.expander("🔍 Error Details"):
                st.code(str(e))
                st.code(traceback.format_exc())
            
            st.caption(f"⏱️ Failed after: {response_time}s")
            st.session_state.chat_history.append(AIMessage(content=error_message))

# Footer with additional information and tips
st.markdown("---")
st.markdown("### 🌟 Travel Agent Capabilities")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h4>🔍 Smart Search</h4>
        <p>Real-time web search for up-to-date travel information, weather, and recommendations.</p>
        <small><strong>Try asking:</strong> "What's the weather in Tokyo?"</small>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h4>📚 Knowledge Base</h4>
        <p>Access to curated travel documents and comprehensive destination guides.</p>
        <small><strong>Try asking:</strong> "Tell me about local customs in Thailand"</small>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <h4>🗺️ Trip Planning</h4>
        <p>Personalized itineraries and travel suggestions based on your preferences and budget.</p>
        <small><strong>Try asking:</strong> "Plan a 5-day trip to Barcelona"</small>
    </div>
    """, unsafe_allow_html=True)

# Additional tips and information
st.markdown("### 💡 Pro Tips")
tips_col1, tips_col2 = st.columns(2)

with tips_col1:
    st.markdown("""
    **🎯 Be Specific**: Include details like budget, travel dates, and preferences for better recommendations.
    
    **🔄 Follow Up**: Ask follow-up questions to refine your travel plans.
    """)

with tips_col2:
    st.markdown("""
    **🌍 Global Coverage**: Ask about destinations worldwide - from popular cities to hidden gems.
    
    **📅 Real-time Info**: Get current weather, events, and travel advisories.
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🌍 <strong>Travel Agent</strong> | Powered by AI Agents | 
        <a href="https://github.com/mini-rag1/mini-rag" target="_blank">View on GitHub</a></p>
    </div>
    """, 
    unsafe_allow_html=True
)
if st.session_state.chat_history:
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📊 Chat Statistics")
        user_messages = len([msg for msg in st.session_state.chat_history if isinstance(msg, HumanMessage)])
        ai_messages = len([msg for msg in st.session_state.chat_history if isinstance(msg, AIMessage)])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("👤 Your Messages", user_messages)
        with col2:
            st.metric("🤖 AI Responses", ai_messages)
        
        st.markdown(f"**🕒 Session Started**: {datetime.now().strftime('%H:%M')}")
        
        # Export chat functionality
        if st.button("📥 Export Chat", use_container_width=True):
            chat_export = []
            for msg in st.session_state.chat_history:
                if isinstance(msg, HumanMessage):
                    chat_export.append(f"👤 User: {msg.content}")
                elif isinstance(msg, AIMessage):
                    chat_export.append(f"🤖 Assistant: {msg.content}")
            
            export_text = "\n\n".join(chat_export)
            st.download_button(
                label="💾 Download Chat History",
                data=export_text,
                file_name=f"travel_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

# Initialize session state for ratings
if "message_ratings" not in st.session_state:
    st.session_state.message_ratings = {}

# Add feedback section in sidebar
with st.sidebar:
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 💭 Feedback")
        
        feedback = st.text_area(
            "Share your thoughts about this travel agent:",
            placeholder="Your feedback helps us improve...",
            height=100
        )
        
        if st.button("📤 Submit Feedback", use_container_width=True):
            if feedback.strip():
                st.success("Thank you for your feedback! 🙏")
                # Here you could save feedback to a file or database
            else:
                st.warning("Please enter some feedback before submitting.")
