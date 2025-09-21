# 🌍 AI Travel Assistant Frontend

A beautiful and intuitive Streamlit-based frontend for the AI Travel Assistant that helps you plan trips, find destinations, and get travel recommendations.

## ✨ Features

- **💬 Interactive Chat Interface**: Natural conversation with the AI travel assistant
- **🎨 Modern UI**: Beautiful, responsive design with custom styling
- **🚀 Quick Start Buttons**: Pre-defined queries for common travel questions
- **📊 Chat Statistics**: Track your conversation metrics
- **📥 Export Functionality**: Download your chat history
- **🔧 System Status**: Real-time monitoring of agent and environment status
- **📱 Responsive Design**: Works great on desktop and mobile devices

## 🛠️ Setup

### Prerequisites
- Python 3.8+
- Required API keys (see main project README)

### Quick Start

1. **Using PowerShell (Recommended for Windows)**:
   ```powershell
   .\start.ps1
   ```

2. **Using Command Prompt**:
   ```cmd
   start.bat
   ```

3. **Manual Start**:
   ```bash
   pip install -r requirements_frontend.txt
   streamlit run app.py
   ```

## 🎯 Usage

### Getting Started
1. Open your browser to `http://localhost:8501`
2. Use the quick start buttons or type your own travel questions
3. Chat naturally with the AI assistant about your travel needs

### Example Queries
- "What are the best places to visit in Japan?"
- "Suggest a 7-day itinerary for Paris"
- "What's the weather like in Bali right now?"
- "Find budget hotels in Rome"
- "Plan a romantic getaway to Santorini"

### Features Overview

#### 🔍 Smart Search
Ask about current travel information, weather, events, and real-time data.

#### 📚 Knowledge Base
Access curated travel documents and destination guides stored in the system.

#### 🗺️ Trip Planning
Get personalized itineraries and recommendations based on your preferences.

#### 💬 Chat Management
- View conversation history
- Export chat transcripts
- Clear chat history
- Real-time response timing

## 🎨 UI Components

### Sidebar Features
- **System Status**: Monitor agent and environment health
- **Quick Actions**: Clear chat, export history
- **Chat Statistics**: Message counts and session info
- **Feedback System**: Share your experience

### Main Interface
- **Gradient Header**: Eye-catching welcome section
- **Chat Messages**: Distinct styling for user and AI messages
- **Error Handling**: Graceful error display with troubleshooting
- **Loading States**: Visual feedback during processing
- **Response Timing**: Performance metrics for each query

### Footer Information
- **Capability Cards**: Overview of assistant features
- **Pro Tips**: Best practices for using the assistant
- **Links**: Additional resources and project information

## 🔧 Troubleshooting

### Common Issues

1. **Module Import Error**:
   - Ensure you're running from the correct directory
   - Check that all dependencies are installed
   - Verify API keys are properly configured

2. **API Key Errors**:
   - Confirm all required API keys are in the `.env` file
   - Check that API keys are valid and have proper permissions

3. **Streamlit Issues**:
   - Try restarting the application
   - Clear browser cache
   - Check console for detailed error messages

## 📁 File Structure

```
frontend/
├── app.py                    # Main Streamlit application
├── config_frontend.py       # Frontend configuration helpers
├── requirements_frontend.txt # Frontend dependencies
├── start.bat                # Windows batch startup script
├── start.ps1                # PowerShell startup script
└── README.md                # This file
```

## 🔮 Future Enhancements

- **🗺️ Interactive Maps**: Visual destination mapping
- **📊 Analytics Dashboard**: Travel insights and statistics
- **🎨 Theme Customization**: Multiple UI themes
- **🔄 Auto-refresh**: Real-time updates for travel information
- **📝 Trip Planning Tools**: Advanced itinerary builders
- **🔗 Social Sharing**: Share travel plans and recommendations

---

**Happy Travels!** ✈️🌍
