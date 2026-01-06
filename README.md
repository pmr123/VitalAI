# VitalAI - Health Intelligence Platform

A comprehensive health AI platform that combines Retrieval Augmented Generation (RAG), local large language models, multi-agent systems, and voice AI to provide personalized health insights and medical information.

## Overview

VitalAI is an intelligent health platform that processes wearable device data, provides medical knowledge through RAG-powered search, manages medications and reminders, and offers personalized health advice through a multi-agent AI system. The platform supports voice interaction and multilingual communication, making health information accessible to users worldwide.

## Features

### Health Data Management
- **Wearable Data Simulation** - Realistic synthetic health metrics including heart rate, HRV, steps, sleep duration, and calories burned
- **Health Scoring System** - Calculates Cardio-Metabolic Index (CMI) and component scores (Cardiac, Activity, Recovery, Metabolic)
- **Interactive Dashboard** - View health scores, metrics, and trends with advanced filtering options
- **Data Visualization** - Filter by metric type, view mode (daily averages or raw data), and time range

### AI-Powered Health Assistant
- **Multi-Agent System** - Intelligent routing to specialist agents based on query intent
- **RAG-Powered Knowledge Base** - 13 curated medical documents with vector search for accurate health information
- **Context-Aware Responses** - AI assistant uses your personal health data in responses
- **Source Citations** - Knowledge Expert agent provides source references for medical information
- **MCP (Model Context Protocol)** - Lightweight custom MCP server implementation for tool discovery and execution
- **Tool System** - 8 specialized tools for health data, medications, and reminders accessible via LLM function calling

### Medication & Reminder Management
- **Medication Tracking** - Add, list, and manage medications with dosage information
- **Smart Reminders** - Schedule medication reminders with customizable times and days
- **Background Scheduler** - Automatic reminder notifications triggered at scheduled times
- **Real-Time Notifications** - In-app notification system for due reminders

### Voice & Multilingual Support
- **Speech-to-Text** - Voice input using OpenAI Whisper for natural conversation
- **Text-to-Speech** - Browser-based TTS with on-demand audio playback
- **Multi-Language Support** - Translation support for 7 languages (English, Spanish, Hindi, Chinese, Arabic, Portuguese, French)
- **Voice UI** - Integrated microphone button and language selector in chat interface

### User Onboarding & Authentication
- **Email-Based Authentication** - Simple login system using email addresses
- **AI-Guided Onboarding** - Conversational onboarding flow that collects user information naturally
- **Smart Data Extraction** - Supports multiple input formats for height (cm, feet'inches", meters), weight (kg, lbs), and age
- **Profile Creation** - Automatically creates user profile from onboarding responses


### Personalized AI
- **Digital Clone Agent** - Personalized AI that learns user patterns and preferences
- **Pattern Recognition** - Analyzes health trends and provides personalized recommendations
- **User-Specific Insights** - Tailored advice based on individual health data and history

## Architecture

### System Flow

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                       │
│  (Dashboard | Chat | Onboarding | Login)                    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    Flask Application                        │
│  - Route Handling                                           │
│  - Session Management                                       │
│  - Authentication & Authorization                           │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   Database   │ │  Vector DB   │ │   Services   │
│   (SQLite)   │ │  (ChromaDB)  │ │  (Business   │
│              │ │              │ │   Logic)     │
└──────────────┘ └──────────────┘ └───────┬──────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
                    ▼                     ▼                     ▼
            ┌──────────────┐    ┌──────────────┐    ┌───────────────┐
            │  RAG Service │    │  LLM Service │    │ Agent System  │
            │  (Embeddings)│    │   (Ollama)   │    │ (Orchestrator)│
            └──────────────┘    └──────────────┘    └───────┬───────┘
                                                             │
                                                             ▼
                                                    ┌─────────────────┐
                                                    │  Supervisor     │
                                                    │  Agent          │
                                                    │  (Routing)      │
                                                    └────────┬────────┘
                                                             │
                    ┌────────────────────────────────────────┼────────────────────────────────────────┐
                    │                                        │                                        │
                    ▼                                        ▼                                        ▼
        ┌──────────────────┐                    ┌──────────────────┐                    ┌──────────────────┐
        │  Health Analyst  │                    │ Medication       │                    │  Knowledge       │
        │  Agent           │                    │ Manager Agent    │                    │  Expert Agent    │
        └──────────────────┘                    └──────────────────┘                    └──────────────────┘
                    │                                        │                                        │
                    └────────────────────────────────────────┼────────────────────────────────────────┘
                                                             │
                                                             ▼
                                                    ┌─────────────────┐
                                                    │  Digital Clone  │
                                                    │  Agent          │
                                                    └────────┬────────┘
                                                             │
                                                             ▼
                                                    ┌─────────────────┐
                                                    │  Response       │
                                                    │  + Metadata     │
                                                    └─────────────────┘
                                                             
                                                             
                    ┌─────────────────────────────────────────────────────────┐
                    │  Onboarding Agent (Separate Flow - /onboarding route)   │
                    │  - Not routed by Supervisor                             │
                    │  - Handles user profile setup only                      │
                    └─────────────────────────────────────────────────────────┘
```

### Multi-Agent System Flow

```
User Query
    │
    ▼
┌─────────────────┐
│  Supervisor     │  ← Intent Classification
│  Agent          │     (Pattern Matching + LLM)
└────────┬────────┘
         │
         │ Routes to one of:
         │
         ├─→ Health Analyst Agent
         │   (health scores, metrics, trends)
         │
         ├─→ Medication Manager Agent
         │   (medications, reminders)
         │
         ├─→ Knowledge Expert Agent
         │   (medical information, RAG)
         │
         └─→ Digital Clone Agent
             (personalized advice)
         │
         ▼
    Response + Metadata
    (Agent Type, Tools Used, Sources)

Note: Onboarding Agent operates independently on /onboarding route
and is not routed through the Supervisor Agent.
```

### RAG Pipeline Flow

```
User Query
    │
    ▼
┌─────────────────┐
│  Query          │
│  Embedding      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vector Search  │  ← ChromaDB
│  (Similarity)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Context        │
│  Retrieval      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM Generation │  ← Ollama
│  (with Context) │
└────────┬────────┘
         │
         ▼
    Response + Sources
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Web Framework** | Flask + Jinja2 | Backend server and templating |
| **Database** | SQLite + SQLAlchemy | Relational data storage |
| **Vector Database** | ChromaDB | Semantic search for knowledge base |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | Text embedding generation |
| **LLM** | Ollama (Llama 3.1 8B) | Local language model inference |
| **Speech-to-Text** | OpenAI Whisper | Voice transcription |
| **Text-to-Speech** | Web Speech API | Browser-based audio synthesis |
| **Translation** | NLLB-200 (Facebook) | Multi-language text translation |
| **Task Scheduler** | APScheduler | Background reminder processing |
| **Frontend** | Jinja2 + HTMX | Server-side rendering with dynamic updates |
| **MCP Server** | Custom Implementation | Lightweight Model Context Protocol server for tool discovery and execution |

## Prerequisites

- Python 3.10 or higher
- Ollama installed and running
- Approximately 10GB disk space for models
- GPU recommended for faster LLM inference (optional but recommended)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd VitalAI
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/WSL
# or: venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Ollama

In a separate terminal, start Ollama and pull the required model:

```bash
# Start Ollama server
ollama serve

# Pull the model (in another terminal)
ollama pull llama3.1:8b-instruct-q4_K_M
```

### 5. Initialize Database

```bash
python app.py
```

The application will automatically:
- Create the database if it doesn't exist
- Initialize the knowledge base vector database
- Load medical documents into the RAG system

## Usage

### First Time Setup

1. **Start the Application**
   ```bash
   python app.py
   ```
   The application will be available at `http://localhost:5000`

2. **Login**
   - Navigate to `http://localhost:5000/login`
   - Enter your email address (and optional name for new users)
   - New accounts are created automatically

3. **Complete Onboarding**
   - Answer questions about your age, gender, height, and weight
   - Optionally provide health goals, medical conditions, or medications
   - Your profile is created automatically upon completion

4. **Access Dashboard**
   - View your health dashboard
   - Generate test data to see health scores in action
   - Explore health metrics with filtering options

### Using the Chat Interface

1. **Navigate to Chat**
   - Go to `http://localhost:5000/chat`
   - The Supervisor Agent automatically routes queries to appropriate specialists

2. **Ask Questions**
   - **Health Data**: "What are my current health scores?"
   - **Medications**: "Add aspirin 100mg to my medications"
   - **Reminders**: "Set a reminder for vitamins at 8:00 AM"
   - **Medical Info**: "What is HRV and why does it matter?"
   - **Personalized Advice**: "Based on my patterns, what should I focus on?"

3. **Voice Input**
   - Click the microphone button to record voice messages
   - Select your preferred language from the dropdown
   - Responses can be played as audio using the TTS button

4. **Agent Selection**
   - Use the agent dropdown to force a specific agent
   - Leave on "Auto" to let the Supervisor route automatically

### Dashboard Features

- **Health Scores**: View CMI, Cardiac, Activity, Recovery, and Metabolic scores
- **Data Filtering**: Filter by metric type, view mode, and time range
- **Data Generation**: Generate synthetic health data for testing
- **User Profile**: View your profile information including BMI

## API Documentation

### Authentication

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/login` | GET | Display login page |
| `/login` | POST | Authenticate user (email-based) |
| `/logout` | POST | Logout current user |
| `/api/auth/status` | GET | Get current authentication status |

### User Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/user/<id>` | GET | Get user profile |
| `/api/user/<id>/health-summary` | GET | Get health scores summary |
| `/api/user/<id>/health-data` | GET | Get recent health data |
| `/api/user/<id>/health-data/history` | GET | Get filtered health data history |
| `/api/simulate-data` | POST | Generate synthetic health data |

### Onboarding

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/onboarding` | GET | Display onboarding page |
| `/api/onboarding/start` | POST | Start onboarding flow |
| `/api/onboarding/chat` | POST | Process onboarding conversation |
| `/api/onboarding/complete` | POST | Complete onboarding and create profile |
| `/api/onboarding/status` | GET | Get onboarding status |

### Chat & AI

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Send message to AI (legacy endpoint) |
| `/api/chat/search` | POST | Search knowledge base directly |
| `/api/agents` | GET | List all available agents |
| `/api/agents/chat` | POST | Send message through agent system |
| `/api/agents/classify` | POST | Classify intent without processing |
| `/api/agents/history` | GET | Get conversation history |
| `/api/agents/history/clear` | POST | Clear conversation history |

### RAG System

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/rag/status` | GET | Check RAG and LLM status |
| `/api/rag/reload` | POST | Reload knowledge base |

### MCP (Model Context Protocol)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/mcp/info` | GET | Get MCP server information |
| `/api/mcp/tools` | GET | List all available tools |
| `/api/mcp/tools/<name>` | GET | Get specific tool schema |
| `/api/mcp/execute` | POST | Execute a tool |
| `/api/mcp/batch` | POST | Execute multiple tools |

### Voice & Translation

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/voice/transcribe` | POST | Transcribe audio to text (Whisper) |
| `/api/voice/synthesize` | POST | Convert text to speech audio |
| `/api/voice/status` | GET | Check STT/TTS availability |
| `/api/translate` | POST | Translate text between languages |
| `/api/translate/languages` | GET | Get list of supported languages |

### Notifications

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/notifications` | GET | Get pending notifications |
| `/api/notifications/<id>/acknowledge` | POST | Acknowledge a notification |

### Debug Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/debug/reminders` | GET | Check all reminders in database |
| `/api/debug/scheduler/check` | POST | Manually trigger reminder check |
| `/api/debug/notifications` | GET | Check pending notifications queue |

## MCP Tools

The platform exposes 8 tools through the Model Context Protocol:

| Tool | Agent | Description |
|------|-------|-------------|
| `get_health_summary` | Health Analyst | Get user's health scores (CMI and components) |
| `get_health_metrics` | Health Analyst | Get specific health metrics with filtering |
| `get_user_profile` | Health Analyst | Get user profile information |
| `add_medication` | Medication Manager | Add new medication to user's list |
| `list_medications` | Medication Manager | List all user medications |
| `set_medication_reminder` | Medication Manager | Create medication reminder |
| `list_reminders` | Medication Manager | List all user reminders |
| `search_knowledge_base` | Knowledge Expert | Search medical knowledge base |

## Project Structure

```
VitalAI/
├── app.py                    # Flask application entry point
├── config.py                 # Application configuration
├── extensions.py             # Flask extensions initialization
├── requirements.txt          # Python dependencies
│
├── models/                   # Database models
│   ├── __init__.py          # Model exports
│   ├── user.py              # User profile model
│   ├── health_data.py       # Health metrics and scores
│   └── medication.py        # Medications and reminders
│
├── services/                 # Business logic services
│   ├── __init__.py
│   ├── data_simulator.py    # Synthetic health data generation
│   ├── health_scoring.py    # CMI and health score calculations
│   ├── rag_service.py       # RAG pipeline implementation
│   ├── llm_service.py       # Ollama LLM integration
│   ├── chat_service.py      # Chat orchestration
│   ├── mcp_tools.py         # MCP tool definitions and handlers
│   ├── mcp_server.py        # MCP server implementation
│   ├── reminder_scheduler.py # Background reminder processing
│   ├── voice_service.py     # Speech-to-text (Whisper)
│   ├── translation_service.py # Multi-language translation
│   ├── onboarding_service.py # User onboarding flow
│   └── agents/              # Multi-agent system
│       ├── __init__.py
│       ├── base_agent.py    # Base agent class
│       ├── supervisor_agent.py # Intent classification and routing
│       ├── health_analyst_agent.py # Health data analysis
│       ├── medication_agent.py # Medication management
│       ├── knowledge_agent.py # RAG-powered knowledge
│       ├── digital_clone_agent.py # Personalized AI
│       ├── onboarding_agent.py # User onboarding
│       └── orchestrator.py  # Agent coordination
│
├── data/
│   ├── knowledge_base/      # Medical knowledge documents (markdown)
│   │   ├── health_metrics/  # Heart rate, HRV, sleep, activity
│   │   ├── health_scores/  # CMI, score interpretation
│   │   ├── medications/    # Drug information, interactions
│   │   └── wellness/       # Exercise, sleep, stress management
│   ├── vectordb/           # ChromaDB vector database storage
│   └── *.db                # SQLite database files
│
├── templates/               # HTML templates (Jinja2)
│   ├── base.html           # Base template
│   ├── dashboard.html     # Health dashboard
│   ├── chat.html          # AI chat interface
│   ├── login.html         # Login page
│   └── onboarding.html    # Onboarding interface
│
└── static/                 # Static assets
    └── css/
        └── style.css      # Application styles
```

## Agent System

The platform uses a multi-agent architecture with specialized agents:

### Supervisor Agent
- Routes user queries to appropriate specialist agents
- Uses priority pattern matching for fast, deterministic routing
- Falls back to LLM-based classification for ambiguous queries
- Maintains conversation context across interactions

### Health Analyst Agent
- Analyzes user health data, scores, and trends
- Provides insights on health metrics and patterns
- Uses tools: `get_health_summary`, `get_health_metrics`, `get_user_profile`

### Medication Manager Agent
- Handles medication and reminder management
- Creates and manages medication schedules
- Uses tools: `add_medication`, `list_medications`, `set_medication_reminder`, `list_reminders`

### Knowledge Expert Agent
- Provides medical information using RAG
- Searches knowledge base and cites sources
- Uses tool: `search_knowledge_base`

### Digital Clone Agent
- Provides personalized advice based on user patterns
- Learns from user history and preferences
- Generates context-aware recommendations

### Onboarding Agent
- Guides new users through profile setup
- Extracts structured data from natural conversation
- Supports multiple input formats for health metrics
- **Note**: Operates independently on `/onboarding` route, not routed through Supervisor Agent

## Troubleshooting

### Ollama Connection Issues

**Problem**: "Ollama not connected" error

**Solution**:
```bash
# Ensure Ollama is running
ollama serve

# Verify model is available
ollama list

# Check if model is pulled
ollama pull llama3.1:8b-instruct-q4_K_M
```

### Slow Response Times

**Problem**: LLM responses take several minutes

**Possible Causes**:
1. Ollama running on CPU instead of GPU
2. Model not loaded into memory
3. First request (model loading)

**Solutions**:

1. **Enable GPU acceleration (WSL2)**:
   ```bash
   # Add CUDA library path
   echo 'export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   
   # Restart Ollama
   pkill ollama && ollama serve &
   
   # Verify GPU is working
   ollama run llama3.1:8b-instruct-q4_K_M "Hi" --verbose 2>&1 | grep -i gpu
   ```

2. **Use smaller model for testing**:
   ```bash
   ollama pull phi3:mini
   # Then update config.py: OLLAMA_MODEL = "phi3:mini"
   ```

### Knowledge Base Not Loading

**Problem**: RAG system shows empty knowledge base

**Solution**:
```bash
# Force reload knowledge base
curl -X POST http://localhost:5000/api/rag/reload

# Check status
curl http://localhost:5000/api/rag/status
```

### Voice Transcription Errors

**Problem**: "Permission denied: 'ffmpeg'" error

**Solution**:
```bash
# Install ffmpeg
# Ubuntu/Debian/WSL2
sudo apt-get update
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows (via WSL2)
sudo apt-get install ffmpeg
```

Note: The browser automatically converts audio to WAV format, which should work with Whisper in most cases. If errors persist, install ffmpeg as shown above.

### Database Issues

**Problem**: Database errors or missing tables

**Solution**:
```bash
# Initialize database
flask init-db

# Or let the app create it automatically on startup
python app.py
```

### Reminder Notifications Not Triggering

**Problem**: Reminders are set but notifications don't appear

**Solution**:
1. Check reminder scheduler is running (should start automatically)
2. Verify reminder time has passed
3. Check debug endpoints:
   ```bash
   curl http://localhost:5000/api/debug/reminders
   curl -X POST http://localhost:5000/api/debug/scheduler/check
   ```

## Development

### Running in Debug Mode

```bash
FLASK_DEBUG=1 python app.py
```

### Database Commands

```bash
# Initialize database
flask init-db

# Seed sample data
flask seed-data
```

### Logging

The application uses Python's logging module. Log levels can be configured in `app.py`. Default level is INFO.


