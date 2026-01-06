"""
VitalAI - Health Intelligence Platform
Main Flask Application
"""
from flask import Flask, render_template, jsonify, request, session, redirect, url_for
from extensions import db, cors
import config
import logging
from functools import wraps

# Configure logging - INFO for important events, DEBUG for detailed tracing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Set specific loggers to appropriate levels
logging.getLogger('services.reminder_scheduler').setLevel(logging.INFO)
logging.getLogger('services.agents').setLevel(logging.INFO)

# Create logger for this module
logger = logging.getLogger(__name__)


def create_app():
    """Application factory pattern"""
    app = Flask(__name__)
    app.config.from_object(config)
    
    # Initialize extensions with app
    db.init_app(app)
    cors.init_app(app)
    
    # Ensure data directories exist
    config.DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    
    return app


# Create app instance
app = create_app()

# Import models after app is created to avoid circular imports
from models import User, HealthData, Medication, Reminder, HealthScore

# =============================================================================
# Authentication Helpers
# =============================================================================

def get_current_user_id():
    """Get current user ID from session, or None if not logged in"""
    return session.get('user_id', None)

def get_current_user():
    """Get current user object from session, or None if not logged in"""
    user_id = get_current_user_id()
    if user_id:
        return User.query.get(user_id)
    return None

def login_required(f):
    """Decorator to require login for a route"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user_id = get_current_user_id()
        if user_id is None:
            # Check if user exists, if not redirect to onboarding
            return redirect(url_for('onboarding_page'))
        return f(*args, **kwargs)
    return decorated_function

def requires_onboarding(f):
    """Decorator to check if user needs onboarding"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        user = get_current_user()
        if user and not user.onboarding_completed:
            return redirect(url_for('onboarding_page'))
        return f(*args, **kwargs)
    return decorated_function

# =============================================================================
# Routes
# =============================================================================

@app.route("/")
@login_required
@requires_onboarding
def index():
    """Main dashboard page"""
    return render_template("dashboard.html")


@app.route("/chat")
@login_required
@requires_onboarding
def chat_page():
    """Chat interface page"""
    return render_template("chat.html")


@app.route("/onboarding")
def onboarding_page():
    """Onboarding page - shown to new users or users who haven't completed onboarding"""
    user = get_current_user()
    # If user is logged in and has completed onboarding, redirect to dashboard
    if user and user.onboarding_completed:
        return redirect(url_for('index'))
    return render_template("onboarding.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    """Simple login - just enter email to find/create user"""
    if request.method == "POST":
        data = request.get_json() if request.is_json else request.form
        email = data.get('email', '').strip().lower()
        name = data.get('name', '').strip()
        
        if not email:
            return jsonify({"error": "Email is required"}), 400
        
        # Find existing user by email, or create new one
        user = User.query.filter_by(email=email).first()
        
        if not user:
            # Create new user
            if not name:
                name = email.split('@')[0]  # Use email prefix as default name
            
            user = User(
                name=name,
                email=email,
                onboarding_completed=False
            )
            db.session.add(user)
            db.session.commit()
            logger.info(f"Created new user: {user.id} - {user.email}")
        
        # Set session
        session['user_id'] = user.id
        session.permanent = True
        
        if request.is_json:
            return jsonify({
                "success": True,
                "user_id": user.id,
                "name": user.name,
                "onboarding_completed": user.onboarding_completed
            })
        else:
            if user.onboarding_completed:
                return redirect(url_for('index'))
            else:
                return redirect(url_for('onboarding_page'))
    
    # GET request - show login page
    return render_template("login.html")


@app.route("/logout", methods=["POST"])
def logout():
    """Logout user"""
    session.pop('user_id', None)
    if request.is_json:
        return jsonify({"success": True})
    return redirect(url_for('login'))


@app.route("/api/auth/status")
def auth_status():
    """Get current authentication status"""
    user = get_current_user()
    if user:
        return jsonify({
            "authenticated": True,
            "user_id": user.id,
            "name": user.name,
            "email": user.email,
            "onboarding_completed": user.onboarding_completed
        })
    return jsonify({
        "authenticated": False,
        "user_id": None
    })


@app.route("/health")
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "ollama_host": config.OLLAMA_HOST,
        "model": config.OLLAMA_MODEL
    })


# =============================================================================
# API Routes
# =============================================================================

@app.route("/api/user/<int:user_id>/health-data")
def get_user_health_data(user_id):
    """Get user's health data"""
    # Get latest health data for user
    health_data = HealthData.query.filter_by(user_id=user_id)\
        .order_by(HealthData.timestamp.desc())\
        .limit(100)\
        .all()
    
    return jsonify({
        "user_id": user_id,
        "data": [h.to_dict() for h in health_data]
    })


@app.route("/api/user/<int:user_id>/health-summary")
def get_health_summary(user_id):
    """Get user's health summary with scores"""
    from services.health_scoring import calculate_health_summary
    
    summary = calculate_health_summary(user_id)
    return jsonify(summary)


@app.route("/api/simulate-data", methods=["POST"])
@login_required
def simulate_data():
    """Generate simulated wearable data for a user"""
    from services.data_simulator import WearableSimulator, ensure_demo_user
    
    data = request.get_json() or {}
    user_id = get_current_user_id()  # Use session user_id
    days = data.get("days", 7)
    
    # Ensure demo user exists (needed for metabolic score calculation)
    user = ensure_demo_user(user_id)
    
    simulator = WearableSimulator()
    generated = simulator.generate_and_store(user_id, days=days)
    
    return jsonify({
        "status": "success",
        "records_generated": generated,
        "user_id": user_id,
        "user_name": user.name,
        "days": days
    })


@app.route("/api/user/<int:user_id>")
def get_user(user_id):
    """Get user profile"""
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify(user.to_dict())


@app.route("/api/user/<int:user_id>/health-data/history")
@login_required
def get_health_history(user_id):
    """Get detailed health data history with filtering"""
    # Verify user can only access their own data
    current_user_id = get_current_user_id()
    if user_id != current_user_id:
        return jsonify({"error": "Unauthorized"}), 403
    metric_type = request.args.get("metric")
    days = int(request.args.get("days", 7))
    
    from datetime import datetime, timedelta
    cutoff = datetime.now() - timedelta(days=days)
    
    query = HealthData.query.filter(
        HealthData.user_id == user_id,
        HealthData.timestamp >= cutoff
    )
    
    if metric_type:
        query = query.filter(HealthData.metric_type == metric_type)
    
    health_data = query.order_by(HealthData.timestamp.desc()).all()
    
    return jsonify({
        "user_id": user_id,
        "days": days,
        "metric_filter": metric_type,
        "count": len(health_data),
        "data": [h.to_dict() for h in health_data]
    })


# =============================================================================
# Chat API Routes
# =============================================================================

@app.route("/api/chat", methods=["POST"])
@login_required
def chat():
    """Send a message to the AI assistant"""
    from services.chat_service import get_chat_service
    
    data = request.get_json() or {}
    message = data.get("message", "").strip()
    user_id = get_current_user_id()  # Use session user_id
    chat_history = data.get("history", [])
    
    if not message:
        return jsonify({"error": "Message is required"}), 400
    
    chat_service = get_chat_service()
    result = chat_service.chat(
        user_id=user_id,
        message=message,
        chat_history=chat_history,
        use_rag=True
    )
    
    return jsonify(result)


@app.route("/api/chat/search", methods=["POST"])
def search_knowledge_base():
    """Search the knowledge base without generating a response"""
    from services.rag_service import get_rag_service
    
    data = request.get_json() or {}
    query = data.get("query", "").strip()
    top_k = data.get("top_k", 5)
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    rag = get_rag_service()
    results = rag.search(query, top_k=top_k)
    
    return jsonify({
        "query": query,
        "results": results
    })


@app.route("/api/rag/status")
def rag_status():
    """Get RAG service status"""
    from services.rag_service import get_rag_service
    from services.llm_service import get_llm_service
    
    rag = get_rag_service()
    llm = get_llm_service()
    
    return jsonify({
        "rag": rag.get_stats(),
        "llm": llm.check_connection()
    })


@app.route("/api/rag/reload", methods=["POST"])
def reload_knowledge_base():
    """Reload the knowledge base"""
    from services.rag_service import init_knowledge_base
    
    count = init_knowledge_base(force_reload=True)
    
    return jsonify({
        "status": "success",
        "chunks_loaded": count
    })


# =============================================================================
# MCP (Model Context Protocol) API Routes
# =============================================================================

@app.route("/api/mcp/info")
def mcp_info():
    """Get MCP server information"""
    from services.mcp_server import get_mcp_server
    
    mcp = get_mcp_server()
    return jsonify(mcp.get_server_info())


@app.route("/api/mcp/tools")
def mcp_list_tools():
    """List all available MCP tools"""
    from services.mcp_server import get_mcp_server
    
    mcp = get_mcp_server()
    tools = mcp.list_tools()
    
    return jsonify({
        "tool_count": len(tools),
        "tools": tools
    })


@app.route("/api/mcp/tools/<tool_name>")
def mcp_get_tool(tool_name):
    """Get schema for a specific tool"""
    from services.mcp_server import get_mcp_server
    
    mcp = get_mcp_server()
    schema = mcp.get_tool_schema(tool_name)
    
    if not schema:
        return jsonify({"error": f"Tool not found: {tool_name}"}), 404
    
    return jsonify(schema)


@app.route("/api/mcp/execute", methods=["POST"])
def mcp_execute_tool():
    """Execute an MCP tool"""
    from services.mcp_server import get_mcp_server
    
    data = request.get_json() or {}
    tool_name = data.get("tool_name") or data.get("name")
    arguments = data.get("arguments") or data.get("parameters", {})
    
    if not tool_name:
        return jsonify({"error": "tool_name is required"}), 400
    
    mcp = get_mcp_server()
    result = mcp.execute(tool_name, arguments)
    
    return jsonify(result.to_dict())


@app.route("/api/mcp/batch", methods=["POST"])
def mcp_batch_execute():
    """Execute multiple MCP tools in batch"""
    from services.mcp_server import get_mcp_server
    
    data = request.get_json() or []
    
    if not isinstance(data, list):
        return jsonify({"error": "Request body must be an array of tool calls"}), 400
    
    mcp = get_mcp_server()
    results = mcp.handle_request(data)
    
    return jsonify({"results": results})


# =============================================================================
# Reminder Notification API Routes
# =============================================================================

@app.route("/api/notifications")
@login_required
def get_notifications():
    """Get pending reminder notifications for a user"""
    from services.reminder_scheduler import get_pending_notifications
    
    user_id = get_current_user_id()  # Use session user_id
    notifications = get_pending_notifications(user_id)
    
    return jsonify({
        "user_id": user_id,
        "count": len(notifications),
        "notifications": notifications
    })


@app.route("/api/notifications/<int:notification_id>/acknowledge", methods=["POST"])
def acknowledge_notification(notification_id):
    """Acknowledge a notification"""
    from services.reminder_scheduler import acknowledge_notification
    
    success = acknowledge_notification(notification_id)
    
    return jsonify({
        "success": success,
        "notification_id": notification_id
    })


# =============================================================================
# Agent API Routes (Phase 4)
# =============================================================================

@app.route("/api/agents")
def list_agents():
    """List all available agents in the multi-agent system"""
    from services.agents import get_orchestrator
    
    orchestrator = get_orchestrator()
    agents = orchestrator.get_agent_info()
    
    return jsonify({
        "agent_count": len(agents),
        "agents": agents
    })


@app.route("/api/agents/chat", methods=["POST"])
@login_required
def agent_chat():
    """Send a message through the multi-agent system"""
    from services.chat_service import get_chat_service
    
    data = request.get_json() or {}
    message = data.get("message", "").strip()
    user_id = get_current_user_id()  # Use session user_id
    chat_history = data.get("history", [])
    force_agent = data.get("agent")  # Optional: force a specific agent
    
    if not message:
        return jsonify({"error": "Message is required"}), 400
    
    chat_service = get_chat_service()
    
    # If specific agent requested, use it directly
    if force_agent:
        result = chat_service.chat_with_agent(
            user_id=user_id,
            message=message,
            agent_type=force_agent,
            chat_history=chat_history
        )
    else:
        # Let the supervisor route to the appropriate agent
        result = chat_service.chat(
            user_id=user_id,
            message=message,
            chat_history=chat_history,
            use_agents=True
        )
    
    return jsonify(result)


@app.route("/api/agents/classify", methods=["POST"])
def classify_intent():
    """Classify a message's intent without processing it"""
    from services.agents import get_orchestrator
    
    data = request.get_json() or {}
    message = data.get("message", "").strip()
    
    if not message:
        return jsonify({"error": "Message is required"}), 400
    
    orchestrator = get_orchestrator()
    classification = orchestrator.supervisor.classify_intent(message)
    
    return jsonify({
        "message": message,
        "classification": {
            "agent": classification["primary_agent"].value,
            "confidence": classification["confidence"],
            "method": classification.get("method", "unknown"),
            "reasoning": classification.get("reasoning", "")
        }
    })


@app.route("/api/agents/history")
@login_required
def agent_history():
    """Get the current conversation history from the orchestrator"""
    from services.agents import get_orchestrator
    
    user_id = get_current_user_id()  # Use session user_id
    orchestrator = get_orchestrator(user_id)
    
    return jsonify({
        "user_id": user_id,
        "history_length": len(orchestrator._conversation_history),
        "history": orchestrator._conversation_history
    })


@app.route("/api/agents/history/clear", methods=["POST"])
@login_required
def clear_agent_history():
    """Clear the orchestrator's conversation history"""
    from services.agents import get_orchestrator
    
    user_id = get_current_user_id()  # Use session user_id
    orchestrator = get_orchestrator(user_id)
    orchestrator.clear_history()
    
    return jsonify({
        "success": True,
        "user_id": user_id
    })


@app.route("/api/debug/reminders")
@login_required
def debug_reminders():
    """Debug endpoint to check all reminders in database"""
    from models import Reminder
    
    user_id = get_current_user_id()  # Use session user_id
    
    all_reminders = Reminder.query.filter_by(user_id=user_id).all()
    active_reminders = Reminder.query.filter_by(user_id=user_id, active=True).all()
    
    return jsonify({
        "user_id": user_id,
        "total_reminders": len(all_reminders),
        "active_reminders": len(active_reminders),
        "all_reminders": [
            {
                "id": r.id,
                "title": r.title,
                "time": r.scheduled_time.strftime("%H:%M") if r.scheduled_time else None,
                "days": r.days_of_week,
                "active": r.active,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "last_sent": r.last_sent.isoformat() if r.last_sent else None
            }
            for r in all_reminders
        ]
    })


@app.route("/api/debug/scheduler/check", methods=["POST"])
def debug_scheduler_check():
    """Manually trigger a reminder check (for testing)"""
    from services.reminder_scheduler import check_due_reminders
    
    try:
        check_due_reminders()
        return jsonify({
            "success": True,
            "message": "Reminder check executed. Check server logs for details."
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/debug/notifications")
@login_required
def debug_notifications():
    """Debug endpoint to check pending notifications"""
    from services.reminder_scheduler import get_pending_notifications, pending_notifications
    
    user_id = get_current_user_id()  # Use session user_id
    user_notifications = get_pending_notifications(user_id)
    all_notifications = list(pending_notifications)
    
    return jsonify({
        "user_id": user_id,
        "user_notifications_count": len(user_notifications),
        "all_notifications_count": len(all_notifications),
        "user_notifications": user_notifications,
        "all_notifications": all_notifications
    })


# =============================================================================
# Voice & Translation API Endpoints
# =============================================================================

@app.route("/api/voice/transcribe", methods=["POST"])
def transcribe_audio():
    """Transcribe audio to text using Whisper"""
    from services.voice_service import get_voice_service
    
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    language = request.form.get('language', None)  # Optional: 'en', 'es', etc.
    
    if audio_file.filename == '':
        return jsonify({"error": "Empty audio file"}), 400
    
    try:
        voice_service = get_voice_service()
        
        if not voice_service.is_stt_available():
            return jsonify({
                "error": "Speech-to-text not available",
                "message": "Whisper model not loaded. Check server logs."
            }), 503
        
        # Read audio data
        audio_data = audio_file.read()
        
        # Transcribe
        text, detected_lang = voice_service.transcribe_audio(audio_data, language=language)
        
        return jsonify({
            "success": True,
            "text": text,
            "detected_language": detected_lang
        })
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/voice/synthesize", methods=["POST"])
def synthesize_speech():
    """Convert text to speech audio"""
    from services.voice_service import get_voice_service
    
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Text is required"}), 400
    
    text = data['text']
    output_format = data.get('format', 'wav')
    
    if not text.strip():
        return jsonify({"error": "Text cannot be empty"}), 400
    
    try:
        voice_service = get_voice_service()
        
        if not voice_service.is_tts_available():
            return jsonify({
                "error": "Text-to-speech not available",
                "message": "TTS engine not initialized. Check server logs."
            }), 503
        
        # Generate speech
        audio_data = voice_service.text_to_speech(text, output_format=output_format)
        
        # Return audio as response
        from flask import Response
        mimetype = 'audio/wav' if output_format == 'wav' else 'audio/mpeg'
        
        return Response(
            audio_data,
            mimetype=mimetype,
            headers={
                'Content-Disposition': f'attachment; filename=speech.{output_format}'
            }
        )
        
    except Exception as e:
        logger.error(f"TTS synthesis error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/voice/status")
def voice_status():
    """Get status of voice services"""
    from services.voice_service import get_voice_service
    
    voice_service = get_voice_service()
    
    return jsonify({
        "stt_available": voice_service.is_stt_available(),
        "tts_available": voice_service.is_tts_available()
    })


@app.route("/api/translate", methods=["POST"])
def translate_text():
    """Translate text from source language to target language"""
    from services.translation_service import get_translation_service
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body is required"}), 400
    
    text = data.get('text', '')
    source_lang = data.get('source_lang', 'en')
    target_lang = data.get('target_lang', 'en')
    
    if not text.strip():
        return jsonify({"error": "Text is required"}), 400
    
    try:
        translation_service = get_translation_service()
        
        if not translation_service.is_available():
            return jsonify({
                "error": "Translation service not available",
                "message": "Translation model not loaded. Check server logs."
            }), 503
        
        # Translate
        translated_text = translation_service.translate(
            text,
            source_lang=source_lang,
            target_lang=target_lang
        )
        
        return jsonify({
            "success": True,
            "original_text": text,
            "translated_text": translated_text,
            "source_lang": source_lang,
            "target_lang": target_lang
        })
        
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/translate/languages")
def get_supported_languages():
    """Get list of supported languages for translation"""
    from services.translation_service import get_translation_service
    
    translation_service = get_translation_service()
    languages = translation_service.get_supported_languages()
    
    return jsonify({
        "languages": languages,
        "default": "en"
    })


# =============================================================================
# Onboarding API Endpoints
# =============================================================================

@app.route("/api/onboarding/start", methods=["POST"])
@login_required
def start_onboarding():
    """Start or resume onboarding flow"""
    from services.onboarding_service import get_onboarding_service
    
    user_id = get_current_user_id()
    service = get_onboarding_service(user_id)
    
    # Get current state
    state = service.get_state()
    
    # If already complete, return that
    if state['completed']:
        return jsonify({
            "message": "Onboarding already completed",
            "completed": True
        })
    
    # Get user name for personalized welcome
    user = get_current_user()
    user_name = user.name if user else "there"
    
    # Start with welcome message
    welcome_message = f"Hello {user_name}! Welcome to VitalAI. I'm here to help you get started with your health journey. Let's begin by learning a bit about you. How old are you?"
    
    return jsonify({
        "message": welcome_message,
        "step": state['step'],
        "onboarding_state": state['onboarding_state']
    })


@app.route("/api/onboarding/chat", methods=["POST"])
@login_required
def onboarding_chat():
    """Process a message during onboarding"""
    from services.onboarding_service import get_onboarding_service
    
    user_id = get_current_user_id()
    data = request.get_json()
    
    if not data or 'message' not in data:
        return jsonify({"error": "Message is required"}), 400
    
    message = data['message']
    service = get_onboarding_service(user_id)
    
    # Process message
    result = service.process_message(message)
    
    return jsonify({
        "response": result['response'],
        "step": result['step'],
        "onboarding_state": result['onboarding_state'],
        "completed": result['completed']
    })


@app.route("/api/onboarding/complete", methods=["POST"])
@login_required
def complete_onboarding():
    """Complete onboarding and create user profile"""
    from services.onboarding_service import get_onboarding_service
    
    user_id = get_current_user_id()
    data = request.get_json() or {}
    
    service = get_onboarding_service(user_id)
    
    # Complete onboarding with any final data
    final_data = data.get('final_data', {})
    user = service.complete_onboarding(final_data)
    
    return jsonify({
        "success": True,
        "user": user.to_dict(),
        "message": "Onboarding completed successfully!"
    })


@app.route("/api/onboarding/status")
@login_required
def onboarding_status():
    """Get current onboarding status"""
    from services.onboarding_service import get_onboarding_service
    
    user_id = get_current_user_id()
    service = get_onboarding_service(user_id)
    
    state = service.get_state()
    
    return jsonify({
        "step": state['step'],
        "onboarding_state": state['onboarding_state'],
        "completed": state['completed']
    })


# =============================================================================
# CLI Commands
# =============================================================================

@app.cli.command("init-db")
def init_db():
    """Initialize the database."""
    db.create_all()
    print("Database initialized!")


@app.cli.command("seed-data")
def seed_data():
    """Seed database with sample data."""
    from services.data_simulator import seed_sample_user
    seed_sample_user()
    print("Sample data seeded!")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Create tables if they don't exist
    with app.app_context():
        db.create_all()
        
        # Initialize RAG knowledge base
        print("Initializing RAG knowledge base...")
        from services.rag_service import init_knowledge_base
        chunk_count = init_knowledge_base()
        print(f"Knowledge base ready with {chunk_count} chunks.")
    
    # Initialize MCP server
    print("Initializing MCP server...")
    from services.mcp_server import get_mcp_server
    mcp = get_mcp_server()
    print(f"MCP server ready with {len(mcp.list_tools())} tools.")
    
    # Initialize reminder scheduler
    print("Initializing reminder scheduler...")
    from services.reminder_scheduler import init_scheduler
    scheduler = init_scheduler(app)
    print("Reminder scheduler active - checking every minute.")
    
    # Initialize multi-agent system
    print("Initializing multi-agent system...")
    from services.agents import get_orchestrator
    orchestrator = get_orchestrator()
    print(f"Agent system ready with {len(orchestrator.get_agent_info())} agents.")
    
    # Initialize voice services (lazy loading - models load on first use)
    print("Initializing voice services...")
    from services.voice_service import get_voice_service
    from services.translation_service import get_translation_service
    voice_service = get_voice_service()
    translation_service = get_translation_service()
    print("Voice services initialized (models load on first use).")
    
    print(f"""
    ================================================================
                    VitalAI Health Platform
              Multi-Agent System + Voice AI
    ================================================================
      Dashboard:  http://localhost:5000
      Chat:       http://localhost:5000/chat
      Health:     http://localhost:5000/health
    ----------------------------------------------------------------
      Agents:
      - Supervisor (routes queries)
      - Health Analyst (data & scores)
      - Medication Manager (meds & reminders)
      - Knowledge Expert (RAG-powered)
      - Digital Clone (personalized)
    ----------------------------------------------------------------
      Voice & Translation:
      - Speech-to-Text: Whisper (STT)
      - Text-to-Speech: pyttsx3 (TTS)
      - Translation: NLLB-200 (7 languages)
    ----------------------------------------------------------------
      API:
      - /api/agents        (list agents)
      - /api/agents/chat   (agent chat)
      - /api/voice/transcribe  (STT)
      - /api/voice/synthesize  (TTS)
      - /api/translate     (translation)
      - /api/rag/status    (RAG status)
      - /api/mcp/tools     (MCP tools)
    ================================================================
    """)
    
    app.run(host="0.0.0.0", port=5000, debug=config.DEBUG)

