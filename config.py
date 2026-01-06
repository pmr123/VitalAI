"""
VitalAI Configuration
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.absolute()

# Database
DATABASE_PATH = BASE_DIR / "data" / "vitalai.db"
SQLALCHEMY_DATABASE_URI = f"sqlite:///{DATABASE_PATH}"
SQLALCHEMY_TRACK_MODIFICATIONS = False

# Ollama Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
# Use smaller model for faster responses (change to llama3.1:8b-instruct-q4_K_M for better quality)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")  # Fast: phi3:mini or llama3.2:3b
OLLAMA_MODEL_SMALL = "phi3:mini"  # Fallback for faster responses

# Embedding Model (sentence-transformers)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# ChromaDB
CHROMA_PERSIST_DIR = BASE_DIR / "data" / "vectordb"

# Flask
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
DEBUG = os.getenv("FLASK_DEBUG", "1") == "1"

# Health Scoring Weights (can be adjusted)
HEALTH_SCORE_WEIGHTS = {
    "cardiac": 0.30,
    "activity": 0.25,
    "recovery": 0.25,
    "metabolic": 0.20
}

# Supported Languages
SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish", 
    "hi": "Hindi",
    "zh": "Chinese",
    "ar": "Arabic",
    "pt": "Portuguese",
    "fr": "French"
}

# Default User Settings
DEFAULT_LANGUAGE = "en"
DEFAULT_TIMEZONE = "UTC"

