"""
Flask Extensions - Initialized separately to avoid circular imports
"""
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

# Initialize extensions without app
db = SQLAlchemy()
cors = CORS()

