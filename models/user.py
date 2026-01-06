"""
User Model
"""
from datetime import datetime
from extensions import db
import config


class User(db.Model):
    """User profile and preferences"""
    __tablename__ = "users"
    
    id = db.Column(db.Integer, primary_key=True)
    
    # Basic Info
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    
    # Demographics
    age = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.String(20), nullable=True)  # male, female, other
    height_cm = db.Column(db.Float, nullable=True)
    weight_kg = db.Column(db.Float, nullable=True)
    
    # Health Profile
    conditions = db.Column(db.JSON, default=list)  # ["diabetes", "hypertension"]
    allergies = db.Column(db.JSON, default=list)   # ["penicillin"]
    blood_type = db.Column(db.String(10), nullable=True)
    
    # Preferences
    preferred_language = db.Column(db.String(10), default=config.DEFAULT_LANGUAGE)
    timezone = db.Column(db.String(50), default=config.DEFAULT_TIMEZONE)
    
    # Goals
    health_goals = db.Column(db.JSON, default=list)  # ["lose weight", "sleep better"]
    daily_step_goal = db.Column(db.Integer, default=10000)
    
    # Onboarding
    onboarding_completed = db.Column(db.Boolean, default=False)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    health_data = db.relationship("HealthData", backref="user", lazy="dynamic")
    medications = db.relationship("Medication", backref="user", lazy="dynamic")
    reminders = db.relationship("Reminder", backref="user", lazy="dynamic")
    
    def __repr__(self):
        return f"<User {self.id}: {self.name}>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "age": self.age,
            "gender": self.gender,
            "height_cm": self.height_cm,
            "weight_kg": self.weight_kg,
            "conditions": self.conditions or [],
            "allergies": self.allergies or [],
            "preferred_language": self.preferred_language,
            "health_goals": self.health_goals or [],
            "daily_step_goal": self.daily_step_goal,
            "onboarding_completed": self.onboarding_completed,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
    
    @property
    def bmi(self):
        """Calculate BMI if height and weight are available"""
        if self.height_cm and self.weight_kg:
            height_m = self.height_cm / 100
            return round(self.weight_kg / (height_m ** 2), 1)
        return None

