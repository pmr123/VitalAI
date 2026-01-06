"""
Medication and Reminder Models
"""
from datetime import datetime, time
from extensions import db


class Medication(db.Model):
    """User's medications"""
    __tablename__ = "medications"
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    
    # Medication details
    name = db.Column(db.String(200), nullable=False)
    dosage = db.Column(db.String(100), nullable=True)  # "500mg"
    frequency = db.Column(db.String(100), nullable=True)  # "twice daily"
    instructions = db.Column(db.Text, nullable=True)  # "take with food"
    
    # Schedule
    times = db.Column(db.JSON, default=list)  # ["08:00", "20:00"]
    days_of_week = db.Column(db.JSON, default=list)  # [0,1,2,3,4,5,6] = everyday
    
    # Duration
    start_date = db.Column(db.Date, nullable=True)
    end_date = db.Column(db.Date, nullable=True)  # null = ongoing
    
    # Status
    active = db.Column(db.Boolean, default=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    reminders = db.relationship("Reminder", backref="medication", lazy="dynamic")
    
    def __repr__(self):
        return f"<Medication {self.id}: {self.name}>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "dosage": self.dosage,
            "frequency": self.frequency,
            "instructions": self.instructions,
            "times": self.times or [],
            "days_of_week": self.days_of_week or [],
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "active": self.active
        }


class Reminder(db.Model):
    """Scheduled reminders (medication, check-ins, etc.)"""
    __tablename__ = "reminders"
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    medication_id = db.Column(db.Integer, db.ForeignKey("medications.id"), nullable=True)
    
    # Reminder type
    reminder_type = db.Column(db.String(50), nullable=False)  # medication, checkin, custom
    
    # Content
    title = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text, nullable=True)
    
    # Schedule
    scheduled_time = db.Column(db.Time, nullable=False)
    days_of_week = db.Column(db.JSON, default=list)  # [0,1,2,3,4,5,6]
    
    # Delivery
    channel = db.Column(db.String(50), default="push")  # push, sms, voice
    
    # Status
    active = db.Column(db.Boolean, default=True)
    
    # Tracking
    last_sent = db.Column(db.DateTime, nullable=True)
    last_acknowledged = db.Column(db.DateTime, nullable=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Reminder {self.id}: {self.title}>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "medication_id": self.medication_id,
            "reminder_type": self.reminder_type,
            "title": self.title,
            "message": self.message,
            "scheduled_time": self.scheduled_time.isoformat() if self.scheduled_time else None,
            "days_of_week": self.days_of_week or [],
            "channel": self.channel,
            "active": self.active
        }


class ReminderType:
    """Standard reminder types"""
    MEDICATION = "medication"
    HEALTH_CHECKIN = "checkin"
    HYDRATION = "hydration"
    EXERCISE = "exercise"
    CUSTOM = "custom"

