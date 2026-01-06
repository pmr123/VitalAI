"""
Health Data Model - Stores wearable/sensor data
"""
from datetime import datetime
from extensions import db


class HealthData(db.Model):
    """Time-series health data from wearables"""
    __tablename__ = "health_data"
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, index=True)
    
    # Metric type and value
    metric_type = db.Column(db.String(50), nullable=False, index=True)
    value = db.Column(db.Float, nullable=False)
    unit = db.Column(db.String(20), nullable=True)
    
    # Optional metadata
    source = db.Column(db.String(50), default="simulator")  # simulator, fitbit, apple_watch
    quality = db.Column(db.Float, default=1.0)  # Data quality score 0-1
    
    # Index for faster queries
    __table_args__ = (
        db.Index("idx_user_metric_time", "user_id", "metric_type", "timestamp"),
    )
    
    def __repr__(self):
        return f"<HealthData {self.metric_type}: {self.value} at {self.timestamp}>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "metric_type": self.metric_type,
            "value": self.value,
            "unit": self.unit,
            "source": self.source
        }


# Metric type constants for consistency
class MetricType:
    """Standard metric types"""
    HEART_RATE = "heart_rate"           # BPM
    RESTING_HR = "resting_hr"           # BPM
    HRV = "hrv"                         # ms (heart rate variability)
    STEPS = "steps"                     # count
    CALORIES = "calories"               # kcal
    ACTIVE_MINUTES = "active_minutes"   # minutes
    SLEEP_DURATION = "sleep_duration"   # hours
    SLEEP_DEEP = "sleep_deep"           # hours
    SLEEP_LIGHT = "sleep_light"         # hours
    SLEEP_REM = "sleep_rem"             # hours
    SLEEP_SCORE = "sleep_score"         # 0-100
    SPO2 = "spo2"                       # percentage
    STRESS_LEVEL = "stress_level"       # 0-100
    BODY_TEMP = "body_temp"             # celsius
    BLOOD_PRESSURE_SYS = "bp_systolic"  # mmHg
    BLOOD_PRESSURE_DIA = "bp_diastolic" # mmHg
    WEIGHT = "weight"                   # kg


class HealthScore(db.Model):
    """Daily aggregated health scores"""
    __tablename__ = "health_scores"
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    date = db.Column(db.Date, nullable=False, index=True)
    
    # Component scores (0-100)
    cardiac_score = db.Column(db.Integer, nullable=True)
    activity_score = db.Column(db.Integer, nullable=True)
    recovery_score = db.Column(db.Integer, nullable=True)
    metabolic_score = db.Column(db.Integer, nullable=True)
    
    # Overall score
    cmi_score = db.Column(db.Integer, nullable=True)  # Cardio-metabolic index
    
    # AI-generated insights
    insights = db.Column(db.JSON, default=list)
    
    # Metadata
    calculated_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        db.UniqueConstraint("user_id", "date", name="unique_user_date_score"),
    )
    
    def __repr__(self):
        return f"<HealthScore user={self.user_id} date={self.date} cmi={self.cmi_score}>"
    
    def to_dict(self):
        return {
            "user_id": self.user_id,
            "date": self.date.isoformat(),
            "cardiac_score": self.cardiac_score,
            "activity_score": self.activity_score,
            "recovery_score": self.recovery_score,
            "metabolic_score": self.metabolic_score,
            "cmi_score": self.cmi_score,
            "insights": self.insights or []
        }

