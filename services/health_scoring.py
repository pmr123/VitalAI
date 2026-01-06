"""
Health Scoring Service
Calculates Cardio-Metabolic Index (CMI) and other health scores.
"""
from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy import func

from extensions import db
from models import HealthData, HealthScore, MetricType, User
import config


def calculate_cardiac_score(user_id: int, days: int = 7) -> Optional[int]:
    """
    Calculate cardiac health score (0-100) based on:
    - Resting heart rate (lower is better, up to a point)
    - Heart rate variability (higher is better)
    - Heart rate patterns
    """
    from app import app
    
    with app.app_context():
        cutoff = datetime.now() - timedelta(days=days)
        
        # Get resting HR data
        rhr_data = HealthData.query.filter(
            HealthData.user_id == user_id,
            HealthData.metric_type == MetricType.RESTING_HR,
            HealthData.timestamp >= cutoff
        ).all()
        
        # Get HRV data
        hrv_data = HealthData.query.filter(
            HealthData.user_id == user_id,
            HealthData.metric_type == MetricType.HRV,
            HealthData.timestamp >= cutoff
        ).all()
        
        if not rhr_data:
            return None
        
        avg_rhr = sum(d.value for d in rhr_data) / len(rhr_data)
        avg_hrv = sum(d.value for d in hrv_data) / len(hrv_data) if hrv_data else 50
        
        # Resting HR score: 60 BPM = 100, 80 BPM = 50, 100 BPM = 0
        rhr_score = max(0, min(100, 100 - (avg_rhr - 60) * 2.5))
        
        # HRV score: 80ms = 100, 40ms = 50, 20ms = 0
        hrv_score = max(0, min(100, (avg_hrv - 20) * (100 / 60)))
        
        # Combined cardiac score
        cardiac = int(rhr_score * 0.6 + hrv_score * 0.4)
        return cardiac


def calculate_activity_score(user_id: int, days: int = 7) -> Optional[int]:
    """
    Calculate activity score (0-100) based on:
    - Daily steps vs goal
    - Active minutes
    - Consistency
    """
    from app import app
    
    with app.app_context():
        cutoff = datetime.now() - timedelta(days=days)
        
        # Get user's step goal
        user = User.query.get(user_id)
        step_goal = user.daily_step_goal if user else 10000
        
        # Get steps data
        steps_data = HealthData.query.filter(
            HealthData.user_id == user_id,
            HealthData.metric_type == MetricType.STEPS,
            HealthData.timestamp >= cutoff
        ).all()
        
        if not steps_data:
            return None
        
        daily_steps = [d.value for d in steps_data]
        avg_steps = sum(daily_steps) / len(daily_steps)
        
        # Goal achievement score
        goal_pct = min(1.5, avg_steps / step_goal)  # Cap at 150%
        goal_score = goal_pct * 70  # Max 105 from goal
        
        # Consistency bonus (low variance = good)
        if len(daily_steps) > 1:
            variance = sum((s - avg_steps) ** 2 for s in daily_steps) / len(daily_steps)
            std_dev = variance ** 0.5
            consistency = max(0, 1 - (std_dev / avg_steps)) if avg_steps > 0 else 0
            consistency_bonus = consistency * 30
        else:
            consistency_bonus = 15
        
        activity = int(min(100, goal_score + consistency_bonus))
        return activity


def calculate_recovery_score(user_id: int, days: int = 7) -> Optional[int]:
    """
    Calculate recovery score (0-100) based on:
    - Sleep duration
    - Sleep quality/stages
    - Sleep consistency
    """
    from app import app
    
    with app.app_context():
        cutoff = datetime.now() - timedelta(days=days)
        
        # Get sleep data
        sleep_data = HealthData.query.filter(
            HealthData.user_id == user_id,
            HealthData.metric_type == MetricType.SLEEP_DURATION,
            HealthData.timestamp >= cutoff
        ).all()
        
        sleep_scores = HealthData.query.filter(
            HealthData.user_id == user_id,
            HealthData.metric_type == MetricType.SLEEP_SCORE,
            HealthData.timestamp >= cutoff
        ).all()
        
        deep_sleep = HealthData.query.filter(
            HealthData.user_id == user_id,
            HealthData.metric_type == MetricType.SLEEP_DEEP,
            HealthData.timestamp >= cutoff
        ).all()
        
        if not sleep_data:
            return None
        
        avg_duration = sum(d.value for d in sleep_data) / len(sleep_data)
        avg_score = sum(d.value for d in sleep_scores) / len(sleep_scores) if sleep_scores else 70
        avg_deep = sum(d.value for d in deep_sleep) / len(deep_sleep) if deep_sleep else 1.0
        
        # Duration score: 7-8 hours = optimal
        if avg_duration < 5:
            duration_score = avg_duration * 10
        elif avg_duration < 7:
            duration_score = 50 + (avg_duration - 5) * 20
        elif avg_duration <= 9:
            duration_score = 90 + (8 - abs(avg_duration - 7.5)) * 5
        else:
            duration_score = 80 - (avg_duration - 9) * 10
        
        # Deep sleep score: 1.5 hours = optimal
        deep_score = min(100, (avg_deep / 1.5) * 100)
        
        # Combined recovery score
        recovery = int(
            duration_score * 0.4 +
            avg_score * 0.35 +
            deep_score * 0.25
        )
        return max(0, min(100, recovery))


def calculate_metabolic_score(user_id: int) -> Optional[int]:
    """
    Calculate metabolic score (0-100) based on:
    - BMI (if available)
    - Activity level
    - Calorie balance (simplified)
    """
    from app import app
    
    with app.app_context():
        user = User.query.get(user_id)
        if not user:
            return None
        
        # Start with base score
        base_score = 70
        
        # BMI component
        if user.bmi:
            bmi = user.bmi
            if 18.5 <= bmi <= 24.9:
                bmi_score = 100
            elif 25 <= bmi <= 29.9:
                bmi_score = 70 - (bmi - 25) * 4
            elif bmi > 29.9:
                bmi_score = 50 - (bmi - 30) * 2
            else:  # Underweight
                bmi_score = 70 - (18.5 - bmi) * 5
            
            base_score = bmi_score * 0.5 + base_score * 0.5
        
        # Activity contribution (use activity score if calculated)
        activity_score = calculate_activity_score(user_id, days=7)
        if activity_score:
            base_score = base_score * 0.7 + activity_score * 0.3
        
        return max(0, min(100, int(base_score)))


def calculate_cmi(
    cardiac: Optional[int],
    activity: Optional[int],
    recovery: Optional[int],
    metabolic: Optional[int]
) -> Optional[int]:
    """
    Calculate Cardio-Metabolic Index from component scores.
    Uses configurable weights.
    """
    weights = config.HEALTH_SCORE_WEIGHTS
    
    scores = {
        "cardiac": cardiac,
        "activity": activity,
        "recovery": recovery,
        "metabolic": metabolic
    }
    
    # Calculate weighted average, handling None values
    total_weight = 0
    weighted_sum = 0
    
    for key, score in scores.items():
        if score is not None:
            weighted_sum += score * weights[key]
            total_weight += weights[key]
    
    if total_weight == 0:
        return None
    
    return int(weighted_sum / total_weight)


def calculate_health_summary(user_id: int, days: int = 7) -> dict:
    """
    Calculate complete health summary for a user.
    """
    cardiac = calculate_cardiac_score(user_id, days)
    activity = calculate_activity_score(user_id, days)
    recovery = calculate_recovery_score(user_id, days)
    metabolic = calculate_metabolic_score(user_id)
    cmi = calculate_cmi(cardiac, activity, recovery, metabolic)
    
    # Generate simple insights
    insights = []
    
    if cardiac is not None:
        if cardiac >= 80:
            insights.append("Your cardiac health is excellent!")
        elif cardiac < 60:
            insights.append("Consider more cardio exercise to improve heart health.")
    
    if activity is not None:
        if activity >= 80:
            insights.append("Great job staying active!")
        elif activity < 50:
            insights.append("Try to increase your daily steps.")
    
    if recovery is not None:
        if recovery >= 80:
            insights.append("Your sleep quality is great!")
        elif recovery < 60:
            insights.append("Focus on improving sleep duration and quality.")
    
    return {
        "user_id": user_id,
        "period_days": days,
        "scores": {
            "cardiac": cardiac,
            "activity": activity,
            "recovery": recovery,
            "metabolic": metabolic,
            "cmi": cmi
        },
        "insights": insights,
        "calculated_at": datetime.now().isoformat()
    }


def get_score_interpretation(score: Optional[int]) -> str:
    """Get human-readable interpretation of a score."""
    if score is None:
        return "Not enough data"
    elif score >= 85:
        return "Excellent"
    elif score >= 70:
        return "Good"
    elif score >= 50:
        return "Fair"
    else:
        return "Needs Improvement"

