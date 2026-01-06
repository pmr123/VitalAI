"""
Wearable Data Simulator
Generates realistic synthetic health data when no real device is available.
"""
import random
import math
from datetime import datetime, timedelta
from typing import Optional

from extensions import db
from models import User, HealthData, HealthScore, MetricType


class WearableSimulator:
    """
    Generates realistic synthetic wearable data.
    
    Simulates:
    - Heart rate (with circadian rhythm and activity variation)
    - Steps (with daily patterns)
    - Sleep (stages and quality)
    - HRV (stress indicator)
    - Other vitals
    """
    
    def __init__(self, seed: Optional[int] = None):
        if seed:
            random.seed(seed)
    
    def generate_heart_rate(
        self, 
        hour: int, 
        base_hr: int = 70,
        is_sleeping: bool = False,
        activity_level: float = 0.0  # 0-1
    ) -> float:
        """
        Generate realistic heart rate based on time and activity.
        
        - Lower during sleep (especially deep sleep)
        - Higher during day, peaks with activity
        - Natural circadian variation
        """
        # Circadian rhythm: lower at night, higher during day
        circadian = math.sin((hour - 6) * math.pi / 12) * 5
        
        if is_sleeping:
            # Sleep: 50-65 BPM typically
            base = base_hr - 15
            variation = random.gauss(0, 3)
        else:
            # Awake: activity affects HR
            activity_boost = activity_level * 50  # Up to +50 BPM for high activity
            variation = random.gauss(0, 5)
            base = base_hr + circadian + activity_boost
        
        return max(45, min(180, base + variation))
    
    def generate_steps_for_hour(self, hour: int, is_weekend: bool = False) -> int:
        """
        Generate realistic hourly step count.
        
        - Sleep hours (0-6): ~0 steps
        - Morning rush (7-9): moderate
        - Work hours (9-17): varies
        - Evening (17-21): moderate activity
        - Night (21-24): low
        """
        if hour < 6:
            return random.randint(0, 10)  # Sleep
        elif hour < 9:
            return random.randint(200, 800)  # Morning routine
        elif hour < 12:
            return random.randint(100, 500) if not is_weekend else random.randint(300, 1000)
        elif hour < 14:
            return random.randint(300, 800)  # Lunch walk
        elif hour < 17:
            return random.randint(100, 400)  # Afternoon
        elif hour < 20:
            return random.randint(400, 1500)  # Evening activity
        else:
            return random.randint(50, 200)  # Winding down
    
    def generate_sleep_data(self, sleep_hours: float = 7.5) -> dict:
        """
        Generate sleep stage breakdown.
        
        Typical distribution:
        - Deep sleep: 15-20%
        - REM: 20-25%
        - Light sleep: 55-60%
        """
        deep_pct = random.uniform(0.13, 0.22)
        rem_pct = random.uniform(0.18, 0.27)
        light_pct = 1 - deep_pct - rem_pct
        
        deep = round(sleep_hours * deep_pct, 1)
        rem = round(sleep_hours * rem_pct, 1)
        light = round(sleep_hours * light_pct, 1)
        
        # Sleep score based on total and deep sleep
        score = min(100, int(
            (sleep_hours / 8) * 50 +  # Duration component
            (deep / 1.5) * 30 +        # Deep sleep component
            random.randint(-10, 10)    # Variation
        ))
        
        return {
            "duration": sleep_hours,
            "deep": deep,
            "rem": rem,
            "light": light,
            "score": max(30, min(100, score))
        }
    
    def generate_hrv(self, stress_level: float = 0.5, age: int = 35) -> float:
        """
        Generate Heart Rate Variability (ms).
        
        - Higher HRV = better recovery, lower stress
        - Decreases with age
        - Decreases with stress
        """
        # Base HRV decreases with age
        age_factor = max(0.5, 1 - (age - 20) * 0.01)
        base_hrv = 60 * age_factor
        
        # Stress reduces HRV
        stress_factor = 1 - (stress_level * 0.4)
        
        hrv = base_hrv * stress_factor + random.gauss(0, 8)
        return max(20, min(120, hrv))
    
    def generate_day_data(
        self, 
        user_id: int, 
        date: datetime,
        user_profile: Optional[dict] = None
    ) -> list[HealthData]:
        """
        Generate a full day's worth of health data.
        """
        profile = user_profile or {}
        base_hr = profile.get("base_hr", 70)
        age = profile.get("age", 35)
        is_weekend = date.weekday() >= 5
        
        records = []
        daily_steps = 0
        
        # Generate hourly data
        for hour in range(24):
            timestamp = date.replace(hour=hour, minute=30)
            
            is_sleeping = hour < 6 or hour >= 23
            
            # Determine activity level based on hour
            if is_sleeping:
                activity = 0.0
            elif 17 <= hour <= 19:
                activity = random.uniform(0.3, 0.7)  # Evening exercise window
            else:
                activity = random.uniform(0.0, 0.3)
            
            # Heart rate
            hr = self.generate_heart_rate(hour, base_hr, is_sleeping, activity)
            records.append(HealthData(
                user_id=user_id,
                timestamp=timestamp,
                metric_type=MetricType.HEART_RATE,
                value=round(hr, 1),
                unit="bpm",
                source="simulator"
            ))
            
            # Steps (accumulated)
            hour_steps = self.generate_steps_for_hour(hour, is_weekend)
            daily_steps += hour_steps
            
            if hour == 23:  # Record daily total at end of day
                records.append(HealthData(
                    user_id=user_id,
                    timestamp=timestamp,
                    metric_type=MetricType.STEPS,
                    value=daily_steps,
                    unit="count",
                    source="simulator"
                ))
            
            # HRV (a few times per day)
            if hour in [7, 14, 22]:
                stress = random.uniform(0.2, 0.8)
                hrv = self.generate_hrv(stress, age)
                records.append(HealthData(
                    user_id=user_id,
                    timestamp=timestamp,
                    metric_type=MetricType.HRV,
                    value=round(hrv, 1),
                    unit="ms",
                    source="simulator"
                ))
        
        # Sleep data (recorded in morning)
        sleep_hours = random.gauss(7.0, 1.0)
        sleep_hours = max(4, min(10, sleep_hours))
        sleep_data = self.generate_sleep_data(sleep_hours)
        
        morning = date.replace(hour=7, minute=0)
        
        records.append(HealthData(
            user_id=user_id,
            timestamp=morning,
            metric_type=MetricType.SLEEP_DURATION,
            value=sleep_data["duration"],
            unit="hours",
            source="simulator"
        ))
        records.append(HealthData(
            user_id=user_id,
            timestamp=morning,
            metric_type=MetricType.SLEEP_DEEP,
            value=sleep_data["deep"],
            unit="hours",
            source="simulator"
        ))
        records.append(HealthData(
            user_id=user_id,
            timestamp=morning,
            metric_type=MetricType.SLEEP_SCORE,
            value=sleep_data["score"],
            unit="score",
            source="simulator"
        ))
        
        # Resting heart rate (morning measurement)
        resting_hr = base_hr - 5 + random.gauss(0, 3)
        records.append(HealthData(
            user_id=user_id,
            timestamp=morning,
            metric_type=MetricType.RESTING_HR,
            value=round(resting_hr, 1),
            unit="bpm",
            source="simulator"
        ))
        
        # Calories (end of day)
        base_calories = 1800 + (daily_steps * 0.04)  # ~0.04 cal per step
        calories = base_calories + random.gauss(0, 100)
        records.append(HealthData(
            user_id=user_id,
            timestamp=date.replace(hour=23, minute=59),
            metric_type=MetricType.CALORIES,
            value=round(calories),
            unit="kcal",
            source="simulator"
        ))
        
        return records
    
    def generate_and_store(self, user_id: int, days: int = 7) -> int:
        """
        Generate and store data for multiple days.
        Returns number of records created.
        """
        from app import app
        
        records = []
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Get user profile if exists
        with app.app_context():
            user = User.query.get(user_id)
            profile = {
                "age": user.age if user and user.age else 35,
                "base_hr": 70
            } if user else {}
            
            for day_offset in range(days):
                date = today - timedelta(days=day_offset)
                day_records = self.generate_day_data(user_id, date, profile)
                records.extend(day_records)
            
            # Bulk insert
            db.session.bulk_save_objects(records)
            db.session.commit()
        
        return len(records)


def ensure_demo_user(user_id: int = 1) -> User:
    """
    Ensure a demo user exists with the given ID.
    Creates one if it doesn't exist.
    Returns the user object.
    """
    user = User.query.get(user_id)
    if user:
        return user
    
    # Create demo user
    user = User(
        id=user_id,
        name="Demo User",
        email=f"demo{user_id}@vitalai.com",
        age=32,
        gender="male",
        height_cm=175,
        weight_kg=75,
        conditions=[],
        allergies=[],
        preferred_language="en",
        health_goals=["improve sleep", "increase activity"],
        daily_step_goal=10000,
        onboarding_completed=True
    )
    db.session.add(user)
    db.session.commit()
    
    return user


def seed_sample_user():
    """Create a sample user with data for testing."""
    from app import app
    
    with app.app_context():
        # Check if sample user exists
        existing = User.query.filter_by(email="demo@vitalai.com").first()
        if existing:
            print(f"Sample user already exists: {existing.name}")
            return existing
        
        # Create sample user
        user = User(
            name="Demo User",
            email="demo@vitalai.com",
            age=32,
            gender="male",
            height_cm=175,
            weight_kg=75,
            conditions=["none"],
            allergies=[],
            preferred_language="en",
            health_goals=["improve sleep", "increase activity"],
            daily_step_goal=10000,
            onboarding_completed=True
        )
        db.session.add(user)
        db.session.commit()
        
        print(f"Created sample user: {user.name} (ID: {user.id})")
        
        # Generate 14 days of data
        simulator = WearableSimulator(seed=42)
        records = simulator.generate_and_store(user.id, days=14)
        print(f"Generated {records} health data records")
        
        return user


if __name__ == "__main__":
    # Test the simulator
    seed_sample_user()

