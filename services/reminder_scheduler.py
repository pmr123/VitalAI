"""
Reminder Scheduler Service

Uses APScheduler to check for due reminders and trigger notifications.
"""
from datetime import datetime, time as time_type
from typing import List, Dict, Any, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)

# Store pending notifications (simple in-memory queue for demo)
# In production, use Redis or a message queue
pending_notifications: deque = deque(maxlen=100)


def check_due_reminders():
    """
    Check for reminders that are due now.
    Called every minute by the scheduler.
    """
    from app import app
    from models import Reminder, User
    from extensions import db
    
    with app.app_context():
        now = datetime.now()
        current_time = now.time()
        current_day = now.weekday()  # 0=Monday, 6=Sunday
        
        # Get current hour and minute for comparison
        current_hm = (current_time.hour, current_time.minute)
        
        logger.debug(f"[ReminderScheduler] Checking reminders at {current_time.strftime('%H:%M:%S')} on day {current_day} ({now.strftime('%A')})")
        
        # Find active reminders - query ALL active reminders, not just for a specific user
        active_reminders = Reminder.query.filter_by(active=True).all()
        
        logger.debug(f"[ReminderScheduler] Found {len(active_reminders)} active reminders in database")
        
        for reminder in active_reminders:
            # Check if today is a scheduled day
            days = reminder.days_of_week or [0, 1, 2, 3, 4, 5, 6]
            
            # Handle case where days might be stored as a JSON string
            if isinstance(days, str):
                import json
                try:
                    days = json.loads(days)
                except:
                    days = [0, 1, 2, 3, 4, 5, 6]  # Default to every day
            
            # Convert to integers safely
            try:
                days = [int(d) for d in days]
            except (ValueError, TypeError):
                days = [0, 1, 2, 3, 4, 5, 6]
            
            logger.debug(f"[ReminderScheduler] Reminder {reminder.id} ({reminder.title}): scheduled for {reminder.scheduled_time.strftime('%H:%M')} on days {days}")
            
            if current_day not in days:
                logger.debug(f"[ReminderScheduler] Reminder {reminder.id}: today (day {current_day}) not in scheduled days {days}")
                continue
            
            # Check if it's time (compare hour and minute)
            scheduled_hm = (reminder.scheduled_time.hour, reminder.scheduled_time.minute)
            
            logger.debug(f"[ReminderScheduler] Reminder {reminder.id}: comparing current {current_hm} with scheduled {scheduled_hm}")
            
            # Allow a 1-minute window (check if within the same minute)
            if current_hm == scheduled_hm:
                # Check if we already sent this reminder in the last minute
                if reminder.last_sent:
                    time_since_last = (now - reminder.last_sent).total_seconds()
                    logger.debug(f"[ReminderScheduler] Reminder {reminder.id}: last sent {time_since_last:.0f} seconds ago")
                    if time_since_last < 60:
                        logger.debug(f"[ReminderScheduler] Reminder {reminder.id}: already sent recently, skipping")
                        continue  # Already sent recently
                
                # Trigger the reminder!
                logger.info(f"[ReminderScheduler] Triggering reminder {reminder.id}: {reminder.title} at {current_time.strftime('%H:%M')}")
                try:
                    trigger_reminder(reminder)
                    
                    # Update last_sent
                    reminder.last_sent = now
                    db.session.commit()
                    
                    logger.info(f"[ReminderScheduler] Successfully triggered reminder {reminder.id}: {reminder.title}")
                except Exception as e:
                    logger.error(f"[ReminderScheduler] Error triggering reminder {reminder.id}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                logger.debug(f"[ReminderScheduler] Reminder {reminder.id}: time mismatch ({current_hm} != {scheduled_hm})")


def trigger_reminder(reminder) -> Dict[str, Any]:
    """
    Trigger a reminder notification.
    Adds to pending notifications queue.
    """
    from models import User
    
    user = User.query.get(reminder.user_id)
    
    notification = {
        "id": reminder.id,
        "user_id": reminder.user_id,
        "user_name": user.name if user else "Unknown",
        "title": reminder.title,
        "message": reminder.message,
        "medication": reminder.medication.name if reminder.medication else None,
        "channel": reminder.channel,
        "triggered_at": datetime.now().isoformat(),
        "acknowledged": False
    }
    
    pending_notifications.append(notification)
    logger.debug(f"Added notification to queue: {notification['title']}")
    
    return notification


def get_pending_notifications(user_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Get pending notifications, optionally filtered by user.
    """
    if user_id is None:
        return list(pending_notifications)
    
    return [n for n in pending_notifications if n["user_id"] == user_id]


def acknowledge_notification(notification_id: int) -> bool:
    """
    Mark a notification as acknowledged.
    """
    for notification in pending_notifications:
        if notification["id"] == notification_id:
            notification["acknowledged"] = True
            return True
    return False


def clear_acknowledged_notifications():
    """
    Remove acknowledged notifications from the queue.
    """
    global pending_notifications
    pending_notifications = deque(
        [n for n in pending_notifications if not n["acknowledged"]],
        maxlen=100
    )


def init_scheduler(app):
    """
    Initialize the APScheduler for reminder checking.
    Call this from app.py during startup.
    """
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.interval import IntervalTrigger
    
    scheduler = BackgroundScheduler()
    
    # Check for due reminders every minute
    scheduler.add_job(
        func=check_due_reminders,
        trigger=IntervalTrigger(minutes=1),
        id='check_reminders',
        name='Check for due reminders',
        replace_existing=True,
        max_instances=1  # Prevent overlapping runs
    )
    
    # Also run immediately on startup (with small delay to ensure app context is ready)
    from datetime import timedelta
    scheduler.add_job(
        func=check_due_reminders,
        trigger='date',
        run_date=datetime.now() + timedelta(seconds=2),
        id='check_reminders_startup',
        name='Initial reminder check'
    )
    
    scheduler.start()
    logger.info("Reminder scheduler started - checking every minute")
    
    return scheduler

