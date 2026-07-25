from celery import Celery
from celery.schedules import crontab
import os

def make_celery(app_name=__name__):
    backend = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/2')
    broker = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/1')

    celery_app = Celery(app_name, backend=backend, broker=broker)
    
    celery_app.conf.beat_schedule = {
        'daily-reminders': {
            'task': 'tasks.daily_reminders',
            'schedule': crontab(hour=9, minute=0),
        },
        'monthly-report': {
            'task': 'tasks.monthly_report',
            'schedule': crontab(day_of_month=1, hour=9, minute=0),
        }
    }
    return celery_app

celery = make_celery()
