from celery_worker import celery
import csv
import smtplib
from email.message import EmailMessage
import os
import requests
from datetime import datetime

# You can adjust SMTP settings here or pass them as environment variables
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = 587
SMTP_USER = os.environ.get("SMTP_USER", "email@example.com")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "password")

def _send_email(recipient, subject, text, html=None):
    """Send configured mail; return False when no SMTP credentials are configured."""
    if not recipient or SMTP_PASSWORD == 'password':
        return False
    msg = EmailMessage()
    msg['Subject'], msg['From'], msg['To'] = subject, SMTP_USER, recipient
    msg.set_content(text)
    if html:
        msg.add_alternative(html, subtype='html')
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=20) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)
    return True

@celery.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={'max_retries': 2})
def export_applications_csv(self, job_id):
    from app import create_app
    from models import Application, ExportJob
    from extensions import db
    app = create_app()
    with app.app_context():
        job = db.session.get(ExportJob, job_id)
        if not job:
            return None
        job.status = 'Processing'
        db.session.commit()
        apps = Application.query.filter_by(student_id=job.student_id).all()
        filename = f"exported_apps_{job.student_id}_{job.id}.csv"
        export_dir = os.path.join(app.root_path, 'uploads')
        os.makedirs(export_dir, exist_ok=True)
        filepath = os.path.join(export_dir, filename)
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Application ID', 'Student ID', 'Drive ID', 'Company Name', 'Job Title', 'Application Date', 'Status']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for a in apps:
                writer.writerow({
                    'Application ID': a.id,
                    'Student ID': job.student_id,
                    'Drive ID': a.drive_id,
                    'Company Name': a.drive.company.name if a.drive and a.drive.company else '',
                    'Job Title': a.drive.job_title if a.drive else '',
                    'Application Date': a.application_date.isoformat(),
                    'Status': a.status
                })
        
        job.status = 'Completed'
        job.filename = filename
        job.completed_at = datetime.utcnow()
        db.session.commit()
        _send_email(job.student.email, 'Placement applications export ready', 'Your CSV export is ready to download from the Placement Portal.')
        print(f"Exported to {filepath}")
    return filename

@celery.task
def daily_reminders():
    # Send daily reminders to students about upcoming application deadlines via G-Chat Webhook
    from app import create_app
    from models import PlacementDrive, User
    
    app = create_app()
    with app.app_context():
        now = datetime.utcnow()
        sent = 0
        drives = PlacementDrive.query.filter_by(status='Approved').all()
        for drive in drives:
            if 0 < (drive.application_deadline - now).total_seconds() < 86400:
                message = f"Reminder: {drive.job_title} by {drive.company.name} closes soon! Apply before {drive.application_deadline}."
                students = User.query.filter_by(role='student', is_active=True).all()
                for student in students:
                    _send_email(student.email, 'Placement deadline reminder', message)
                    sent += 1
                webhook_url = os.environ.get("GCHAT_WEBHOOK_URL", "")
                if webhook_url:
                    requests.post(webhook_url, json={"text": message}, timeout=10)
        return {'notifications': sent}

@celery.task
def monthly_report():
    print("Generating monthly report...")
    from app import create_app
    from models import PlacementDrive, Application, User
    app = create_app()
    with app.app_context():
        drives_count = PlacementDrive.query.count()
        apps_count = Application.query.count()
        selected_count = Application.query.filter_by(status='Selected').count()
        admin = User.query.filter_by(role='admin').first()
        
        # HTML template string
        html_content = f"""
        <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; }}
                    .container {{ padding: 20px; border: 1px solid #ccc; }}
                    h2 {{ color: #4F46E5; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>Monthly Placement Activity Report</h2>
                    <p>Here is the summary of the placement activities for this month:</p>
                    <ul>
                        <li><strong>Total Drives Conducted:</strong> {drives_count}</li>
                        <li><strong>Total Applications Received:</strong> {apps_count}</li>
                        <li><strong>Total Students Selected:</strong> {selected_count}</li>
                    </ul>
                    <p>Regards,<br>Placement Portal Automated System</p>
                </div>
            </body>
        </html>
        """
        
        sent = _send_email(
            admin.email if admin else "admin@institute.edu",
            'Monthly Placement Activity Report',
            'Your monthly placement activity report is attached as HTML.',
            html_content
        )
        return {'email_sent': sent, 'drives': drives_count, 'applications': apps_count, 'selected': selected_count}
