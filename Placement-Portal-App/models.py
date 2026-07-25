from extensions import db
from datetime import datetime

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    role = db.Column(db.String(50), nullable=False) # admin, company, student
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    is_approved = db.Column(db.Boolean, default=True) # False for companies initially

    def to_dict(self):
        return {
            'id': self.id,
            'role': self.role,
            'email': self.email,
            'name': self.name,
            'is_active': self.is_active,
            'is_approved': self.is_approved
        }

class StudentProfile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    cgpa = db.Column(db.Float, default=0.0)
    branch = db.Column(db.String(100), default='')
    year = db.Column(db.Integer, default=1)
    resume_link = db.Column(db.String(255), default='')
    user = db.relationship('User', backref=db.backref('student_profile', uselist=False))

    def to_dict(self):
        return {
            'cgpa': self.cgpa,
            'branch': self.branch,
            'year': self.year,
            'resume_link': self.resume_link
        }

class CompanyProfile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    hr_contact = db.Column(db.String(100), default='')
    website = db.Column(db.String(255), default='')
    user = db.relationship('User', backref=db.backref('company_profile', uselist=False))

    def to_dict(self):
        return {
            'hr_contact': self.hr_contact,
            'website': self.website
        }

class PlacementDrive(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    company_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    job_title = db.Column(db.String(255), nullable=False)
    job_description = db.Column(db.Text, nullable=False)
    eligibility_branch = db.Column(db.String(255), default='Any')
    eligibility_cgpa = db.Column(db.Float, default=0.0)
    eligibility_year = db.Column(db.Integer, default=4)
    application_deadline = db.Column(db.DateTime, nullable=False)
    status = db.Column(db.String(50), default='Pending') # Pending, Approved, Closed
    company = db.relationship('User', backref=db.backref('drives', lazy=True))

    def to_dict(self):
        return {
            'id': self.id,
            'company_id': self.company_id,
            'company_name': self.company.name if self.company else '',
            'job_title': self.job_title,
            'job_description': self.job_description,
            'eligibility_branch': self.eligibility_branch,
            'eligibility_cgpa': self.eligibility_cgpa,
            'eligibility_year': self.eligibility_year,
            'application_deadline': self.application_deadline.isoformat() if self.application_deadline else None,
            'status': self.status
        }

class Application(db.Model):
    __table_args__ = (db.UniqueConstraint('student_id', 'drive_id', name='uq_student_drive_application'),)
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    drive_id = db.Column(db.Integer, db.ForeignKey('placement_drive.id'), nullable=False)
    application_date = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(50), default='Applied') # Applied, Shortlisted, Selected, Rejected
    student = db.relationship('User', backref=db.backref('applications', lazy=True))
    drive = db.relationship('PlacementDrive', backref=db.backref('applications', lazy=True))

    def to_dict(self):
        return {
            'id': self.id,
            'student_id': self.student_id,
            'student_name': self.student.name if self.student else '',
            'drive_id': self.drive_id,
            'job_title': self.drive.job_title if self.drive else '',
            'company_name': self.drive.company.name if self.drive and self.drive.company else '',
            'application_date': self.application_date.isoformat() if self.application_date else None,
            'status': self.status
        }

class ExportJob(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    status = db.Column(db.String(20), nullable=False, default='Pending')
    filename = db.Column(db.String(255), default='')
    error_message = db.Column(db.String(255), default='')
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    completed_at = db.Column(db.DateTime)
    student = db.relationship('User', backref=db.backref('export_jobs', lazy=True))

    def to_dict(self):
        return {
            'id': self.id,
            'status': self.status,
            'download_url': f'/uploads/{self.filename}' if self.status == 'Completed' and self.filename else None,
            'error_message': self.error_message or None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
        }
