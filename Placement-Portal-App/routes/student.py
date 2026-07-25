from flask import Blueprint, request, jsonify
from extensions import db, cache
from models import User, StudentProfile, PlacementDrive, Application, ExportJob
from flask_jwt_extended import jwt_required, get_jwt_identity, get_jwt
from datetime import datetime

student_bp = Blueprint('student', __name__)

def is_student():
    return get_jwt().get('role') == 'student'

def student_drives_cache_key():
    return f'approved_drives_{get_jwt_identity()}'

import os
from werkzeug.utils import secure_filename

@student_bp.route('/profile', methods=['GET', 'POST'])
@jwt_required()
def profile():
    if not is_student(): return jsonify({'error': 'Unauthorized'}), 403
    identity = get_jwt_identity()
    profile = StudentProfile.query.filter_by(user_id=int(identity)).first()
    
    if request.method == 'GET':
        return jsonify(profile.to_dict()), 200
        
    data = request.get_json()
    profile.cgpa = float(data.get('cgpa', profile.cgpa))
    profile.branch = data.get('branch', profile.branch)
    profile.year = int(data.get('year', profile.year))
    db.session.commit()
    return jsonify({'message': 'Profile updated'}), 200

@student_bp.route('/upload_resume', methods=['POST'])
@jwt_required()
def upload_resume():
    if not is_student(): return jsonify({'error': 'Unauthorized'}), 403
    identity = get_jwt_identity()
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        filename = secure_filename(file.filename)
        # Create uploads folder if not exists
        os.makedirs('uploads', exist_ok=True)
        filepath = os.path.join('uploads', f"student_{identity}_{filename}")
        file.save(filepath)
        
        # update profile
        profile = StudentProfile.query.filter_by(user_id=int(identity)).first()
        profile.resume_link = filepath
        db.session.commit()
        
        return jsonify({'message': 'Resume uploaded successfully', 'filepath': filepath}), 200

@student_bp.route('/drives', methods=['GET'])
@jwt_required()
@cache.cached(timeout=60, key_prefix=student_drives_cache_key)
def drives():
    if not is_student(): return jsonify({'error': 'Unauthorized'}), 403
    profile = StudentProfile.query.filter_by(user_id=int(get_jwt_identity())).first()
    drives = PlacementDrive.query.filter_by(status='Approved').all()
    applied_drive_ids = {
        application.drive_id
        for application in Application.query.filter_by(student_id=int(get_jwt_identity())).all()
    }
    result = []
    for drive in drives:
        item = drive.to_dict()
        reasons = []
        if profile.cgpa < drive.eligibility_cgpa:
            reasons.append(f'Minimum CGPA is {drive.eligibility_cgpa}')
        if drive.eligibility_branch != 'Any' and profile.branch != drive.eligibility_branch:
            reasons.append(f'Eligible branch: {drive.eligibility_branch}')
        if profile.year < drive.eligibility_year:
            reasons.append(f'Year {drive.eligibility_year} or above required')
        if drive.application_deadline and drive.application_deadline < datetime.utcnow():
            reasons.append('Application deadline has passed')
        item['is_eligible'] = not reasons
        item['eligibility_message'] = '; '.join(reasons)
        item['has_applied'] = drive.id in applied_drive_ids
        result.append(item)
    return jsonify(result), 200

@student_bp.route('/applications', methods=['GET', 'POST'])
@jwt_required()
def applications():
    if not is_student(): return jsonify({'error': 'Unauthorized'}), 403
    identity = get_jwt_identity()
    
    if request.method == 'GET':
        apps = Application.query.filter_by(student_id=int(identity)).all()
        return jsonify([a.to_dict() for a in apps]), 200
        
    data = request.get_json(silent=True) or {}
    drive_id = data.get('drive_id')
    
    # Check if already applied
    existing = Application.query.filter_by(student_id=int(identity), drive_id=drive_id).first()
    if existing:
        return jsonify({'error': 'Already applied to this drive'}), 400
        
    # Check eligibility
    drive = db.session.get(PlacementDrive, drive_id)
    if not drive or drive.status != 'Approved':
        return jsonify({'error': 'Drive not available'}), 404
        
    profile = StudentProfile.query.filter_by(user_id=int(identity)).first()
    if profile.cgpa < drive.eligibility_cgpa:
        return jsonify({'error': 'CGPA too low'}), 400
    if drive.eligibility_branch != 'Any' and profile.branch != drive.eligibility_branch:
        return jsonify({'error': 'Branch not eligible'}), 400
    if profile.year < drive.eligibility_year:
        return jsonify({'error': f'Year {drive.eligibility_year} or above is required'}), 400
    if drive.application_deadline and drive.application_deadline < datetime.utcnow():
        return jsonify({'error': 'Application deadline has passed'}), 400
        
    app_record = Application(
        student_id=int(identity),
        drive_id=drive_id
    )
    db.session.add(app_record)
    try:
        db.session.commit()
    except Exception:
        db.session.rollback()
        return jsonify({'error': 'Already applied to this drive'}), 400
    cache.delete(student_drives_cache_key())
    return jsonify({'message': 'Applied successfully'}), 201

@student_bp.route('/export', methods=['POST'])
@jwt_required()
def export_csv():
    if not is_student(): return jsonify({'error': 'Unauthorized'}), 403
    identity = get_jwt_identity()
    job = ExportJob(student_id=int(identity), status='Pending')
    db.session.add(job)
    db.session.commit()
    from tasks import export_applications_csv
    try:
        export_applications_csv.delay(job.id)
        return jsonify({'message': 'Export started', 'job': job.to_dict()}), 202
    except Exception as exc:
        job.status = 'Failed'
        job.error_message = str(exc)[:255]
        db.session.commit()
        return jsonify({'error': 'Export worker is unavailable. Start the Celery worker and try again.'}), 503

@student_bp.route('/export/<int:job_id>', methods=['GET'])
@jwt_required()
def export_status(job_id):
    if not is_student(): return jsonify({'error': 'Unauthorized'}), 403
    job = db.session.get(ExportJob, job_id)
    if not job or job.student_id != int(get_jwt_identity()):
        return jsonify({'error': 'Not found'}), 404
    return jsonify(job.to_dict()), 200

