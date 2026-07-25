from flask import Blueprint, request, jsonify
from extensions import db, cache
from models import User, CompanyProfile, PlacementDrive, Application, StudentProfile
from flask_jwt_extended import jwt_required, get_jwt_identity, get_jwt
from datetime import datetime

company_bp = Blueprint('company', __name__)

def is_company():
    return get_jwt().get('role') == 'company'

@company_bp.route('/profile', methods=['GET', 'POST'])
@jwt_required()
def profile():
    if not is_company(): return jsonify({'error': 'Unauthorized'}), 403
    identity = get_jwt_identity()
    profile = CompanyProfile.query.filter_by(user_id=int(identity)).first()
    
    if request.method == 'GET':
        return jsonify(profile.to_dict()), 200
        
    data = request.get_json(silent=True) or {}
    profile.hr_contact = data.get('hr_contact', profile.hr_contact)
    profile.website = data.get('website', profile.website)
    db.session.commit()
    return jsonify({'message': 'Profile updated'}), 200

@company_bp.route('/drives', methods=['GET', 'POST'])
@jwt_required()
def drives():
    if not is_company(): return jsonify({'error': 'Unauthorized'}), 403
    identity = get_jwt_identity()
    user = db.session.get(User, int(identity))
    
    if request.method == 'GET':
        drives = PlacementDrive.query.filter_by(company_id=int(identity)).all()
        res = []
        for d in drives:
            base = d.to_dict()
            base['applicant_count'] = len(d.applications) if d.applications else 0
            res.append(base)
        return jsonify(res), 200
        
    if not user.is_approved:
        return jsonify({'error': 'Company profile not approved by admin yet'}), 403

    data = request.get_json(silent=True) or {}
    if not data.get('job_title') or not data.get('job_description'):
        return jsonify({'error': 'Job title and description are required'}), 400
    deadline = data.get('application_deadline')
    if deadline:
        deadline = datetime.fromisoformat(deadline).replace(tzinfo=None)
    else:
        return jsonify({'error': 'Application deadline is required'}), 400

    if deadline <= datetime.utcnow():
        return jsonify({'error': 'Application deadline must be in the future'}), 400

    drive = PlacementDrive(
        company_id=int(identity),
        job_title=data.get('job_title'),
        job_description=data.get('job_description'),
        eligibility_branch=data.get('eligibility_branch', 'Any') or 'Any',
        eligibility_cgpa=float(data.get('eligibility_cgpa', 0.0)),
        eligibility_year=int(data.get('eligibility_year', 4)),
        application_deadline=deadline,
        status='Pending'
    )
    db.session.add(drive)
    db.session.commit()
    return jsonify({'message': 'Drive created, pending admin approval'}), 201

@company_bp.route('/drives/<int:drive_id>/close', methods=['POST'])
@jwt_required()
def close_drive(drive_id):
    if not is_company(): return jsonify({'error': 'Unauthorized'}), 403
    drive = db.session.get(PlacementDrive, drive_id)
    if not drive or drive.company_id != int(get_jwt_identity()):
        return jsonify({'error': 'Not found or unauthorized'}), 404
    drive.status = 'Closed'
    db.session.commit()
    cache.clear()
    return jsonify({'message': 'Drive closed'}), 200

@company_bp.route('/drives/<int:drive_id>/applications', methods=['GET'])
@jwt_required()
def applications(drive_id):
    if not is_company(): return jsonify({'error': 'Unauthorized'}), 403
    identity = get_jwt_identity()
    drive = db.session.get(PlacementDrive, drive_id)
    if not drive or drive.company_id != int(identity):
        return jsonify({'error': 'Not found or unauthorized'}), 404
        
    apps = Application.query.filter_by(drive_id=drive_id).all()
    # add student details to application dict
    res = []
    for a in apps:
        base = a.to_dict()
        sp = StudentProfile.query.filter_by(user_id=a.student_id).first()
        if sp:
            base['student_profile'] = sp.to_dict()
        res.append(base)
        
    return jsonify(res), 200

@company_bp.route('/applications/<int:app_id>/status', methods=['POST'])
@jwt_required()
def update_application_status(app_id):
    if not is_company(): return jsonify({'error': 'Unauthorized'}), 403
    identity = get_jwt_identity()
    app_record = db.session.get(Application, app_id)
    if not app_record or app_record.drive.company_id != int(identity):
        return jsonify({'error': 'Not found or unauthorized'}), 404
        
    data = request.get_json()
    new_status = data.get('status')
    if new_status in ['Shortlisted', 'Interview Scheduled', 'Selected', 'Rejected']:
        app_record.status = new_status
        db.session.commit()
        return jsonify({'message': f'Status updated to {new_status}'}), 200
        
    return jsonify({'error': 'Invalid status'}), 400
