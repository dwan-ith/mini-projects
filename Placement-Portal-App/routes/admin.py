from flask import Blueprint, request, jsonify
from extensions import db
from extensions import cache
from models import User, PlacementDrive, Application
from flask_jwt_extended import jwt_required, get_jwt

admin_bp = Blueprint('admin', __name__)

def is_admin():
    return get_jwt().get('role') == 'admin'

@admin_bp.route('/dashboard', methods=['GET'])
@jwt_required()
def dashboard():
    if not is_admin(): return jsonify({'error': 'Unauthorized'}), 403
    students = User.query.filter_by(role='student').count()
    companies = User.query.filter_by(role='company').count()
    drives = PlacementDrive.query.count()
    applications = Application.query.count()
    selected = Application.query.filter_by(status='Selected').count()
    return jsonify({
        'students': students,
        'companies': companies,
        'drives': drives,
        'applications': applications,
        'selected': selected
    }), 200

@admin_bp.route('/companies', methods=['GET'])
@jwt_required()
def get_companies():
    if not is_admin(): return jsonify({'error': 'Unauthorized'}), 403
    companies = User.query.filter_by(role='company').all()
    return jsonify([c.to_dict() for c in companies]), 200

@admin_bp.route('/students', methods=['GET'])
@jwt_required()
def get_students():
    if not is_admin(): return jsonify({'error': 'Unauthorized'}), 403
    students = User.query.filter_by(role='student').all()
    return jsonify([s.to_dict() for s in students]), 200

@admin_bp.route('/approve_company/<int:company_id>', methods=['POST'])
@jwt_required()
def approve_company(company_id):
    if not is_admin(): return jsonify({'error': 'Unauthorized'}), 403
    company = User.query.filter_by(id=company_id, role='company').first()
    if not company: return jsonify({'error': 'Not found'}), 404
    company.is_approved = True
    db.session.commit()
    return jsonify({'message': 'Company approved'}), 200

@admin_bp.route('/toggle_active/<int:user_id>', methods=['POST'])
@jwt_required()
def toggle_active(user_id):
    if not is_admin(): return jsonify({'error': 'Unauthorized'}), 403
    user = db.session.get(User, user_id)
    if not user: return jsonify({'error': 'Not found'}), 404
    if user.role == 'admin': return jsonify({'error': 'Cannot modify admin'}), 400
    user.is_active = not user.is_active
    db.session.commit()
    return jsonify({'message': 'User status updated', 'is_active': user.is_active}), 200

@admin_bp.route('/reject_company/<int:company_id>', methods=['POST'])
@jwt_required()
def reject_company(company_id):
    if not is_admin(): return jsonify({'error': 'Unauthorized'}), 403
    company = User.query.filter_by(id=company_id, role='company').first()
    if not company: return jsonify({'error': 'Not found'}), 404
    db.session.delete(company)
    db.session.commit()
    return jsonify({'message': 'Company rejected and removed'}), 200

@admin_bp.route('/drives', methods=['GET'])
@jwt_required()
def get_drives():
    if not is_admin(): return jsonify({'error': 'Unauthorized'}), 403
    drives = PlacementDrive.query.all()
    # attach applicant count
    res = []
    for d in drives:
        base = d.to_dict()
        base['applicant_count'] = len(d.applications) if d.applications else 0
        res.append(base)
    return jsonify(res), 200

@admin_bp.route('/approve_drive/<int:drive_id>', methods=['POST'])
@jwt_required()
def approve_drive(drive_id):
    if not is_admin(): return jsonify({'error': 'Unauthorized'}), 403
    drive = db.session.get(PlacementDrive, drive_id)
    if not drive: return jsonify({'error': 'Not found'}), 404
    drive.status = 'Approved'
    db.session.commit()
    cache.clear()
    return jsonify({'message': 'Drive approved'}), 200

@admin_bp.route('/reject_drive/<int:drive_id>', methods=['POST'])
@jwt_required()
def reject_drive(drive_id):
    if not is_admin(): return jsonify({'error': 'Unauthorized'}), 403
    drive = db.session.get(PlacementDrive, drive_id)
    if not drive: return jsonify({'error': 'Not found'}), 404
    drive.status = 'Rejected'
    db.session.commit()
    cache.clear()
    return jsonify({'message': 'Drive rejected'}), 200

@admin_bp.route('/applications', methods=['GET'])
@jwt_required()
def get_applications():
    if not is_admin(): return jsonify({'error': 'Unauthorized'}), 403
    apps = Application.query.all()
    res = []
    for a in apps:
        base = a.to_dict()
        base['student_name'] = a.student.name if a.student else ''
        base['company_name'] = a.drive.company.name if a.drive and a.drive.company else ''
        base['job_title'] = a.drive.job_title if a.drive else ''
        res.append(base)
    return jsonify(res), 200
