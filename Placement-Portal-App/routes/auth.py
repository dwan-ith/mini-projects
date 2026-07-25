from flask import Blueprint, request, jsonify
from extensions import db
from models import User, StudentProfile, CompanyProfile
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json(silent=True) or {}
    role = data.get('role')
    email = data.get('email')
    password = data.get('password')
    name = data.get('name')

    if role not in ['student', 'company']:
        return jsonify({'error': 'Invalid role'}), 400
    if not all(isinstance(value, str) and value.strip() for value in (email, password, name)):
        return jsonify({'error': 'Name, email, and password are required'}), 400
    if len(password) < 6:
        return jsonify({'error': 'Password must contain at least 6 characters'}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already exists'}), 400

    user = User(
        role=role,
        email=email,
        password=generate_password_hash(password),
        name=name,
        is_active=True,
        is_approved=True if role == 'student' else False
    )
    db.session.add(user)
    db.session.commit()

    if role == 'student':
        student_profile = StudentProfile(user_id=user.id)
        db.session.add(student_profile)
    elif role == 'company':
        company_profile = CompanyProfile(user_id=user.id)
        db.session.add(company_profile)
    
    db.session.commit()

    return jsonify({'message': 'Registration successful'}), 201

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json(silent=True) or {}
    email = data.get('email')
    password = data.get('password')

    user = User.query.filter_by(email=email).first()

    if not user or not check_password_hash(user.password, password):
        return jsonify({'error': 'Invalid credentials'}), 401

    if not user.is_active:
        return jsonify({'error': 'Account is deactivated or blacklisted'}), 403

    # JWT "sub" must be a string with current PyJWT releases.  Keep
    # authorization metadata in claims instead of using a JSON object as sub.
    access_token = create_access_token(
        identity=str(user.id),
        additional_claims={'role': user.role, 'is_approved': user.is_approved}
    )
    return jsonify({
        'token': access_token,
        'user': user.to_dict()
    }), 200

@auth_bp.route('/me', methods=['GET'])
@jwt_required()
def me():
    current_user = get_jwt_identity()
    user = db.session.get(User, int(current_user))
    if not user:
        return jsonify({'error': 'User not found'}), 404
        
    return jsonify({'user': user.to_dict()}), 200
