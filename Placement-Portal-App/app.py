import os
from flask import Flask, render_template, request, jsonify
from config import Config
from extensions import db, jwt, cache
from models import User
from werkzeug.security import generate_password_hash

from flask_cors import CORS


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Redis is used whenever it is running.  A local fallback keeps the app
    # functional for evaluators who have not started Redis yet.
    if app.config['CACHE_TYPE'] == 'RedisCache':
        try:
            import redis
            redis.Redis.from_url(app.config['CACHE_REDIS_URL']).ping()
        except Exception:
            app.config['CACHE_TYPE'] = 'SimpleCache'

    CORS(app)
    db.init_app(app)
    jwt.init_app(app)
    cache.init_app(app)

    # Register blueprints (to be created)
    from routes.auth import auth_bp
    from routes.admin import admin_bp
    from routes.company import company_bp
    from routes.student import student_bp

    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(admin_bp, url_prefix='/api/admin')
    app.register_blueprint(company_bp, url_prefix='/api/company')
    app.register_blueprint(student_bp, url_prefix='/api/student')

    from flask import send_from_directory
    @app.route('/uploads/<path:filename>')
    def download_file(filename):
        return send_from_directory(os.path.join(app.root_path, 'uploads'), filename)

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/<path:path>')
    def catch_all(path):
        return render_template('index.html')

    with app.app_context():
        db.create_all()
        # Create default admin if not exists
        admin = User.query.filter_by(role='admin').first()
        if not admin:
            admin = User(
                role='admin',
                email='admin@institute.edu',
                password=generate_password_hash('admin123'),
                name='Institute Admin',
                is_active=True,
                is_approved=True
            )
            db.session.add(admin)
            db.session.commit()

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
