"""
Database configuration and SQLAlchemy setup for the melanoma detection system.
"""
import os
from flask_sqlalchemy import SQLAlchemy

# Initialize SQLAlchemy without a Flask instance
db = SQLAlchemy()

def init_db(app):
    """
    Initialize the database with the Flask application.
    
    Args:
        app: Flask application instance
    """
    # Configure database URI from environment variable
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
    
    # Configure database connection options
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
    }
    
    # Initialize the app with the SQLAlchemy extension
    db.init_app(app)
    
    # Create tables if they don't exist
    with app.app_context():
        db.create_all()