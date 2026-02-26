import os
import jwt
from functools import wraps
from flask import request, jsonify, current_app
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()
JWT_SECRET = os.getenv("JWT_SECRET", "supersecretkey")

# Predefined user roles — extend by adding to .env or a users table
USERS = {
    os.getenv("ADMIN_USER", "admin"): {
        "password": os.getenv("ADMIN_PASS", "password"),
        "role": "ADMIN",
        "name": "Administrator"
    },
    os.getenv("MANAGER_USER", "manager"): {
        "password": os.getenv("MANAGER_PASS", "manager123"),
        "role": "MANAGER",
        "name": "Branch Manager"
    },
    os.getenv("OFFICER_USER", "officer"): {
        "password": os.getenv("OFFICER_PASS", "officer123"),
        "role": "OFFICER",
        "name": "Loan Officer"
    },
}


def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT")
    )


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing token"}), 401
        token = auth_header.split(" ")[1]
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            request.current_user = payload.get("sub")
            request.current_role = payload.get("role", "OFFICER")
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
        return f(*args, **kwargs)
    return decorated


def role_required(*allowed_roles):
    """Decorator factory: restricts endpoint to one or more roles."""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return jsonify({"error": "Missing token"}), 401
            token = auth_header.split(" ")[1]
            try:
                payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
                role = payload.get("role", "OFFICER")
                if role not in allowed_roles:
                    return jsonify({"error": f"Access denied — requires {' or '.join(allowed_roles)}"}), 403
                request.current_user = payload.get("sub")
                request.current_role = role
            except jwt.ExpiredSignatureError:
                return jsonify({"error": "Token expired"}), 401
            except jwt.InvalidTokenError:
                return jsonify({"error": "Invalid token"}), 401
            return f(*args, **kwargs)
        return decorated
    return decorator


def get_admin_users():
    """GET /admin/users — ADMIN only"""
    users_list = [
        {"username": u, "name": d["name"], "role": d["role"]}
        for u, d in USERS.items()
        if u  # skip empty env vars
    ]
    return jsonify({"users": users_list})


def get_me():
    """GET /me — return current user info"""
    return jsonify({
        "username": request.current_user,
        "role": request.current_role
    })
