from werkzeug.security import generate_password_hash, check_password_hash
import sys
import os
import psycopg2
from psycopg2 import pool

# ===== Database Config =====
root_path = '/home/coder/trong/computervision/checkin_face_anti_spoofing/'
# Load từ file .env
db_config = {}
env_file = os.path.join(root_path, '.env')
if os.path.exists(env_file):
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                db_config[key.strip()] = value.strip()

# Database connection pool
def get_db_connection():
    try:
        db_pool = psycopg2.pool.ThreadedConnectionPool(
            1, 10,
            host=db_config.get('HOST', 'localhost'),
            port=db_config.get('PORT', '5432'),
            user=db_config.get('USER', 'postgres'),
            password=db_config.get('PASSWORD', ''),
            dbname=db_config.get('DBNAME', 'checkin')
        )
        print("✅ Database connection pool created successfully")
    except Exception as e:
        print(f"❌ Failed to create database pool: {e}")
        db_pool = None
    return db_pool

if __name__ == "__main__":
    # Test connection pool
    pool = get_db_connection()
    if pool:
        conn = pool.getconn()
        if conn:
            print("✅ Successfully obtained a connection from the pool")
            pool.putconn(conn)
        pool.closeall()