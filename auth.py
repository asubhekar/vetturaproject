import psycopg2
from psycopg2 import sql
from huggingface_hub import HfApi
import hashlib

def get_db_connection():
    return psycopg2.connect(
        dbname="user_database",
        user="atharvsubhekar",
        password="",
        host="localhost"
    )

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def validate_hf_token(token):
    try:
        api = HfApi(token=token)
        api.whoami()
        return True
    except:
        return False

def register_user(username, password, hf_token, hf_username):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            sql.SQL("INSERT INTO users (username, password, hf_token, hf_username) VALUES (%s, %s, %s, %s)"),
            [username, hash_password(password), hf_token, hf_username]
        )
        conn.commit()
        return True
    except psycopg2.Error:
        conn.rollback()
        return False
    finally:
        cur.close()
        conn.close()

def authenticate_user(username, password):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT password FROM users WHERE username = %s", (username,))
    result = cur.fetchone()
    cur.close()
    conn.close()
    if result:
        return result[0] == hash_password(password)
    return False
