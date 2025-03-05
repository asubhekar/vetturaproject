import sqlalchemy
from google.cloud.sql.connector import Connector, IPTypes
from huggingface_hub import HfApi
import hashlib

INSTANCE_CONNECTION_NAME = "quiet-canto-451319-v0:us-central1:photographyagent"
DB_USER = "postgres"
DB_PASS = "Tambourinet$64"
DB_NAME = "user_database"

def get_db_connection():
    def getconn():
        with Connector() as connector:
            conn = connector.connect(
                INSTANCE_CONNECTION_NAME,
                "pg8000",
                user=DB_USER,
                password=DB_PASS,
                db=DB_NAME,
                ip_type=IPTypes.PUBLIC,
            )
            return conn

    engine = sqlalchemy.create_engine(
        "postgresql+pg8000://",
        creator=getconn,
    )
    return engine.connect()

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
    with get_db_connection() as conn:
        try:
            conn.execute(
                sqlalchemy.text("INSERT INTO users (username, password, hf_token, hf_username) VALUES (:username, :password, :hf_token, :hf_username)"),
                {"username": username, "password": hash_password(password), "hf_token": hf_token, "hf_username": hf_username}
            )
            conn.commit()
            return True
        except sqlalchemy.exc.IntegrityError:
            conn.rollback()
            return False

def authenticate_user(username, password):
    with get_db_connection() as conn:
        result = conn.execute(
            sqlalchemy.text("SELECT password FROM users WHERE username = :username"),
            {"username": username}
        ).fetchone()
        if result:
            return result[0] == hash_password(password)
        return False
