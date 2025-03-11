import replicate
from fastapi import HTTPException
import sqlalchemy
from google.cloud.sql.connector import Connector, IPTypes
import requests
import os

#openai_key = os.getenv("OPEN_API_KEY")
#sendgrid_key = os.getenv("SENDGRID_KEY")
replicate_key = os.getenv("REPLICATE_KEY")

INSTANCE_CONNECTION_NAME = "quiet-canto-451319-v0:us-central1:photographyagent"
DB_USER = "postgres"
DB_PASS = os.getenv("DB_PASSWORD")
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

def get_user_models(username):
    with get_db_connection() as conn:
        try:
            result = conn.execute(
                sqlalchemy.text("SELECT model_name, model_trigger, model_description FROM user_models WHERE username = :username"),
                {"username": username}
            )
            models = result.fetchall()
            return [{"name": m[0], "trigger": m[1], "description": m[2]} for m in models]
        except sqlalchemy.exc.SQLAlchemyError as e:
            print(f"Error fetching user models: {e}")
            return []

def get_latest_model_version(model_owner, model_name):
    headers = {
        "Authorization": f"Token {replicate_key}"
    }
    response = requests.get(
        f"https://api.replicate.com/v1/models/{model_owner}/{model_name}/versions",
        headers=headers
    )
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Failed to fetch model versions: {response.text}")
    versions = response.json()["results"]
    if not versions:
        raise HTTPException(status_code=404, detail=f"No versions found for model {model_owner}/{model_name}")
    return versions[0]["id"]

def run_inference(model_name, prompt):
    replicate_client = replicate.Client(api_token=replicate_key)
    prompt += "Realistic"
    try:
        latest_version = get_latest_model_version("asubhekar", model_name)
        model = f"asubhekar/{model_name}:{latest_version}"
        output = replicate_client.run(
            model,
            input={
                "model": "dev",
                "go_fast": False,
                "lora_scale": 1,
                "megapixels": "1",
                "num_outputs": 1,
                "aspect_ratio": "1:1",
                "output_format": "webp",
                "guidance_scale": 3,
                "output_quality": 80,
                "prompt_strength": 0.8,
                "extra_lora_scale": 1,
                "num_inference_steps": 28,
                "prompt": prompt
            }
        )
        if output and isinstance(output[0], replicate.helpers.FileOutput):
            return str(output[0])  
        else:
            raise ValueError("Unexpected output format from inference")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")