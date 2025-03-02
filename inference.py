import replicate
from fastapi import HTTPException
import psycopg2
from psycopg2 import sql
import requests

def get_db_connection():
    return psycopg2.connect(
        dbname="user_database",
        user="atharvsubhekar",
        password="",
        host="localhost"
    )

def get_user_models(username):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            sql.SQL("SELECT model_name, model_trigger, model_description FROM user_models WHERE username = %s"),
            [username]
        )
        models = cur.fetchall()
        return [{"name": m[0], "trigger": m[1], "description": m[2]} for m in models]
    except psycopg2.Error as e:
        print(f"Error fetching user models: {e}")
        return []
    finally:
        cur.close()
        conn.close()

def get_latest_model_version(model_owner, model_name, api_token):
    headers = {
        "Authorization": f"Token {api_token}"
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
    replicate_client = replicate.Client(api_token="r8_EBW5V3SAopFXqoSKwSldKIzzu8TsMXM0wCia4")
    try:
        latest_version = get_latest_model_version("asubhekar", model_name,"r8_EBW5V3SAopFXqoSKwSldKIzzu8TsMXM0wCia4")
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
            return str(output[0])  # Convert FileOutput to string to get the URL
        else:
            raise ValueError("Unexpected output format from inference")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")