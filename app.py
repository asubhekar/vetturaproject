import os
import zipfile
import tempfile
from typing import List
import requests
import base64
import asyncio
import time

import replicate
from huggingface_hub import HfApi
from openai import OpenAI 

from fastapi import UploadFile, BackgroundTasks

import sqlalchemy
from google.cloud.sql.connector import Connector, IPTypes

from fastapi.templating import Jinja2Templates
import sendgrid
from sendgrid.helpers.mail import Mail, Email, To, Content



templates = Jinja2Templates(directory="templates")

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

def fetch_db(username):
    with get_db_connection() as conn:
        result = conn.execute(sqlalchemy.text("SELECT hf_token, hf_username FROM users WHERE LOWER(TRIM(username)) = LOWER(:username)"), {"username": username})
        return result.fetchone()

def insert_model_to_db(username, model_name, trigger_word, description):
    with get_db_connection() as conn:
        try:
            conn.execute(
                sqlalchemy.text("INSERT INTO user_models (username, model_name, model_trigger, model_description) VALUES (:username, :model_name, :trigger_word, :description)"),
                {"username": username, "model_name": model_name, "trigger_word": trigger_word, "description": description}
            )
            conn.commit()
        except sqlalchemy.exc.IntegrityError as e:
            print(f"Error inserting model into database: {e}")
            conn.rollback()

def generate_llm_caption(image_path, person_type):
    """Generates codenames and descriptions using GPT-4 Vision"""
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    openai_client = OpenAI(api_key="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")
    prompt = f"""Analyze this {person_type} photo and generate a detailed description."""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ],
            }
        ],
    )
    return response.choices[0].message.content

async def zip_and_upload_images(images: List[UploadFile], hf_user: str, hf_token: str, model_name: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, img in enumerate(images):
            img_filename = f"image{i+1}{os.path.splitext(img.filename)[1]}"
            img_path = os.path.join(temp_dir, img_filename)
            contents = await img.read()
            with open(img_path, "wb") as f:
                f.write(contents)
            
            person_type = "subject1" if i < 5 else "subject2" if i < 10 else "both subjects"
            description = generate_llm_caption(img_path, person_type)
            desc_filename = f"image{i+1}.txt"
            desc_path = os.path.join(temp_dir, desc_filename)
            with open(desc_path, "w") as f:
                f.write(description)
        
        zip_path = os.path.join(temp_dir, "data.zip")
        with zipfile.ZipFile(zip_path, "w") as zip_file:
            for file in os.listdir(temp_dir):
                if file.startswith("image"):
                    zip_file.write(os.path.join(temp_dir, file), file)
        
        repo_id = f"{hf_user}/{model_name}_dataset"
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        api.upload_file(
            path_or_fileobj=zip_path,
            path_in_repo="data.zip",
            repo_id=repo_id,
            repo_type="dataset",
            token=hf_token
        )

    return zip_path

async def flux_training(hf_user, hf_token, model_name, triggerword):
    repo_id = f"{hf_user}/{model_name}_dataset"
    api = HfApi(token=hf_token)
    zip_path = api.hf_hub_download(repo_id=repo_id, filename="data.zip", repo_type="dataset")
    
    replicate_client = replicate.Client(api_token="r8_EBW5V3SAopFXqoSKwSldKIzzu8TsMXM0wCia4")
    model = replicate_client.models.create(
        owner="asubhekar",
        name=model_name,
        visibility="private",
        hardware="gpu-t4", 
        description="A fine-tuned FLUX.1 model"
    )
    replicate_destination = f"{model.owner}/{model.name}"
    
    training = replicate_client.trainings.create(
        destination=replicate_destination,
        version="ostris/flux-dev-lora-trainer:b6af14222e6bd9be257cbc1ea4afda3cd0503e1133083b9d1de0364d8568e6ef",
        input={
            "steps": 1000,
            "lora_rank": 16,
            "optimizer": "adamw8bit",
            "batch_size": 1,
            "resolution": "512,768,1024",
            "autocaption": False,
            "input_images": open(zip_path,"rb"),
            "trigger_word": triggerword,
            "learning_rate": 0.0004,
            "caption_dropout_rate": 0.05,
            "cache_latents_to_disk": False,
            "gradient_checkpointing": False
        },
    )

    while training.status != "succeeded":
        await asyncio.sleep(10)
        training.reload()
        print(training.status)

    response = requests.get(training.output["weights"])
    training_repo_id = f"{hf_user}/{model_name}"
    api.create_repo(repo_id=training_repo_id, repo_type="model", exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tar") as temp_file:
        temp_file.write(response.content)
        temp_file_path = temp_file.name
        
    try:
        api.upload_file(
            path_or_fileobj=temp_file_path,
            path_in_repo="model.safetensors",
            repo_id=training_repo_id,
            repo_type="model"
        )
    finally:
        os.unlink(temp_file_path)

    return training

def send_completion_email(to_email, model_name):
    sg_apikey = "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"
    sg = sendgrid.SendGridAPIClient(api_key=sg_apikey)
    from_email = Email("subhuatharva@gmail.com") 
    subject = f"Training Completed for Model: {model_name}"

    with open("templates/email.html", "r") as f:
        html_content = f.read()
    
    image_path = "templates/DreamWeaver.png"  
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    html_content = html_content.replace(
        'src="http://cdn.mcauto-images-production.sendgrid.net/954c252fedab403f/7dc41373-34dd-434e-a1d8-0b85cfcefa0a/256x359.png"',
        f'src="data:image/png;base64,{encoded_image}"'
    )

    
    html_content = html_content.replace("{{model_name}}", model_name)

    content = Content("text/html", html_content)
    mail = Mail(from_email, To(to_email), subject, content)
    response = sg.client.mail.send.post(request_body=mail.get())
    print(f"Email sent with status code: {response.status_code}")

def run_training(hf_user, hf_token, model_name, trigger, username, email_id):
    training = flux_training(hf_user, hf_token, model_name, triggerword=trigger)
    while training.status != "succeeded":
        training.reload()
        time.sleep(10)
    description = f"Model trained on 15 images with trigger word: {trigger}"
    print(description)
    insert_model_to_db(username, model_name, trigger, description)
    if email_id != None:
        send_completion_email(email_id, model_name)

async def process_images_and_train(background_tasks: BackgroundTasks, username, model_name, images, trigger, email_id = None):
    hf_token, hf_user = fetch_db(username)
    await zip_and_upload_images(images, hf_user, hf_token, model_name)
    background_tasks.add_task(run_training, hf_user, hf_token, model_name, trigger, username, email_id)
    return "Your AI is learning. You will be notified via email once it is completed. Once trained, the model can be found in your HuggingFace account."


