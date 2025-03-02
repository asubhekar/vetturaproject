import os
import zipfile
import replicate
from huggingface_hub import HfApi, upload_file
import tempfile
import psycopg2
import requests
from psycopg2 import sql
from fastapi import UploadFile
from typing import List
from openai import OpenAI 
import base64
import time

def get_db_connection():
    return psycopg2.connect(
        dbname="user_database",
        user="atharvsubhekar",
        password="",
        host="localhost"
    )

def fetch_db(username):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(f"SELECT hf_token, hf_username FROM users WHERE LOWER(TRIM(username)) = LOWER('{username}')")
    result = cur.fetchone()
    cur.close()
    conn.close()
    return result

def insert_model_to_db(username, model_name, trigger_word, description):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            sql.SQL("INSERT INTO user_models (username, model_name, model_trigger, model_description) VALUES (%s, %s, %s, %s)"),
            [username, model_name, trigger_word, description]
        )
        conn.commit()
    except psycopg2.Error as e:
        print(f"Error inserting model into database: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

def generate_llm_caption(image_path, person_type):
    """Generates codenames and descriptions using GPT-4 Vision"""
    # Read and encode the image
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
            # Save image
            img_filename = f"image{i+1}{os.path.splitext(img.filename)[1]}"
            img_path = os.path.join(temp_dir, img_filename)
            contents = await img.read()
            with open(img_path, "wb") as f:
                f.write(contents)
            
            # Generate and save description
            person_type = "subject1" if i < 5 else "subject2" if i < 10 else "both subjects"
            description = generate_llm_caption(img_path, person_type)
            desc_filename = f"image{i+1}.txt"
            desc_path = os.path.join(temp_dir, desc_filename)
            with open(desc_path, "w") as f:
                f.write(description)
        
        # Create zip file
        zip_path = os.path.join(temp_dir, "data.zip")
        with zipfile.ZipFile(zip_path, "w") as zip_file:
            for file in os.listdir(temp_dir):
                if file.startswith("image"):
                    zip_file.write(os.path.join(temp_dir, file), file)
        
        # Upload to Hugging Face
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

def flux_training(hf_user, hf_token, model_name, triggerword):
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

    # Wait for training to complete
    while training.status != "succeeded":
        training.reload()
        time.sleep(10)
        print(training.status)

    # Download the file from the URL
    response = requests.get(training.output["weights"])
    training_repo_id = f"{hf_user}/{model_name}"
    api.create_repo(repo_id=training_repo_id, repo_type="model", exist_ok=True)
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tar") as temp_file:
        temp_file.write(response.content)
        temp_file_path = temp_file.name
        
    try:
        # Upload the local file to Hugging Face Hub
        api.upload_file(
            path_or_fileobj=temp_file_path,
            path_in_repo="model.safetensors",
            repo_id=training_repo_id,
            repo_type="model"
        )
    finally:
        # Delete the temporary file
        os.unlink(temp_file_path)

    return training

async def process_images_and_train(username, model_name, images, trigger):
    hf_token, hf_user = fetch_db(username)
    
    await zip_and_upload_images(images, hf_user, hf_token, model_name)
    
    training = flux_training(hf_user, hf_token, model_name, triggerword=trigger)

    description = f"Model trained on {len(images)} images with trigger word: {trigger}"
    insert_model_to_db(username, model_name, trigger, description) # Change model name to model name + version
    
    return f"Training completed successfully! Training ID: {training.id}"
