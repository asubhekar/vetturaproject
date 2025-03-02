# main.py

from fastapi import FastAPI, Request, Form, Depends, HTTPException, status, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.middleware.sessions import SessionMiddleware
from auth import authenticate_user, register_user, validate_hf_token
from app import process_images_and_train
from inference import run_inference, get_user_models
from typing import List

import secrets

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="your-secret-key")

templates = Jinja2Templates(directory="templates")
security = HTTPBasic()

@app.get("/", response_class=HTMLResponse)
async def login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/")
async def login_post(request: Request, username: str = Form(...), password: str = Form(...)):
    if authenticate_user(username, password):
        request.session["username"] = username
        return RedirectResponse(url="/app", status_code=status.HTTP_303_SEE_OTHER)
    return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})

@app.get("/register", response_class=HTMLResponse)
async def register(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register")
async def register_post(request: Request, username: str = Form(...), password: str = Form(...), 
                        hf_username: str = Form(...), hf_token: str = Form(...)):
    if validate_hf_token(hf_token):
        if register_user(username, password, hf_token, hf_username):
            return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
        return templates.TemplateResponse("register.html", {"request": request, "error": "Username already exists"})
    return templates.TemplateResponse("register.html", {"request": request, "error": "Invalid Hugging Face token"})

@app.get("/app", response_class=HTMLResponse)
async def app_page(request: Request):
    username = request.session.get("username")
    if not username:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    user_models = get_user_models(username)
    return templates.TemplateResponse("app.html", {"request": request, "username": username, "user_models": user_models})

@app.post("/process")
async def process(request: Request, model_name: str = Form(...), images: List[UploadFile] = File(...), trigger_word: str = Form(...)):
    username = request.session.get("username")
    if not username:
        raise HTTPException(status_code=401, detail="Not authenticated")
    result = await process_images_and_train(username, model_name, images, trigger_word)
    user_models = get_user_models(username)
    return templates.TemplateResponse("app.html", {"request": request, "username": username, "result": result, "user_models": user_models})

@app.post("/inference")
async def inference(request: Request, model_name: str = Form(...), prompt: str = Form(...)):
    username = request.session.get("username")
    if not username:
        raise HTTPException(status_code=401, detail="Not authenticated")
    inference_result = run_inference(model_name, prompt)
    user_models = get_user_models(username)
    return templates.TemplateResponse("app.html", {
        "request": request,
        "username": username,
        "user_models": user_models,
        "inference_result": inference_result  
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
