from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import shutil
import os
from typing import Optional, List
from src.gradio_demo import SadTalker
import uuid
import glob
from urllib.parse import quote
app = FastAPI(title="SadTalker API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create required directories
UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"
STATIC_DIR = "static"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Initialize SadTalker
sad_talker = SadTalker(checkpoint_path='checkpoints', config_path='src/config', lazy_load=True)

@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        with open(os.path.join(STATIC_DIR, "index.html")) as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="Error: index.html not found in static directory", status_code=404)

@app.post("/generate")
async def generate_video(
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
    preprocess: str = Form("crop"),
    still_mode: bool = Form(False),
    use_enhancer: bool = Form(False),
    batch_size: int = Form(1),
    size: int = Form(256),
    pose_style: int = Form(0)
):
    if not image.filename or not audio.filename:
        raise HTTPException(status_code=400, detail="Both image and audio files are required")
        
    # Generate unique IDs for files
    image_id = str(uuid.uuid4())
    audio_id = str(uuid.uuid4())
    
    # Save uploaded files
    image_path = os.path.join(UPLOAD_DIR, f"{image_id}_{image.filename}")
    audio_path = os.path.join(UPLOAD_DIR, f"{audio_id}_{audio.filename}")
    
    try:
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        
        # Generate video using SadTalker
        result_path = sad_talker.test(
            source_image=image_path,
            driven_audio=audio_path,
            preprocess=preprocess,
            still_mode=still_mode,
            use_enhancer=use_enhancer,
            batch_size=batch_size,
            size=size,
            pose_style=pose_style,
            result_dir=RESULTS_DIR
        )
        
        if not os.path.exists(result_path):
            raise HTTPException(status_code=500, detail="Failed to generate video")
            
        return FileResponse(
            result_path,
            media_type="video/mp4",
            filename="generated_video.mp4"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup uploaded files
        if os.path.exists(image_path):
            os.remove(image_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)

@app.get("/videos", response_model=List[dict])
async def list_videos():
    videos = []
    for dir_path in glob.glob(os.path.join(RESULTS_DIR, "*")):
        if os.path.isdir(dir_path):
            dir_id = os.path.basename(dir_path)
            input_dir = os.path.join(dir_path, "input")
            
            # Find input files
            input_image = None
            input_audio = None
            if os.path.exists(input_dir):
                for file in os.listdir(input_dir):
                    file_path = os.path.join(input_dir, file)
                    encoded_file = quote(file)
                    if file.endswith(('.jpg', '.jpeg', '.png')):
                        input_image = f"/video/{dir_id}/input/{encoded_file}"
                    elif file.endswith(('.wav', '.mp3')):
                        input_audio = f"/video/{dir_id}/input/{encoded_file}"
            
            # Find output video
            video_files = [f for f in glob.glob(os.path.join(dir_path, "*.mp4")) 
                         if "##" in os.path.basename(f)]
            
            for video_file in video_files:
                video_name = os.path.basename(video_file)
                encoded_video_name = quote(video_name)
                videos.append({
                    "id": dir_id,
                    "name": video_name,
                    "display_name": video_name.split("##")[0],  # For display purposes
                    "url": f"/video/{dir_id}/output/{encoded_video_name}",
                    "input_image": input_image,
                    "input_audio": input_audio,
                    "timestamp": os.path.getctime(video_file)
                })
    
    # Sort by timestamp, newest first
    videos.sort(key=lambda x: x["timestamp"], reverse=True)
    return videos

@app.get("/video/{dir_id}/{type}/{file_name:path}")  # Added :path to handle special characters
async def get_input_file(dir_id: str, type: str, file_name: str):
    if type == "input":
        file_path = os.path.join(RESULTS_DIR, dir_id, "input", file_name)
    elif type == "output":
        file_path = os.path.join(RESULTS_DIR, dir_id, file_name)
    else:
        raise HTTPException(status_code=400, detail="Invalid type. Must be 'input' or 'output'")
        
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
    media_type = None
    if file_name.endswith(('.jpg', '.jpeg', '.png')):
        media_type = f"image/{file_name.split('.')[-1]}"
    elif file_name.endswith('.wav'):
        media_type = "audio/wav"
    elif file_name.endswith('.mp3'):
        media_type = "audio/mpeg"
    elif file_name.endswith('.mp4'):
        media_type = "video/mp4"
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")
        
    return FileResponse(file_path, media_type=media_type) 