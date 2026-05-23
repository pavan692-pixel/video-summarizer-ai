from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import shutil
import os

from app.services.video_processor import (
    detect_scenes,
    create_summary_video
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def home():
    return {"message": "Video Summarizer API Running"}


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):

    file_path = f"{UPLOAD_DIR}/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    timestamps = detect_scenes(file_path)

    summary_path = create_summary_video(
        file_path,
        timestamps
    )

    return {
        "filename": file.filename,
        "scene_count": len(timestamps),
        "timestamps": timestamps[:10],
        "summary_video": summary_path
    }