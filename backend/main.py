from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import pytesseract
import cv2
import numpy as np
import io

app = FastAPI()

# Allow frontend (React Native) access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload/")
async def extract_text(file: UploadFile = File(...)):
    # Convert uploaded file to image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(image)
    
    # Preprocess with OpenCV
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    text = pytesseract.image_to_string(gray)

    return {"text": text}
