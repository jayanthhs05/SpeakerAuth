from io import BytesIO
import os
from torch import mean
import numpy as np
import torchaudio
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from speechbrain.inference import SpeakerRecognition
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn
from fastapi.encoders import jsonable_encoder
import whisper

app = FastAPI()

# Enable CORS for cross-device communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
verification_system = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", 
    savedir="pretrained_models/"
)
model = whisper.load_model("medium")

# Configuration
THRESHOLD = 0.4
SAMPLE_RATE = 16000
USERS_DIR = "users/"
os.makedirs(USERS_DIR, exist_ok=True)

def process_audio(audio_data: bytes):
    """Process uploaded audio data into waveform"""
    try:
        audio_buffer = BytesIO(audio_data)
        waveform, orig_freq = torchaudio.load(audio_buffer)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = mean(waveform, dim=0, keepdim=True)
            
        # Resample if necessary
        if orig_freq != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform, 
                orig_freq, 
                SAMPLE_RATE
            )
            
        return waveform
    except Exception as e:
        raise HTTPException(400, f"Audio processing failed: {str(e)}")

def verify_speaker(profile_path: str, waveform):
    """Verify speaker against stored profile"""
    try:
        enrolled_embedding = np.load(profile_path)
        new_embedding = verification_system.encode_batch(waveform)
        new_embedding = new_embedding.squeeze().detach().numpy()
        
        # Calculate similarity score
        similarity_score = cosine_similarity(
            enrolled_embedding.reshape(1, -1),
            new_embedding.reshape(1, -1)
        )[0][0]
        
        return {
            "authenticated": bool(similarity_score >= THRESHOLD),
            "similarity_score": float(similarity_score),
            "threshold": float(THRESHOLD)
        }
    except Exception as e:
        raise HTTPException(500, f"Verification failed: {str(e)}")

@app.post("/enroll")
async def enroll_user(user_id: str = Form(...), file: UploadFile = File(...)):
    """Enroll a new user with voice sample"""
    if os.path.exists(os.path.join(USERS_DIR, f"{user_id}.npy")):
        raise HTTPException(400, "User already exists")
    
    try:
        # Read and process audio
        audio_data = await file.read()
        waveform = process_audio(audio_data)
        
        # Generate and save embedding
        embedding = verification_system.encode_batch(waveform)
        embedding = embedding.detach().numpy()
        np.save(os.path.join(USERS_DIR, f"{user_id}.npy"), embedding)
        
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/verify")
async def verify_user(user_id: str = Form(...), file: UploadFile = File(...)):
    profile_path = os.path.join(USERS_DIR, f"{user_id}.npy")
    if not os.path.exists(profile_path):
        raise HTTPException(404, "User not found")
    
    try:
        audio_data = await file.read()
        waveform = process_audio(audio_data)
        verification_result = verify_speaker(profile_path, waveform)
        audio_np = waveform.squeeze().numpy().astype(np.float32)
        result = model.transcribe(audio_np)
        verification_result["transcription"] = result["text"]
        
        return jsonable_encoder(verification_result)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/health")
async def health_check():
    """Service health check"""
    return {"status": "ok", "model_loaded": True}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
