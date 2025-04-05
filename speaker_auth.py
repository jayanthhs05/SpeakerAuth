import os
from torch import mean
import numpy as np
import torchaudio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from speechbrain.inference import SpeakerRecognition
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn
from fastapi.encoders import jsonable_encoder

app = FastAPI()
verification_system = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/"
)
THRESHOLD = 0.6
SAMPLE_RATE = 16000
USERS_DIR = "users/"
os.makedirs(USERS_DIR, exist_ok=True)

class EnrollRequest(BaseModel):
    user_id: str
    audio_path: str

class VerifyRequest(BaseModel):
    user_id: str
    audio_path: str

def load_and_process_audio(file_path):
    try:
        waveform, orig_freq = torchaudio.load(file_path)
        if waveform.shape[0] > 1:
            waveform = mean(waveform, dim=0, keepdim=True)
        if orig_freq != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, orig_freq, SAMPLE_RATE)
        return waveform
    except Exception as e:
        print(e)

def save_embedding(user_id, embedding):
    np.save(os.path.join(USERS_DIR, f"{user_id}.npy"), embedding)

@app.post("/enroll")
async def enroll_user(request: EnrollRequest):
    if os.path.exists(os.path.join(USERS_DIR, f"{request.user_id}.npy")):
        raise HTTPException(400, "User exists")
    try:
        waveform = load_and_process_audio(request.audio_path)
        embedding = verification_system.encode_batch(waveform)
        save_embedding(request.user_id, embedding.detach().numpy())
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/verify")
async def verify_user(request: VerifyRequest):
    profile_path = os.path.join(USERS_DIR, f"{request.user_id}.npy")
    if not os.path.exists(profile_path):
        raise HTTPException(404, "User not found")
    try:
        enrolled_embedding = np.load(profile_path).squeeze()
        waveform = load_and_process_audio(request.audio_path)
        new_embedding = verification_system.encode_batch(waveform).squeeze().detach().numpy()
        enrolled_embedding = enrolled_embedding.reshape(1, -1) if enrolled_embedding.ndim == 1 else enrolled_embedding
        new_embedding = new_embedding.reshape(1, -1) if new_embedding.ndim == 1 else new_embedding
        similarity_score = cosine_similarity(enrolled_embedding, new_embedding)[0][0]
        is_authenticated = (similarity_score >= THRESHOLD)
        return jsonable_encoder({
            "authenticated": bool(is_authenticated),
            "similarity_score": float(similarity_score),
            "threshold": float(THRESHOLD)
        })
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
