from io import BytesIO
import os
import torch
import numpy as np
import torchaudio
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from speechbrain.inference import SpeakerRecognition
import uvicorn
from fastapi.encoders import jsonable_encoder
import whisper


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
THRESHOLD = 0.4
SAMPLE_RATE = 16000
USERS_DIR = "users/"
os.makedirs(USERS_DIR, exist_ok=True)

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


verification_system = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/",
    run_opts={"device": device},
).eval()

whisper_model = whisper.load_model("medium", device=device)
if device == "cuda":
    whisper_model = whisper_model.half()


def process_audio(audio_data: bytes) -> torch.Tensor:
    """Process uploaded audio data into waveform"""
    try:
        audio_buffer = BytesIO(audio_data)
        waveform, orig_freq = torchaudio.load(audio_buffer)

        waveform = torchaudio.functional.resample(
            waveform.mean(dim=0, keepdim=True),
            orig_freq=orig_freq,
            new_freq=SAMPLE_RATE,
        )

        return waveform.to(device, non_blocking=True)
    except Exception as e:
        raise HTTPException(400, f"Audio processing failed: {str(e)}")


def verify_speaker(profile_path: str, waveform: torch.Tensor) -> dict:
    try:
        enrolled_embedding = torch.load(profile_path, map_location=device)

        enrolled_embedding = enrolled_embedding.view(1, -1)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == "cuda")):
            new_embedding = verification_system.encode_batch(waveform)

        new_embedding = new_embedding.reshape(1, -1)

        similarity_score = torch.cosine_similarity(
            enrolled_embedding, new_embedding, dim=1
        ).item()

        return {
            "authenticated": similarity_score >= THRESHOLD,
            "similarity_score": similarity_score,
            "threshold": THRESHOLD,
        }   
    except Exception as e:
        raise HTTPException(500, f"Verification failed: {str(e)}")


@app.post("/enroll")
async def enroll_user(user_id: str = Form(...), file: UploadFile = File(...)):
    """Enroll a new user with voice sample"""
    profile_path = os.path.join(USERS_DIR, f"{user_id}.pt")
    # if os.path.exists(profile_path):
    #     raise HTTPException(400, "User already exists")

    try:
        audio_data = await file.read()
        waveform = process_audio(audio_data)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == "cuda")):
            embedding = verification_system.encode_batch(waveform)

        torch.save(embedding.cpu(), profile_path)
        torch.cuda.empty_cache()

        return {"status": "success"}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/verify")
async def verify_user(user_id: str = Form(...), file: UploadFile = File(...)):
    """Verify a user's voice sample against their enrolled profile"""
    profile_path = os.path.join(USERS_DIR, f"{user_id}.pt")
    if not os.path.exists(profile_path):
        raise HTTPException(404, "User not found")

    try:
        audio_data = await file.read()
        waveform = process_audio(audio_data)

        verification_result = verify_speaker(profile_path, waveform)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == "cuda")):
            audio_np = waveform.squeeze().cpu().numpy().astype(np.float32)
            transcription_result = whisper_model.transcribe(
                audio_np, fp16=(device == "cuda")
            )

        verification_result["transcription"] = transcription_result["text"]

        return jsonable_encoder(verification_result)
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        torch.cuda.empty_cache()


@app.get("/health")
async def health_check():
    """Service health check endpoint"""
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "device": device,
        "cuda_mem_allocated": (
            f"{torch.cuda.memory_allocated()/1e9:.2f} GB" if device == "cuda" else None
        ),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
