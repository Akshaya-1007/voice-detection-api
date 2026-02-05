from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import numpy as np
import librosa
import joblib
import os

# ---------------- CONFIG ----------------
SECRET_API_KEY = os.getenv("SECRET_API_KEY")
MODEL_PATH = "voice_detector.pkl"
SCALER_PATH = "scaler.pkl"
TEMP_AUDIO = "temp.mp3"

ALLOWED_LANGUAGES = {"tamil", "english", "hindi", "malayalam", "telugu"}
# --------------------------------------

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

app = FastAPI(title="AI Generated Voice Detection API")

class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)

    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rms = np.mean(librosa.feature.rms(y=y))

    return np.hstack([
        mfccs_mean,
        zcr,
        centroid,
        rolloff,
        bandwidth,
        rms
    ])

@app.post("/api/voice-detection")
def detect_voice(
    request: VoiceRequest,
    x_api_key: str = Header(None)
):
    try:
        # API key
        if x_api_key != SECRET_API_KEY:
            raise ValueError()

        # Language validation
        if request.language.strip().lower() not in ALLOWED_LANGUAGES:
            raise ValueError()

        # Format field validation
        if request.audioFormat.strip().lower() != "mp3":
            raise ValueError()

        # Base64 decode
        audio_bytes = base64.b64decode(request.audioBase64, validate=True)

        # 🚫 Reject non-MP3 formats safely
        if (
            audio_bytes[:4] == b"OggS" or
            audio_bytes[:4] == b"RIFF" or
            audio_bytes[:4] == b"fLaC" or
            audio_bytes[:4] == b"\x00\x00\x01\xba" or
            audio_bytes[:4] == b"\x00\x00\x01\xb3"
        ):
            raise ValueError()

        with open(TEMP_AUDIO, "wb") as f:
            f.write(audio_bytes)

        features = extract_features(TEMP_AUDIO)
        features = scaler.transform([features])

        pred = model.predict(features)[0]
        prob = model.predict_proba(features).max()

        if pred == 1:
            classification = "AI_GENERATED"
            explanation = "Unnatural pitch consistency and synthetic spectral patterns detected"
        else:
            classification = "HUMAN"
            explanation = "Natural pitch variation and human-like spectral characteristics detected"

        return {
            "status": "success",
            "language": request.language,
            "classification": classification,
            "confidenceScore": round(float(prob), 2),
            "explanation": explanation
        }

    except Exception:
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "message": "Invalid API key or malformed request"
            }
        )

    finally:
        if os.path.exists(TEMP_AUDIO):
            os.remove(TEMP_AUDIO)
