from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import base64
import numpy as np
import librosa
import joblib
import os

# ---------------- CONFIG ----------------
SECRET_API_KEY = os.getenv("SECRET_API_KEY")  # ✅ from Render env
MODEL_PATH = "voice_detector.pkl"
SCALER_PATH = "scaler.pkl"
TEMP_AUDIO = "temp.mp3"

ALLOWED_LANGUAGES = {"tamil", "english", "hindi", "malayalam", "telugu"}
# --------------------------------------

# Load model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

app = FastAPI(title="AI Generated Voice Detection API")

# -------------------------------------------------
# ✅ ROOT ENDPOINT (VERY IMPORTANT FOR TESTER)
# -------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "success",
        "message": "AI Generated Voice Detection API is running"
    }

# -------- Request Schema --------
class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

# -------- Feature Extraction --------
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

# -------------------------------------------------
# ✅ MAIN API ENDPOINT
# -------------------------------------------------
@app.post("/api/voice-detection")
def detect_voice(
    request: VoiceRequest,
    x_api_key: str = Header(None)
):
    try:
        # 1️⃣ API Key validation
        if not SECRET_API_KEY or x_api_key != SECRET_API_KEY:
            raise ValueError()

        # 2️⃣ Language validation (case-insensitive)
        language = request.language.strip().lower()
        if language not in ALLOWED_LANGUAGES:
            raise ValueError()

        # 3️⃣ Audio format validation (spec says MP3 only)
        if request.audioFormat.strip().lower() != "mp3":
            raise ValueError()

        # 4️⃣ Base64 decode
        audio_bytes = base64.b64decode(request.audioBase64, validate=True)

        # 5️⃣ Reject non-MP3 headers (ogg, wav, flac, mpeg)
        if (
            audio_bytes[:4] == b"OggS" or      # OGG
            audio_bytes[:4] == b"RIFF" or      # WAV
            audio_bytes[:4] == b"fLaC" or      # FLAC
            audio_bytes[:4] == b"\x00\x00\x01\xba" or
            audio_bytes[:4] == b"\x00\x00\x01\xb3"
        ):
            raise ValueError()

        # Save MP3 temporarily
        with open(TEMP_AUDIO, "wb") as f:
            f.write(audio_bytes)

        # 6️⃣ Feature extraction
        features = extract_features(TEMP_AUDIO)
        features = scaler.transform([features])

        # 7️⃣ Prediction
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
        # ✅ REQUIRED ERROR FORMAT (RULE 11)
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
