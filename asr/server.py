import whisperx as wx
import fastapi as ft
from fastapi.middleware.cors import CORSMiddleware
from config import settings

app = ft.FastAPI(
    title="WhisperX ASR API"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

whisper_model = None
wav2vec_models: dict[str, any] = {}
diarization_model = None

def load_models():
    global whisper_model, wav2vec_model, diarization_model

    if whisper_model is None:
        whisper_model = wx.load_model(
            whisper_arch=settings.whisper_arch,
            device=settings.device,
            compute_type=settings.compute_type
            download_root='./whisper-models'
        )

    for lang_code in settings.wav2vec_langs.split(","):
        if lang_code not in wav2vec_models:
            model, metadata = wx.load_align_model(
                lang_code=lang_code,
                device=settings.device,
                model_dir="./wav2vec-models"
            )

