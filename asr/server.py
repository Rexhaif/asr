import asyncio
import logging
import time
from contextlib import asynccontextmanager
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import fastapi as ft
import numpy as np
import whisperx as wx
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
import sentry_sdk

from .config import settings
from .utils import UnsupportedLanguageException, LimitUploadSize, convert_audio, setup_logging

setup_logging()

logger = logging.getLogger(__name__)

# == Global variables ==
wav2vec_models = {}
whisper_model = None
diarization_model = None
executor = None
# ======================

def load_models():
    global whisper_model, wav2vec_models, diarization_model
    if whisper_model is None:
        logger.info(
            f"Loading Whisper model: {settings.whisper_arch} for {settings.device} device"
        )
        whisper_model = wx.load_model(
            whisper_arch=settings.whisper_arch,
            device=settings.device,
            compute_type=settings.compute_type,
            download_root="./whisper-models",
        )

    for lang_code in settings.wav2vec_langs.split(","):
        if lang_code not in wav2vec_models:
            logger.info(
                f"Loading Wav2Vec model: {lang_code} for {settings.device} device"
            )
            model, metadata = wx.load_align_model(
                language_code=lang_code,
                device=settings.device,
                model_dir="./wav2vec-models",
            )
            wav2vec_models[lang_code] = (model, metadata)

    if diarization_model is None:
        logger.info(f"Loading diarization model for {settings.device} device")
        diarization_model = wx.DiarizationPipeline(
            device=settings.device, use_auth_token=settings.hub_token
        )


@asynccontextmanager
async def lifespan_fn(app: ft.FastAPI):
    global executor
    executor = ProcessPoolExecutor(
        max_workers=1, 
        initializer=load_models
    )
    def shutdown_fn():
        logger.info("Shutting down executor")
        executor.shutdown(wait=False)

    try:
        yield
    except asyncio.exceptions.CancelledError as e:
        logger.info(f"Silenced exception: {str(e)}")
        pass

    shutdown_fn()

if settings.use_sentry:
    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        enable_tracing=True,
        traces_sample_rate=1.0,
        profiles_sample_rate=1.0,
    )

app = ft.FastAPI(
    title="WhisperX ASR API",
    lifespan=lifespan_fn,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    GZipMiddleware,
    compresslevel=9,
)
app.add_middleware(
    LimitUploadSize,
    max_upload_size=settings.max_size_mb * 1024 * 1024,
)


def whisperx_predict(
    audio_input: np.ndarray,
    language_code: str | None = None,
    min_speakers: int = 1,
    max_speakers: int | None = None,
) -> dict[str, any]:

    if language_code is not None:
        if language_code not in wav2vec_models:
            raise UnsupportedLanguageException(language=language_code)

    logger.info("Transcribing audio...")
    transcription = whisper_model.transcribe(
        audio=audio_input,
        batch_size=settings.batch_size,
        language=language_code,
        num_workers=0,
    )

    if transcription["language"] not in wav2vec_models:
        raise UnsupportedLanguageException(language=transcription["language"])

    model, metadata = wav2vec_models[transcription["language"]]
    logger.info(f"Aligning transcript for {transcription['language']}...")
    transcription = wx.align(
        transcript=transcription["segments"],
        model=model,
        align_model_metadata=metadata,
        audio=audio_input,
        device=settings.device,
        return_char_alignments=False,
    )

    logger.info("Diarizing transcript...")
    diarize_segments = diarization_model(
        audio_input, min_speakers=min_speakers, max_speakers=max_speakers
    )

    transcription = wx.assign_word_speakers(diarize_segments, transcription)

    return transcription


@app.post("/asr")
async def asr(
    request: ft.Request,
    language_code: str | None = None,
    min_speakers: int = 1,
    max_speakers: int | None = None,
):

    loop = asyncio.get_running_loop()
    t1 = time.perf_counter()
    input_data: bytes = await request.body()
    t2 = time.perf_counter()
    logger.info(
        f"Received audio data: length={(len(input_data) / 1024 / 1024):.4f} MB / speed={len(input_data) / (t2-t1) / 1024 / 1024:.4f} MB/s"
    )
    input_tensor: np.ndarray = await convert_audio(audio_binary=input_data)

    fn = partial(
        whisperx_predict,
        audio_input=input_tensor,
        language_code=language_code,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    t1 = time.perf_counter()
    transcription = await loop.run_in_executor(
        executor=executor, 
        func=fn
    )
    t2 = time.perf_counter()
    logger.info(f"Took {t2-t1:.3f} seconds to process the audio")

    return transcription
