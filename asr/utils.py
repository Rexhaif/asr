import logging
from asyncio import subprocess as async_subprocess

import fastapi as ft
import numpy as np
import whisperx as wx
from rich.logging import RichHandler

from .config import settings


def setup_logging():

    handler = RichHandler(
        rich_tracebacks=True,
        tracebacks_show_locals=True,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[handler],
    )
    for name in {
        "httpx",
    }:
        logging.getLogger(name).setLevel(logging.WARNING)

    for name in {
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
    }:
        logging.getLogger(name).handlers.clear()
        logging.getLogger(name).propagate = False
        logging.getLogger(name).addHandler(handler)


class UnsupportedLanguageException(ft.HTTPException):

    def __init__(self, language: str):
        super().__init__(
            status_code=400, detail=f"Language {language} is not supported"
        )


async def convert_audio(audio_binary: bytes) -> np.ndarray:
    command = [
        "ffmpeg",
        "-i",
        "pipe:0",
        "-threads",
        "1",
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "pipe:1",
    ]

    process = await async_subprocess.create_subprocess_shell(
        cmd=" ".join(command),
        stdout=async_subprocess.PIPE,
        stderr=async_subprocess.PIPE,
        stdin=async_subprocess.PIPE,
    )

    stdout, stderr = await process.communicate(input=audio_binary)

    if process.returncode != 0:
        raise Exception(stderr.decode())

    return np.frombuffer(stdout, np.int16).flatten().astype(np.float32) / 32768.0
