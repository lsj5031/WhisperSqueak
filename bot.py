import asyncio
import io
import json
import os
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
from aiogram import Bot, Dispatcher, F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from dotenv import load_dotenv
from pydub import AudioSegment

from sse_parser import SSEParser

load_dotenv()


# Job Queue System
class JobStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RETRY = "retry"
    FAILED = "failed"
    COMPLETED = "completed"
    ARCHIVED = "archived"


@dataclass
class Job:
    """Represents a transcription job with retry capabilities."""

    id: str
    user_id: int
    chat_id: int
    message_id: int
    file_id: str
    filename: str
    status: str = JobStatus.PENDING
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    retry_count: int = 0
    next_retry_at: float | None = None
    last_error: str | None = None
    status_message_id: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Job":
        return cls(**data)


# Retry configuration
MAX_RETRIES = 10  # Maximum retry attempts before archiving
BASE_RETRY_DELAY = 5  # Base delay in seconds (5s)
MAX_RETRY_DELAY = 3600  # Max delay (1 hour)
API_DOWN_THRESHOLD = 3600  # 1 hour - if API is down this long, mark as failed

# Transcription configuration
TELEGRAM_MESSAGE_LIMIT = 4000  # Telegram's max message length
MESSAGE_UPDATE_THROTTLE_SEC = 1.0  # Min interval between streaming message updates
PENDING_JOB_GRACE_PERIOD_SEC = 60  # Grace period before retry scheduler picks up pending jobs


def calculate_retry_delay(retry_count: int) -> float:
    """Calculate exponential backoff delay: 5s, 10s, 20s, 40s, ... up to 1 hour."""
    delay = BASE_RETRY_DELAY * (2**retry_count)
    return min(delay, MAX_RETRY_DELAY)


# Job storage paths
JOBS_FILE = Path(os.getenv("JOBS_FILE", "/app/jobs.json"))
ARCHIVE_FILE = Path(os.getenv("ARCHIVE_FILE", "/app/jobs_archive.json"))


def load_jobs() -> dict[str, Job]:
    """Load all jobs from persistent storage."""
    if JOBS_FILE.exists():
        try:
            data = json.loads(JOBS_FILE.read_text())
            return {k: Job.from_dict(v) for k, v in data.items()}
        except Exception as e:
            print(f"Failed to load jobs: {e}", flush=True)
    return {}


def save_jobs(jobs: dict[str, Job]) -> None:
    """Save all jobs to persistent storage."""
    try:
        data = {k: v.to_dict() for k, v in jobs.items()}
        JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
        JOBS_FILE.write_text(json.dumps(data, indent=2))
    except Exception as e:
        print(f"Failed to save jobs: {e}", flush=True)


def load_archive() -> list[dict]:
    """Load archived failed jobs."""
    if ARCHIVE_FILE.exists():
        try:
            return json.loads(ARCHIVE_FILE.read_text())
        except Exception:
            pass
    return []


def archive_job(job: Job, reason: str) -> None:
    """Archive a failed job for later inspection."""
    try:
        archive = load_archive()
        archive_entry = job.to_dict()
        archive_entry["archived_at"] = time.time()
        archive_entry["archive_reason"] = reason
        archive.append(archive_entry)
        ARCHIVE_FILE.parent.mkdir(parents=True, exist_ok=True)
        ARCHIVE_FILE.write_text(json.dumps(archive, indent=2))
        print(f"Archived job {job.id}: {reason}", flush=True)
    except Exception as e:
        print(f"Failed to archive job: {e}", flush=True)


# Global job storage
_jobs: dict[str, Job] = {}
_api_first_failure_time: float | None = None
_api_healthy: bool = True

# Persistent storage for user settings
SETTINGS_FILE = Path(os.getenv("SETTINGS_FILE", "/app/user_settings.json"))


def load_user_settings() -> dict:
    """Load all user settings from file."""
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text())
        except Exception:
            pass
    return {}


def save_user_settings(settings: dict) -> None:
    """Save all user settings to file."""
    try:
        SETTINGS_FILE.write_text(json.dumps(settings, indent=2))
    except Exception as e:
        print(f"Failed to save settings: {e}", flush=True)


DEFAULT_MODEL = "Systran/faster-whisper-large-v3"


def get_user_prefs(user_id: int) -> dict:
    """Get settings for a specific user."""
    settings = load_user_settings()
    return settings.get(str(user_id), {"model": DEFAULT_MODEL})


def set_user_prefs(user_id: int, model: str | None = None) -> None:
    """Update settings for a specific user."""
    settings = load_user_settings()
    uid = str(user_id)
    if uid not in settings:
        settings[uid] = {"model": DEFAULT_MODEL}
    if model is not None:
        settings[uid]["model"] = model
    save_user_settings(settings)


TOKEN = os.getenv("TELEGRAM_TOKEN")
WHISPER_URL = os.getenv("WHISPER_URL", "http://host.docker.internal:18000/v1")
WHISPER_API_KEY = os.getenv("WHISPER_API_KEY", "dummy")
ALLOWED_UIDS = [int(uid.strip()) for uid in os.getenv("ALLOWED_UIDS", "").split(",") if uid.strip()]

if not TOKEN:
    raise ValueError("TELEGRAM_TOKEN env var required")

if not ALLOWED_UIDS:
    raise ValueError("ALLOWED_UIDS env var required (comma-separated user IDs)")

bot = Bot(token=TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)
router = Router()
dp.include_router(router)


@dataclass
class AudioTask:
    """Represents a queued audio transcription task."""

    message: Message
    file_obj: Any
    filename: str
    job_id: str | None = None


# Task queue for sequential audio processing
audio_queue: asyncio.Queue[AudioTask] = asyncio.Queue()


def create_job(message: Message, file_obj: Any, filename: str) -> Job:
    """Create a new job and persist it."""
    job = Job(
        id=str(uuid.uuid4()),
        user_id=message.from_user.id,
        chat_id=message.chat.id,
        message_id=message.message_id,
        file_id=file_obj.file_id,
        filename=filename,
    )
    _jobs[job.id] = job
    save_jobs(_jobs)
    print(f"Created job {job.id} for {filename}", flush=True)
    return job


def update_job(job: Job, **kwargs) -> None:
    """Update job fields and persist."""
    for key, value in kwargs.items():
        if hasattr(job, key):
            setattr(job, key, value)
    job.updated_at = time.time()
    save_jobs(_jobs)


def complete_job(job: Job) -> None:
    """Mark job as completed and remove from active jobs."""
    job.status = JobStatus.COMPLETED
    job.updated_at = time.time()
    del _jobs[job.id]
    save_jobs(_jobs)
    print(f"Completed job {job.id}", flush=True)


def fail_job(job: Job, error: str, should_archive: bool = False) -> None:
    """Mark job as failed with error details."""
    job.status = JobStatus.FAILED
    job.last_error = error
    job.updated_at = time.time()

    if should_archive:
        archive_job(job, f"Max retries exceeded: {error}")
        job.status = JobStatus.ARCHIVED
        del _jobs[job.id]
    save_jobs(_jobs)
    print(f"Failed job {job.id}: {error}", flush=True)


def schedule_retry(job: Job, error: str) -> bool:
    """Schedule a job for retry with exponential backoff. Returns False if max retries exceeded."""
    job.retry_count += 1
    job.last_error = error

    if job.retry_count > MAX_RETRIES:
        fail_job(job, error, should_archive=True)
        return False

    delay = calculate_retry_delay(job.retry_count)
    job.next_retry_at = time.time() + delay
    job.status = JobStatus.RETRY
    job.updated_at = time.time()
    save_jobs(_jobs)

    print(f"Scheduled retry {job.retry_count}/{MAX_RETRIES} for job {job.id} in {delay:.0f}s", flush=True)
    return True


def get_pending_jobs() -> list[Job]:
    """Get jobs that are pending (stale) or ready for retry."""
    now = time.time()
    ready = []
    for job in _jobs.values():
        if job.status == JobStatus.PENDING:
            # Only pick up pending jobs if they've been pending longer than grace period
            # This prevents race conditions with the initial queueing
            if (now - job.created_at) > PENDING_JOB_GRACE_PERIOD_SEC:
                ready.append(job)
        elif job.status == JobStatus.RETRY and job.next_retry_at and job.next_retry_at <= now:
            ready.append(job)
    return sorted(ready, key=lambda j: j.created_at)


def get_in_progress_jobs() -> list[Job]:
    """Get jobs currently being processed."""
    return [j for j in _jobs.values() if j.status == JobStatus.IN_PROGRESS]


# Cache for available models
_available_models: list[str] = []

# Popular models to show first (in priority order)
POPULAR_MODELS = [
    "Systran/faster-whisper-large-v3",
    "Systran/faster-whisper-medium",
    "Systran/faster-whisper-small",
    "Systran/faster-whisper-base",
    "Systran/faster-whisper-tiny",
    "deepdml/faster-whisper-large-v3-turbo-ct2",
    "Systran/faster-distil-whisper-large-v3",
    "Systran/faster-whisper-large-v2",
]


class UserSettings(StatesGroup):
    selecting_model = State()


def is_allowed(user_id: int) -> bool:
    return user_id in ALLOWED_UIDS


async def fetch_available_models(force_refresh: bool = False) -> list[str]:
    """Fetch available models from Whisper server."""
    global _available_models
    if _available_models and not force_refresh:
        return _available_models

    try:
        async with httpx.AsyncClient() as http:
            resp = await http.get(f"{WHISPER_URL}/models", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            # OpenAI-compatible format: {"data": [{"id": "model-name"}, ...]}
            models = [m["id"] for m in data.get("data", [])]
            if models:
                _available_models = models
                return models
    except Exception as e:
        print(f"Failed to fetch models: {e}", flush=True)

    # Fallback to common models if nothing cached
    if not _available_models:
        _available_models = POPULAR_MODELS.copy()
    return _available_models


def get_model_by_index(index: int) -> str | None:
    """Get model name by index from cached list."""
    if 0 <= index < len(_available_models):
        return _available_models[index]
    return None


def get_filtered_models() -> list[tuple[int, str]]:
    """Get available models, with popular ones first."""
    result = []
    added = set()

    # Add popular models first (if available)
    for popular in POPULAR_MODELS:
        if popular in _available_models:
            idx = _available_models.index(popular)
            result.append((idx, popular))
            added.add(popular)

    # Add remaining models
    for idx, model in enumerate(_available_models):
        if model not in added:
            result.append((idx, model))

    return result


def get_user_model(user_id: int) -> str:
    """Get model from persistent storage, validating against available models."""
    prefs = get_user_prefs(user_id)
    model = prefs.get("model", DEFAULT_MODEL)

    # If we have available models and the selected one isn't in the list,
    # fallback to the first available model
    if _available_models and model not in _available_models:
        print(f"Model {model} not available, falling back to {_available_models[0]}", flush=True)
        return _available_models[0]

    return model


@router.message(Command("start"))
async def start_handler(message: Message, state: FSMContext) -> None:
    uid = message.from_user.id
    print(f"User {uid} started WhisperSqueak", flush=True)
    if not is_allowed(uid):
        await message.answer("Access denied. Contact admin.")
        return

    await message.answer(
        "Hi! I'm WhisperSqueak\n\n"
        "Send a voice message or audio file to transcribe.\n\n"
        "Commands:\n"
        "/model - Select transcription model\n"
        "/help - Show help"
    )


@router.message(Command("help"))
async def help_handler(message: Message, state: FSMContext) -> None:
    if not is_allowed(message.from_user.id):
        await message.answer("Access denied. Contact admin.")
        return

    await message.answer(
        "WhisperSqueak - Voice Transcription Bot\n\n"
        "How to use:\n"
        "1. Select a model with /model\n"
        "2. Send a voice message or audio file\n\n"
        "Supported formats: MP3, WAV, OGG, M4A, FLAC, etc."
    )


@router.message(Command("model"))
async def model_command(message: Message, state: FSMContext) -> None:
    if not is_allowed(message.from_user.id):
        await message.answer("Access denied. Contact admin.")
        return

    status = await message.answer("Fetching available models...")
    await fetch_available_models(force_refresh=True)

    # Get user's current model
    prefs = get_user_prefs(message.from_user.id)
    current_model = prefs.get("model", DEFAULT_MODEL)

    # Get popular models with their indices
    filtered = get_filtered_models()

    buttons = []
    for idx, model in filtered:
        # Shorten display name
        display = model.split("/")[-1] if "/" in model else model
        # Mark current selection
        if model == current_model:
            display = f"✓ {display}"
        # Use short index-based callback data (e.g., "m:0")
        buttons.append([InlineKeyboardButton(text=display, callback_data=f"m:{idx}")])

    if not buttons:
        await status.edit_text("No models available. Check server connection.")
        return

    keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)
    await status.edit_text("Choose a model:", reply_markup=keyboard)
    await state.set_state(UserSettings.selecting_model)


@router.callback_query(F.data.startswith("m:"))
async def process_model(callback: CallbackQuery, state: FSMContext) -> None:
    if not is_allowed(callback.from_user.id):
        await callback.answer("Access denied.")
        return

    try:
        idx = int(callback.data.split(":", 1)[1])
        model = get_model_by_index(idx)
        if not model:
            await callback.answer("Model not found.")
            return
    except ValueError:
        await callback.answer("Invalid selection.")
        return

    # Save to persistent storage
    set_user_prefs(callback.from_user.id, model=model)

    model_short = model.split("/")[-1] if "/" in model else model
    await callback.message.edit_text(f"Model set to: {model_short}\n\nSend a voice or audio file to transcribe!")
    await state.set_state(None)
    await callback.answer()


@router.message(F.voice)
async def voice_handler(message: Message, state: FSMContext) -> None:
    print(f"Received voice from {message.from_user.id}", flush=True)
    if not is_allowed(message.from_user.id):
        await message.answer("Access denied. Contact admin.")
        return

    filename = f"{message.voice.file_id}.ogg"
    job = create_job(message, message.voice, filename)
    task = AudioTask(message=message, file_obj=message.voice, filename=filename, job_id=job.id)
    await audio_queue.put(task)
    print(f"Queued voice task (job: {job.id}), queue size: {audio_queue.qsize()}", flush=True)


@router.message(F.audio)
async def audio_handler(message: Message, state: FSMContext) -> None:
    print(f"Received audio from {message.from_user.id}: {message.audio.file_name}", flush=True)
    if not is_allowed(message.from_user.id):
        await message.answer("Access denied. Contact admin.")
        return

    audio = message.audio
    filename = audio.file_name or f"{audio.file_id}.mp3"
    job = create_job(message, audio, filename)
    task = AudioTask(message=message, file_obj=audio, filename=filename, job_id=job.id)
    await audio_queue.put(task)
    print(f"Queued audio task (job: {job.id}), queue size: {audio_queue.qsize()}", flush=True)


@router.message(F.document)
async def document_handler(message: Message, state: FSMContext) -> None:
    doc = message.document
    mime = doc.mime_type or ""
    print(f"Received document from {message.from_user.id}: {doc.file_name} (mime: {mime})", flush=True)

    if not is_allowed(message.from_user.id):
        await message.answer("Access denied. Contact admin.")
        return

    # Accept audio/* and common audio extensions
    is_audio = mime.startswith("audio/") or mime in ("video/mp4", "application/octet-stream")
    if doc.file_name:
        ext = doc.file_name.lower().split(".")[-1]
        is_audio = is_audio or ext in ("mp3", "wav", "ogg", "m4a", "flac", "aac", "wma", "opus")

    if not is_audio:
        print(f"Skipping non-audio document: {mime}", flush=True)
        return

    filename = doc.file_name or f"{doc.file_id}.audio"
    job = create_job(message, doc, filename)
    task = AudioTask(message=message, file_obj=doc, filename=filename, job_id=job.id)
    await audio_queue.put(task)
    print(f"Queued document task (job: {job.id}), queue size: {audio_queue.qsize()}", flush=True)


class APIError(Exception):
    """Custom exception for API errors that may warrant retry."""

    def __init__(self, message: str, is_retryable: bool = True):
        super().__init__(message)
        self.is_retryable = is_retryable


async def check_api_health() -> bool:
    """Check if the Whisper API is responsive."""
    global _api_healthy, _api_first_failure_time

    try:
        async with httpx.AsyncClient(timeout=10) as http:
            resp = await http.get(f"{WHISPER_URL}/models")
            if resp.status_code == 200:
                _api_healthy = True
                _api_first_failure_time = None
                return True
    except Exception as e:
        print(f"API health check failed: {e}", flush=True)

    # Track first failure time
    if _api_first_failure_time is None:
        _api_first_failure_time = time.time()

    _api_healthy = False
    return False


def is_api_down_too_long() -> bool:
    """Check if API has been down longer than threshold (1 hour)."""
    if _api_first_failure_time is None:
        return False
    return (time.time() - _api_first_failure_time) >= API_DOWN_THRESHOLD


async def transcribe_audio_chunk(
    audio_chunk: AudioSegment,
    model: str,
) -> AsyncGenerator[str]:
    """Transcribe a single audio chunk using SSE streaming and yield accumulated text.

    Expected SSE Protocol from Backend:
    ----------------------------------
    The backend should send Server-Sent Events (SSE) with the following format:

    Content-Type: text/event-stream

    Events:
    - Progress updates: `data: {"data": "transcribed text segment"}`
      OR raw text:      `data: transcribed text segment`
    - Completion:       `data: [DONE]`
    - Errors:           `data: [Error: error message here]`

    Example stream:
        data: {"data": "Hello, "}
        data: {"data": "world!"}
        data: [DONE]

    OR (for backends like glm-asr that send raw text):
        data: Hello,
        data: world!
        data: [DONE]

    Notes:
    - Each event is on its own line, prefixed with "data: "
    - Events are separated by newlines
    - JSON format is preferred but raw text is accepted as fallback
    - The function yields accumulated text (not incremental), so callers
      can display the full transcription at any point
    """
    global _api_healthy, _api_first_failure_time

    mp3_buffer = io.BytesIO()
    audio_chunk.export(mp3_buffer, format="mp3")
    mp3_buffer.seek(0)

    form_data = {"model": model, "stream": "true"}

    try:
        async with httpx.AsyncClient(timeout=300) as http:
            async with http.stream(
                "POST",
                f"{WHISPER_URL}/audio/transcriptions",
                headers={"Authorization": f"Bearer {WHISPER_API_KEY}"},
                data=form_data,
                files={"file": ("audio.mp3", mp3_buffer, "audio/mpeg")},
            ) as resp:
                resp.raise_for_status()

                # API is healthy
                _api_healthy = True
                _api_first_failure_time = None

                # Use SSEParser to handle the streaming response
                parser = SSEParser()

                async for chunk in resp.aiter_text():
                    results = parser.feed(chunk)

                    for result in results:
                        if result.error:
                            raise APIError(f"Transcription error: {result.error}", is_retryable=True)

                        if result.is_done:
                            yield parser.get_text()
                            return

                        # Yield current accumulated text for streaming updates
                        if result.accumulated_text:
                            yield result.accumulated_text.strip()

                # Handle any remaining buffer content
                final_result = parser.finalize()
                if final_result:
                    if final_result.error:
                        raise APIError(f"Transcription error: {final_result.error}", is_retryable=True)
                    if not final_result.is_done and final_result.accumulated_text:
                        yield final_result.accumulated_text.strip()

                yield parser.get_text()

    except httpx.ConnectError as e:
        raise APIError(f"Connection failed: {e}", is_retryable=True) from e
    except httpx.TimeoutException as e:
        raise APIError(f"Request timeout: {e}", is_retryable=True) from e
    except httpx.HTTPStatusError as e:
        # 5xx errors are retryable, 4xx are not (except 429)
        if e.response.status_code >= 500 or e.response.status_code == 429:
            raise APIError(f"Server error {e.response.status_code}: {e}", is_retryable=True) from e
        else:
            raise APIError(f"Client error {e.response.status_code}: {e}", is_retryable=False) from e
    except APIError:
        raise
    except Exception as e:
        raise APIError(f"Unexpected error: {e}", is_retryable=True) from e


async def process_audio(message: Message, file_obj: Any, filename: str, job: Job | None = None) -> None:
    uid = message.from_user.id
    print(f"Processing audio for user {uid}: {filename}", flush=True)

    model = get_user_model(uid)
    print(f"Settings - model: {model}", flush=True)

    model_short = model.split("/")[-1] if "/" in model else model

    # Create or retrieve status message
    retry_info = ""
    if job and job.retry_count > 0:
        retry_info = f"\nRetry attempt: {job.retry_count}/{MAX_RETRIES}"

    if job and job.status_message_id:
        # Try to edit existing status message
        try:
            status_msg = await bot.edit_message_text(
                f"Processing: {filename}\n" f"Model: {model_short}{retry_info}",
                chat_id=message.chat.id,
                message_id=job.status_message_id,
            )

            # Create a mock message object with the message_id for editing
            class StatusMessage:
                def __init__(self, chat_id: int, message_id: int):
                    self.chat = type("obj", (object,), {"id": chat_id})()
                    self.message_id = message_id

                async def edit_text(self, text: str):
                    await bot.edit_message_text(text, chat_id=self.chat.id, message_id=self.message_id)

            status_msg = StatusMessage(message.chat.id, job.status_message_id)
        except Exception:
            # Message might be deleted, create new one
            status_msg = await message.reply(
                f"Processing: {filename}\n" f"Model: {model_short}{retry_info}",
            )
            if job:
                update_job(job, status_message_id=status_msg.message_id)
    else:
        status_msg = await message.reply(
            f"Processing: {filename}\n" f"Model: {model_short}{retry_info}",
        )
        if job:
            update_job(job, status_message_id=status_msg.message_id)

    # Mark job as in progress
    if job:
        update_job(job, status=JobStatus.IN_PROGRESS)

    tg_file = await bot.get_file(file_obj.file_id)
    file_path = Path(f"/tmp/{tg_file.file_id}")
    await bot.download_file(tg_file.file_path, file_path)

    try:
        audio = AudioSegment.from_file(file_path)
        duration_sec = len(audio) / 1000
        print(f"Audio duration: {duration_sec:.1f}s", flush=True)

        # Track all messages for multi-message streaming
        all_messages = [status_msg]  # List of message objects
        finalized_length = 0  # Length of text in completed (full) messages
        last_update_time = 0
        full_text = ""  # Complete accumulated text

        async for partial_text in transcribe_audio_chunk(audio, model):
            full_text = partial_text
            # Calculate what goes in current message (text after finalized portion)
            current_msg_text = partial_text[finalized_length:]

            # Check if we need to split into a new message
            while len(current_msg_text) > TELEGRAM_MESSAGE_LIMIT:
                # Find split point at word boundary
                split_idx = current_msg_text.rfind(" ", 0, TELEGRAM_MESSAGE_LIMIT)
                if split_idx == -1:
                    split_idx = TELEGRAM_MESSAGE_LIMIT

                # Finalize the current message with the first part
                finalized_chunk = current_msg_text[:split_idx]
                try:
                    await all_messages[-1].edit_text(finalized_chunk)
                except TelegramBadRequest:
                    pass

                # Track how much text is now finalized
                finalized_length += split_idx

                # Get remaining text for new message
                current_msg_text = current_msg_text[split_idx:].lstrip()
                # Recalculate finalized_length based on current position in full text.
                # After lstrip(), we may have removed whitespace, so we can't just add
                # split_idx to finalized_length. Instead, compute how much of partial_text
                # is NOT in current_msg_text - that's everything that's been finalized.
                finalized_length = len(partial_text) - len(current_msg_text)

                # Create new message for overflow
                if current_msg_text:
                    try:
                        new_msg = await message.reply(current_msg_text + " ⏳")
                        all_messages.append(new_msg)
                    except TelegramBadRequest as e:
                        # Failed to create new message, log and continue with existing messages
                        print(f"Failed to create new message for overflow: {e}", flush=True)

            # Throttle updates on current message
            now = time.time()
            if now - last_update_time >= MESSAGE_UPDATE_THROTTLE_SEC and current_msg_text:
                try:
                    # Show streaming indicator on current message
                    display_text = (
                        current_msg_text + " ⏳"
                        if len(current_msg_text) < TELEGRAM_MESSAGE_LIMIT - 10
                        else current_msg_text
                    )
                    await all_messages[-1].edit_text(display_text)
                except TelegramBadRequest:
                    pass  # Message not modified or other Telegram API error
                last_update_time = now

        # Final result handling - get final current message text
        final_current_text = full_text[finalized_length:] if full_text else ""

        if finalized_length == 0 and not final_current_text.strip():
            await all_messages[-1].edit_text("No speech detected.")
        elif final_current_text.strip():
            # Update the last message with final text (remove streaming indicator)
            try:
                await all_messages[-1].edit_text(final_current_text.strip())
            except TelegramBadRequest:
                pass  # Ignore if message not modified
        elif finalized_length > 0 and len(all_messages) > 1:
            # Edge case: transcription exactly filled previous message(s), leaving
            # current message empty or whitespace-only. The last message was created
            # with overflow text + streaming indicator, but after lstrip the final
            # segment is empty. Delete the empty trailing message.
            try:
                await bot.delete_message(chat_id=all_messages[-1].chat.id, message_id=all_messages[-1].message_id)
            except TelegramBadRequest:
                pass  # Message already deleted or can't be deleted

        # Job completed successfully
        if job:
            complete_job(job)

    except APIError as e:
        print(f"API error processing audio: {e}", flush=True)
        if job:
            if e.is_retryable:
                # Check if API has been down too long
                if is_api_down_too_long():
                    fail_job(job, f"API down for over 1 hour: {e}", should_archive=True)
                    await status_msg.edit_text(
                        f"Transcription failed - API unavailable for over 1 hour.\n"
                        f"Job archived. Error: {str(e)[:200]}"
                    )
                elif schedule_retry(job, str(e)):
                    delay = calculate_retry_delay(job.retry_count)
                    await status_msg.edit_text(
                        f"API error, will retry in {format_delay(delay)}.\n"
                        f"Attempt {job.retry_count}/{MAX_RETRIES}\n"
                        f"Error: {str(e)[:200]}"
                    )
                else:
                    # Max retries exceeded
                    await status_msg.edit_text(
                        f"Transcription failed after {MAX_RETRIES} attempts.\n"
                        f"Job archived. Last error: {str(e)[:200]}"
                    )
            else:
                fail_job(job, str(e), should_archive=True)
                await status_msg.edit_text(f"Transcription failed (non-retryable error).\n" f"Error: {str(e)[:200]}")
        else:
            await status_msg.edit_text(f"Error processing {filename}:\n{str(e)}")
        raise  # Re-raise for the worker to handle

    except Exception as e:
        print(f"Error processing audio: {e}", flush=True)
        if job:
            if schedule_retry(job, str(e)):
                delay = calculate_retry_delay(job.retry_count)
                await status_msg.edit_text(
                    f"Error occurred, will retry in {format_delay(delay)}.\n"
                    f"Attempt {job.retry_count}/{MAX_RETRIES}\n"
                    f"Error: {str(e)[:200]}"
                )
            else:
                await status_msg.edit_text(
                    f"Transcription failed after {MAX_RETRIES} attempts.\n" f"Job archived. Last error: {str(e)[:200]}"
                )
        else:
            await status_msg.edit_text(f"Error processing {filename}:\n{str(e)}")
        raise  # Re-raise for the worker to handle

    finally:
        file_path.unlink(missing_ok=True)


def format_delay(seconds: float) -> str:
    """Format delay in human-readable format."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds / 60)}m {int(seconds % 60)}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"


async def audio_worker() -> None:
    """Worker that processes audio tasks from the queue sequentially."""
    print("Audio worker started", flush=True)
    while True:
        task = await audio_queue.get()
        try:
            print(f"Processing queued task: {task.filename} (job: {task.job_id})", flush=True)

            # Get job if exists
            job = _jobs.get(task.job_id) if task.job_id else None

            await process_audio(task.message, task.file_obj, task.filename, job)
        except (APIError, Exception) as e:
            # Errors are already handled in process_audio with retry scheduling
            print(f"Error in audio worker (handled): {e}", flush=True)
        finally:
            audio_queue.task_done()


async def retry_scheduler() -> None:
    """Background task that checks for jobs ready for retry and re-queues them."""
    print("Retry scheduler started", flush=True)

    while True:
        await asyncio.sleep(5)  # Check every 5 seconds

        try:
            pending_jobs = get_pending_jobs()

            for job in pending_jobs:
                # Skip if already in queue or being processed
                if job.status == JobStatus.IN_PROGRESS:
                    continue

                print(f"Re-queuing job {job.id} for retry (attempt {job.retry_count})", flush=True)

                # Create a mock message for re-processing
                # We need to reconstruct enough context to process the job
                class MockMessage:
                    def __init__(self, job: Job):
                        self.from_user = type("obj", (object,), {"id": job.user_id})()
                        self.chat = type("obj", (object,), {"id": job.chat_id})()
                        self.message_id = job.message_id

                    async def reply(self, text: str):
                        return await bot.send_message(self.chat.id, text, reply_to_message_id=self.message_id)

                class MockFileObj:
                    def __init__(self, file_id: str):
                        self.file_id = file_id

                mock_message = MockMessage(job)
                mock_file = MockFileObj(job.file_id)

                task = AudioTask(
                    message=mock_message,
                    file_obj=mock_file,
                    filename=job.filename,
                    job_id=job.id,
                )
                await audio_queue.put(task)

        except Exception as e:
            print(f"Error in retry scheduler: {e}", flush=True)


async def recover_jobs() -> None:
    """Recover pending and in-progress jobs on bot restart."""
    global _jobs

    _jobs = load_jobs()
    recovered = 0

    for job_id, job in list(_jobs.items()):
        # Reset in-progress jobs to pending (they were interrupted)
        if job.status == JobStatus.IN_PROGRESS:
            job.status = JobStatus.RETRY
            job.retry_count += 1
            if job.retry_count > MAX_RETRIES:
                archive_job(job, "Max retries exceeded after restart recovery")
                del _jobs[job_id]
                continue
            job.next_retry_at = time.time() + calculate_retry_delay(job.retry_count)
            print(f"Reset interrupted job {job.id} to retry", flush=True)

        # Jobs in retry status will be picked up by retry_scheduler
        if job.status in (JobStatus.PENDING, JobStatus.RETRY):
            recovered += 1
            print(f"Recovered job {job.id} ({job.filename}) - status: {job.status}", flush=True)

    save_jobs(_jobs)
    print(f"Recovered {recovered} jobs from previous session", flush=True)


async def main() -> None:
    print("WhisperSqueak is starting...", flush=True)

    # Recover any pending jobs from previous session
    await recover_jobs()

    # Pre-fetch models
    models = await fetch_available_models()
    print(f"Available models: {models}", flush=True)

    # Start audio worker for sequential processing
    asyncio.create_task(audio_worker())

    # Start retry scheduler for exponential backoff retries
    asyncio.create_task(retry_scheduler())

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
