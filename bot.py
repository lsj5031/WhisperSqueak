import asyncio
import io
import json
import os
from pathlib import Path
from typing import Any

import httpx
from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, Message, CallbackQuery
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydub import AudioSegment

load_dotenv()

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
    return settings.get(str(user_id), {"model": DEFAULT_MODEL, "language": "auto"})


def set_user_prefs(user_id: int, model: str | None = None, language: str | None = None) -> None:
    """Update settings for a specific user."""
    settings = load_user_settings()
    uid = str(user_id)
    if uid not in settings:
        settings[uid] = {"model": DEFAULT_MODEL, "language": "auto"}
    if model is not None:
        settings[uid]["model"] = model
    if language is not None:
        settings[uid]["language"] = language
    save_user_settings(settings)

TOKEN = os.getenv("TELEGRAM_TOKEN")
WHISPER_URL = os.getenv("WHISPER_URL", "http://host.docker.internal:18000/v1")
WHISPER_API_KEY = os.getenv("WHISPER_API_KEY", "dummy")
ALLOWED_UIDS = [int(uid.strip()) for uid in os.getenv("ALLOWED_UIDS", "").split(",") if uid.strip()]

# Whisper supported languages (ISO 639-1 codes)
SUPPORTED_LANGUAGES = {
    "auto": "Auto-detect",
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "ko": "Korean",
    "fr": "French",
    "ja": "Japanese",
    "pt": "Portuguese",
    "tr": "Turkish",
    "pl": "Polish",
    "ca": "Catalan",
    "nl": "Dutch",
    "ar": "Arabic",
    "sv": "Swedish",
    "it": "Italian",
    "id": "Indonesian",
    "hi": "Hindi",
    "fi": "Finnish",
    "vi": "Vietnamese",
    "he": "Hebrew",
    "uk": "Ukrainian",
    "el": "Greek",
    "ms": "Malay",
    "cs": "Czech",
    "ro": "Romanian",
    "da": "Danish",
    "hu": "Hungarian",
    "ta": "Tamil",
    "no": "Norwegian",
    "th": "Thai",
    "ur": "Urdu",
    "hr": "Croatian",
    "bg": "Bulgarian",
    "lt": "Lithuanian",
    "la": "Latin",
    "mi": "Maori",
    "ml": "Malayalam",
    "cy": "Welsh",
    "sk": "Slovak",
    "te": "Telugu",
    "fa": "Persian",
    "lv": "Latvian",
    "bn": "Bengali",
    "sr": "Serbian",
    "az": "Azerbaijani",
    "sl": "Slovenian",
    "kn": "Kannada",
    "et": "Estonian",
    "mk": "Macedonian",
    "br": "Breton",
    "eu": "Basque",
    "is": "Icelandic",
    "hy": "Armenian",
    "ne": "Nepali",
    "mn": "Mongolian",
    "bs": "Bosnian",
    "kk": "Kazakh",
    "sq": "Albanian",
    "sw": "Swahili",
    "gl": "Galician",
    "mr": "Marathi",
    "pa": "Punjabi",
    "si": "Sinhala",
    "km": "Khmer",
    "sn": "Shona",
    "yo": "Yoruba",
    "so": "Somali",
    "af": "Afrikaans",
    "oc": "Occitan",
    "ka": "Georgian",
    "be": "Belarusian",
    "tg": "Tajik",
    "sd": "Sindhi",
    "gu": "Gujarati",
    "am": "Amharic",
    "yi": "Yiddish",
    "lo": "Lao",
    "uz": "Uzbek",
    "fo": "Faroese",
    "ht": "Haitian Creole",
    "ps": "Pashto",
    "tk": "Turkmen",
    "nn": "Nynorsk",
    "mt": "Maltese",
    "sa": "Sanskrit",
    "lb": "Luxembourgish",
    "my": "Myanmar",
    "bo": "Tibetan",
    "tl": "Tagalog",
    "mg": "Malagasy",
    "as": "Assamese",
    "tt": "Tatar",
    "haw": "Hawaiian",
    "ln": "Lingala",
    "ha": "Hausa",
    "ba": "Bashkir",
    "jw": "Javanese",
    "su": "Sundanese",
}

if not TOKEN:
    raise ValueError("TELEGRAM_TOKEN env var required")

if not ALLOWED_UIDS:
    raise ValueError("ALLOWED_UIDS env var required (comma-separated user IDs)")

bot = Bot(token=TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)
router = Router()
dp.include_router(router)

client = AsyncOpenAI(
    base_url=WHISPER_URL,
    api_key=WHISPER_API_KEY,
)

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
    selecting_language = State()


def is_allowed(user_id: int) -> bool:
    return user_id in ALLOWED_UIDS


async def fetch_available_models() -> list[str]:
    """Fetch available models from Whisper server."""
    global _available_models
    if _available_models:
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

    # Fallback to common models
    return POPULAR_MODELS.copy()


def get_model_by_index(index: int) -> str | None:
    """Get model name by index from cached list."""
    if 0 <= index < len(_available_models):
        return _available_models[index]
    return None


def get_filtered_models() -> list[tuple[int, str]]:
    """Get popular models that are available, with their indices."""
    result = []
    for popular in POPULAR_MODELS:
        if popular in _available_models:
            idx = _available_models.index(popular)
            result.append((idx, popular))
    return result


def get_user_settings(user_id: int) -> tuple[str, str | None]:
    """Get model and language from persistent storage."""
    prefs = get_user_prefs(user_id)
    model = prefs.get("model", DEFAULT_MODEL)
    lang = prefs.get("language", "auto")
    return model, None if lang == "auto" else lang


def format_settings(model: str, language: str) -> str:
    """Format current settings for display."""
    lang_name = SUPPORTED_LANGUAGES.get(language, language)
    # Shorten model name for display
    model_short = model.split("/")[-1] if "/" in model else model
    return f"Model: {model_short}\nLanguage: {lang_name}"


@router.message(Command("start"))
async def start_handler(message: Message, state: FSMContext) -> None:
    uid = message.from_user.id
    print(f"User {uid} started WhisperSqueak", flush=True)
    if not is_allowed(uid):
        await message.answer("Access denied. Contact admin.")
        return

    prefs = get_user_prefs(uid)
    model = prefs.get("model", DEFAULT_MODEL)
    language = prefs.get("language", "auto")

    await message.answer(
        "Hi! I'm WhisperSqueak\n\n"
        "Send a voice message or upload an audio file to transcribe it.\n\n"
        f"Current settings:\n{format_settings(model, language)}\n\n"
        "Commands:\n"
        "/model - Select transcription model\n"
        "/language - Select input language\n"
        "/help - Show help"
    )


@router.message(Command("help"))
async def help_handler(message: Message, state: FSMContext) -> None:
    if not is_allowed(message.from_user.id):
        await message.answer("Access denied. Contact admin.")
        return

    prefs = get_user_prefs(message.from_user.id)
    model = prefs.get("model", DEFAULT_MODEL)
    language = prefs.get("language", "auto")

    await message.answer(
        "WhisperSqueak - Voice Transcription Bot\n\n"
        "How to use:\n"
        "1. Select a model with /model\n"
        "2. Select language with /language (or use auto-detect)\n"
        "3. Send a voice message or audio file\n\n"
        "Supported formats: MP3, WAV, OGG, M4A, FLAC, etc.\n\n"
        f"Current settings:\n{format_settings(model, language)}\n\n"
        "Tips:\n"
        "- Smaller models (tiny, base) are faster\n"
        "- Larger models (large-v3) are more accurate\n"
        "- Setting a specific language improves accuracy"
    )


@router.message(Command("model"))
async def model_command(message: Message, state: FSMContext) -> None:
    if not is_allowed(message.from_user.id):
        await message.answer("Access denied. Contact admin.")
        return

    status = await message.answer("Fetching available models...")
    await fetch_available_models()

    # Get popular models with their indices
    filtered = get_filtered_models()

    buttons = []
    for idx, model in filtered:
        # Shorten display name
        display = model.split("/")[-1] if "/" in model else model
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


@router.message(Command("language"))
async def language_command(message: Message, state: FSMContext) -> None:
    if not is_allowed(message.from_user.id):
        await message.answer("Access denied. Contact admin.")
        return

    # Show common languages first, then a "more" option
    common_langs = ["auto", "en", "zh", "es", "fr", "de", "ja", "ko", "ru", "pt", "ar", "hi"]

    buttons = []
    row = []
    for code in common_langs:
        name = SUPPORTED_LANGUAGES.get(code, code)
        row.append(InlineKeyboardButton(text=name, callback_data=f"lang:{code}"))
        if len(row) == 3:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)

    buttons.append([InlineKeyboardButton(text="More languages...", callback_data="lang:more")])

    keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)
    await message.answer("Select input language:", reply_markup=keyboard)
    await state.set_state(UserSettings.selecting_language)


@router.callback_query(F.data == "lang:more")
async def show_more_languages(callback: CallbackQuery, state: FSMContext) -> None:
    if not is_allowed(callback.from_user.id):
        await callback.answer("Access denied.")
        return

    # Show all languages in pages
    common = {"auto", "en", "zh", "es", "fr", "de", "ja", "ko", "ru", "pt", "ar", "hi"}
    other_langs = [(k, v) for k, v in SUPPORTED_LANGUAGES.items() if k not in common]
    other_langs.sort(key=lambda x: x[1])  # Sort by name

    buttons = []
    row = []
    for code, name in other_langs[:30]:  # First 30 additional languages
        row.append(InlineKeyboardButton(text=name, callback_data=f"lang:{code}"))
        if len(row) == 3:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)

    buttons.append([InlineKeyboardButton(text="Back", callback_data="lang:back")])

    keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)
    await callback.message.edit_text("Select language:", reply_markup=keyboard)
    await callback.answer()


@router.callback_query(F.data == "lang:back")
async def back_to_common_languages(callback: CallbackQuery, state: FSMContext) -> None:
    if not is_allowed(callback.from_user.id):
        await callback.answer("Access denied.")
        return

    common_langs = ["auto", "en", "zh", "es", "fr", "de", "ja", "ko", "ru", "pt", "ar", "hi"]

    buttons = []
    row = []
    for code in common_langs:
        name = SUPPORTED_LANGUAGES.get(code, code)
        row.append(InlineKeyboardButton(text=name, callback_data=f"lang:{code}"))
        if len(row) == 3:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)

    buttons.append([InlineKeyboardButton(text="More languages...", callback_data="lang:more")])

    keyboard = InlineKeyboardMarkup(inline_keyboard=buttons)
    await callback.message.edit_text("Select input language:", reply_markup=keyboard)
    await callback.answer()


@router.callback_query(F.data.startswith("lang:"))
async def process_language(callback: CallbackQuery, state: FSMContext) -> None:
    if not is_allowed(callback.from_user.id):
        await callback.answer("Access denied.")
        return

    lang_code = callback.data.split(":", 1)[1]
    if lang_code in ("more", "back"):
        return  # Handled by other handlers

    # Save to persistent storage
    set_user_prefs(callback.from_user.id, language=lang_code)

    lang_name = SUPPORTED_LANGUAGES.get(lang_code, lang_code)
    await callback.message.edit_text(f"Language set to: {lang_name}\n\nSend a voice or audio file to transcribe!")
    await state.set_state(None)
    await callback.answer()


@router.message(F.voice)
async def voice_handler(message: Message, state: FSMContext) -> None:
    print(f"Received voice from {message.from_user.id}", flush=True)
    if not is_allowed(message.from_user.id):
        await message.answer("Access denied. Contact admin.")
        return
    await process_audio(message, state, message.voice, f"{message.voice.file_id}.ogg")


@router.message(F.audio)
async def audio_handler(message: Message, state: FSMContext) -> None:
    print(f"Received audio from {message.from_user.id}: {message.audio.file_name}", flush=True)
    if not is_allowed(message.from_user.id):
        await message.answer("Access denied. Contact admin.")
        return
    audio = message.audio
    filename = audio.file_name or f"{audio.file_id}.mp3"
    await process_audio(message, state, audio, filename)


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
    await process_audio(message, state, doc, filename)


async def process_audio(message: Message, state: FSMContext, file_obj: Any, filename: str) -> None:
    uid = message.from_user.id
    print(f"Processing audio for user {uid}: {filename}", flush=True)

    model, language = get_user_settings(uid)
    prefs = get_user_prefs(uid)
    lang_display = prefs.get("language", "auto")
    print(f"Settings - model: {model}, language: {language}", flush=True)

    model_short = model.split("/")[-1] if "/" in model else model
    lang_name = SUPPORTED_LANGUAGES.get(lang_display, lang_display)

    status_msg = await message.answer(
        f"Processing: {filename}\n"
        f"Model: {model_short}\n"
        f"Language: {lang_name}"
    )

    tg_file = await bot.get_file(file_obj.file_id)
    file_path = Path(f"/tmp/{tg_file.file_id}")
    await bot.download_file(tg_file.file_path, file_path)

    try:
        audio = AudioSegment.from_file(file_path)
        mp3_buffer = io.BytesIO()
        audio.export(mp3_buffer, format="mp3")
        mp3_buffer.seek(0)
        file_to_send = ("audio.mp3", mp3_buffer, "audio/mpeg")

        transcript = await client.audio.transcriptions.create(
            model=model,
            file=file_to_send,
            language=language,
            response_format="text",
        )

        result_text = transcript if isinstance(transcript, str) else transcript.text

        if not result_text or not result_text.strip():
            await status_msg.edit_text("No speech detected.")
        elif len(result_text) > 4000:
            await status_msg.edit_text(result_text[:4000])
            remaining = result_text[4000:]
            while remaining:
                chunk = remaining[:4000]
                remaining = remaining[4000:]
                await message.answer(chunk)
        else:
            await status_msg.edit_text(result_text)

    except Exception as e:
        print(f"Error processing audio: {e}", flush=True)
        await status_msg.edit_text(f"Error processing {filename}:\n{str(e)}")
    finally:
        file_path.unlink(missing_ok=True)


async def main() -> None:
    print("WhisperSqueak is starting...", flush=True)
    # Pre-fetch models
    models = await fetch_available_models()
    print(f"Available models: {models}", flush=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
