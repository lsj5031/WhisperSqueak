# WhisperSqueak

![WhisperSqueak](WhisperSqueak.png)

A Telegram bot for audio transcription using [Faster Whisper Server](https://github.com/fedirz/faster-whisper-server).

## Features

- Transcribe voice messages and audio files (MP3, WAV, OGG, M4A, FLAC, etc.)
- **Long audio support** - Automatically splits audio into 1-minute chunks with overlap
- **Progressive display** - See transcription results appear in real-time as chunks complete
- **Sequential task queue** - Multiple audio files are processed one at a time
- Dynamic model selection from your Faster Whisper Server
- Multi-language support with 99+ languages and auto-detection
- Persistent user settings (model & language preferences)
- User whitelist for access control

## Requirements

- Docker
- A running [Faster Whisper Server](https://github.com/fedirz/faster-whisper-server) instance
- Telegram Bot Token (from [@BotFather](https://t.me/BotFather))

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/WhisperSqueak.git
   cd WhisperSqueak
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

3. **Build and run**
   ```bash
   docker build -t whispersqueak .
   docker run -d --name whispersqueak \
     --env-file .env \
     -v ./data:/app/data \
     -e SETTINGS_FILE=/app/data/user_settings.json \
     whispersqueak
   ```

## Configuration

Create a `.env` file with the following variables:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TELEGRAM_TOKEN` | Yes | - | Bot token from @BotFather |
| `ALLOWED_UIDS` | Yes | - | Comma-separated Telegram user IDs |
| `WHISPER_URL` | No | `http://host.docker.internal:18000/v1` | Faster Whisper Server URL |
| `WHISPER_API_KEY` | No | `dummy` | API key (if server requires auth) |
| `SETTINGS_FILE` | No | `/app/user_settings.json` | Path for persistent settings |

### Getting Your Telegram User ID

1. Message [@userinfobot](https://t.me/userinfobot) on Telegram
2. It will reply with your user ID
3. Add this ID to `ALLOWED_UIDS` in your `.env` file

## Bot Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome message and current settings |
| `/model` | Select transcription model |
| `/language` | Select input language (or auto-detect) |
| `/help` | Usage guide and tips |

## Usage

1. Send `/model` to select a Whisper model (default: `faster-whisper-large-v3`)
2. Send `/language` to set input language (default: auto-detect)
3. Send a voice message or audio file
4. Receive the transcription

## Architecture Decisions

### Model Selection
- Models are fetched dynamically from the Faster Whisper Server's `/v1/models` endpoint
- Only popular/common models are shown in the selection menu to avoid Telegram's button limits
- Popular models include: large-v3, medium, small, base, tiny, and turbo variants

### Language Support
- Supports all 99+ languages that Whisper supports
- Auto-detect is the default and works well for most use cases
- Setting a specific language can improve accuracy for that language

### Persistence
- User settings (model, language) are stored in a JSON file
- Mount a volume to `/app/data` to persist across container restarts

### Audio Processing
- All audio formats are converted to MP3 using FFmpeg/pydub before sending to Whisper
- Supports: MP3, WAV, OGG, M4A, FLAC, AAC, WMA, OPUS, and more
- **Chunking**: Audio longer than 1 minute is split into chunks with 2-second overlap to avoid cutting words
- **Progressive results**: Transcription text appears and grows as each chunk completes
- **Task queue**: Multiple files are queued and processed sequentially to avoid server overload

## Known Limitations

### Telegram File Size Limit (20MB)
The Telegram Bot API limits file downloads to **20MB**. This affects:
- Voice messages > 20MB (rare, ~1.5+ hours of Telegram voice)
- Audio files > 20MB (common for longer recordings)

**Estimated file sizes:**
| Duration | Voice (Opus) | MP3 128kbps | MP3 64kbps |
|----------|--------------|-------------|------------|
| 15 min   | ~2MB         | ~14MB       | ~7MB       |
| 1 hour   | ~8MB         | ~56MB       | ~28MB      |
| 1.5 hours| ~12MB        | ~85MB       | ~42MB      |

**Workaround**: Use the Telegram Bot API Local Server (see below).

### Telegram Message Limits
- Maximum message length is 4096 characters
- Long transcriptions are automatically split into multiple messages

### Button Callback Data Limit
- Telegram limits callback data to 64 bytes
- Model names are stored by index to work around this limitation

## Large File Support (Optional)

To transcribe files larger than 20MB, you can run the **Telegram Bot API Local Server**. This removes the file size limit entirely (up to 2GB).

### Setup

1. **Get API credentials** from https://my.telegram.org:
   - Log in with your phone number
   - Go to "API development tools"
   - Create an app to get `api_id` and `api_hash`

2. **Run the Bot API server**:
   ```bash
   docker run -d --name telegram-bot-api \
     -p 8081:8081 \
     -e TELEGRAM_API_ID=<your_api_id> \
     -e TELEGRAM_API_HASH=<your_api_hash> \
     -v /tmp/telegram-bot-api:/var/lib/telegram-bot-api \
     aiogram/telegram-bot-api
   ```

3. **Update WhisperSqueak** to use the local server:

   Add to your `.env`:
   ```
   TELEGRAM_API_URL=http://localhost:8081
   ```

   Or if running both in Docker, use:
   ```
   TELEGRAM_API_URL=http://host.docker.internal:8081
   ```

4. **Modify bot.py** (one-line change):
   ```python
   # Change this:
   bot = Bot(token=TOKEN)

   # To this:
   from aiogram.client.session.aiohttp import AiohttpSession
   session = AiohttpSession(api=TelegramAPIServer.from_base(os.getenv("TELEGRAM_API_URL", "https://api.telegram.org")))
   bot = Bot(token=TOKEN, session=session)
   ```

### Benefits
- Upload/download files up to 2GB
- Faster file transfers (direct connection)
- No changes to bot logic required

## Docker Networking

### WSL2 / Docker Desktop
Use `host.docker.internal` to access services on the host:
```
WHISPER_URL=http://host.docker.internal:18000/v1
```

### Linux (native Docker)
Use the Docker gateway IP or host network mode:
```bash
# Option 1: Gateway IP
WHISPER_URL=http://172.17.0.1:18000/v1

# Option 2: Host network
docker run --network host ...
```

## Development

### Local Testing (Python 3.12+)
```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# For Python 3.13+, also install:
uv pip install audioop-lts

# Run locally
python bot.py
```

### Code Structure
```
WhisperSqueak/
├── bot.py              # Main bot logic
├── Dockerfile          # Container definition
├── requirements.txt    # Python dependencies
├── .env.example        # Environment template
├── .gitignore          # Git ignore rules
└── data/               # Persistent storage (gitignored)
    └── user_settings.json
```

## Troubleshooting

### Bot not responding
- Check logs: `docker logs whispersqueak`
- Verify `TELEGRAM_TOKEN` is correct
- Ensure your user ID is in `ALLOWED_UIDS`

### "No models available"
- Verify Faster Whisper Server is running
- Check `WHISPER_URL` is accessible from the container
- For WSL2, ensure `host.docker.internal` resolves correctly

### Audio not processing
- Check logs for error messages
- Ensure FFmpeg is installed (included in Docker image)
- Verify the file is a supported audio format

### "File is too big" error
- Telegram Bot API limits downloads to 20MB
- See [Large File Support](#large-file-support-optional) for the workaround
- Alternative: Split the audio file before sending

## License

MIT License - See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Credits

- [Faster Whisper Server](https://github.com/fedirz/faster-whisper-server) - OpenAI-compatible Whisper API
- [aiogram](https://github.com/aiogram/aiogram) - Telegram Bot framework
- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition model
