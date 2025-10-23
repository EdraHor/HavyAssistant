# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HavyAssistant is a GPU-accelerated Russian voice assistant with wake word detection, speech recognition, LLM integration (Gemini), and text-to-speech capabilities. Built with PyQt5 for GUI and designed for offline-capable wake word detection with online LLM processing.

## Commands

### Running the Application

```bash
# GUI version (main application)
python main.py

# CLI version (no GUI)
python cli_example.py
```

### Testing

```bash
# Test TTS functionality
python test_tts.py

# Test Chatterbox TTS (if available)
python test_chatterbox.py
```

### Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Note: First run will auto-download required models:
# - Vosk wake word model (~50MB)
# - Faster-Whisper model (75MB - 3GB depending on config)
# - Silero TTS model (downloaded via torch.hub)
```

## Architecture

### Core Pipeline Flow

The assistant operates in a state machine with these key states:

1. **LISTENING_WAKE_WORD** - Vosk continuously processes audio for wake word
2. **RECORDING_COMMAND** - After wake word detected, Faster-Whisper records user command
3. **PROCESSING_LLM** - Command sent to Gemini API for response
4. **SPEAKING** - Silero TTS synthesizes and plays response
5. Back to LISTENING_WAKE_WORD

### Service Architecture

**VoiceAssistantController** (`services/assistant_controller.py`) is the central orchestrator that coordinates:

- **AudioCaptureService** - Captures raw audio from microphone in chunks
- **WakeWordService** - Vosk-based wake word detection (offline)
- **SpeechRecognitionService** - Faster-Whisper GPU-accelerated transcription
- **GeminiService** - Google Gemini LLM API with conversation history
- **TTSService** - Text-to-speech (Silero or Piper engines)

### Data Flow

```
Microphone → AudioCaptureService → [State-dependent routing]
                                    ↓
                    ┌───────────────┴────────────────┐
                    ↓                                ↓
         WakeWordService                  SpeechRecognitionService
         (Vosk - offline)                 (Whisper - GPU)
                    ↓                                ↓
              Wake word detected                Command text
                    ↓                                ↓
              Switch to RECORDING_COMMAND    GeminiService
                                                     ↓
                                              Response text
                                                     ↓
                                              TTSService → Speaker
```

### Configuration System

- **config/settings.yaml** - Single source of truth for all settings
- **utils/config_loader.py** - Singleton Config class with dot-notation access
- Access pattern: `config.get('section.key', default_value)`

Key configuration sections:
- `audio.*` - Sample rate, chunk size, device settings
- `wake_word.*` - Vosk model path and keyword
- `speech_recognition.*` - Whisper model, device (cuda/cpu), sensitivity
- `gemini.*` - API key, model, proxy settings
- `tts.*` - Engine selection (silero/piper), voice, device

### Database Layer

**ConversationDatabase** (`utils/database.py`) - Thread-safe SQLite wrapper:
- Stores conversation sessions and message history
- Three tables: `sessions`, `messages`, `images` (for future use)
- Each new session preserves old history in DB
- GeminiService maintains both in-memory cache and DB persistence

### GUI Architecture

**MainWindow** (`gui/main_window.py`) - PyQt5 main interface:
- Connects controller callbacks to UI updates via Qt signals/slots
- Separate loading overlay for model initialization
- Real-time audio level visualization
- Conversation history display with role-based formatting

### Error Handling

The project implements comprehensive error handling:
- **Global exception hook** in `main.py` catches all unhandled exceptions
- Each service has try-catch blocks with detailed logging
- Callbacks are wrapped to prevent cascading failures
- Timeouts on recording (60s) and API calls (30s)

## Development Guidelines

### Working with TTS

The TTS system uses a plugin architecture:
- Base class: `tts/base_tts.py` (defines `BaseTTS` interface)
- Implementations: `silero_tts.py`, `piper_tts.py`
- Service wrapper: `tts/tts_service.py` (handles engine selection)

To add a new TTS engine:
1. Create new class inheriting from `BaseTTS`
2. Implement: `initialize()`, `synthesize(text) -> bytes`, `cleanup()`
3. Add engine selection logic in `TTSService.initialize()`

### Adding New States

To add assistant states:
1. Add to `AssistantState` enum in `services/assistant_controller.py`
2. Update `_on_audio_data()` routing logic
3. Add state transition logic in relevant callbacks
4. Update GUI status display mapping

### Model Configuration

Models are auto-downloaded on first run to `models/` directory:
- **Vosk**: Small Russian model (vosk-model-small-ru-0.22)
- **Whisper**: Controlled by `speech_recognition.model_name` config
  - Options: tiny (75MB), base (145MB), small (466MB), medium (1.5GB), large-v3 (3GB), large-v3-turbo (1.6GB)
- **Silero**: Downloaded via torch.hub to `models/silero_tts/`

**Important**: First launch with large-v3 model can take 5-10 minutes for download. Subsequent launches are instant due to caching.

### Speech Recognition Sensitivity

The `sensitivity` parameter (1-10) controls voice detection:
- **Higher values** (8-10): More sensitive, picks up quieter speech
- **Lower values** (1-3): Less sensitive, requires louder speech
- Implemented as multiplier on noise floor: `threshold = noise_floor * sensitivity_multiplier`
- Can be adjusted at runtime via `VoiceAssistantController.update_sensitivity()`

### Calibration System

Auto-calibration measures background noise to set dynamic thresholds:
- Triggered 2 seconds after start (if `auto_calibrate: true`)
- Measures RMS over specified duration
- Updates `noise_floor` and recalculates `voice_threshold`
- Manual trigger: `VoiceAssistantController.calibrate_noise_floor(duration)`

### Proxy Configuration

For restricted networks, enable SOCKS5 proxy in config:
```yaml
gemini:
  proxy:
    enabled: true
    type: "socks5"
    host: "127.0.0.1"
    port: 10808
```

## Key Implementation Details

### Thread Safety

- All audio processing happens in separate threads
- Database uses threading locks for concurrent access
- Qt signals/slots ensure thread-safe GUI updates
- Callbacks are invoked within try-catch to prevent crashes

### GPU Utilization

- **Whisper**: Uses CUDA by default, auto-falls back to CPU
- **Silero**: Supports both CUDA and CPU inference
- Warm-up pass on GPU to initialize CUDA context
- Set `device: "cpu"` in config to force CPU mode

### Wake Word Detection Workflow

WakeWordService processes every audio chunk:
1. Vosk recognizer accepts waveform
2. Checks partial and final results for wake word substring
3. On detection: sets `is_recording = True` flag and triggers callback
4. Controller switches to RECORDING_COMMAND state
5. WakeWordService remains paused until controller calls `stop()`

### Conversation Context Management

GeminiService maintains conversation history:
- Session-based: Each `clear_history()` creates new session
- History format matches Gemini API structure (role + parts)
- System prompt injected only at session start
- History length reported as message pairs (user + model)

### Audio Level Monitoring

Both WakeWordService and SpeechRecognitionService calculate RMS:
```python
rms = sqrt(mean(audio_int16^2)) / 32768.0
```
- Normalized to 0.0-1.0 range
- Used for UI visualization and voice detection
- Status labels: "Тишина" (< 0.02), "Шум" (0.02-0.05), "ГОЛОС" (> threshold)

## Troubleshooting

### Common Issues

**Models not downloading**: Check internet connection and disk space. Models download to `./models/` directory.

**CUDA errors**: If GPU errors occur, set `device: "cpu"` in config for both `speech_recognition` and `tts` sections.

**Wake word not detected**:
- Increase microphone volume
- Try different wake word in config
- Check audio device selection in GUI

**Gemini API errors**:
- Verify API key in `config/settings.yaml`
- Check proxy settings if behind firewall
- Ensure proxy service is running (if enabled)

**TTS not working**:
- Verify `tts.enabled: true` in config
- Check speaker device is properly configured
- Try switching engine: `engine: "piper"` or `engine: "silero"`

### Logging

Logs are written to:
- Console output (when `logging.console: true`)
- File: `voice_assistant.log` (rotating log)

Set `logging.level: "DEBUG"` in config for verbose output showing:
- RMS values per audio chunk
- Model loading progress
- State transitions
- API request/response details
