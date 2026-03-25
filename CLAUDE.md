# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python pipeline for building ML-ready audio datasets from Bangladeshi audio sources. It downloads from YouTube and processes manually recorded audio, segments everything into standardized 10-second WAV files (48kHz, mono), validates quality, and tracks metadata.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r mac_requirements.txt   # macOS
# pip install -r requirements.txt     # Linux/CI
```

FFmpeg must be installed separately (`brew install ffmpeg` on macOS).

## Common Commands

### Collect & process YouTube audio
```bash
python scripts/collect_audio.py
```
Downloads new URLs from `data/youtube_urls.csv`, processes only newly downloaded files.

### Process physically collected audio (all categories)
```bash
python scripts/process_physical.py
```
Processes all audio/video files in `ml_data/physically_collected/` organized by category subdirectory.

### Process a single category
```bash
python scripts/process_physical_category.py <category_name>
```

### Validate input CSV / output metadata
```python
from src.processors.data_validator import validate_youtube_csv, validate_processing_metadata
validate_youtube_csv("data/youtube_urls.csv")
validate_processing_metadata("ml_data/processing_metadata.csv")
```

### Run an MLflow experiment
```bash
python experiments/track_experiment.py
mlflow ui --port 5000          # open http://localhost:5000
```

### Run the Streamlit demo
```bash
streamlit run app/demo.py
```

### DVC — reproduce the full pipeline
```bash
dvc repro                      # runs collect → process → quality_check
dvc push                       # push data to remote storage
dvc pull                       # restore data on a new machine
```
To switch from local to cloud storage: `dvc remote modify local_storage url s3://your-bucket/path`

### Process raw YouTube downloads (batch reprocess)
```bash
python scripts/process_raw.py
```

### Convert OPUS files to WAV
```bash
python scripts/convert_opus.py
```

### Run tests
```bash
python -m pytest tests/
python -m pytest tests/test_physical_processor.py  # single test file
```

## Architecture

### Two Input Pipelines

1. **YouTube pipeline**: `data/youtube_urls.csv` → `src/collectors/youtube_collector.py` → `ml_data/raw/` → `src/processors/audio_processor.py` → `ml_data/processed/<category>/`

2. **Physical pipeline**: `ml_data/physically_collected/<category>/` → `src/processors/physical_audio_processor.py` → `ml_data/processed/<category>/`

Both pipelines produce identical output: `<category>_XXXX.wav` segments at 48kHz mono.

### Key Design Decisions

- **Smart numbering**: New segments always continue from the highest existing number in a category directory — never overwrite or gap-fill.
- **Incremental processing**: YouTube collector tracks processed video IDs in `ml_data/download_metadata.json` to avoid redownloading.
- **Segment validation**: Segments are rejected if they are too quiet (below -45dB) or contain less than 30% speech-like content.
- **Multi-format support**: Physical processor handles 8+ audio formats (.opus, .m4a, .mp3, .aac, .wav, .flac, .ogg, .wma) and 8+ video formats (mp4, mov, mkv, webm, avi, etc.) — video files have audio extracted via FFmpeg before processing.

### Configuration

All key parameters are in `config/config.yaml`:
- `segment_duration_ms`: 10000 (10 seconds)
- `sample_rate`: 48000 Hz
- `silence_threshold_db`: -45
- `min_speech_percentage`: 30

### Output Structure

```
ml_data/
├── raw/                          # Downloaded YouTube audio
├── physically_collected/         # Manually collected audio, organized by category
├── processed/
│   ├── <category>/
│   │   ├── <category>_0000.wav
│   │   └── ...
├── download_metadata.json        # YouTube download history (video IDs)
├── processing_metadata.csv       # YouTube processing log
├── physical_processing_metadata.csv
└── quality_report.csv
```

### Quality Control

`src/quality/quality_controller.py` validates processed segments: checks 48kHz sample rate, ~10s duration, and rejects silent/invalid segments. Run after processing to generate `ml_data/quality_report.csv`.

## Git Workflow

After every set of changes, commit and push to GitHub:
```bash
git add <changed files>
git commit -m "descriptive message"
git push origin main
```
