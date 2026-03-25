# Project Summary: Bangladeshi Audio ML Dataset Pipeline

## What It Does

A data collection and preprocessing pipeline that builds ML-ready audio datasets from Bangladeshi audio sources. It automates downloading, segmenting, validating, and organizing audio into standardized clips suitable for training speech recognition, sound event detection, or audio classification models.

## Data Sources

| Source | Input | Tool |
|--------|-------|------|
| YouTube | URLs in `data/youtube_urls.csv` | `yt-dlp` |
| Physical recordings | Files in `ml_data/physically_collected/<category>/` | FFmpeg + pydub |

Supported input formats: `.opus`, `.m4a`, `.mp3`, `.aac`, `.wav`, `.flac`, `.ogg`, `.wma`, `.mp4`, `.mov`, `.mkv`, `.webm`, `.avi`, and more.

## Output

Every processed file becomes a **10-second, 48kHz, mono WAV** segment named `<category>_XXXX.wav` and stored under `ml_data/processed/<category>/`.

## Pipeline Overview

```
YouTube URLs ──► Download (yt-dlp) ──► ml_data/raw/
                                              │
Physical files ──────────────────────────────┤
                                              ▼
                                    Segment into 10s clips
                                              │
                                              ▼
                                    Validate (volume + content)
                                              │
                                              ▼
                                    ml_data/processed/<category>/
```

## Key Features

- **Incremental downloads** — tracks video IDs to never re-download YouTube content
- **Smart numbering** — new segments always continue from the highest existing index; no overwrites, no gaps
- **Quality filtering** — rejects clips below -45 dB or with less than 30% speech-like content
- **Multi-format ingestion** — handles audio and video files transparently via FFmpeg extraction
- **Full metadata tracking** — every processed file is logged with source, duration, dBFS, and speech percentage

## Tech Stack

| Layer | Libraries |
|-------|-----------|
| Audio processing | `pydub`, `librosa`, `soundfile` |
| YouTube download | `yt-dlp` |
| Data handling | `pandas`, `numpy` |
| ML/modeling | `torch`, `torchaudio`, `transformers`, `scikit-learn` |
| Augmentation | `audiomentations` |

## Directory Structure

```
bangladeshi-audio-ml/
├── config/config.yaml                  # Tunable parameters
├── data/youtube_urls.csv               # YouTube source list
├── src/
│   ├── collectors/youtube_collector.py
│   ├── processors/audio_processor.py
│   ├── processors/physical_audio_processor.py
│   └── quality/quality_controller.py
├── scripts/                            # Entry-point scripts
├── tests/                              # Unit tests
└── ml_data/                            # Generated data (not committed)
```

## Current Dataset State

- Categories stored under `ml_data/processed/` (e.g., `bike/`, `bus/`, etc.)
- Each category contains sequentially numbered `.wav` segments
- Processing history tracked in `ml_data/download_metadata.json` and CSV metadata files
