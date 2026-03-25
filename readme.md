# Audio ML Dataset Collection & Processing Pipeline

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Executive Summary

- **What:** A reusable pipeline for building labeled, ML-ready audio datasets from any domain — define your categories, point it at YouTube URLs or local recordings, and get clean 48 kHz mono WAV segments out
- **How:** Two ingestion sources (automated YouTube downloads via yt-dlp + multi-format physical recordings), quality filtering, smart sequential numbering, and MLflow-tracked model training
- **Example dataset built with this pipeline:** 10-class Bangladeshi urban soundscape (bike, bus, CNG auto-rickshaw, traffic jam, siren, etc.) — CNN and Wav2Vec2 classifiers trained end-to-end

---

## Demo

```bash
streamlit run app/demo.py
```

Upload any audio clip and get a predicted class with SHAP feature attribution explanations.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Pipeline Architecture](#pipeline-architecture)
- [Example Dataset: Bangladeshi Urban Audio](#example-dataset-bangladeshi-urban-audio)
- [Model Results](#model-results)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Future Work](#future-work)

---

## How It Works

The pipeline has two ingestion paths that both produce identical standardized output:

**YouTube (automated):** Add URLs + category labels to `data/youtube_urls.csv` → yt-dlp downloads with exponential-backoff retry and video-ID deduplication → segments produced automatically.

**Physical recordings:** Drop audio/video files into `ml_data/physically_collected/<your_category>/` → pipeline handles conversion and segmentation. Supports 8 audio formats (`.opus`, `.m4a`, `.mp3`, `.aac`, `.wav`, `.ogg`, `.flac`, `.wma`) and 8+ video formats via FFmpeg.

Both paths output `<category>_XXXX.wav` segments with smart sequential numbering — safe to add new recordings at any time without overwriting or renumbering existing files.

### Pipeline Parameters

| Parameter | Value | Config key |
|---|---|---|
| Segment length | 10 s | `segment_duration_ms` |
| Sample rate | 48 kHz mono | `sample_rate` |
| Silence rejection | < −45 dBFS | `silence_threshold_db` |
| Min audio content | ≥ 30% non-silent | `min_speech_percentage` |
| Output format | WAV (PCM) | — |

All parameters are in `config/config.yaml` — change them once and the entire pipeline adapts.

---

## Pipeline Architecture

```
data/youtube_urls.csv          ml_data/physically_collected/
  (any categories)               (any audio/video formats)
        │                                   │
  YouTubeAudioCollector            PhysicalAudioProcessor
  (yt-dlp, retry + dedup)          (FFmpeg-backed conversion)
        │                                   │
        └──────────── AudioProcessor ───────┘
                      (10-s segments, 48 kHz mono)
                               │
                      QualityController
                      (silence + content % checks)
                               │
                  ml_data/processed/<category>/
                               │
               ┌───────────────┴───────────────┐
           CNN Classifier              Wav2Vec2 Fine-tune
         (MFCC features)            (Hugging Face Transformers)
               │                               │
          MLflow tracking               MLflow tracking
               └───────────────┬───────────────┘
                         Streamlit Demo
                      (live inference + SHAP)
```

---

## Example Dataset: Bangladeshi Urban Audio

To demonstrate the pipeline, a 10-class urban soundscape dataset was collected from Dhaka, Bangladesh — an acoustic environment not covered by existing public datasets (UrbanSound8K, ESC-50, etc.).

| Category | Description |
|---|---|
| `bike` | Motorcycle engine and horn sounds |
| `bus` | City bus engine and traffic noise |
| `car` | Passenger car sounds |
| `cng_auto` | CNG auto-rickshaw engine sounds |
| `construction_noise` | Construction site machinery |
| `protest` | Crowd and protest sounds |
| `siren` | Emergency vehicle sirens |
| `traffic_jam` | Dense urban traffic ambience |
| `train` | Train engine and rail sounds |
| `truck` | Heavy truck engine sounds |

---

## Model Results

Models trained on the example Bangladeshi urban audio dataset. Full per-class metrics logged via MLflow (`experiments/track_experiment.py`).

| Model | Features | Notes |
|---|---|---|
| CNN | MFCC | See charts below |
| Wav2Vec2 fine-tune | Raw waveform | Hugging Face Transformers |

### Training Curve

![Validation vs Training Curve](selected_outputs/validation_vs_training_curve.png)

### Confusion Matrix (CNN)

![Confusion Matrix](selected_outputs/confusion_cnn.png)

### Per-Class Accuracy (CNN)

![Per-Class Accuracy](selected_outputs/cnn_per_class_accuracy.png)

---

## Tech Stack

| Category | Tools |
|---|---|
| Audio processing | pydub, librosa, FFmpeg |
| ML / Deep Learning | PyTorch, Hugging Face Transformers |
| MLOps | MLflow |
| Data pipeline | yt-dlp, pandas, PyYAML |
| Demo | Streamlit, SHAP |
| Testing | pytest |

---

## Quick Start

```bash
# 1. Clone and set up
git clone https://github.com/MehediHasan-75/bangladesh-audio-ml.git
cd bangladesh-audio-ml
brew install ffmpeg                          # macOS; apt install ffmpeg on Linux
python3 -m venv .venv && source .venv/bin/activate
pip install -r mac_requirements.txt          # Linux: requirements.txt

# 2. Define your categories in data/youtube_urls.csv, then collect
python scripts/collect_audio.py

# 3. Process any physically collected recordings
python scripts/process_physical.py

# 4. Run the demo
streamlit run app/demo.py
```

To use your own categories: edit `data/youtube_urls.csv` with `url,category` rows and create matching subdirectories in `ml_data/physically_collected/`.

---

## Project Structure

```
audio-ml-pipeline/
├── app/                    # Streamlit demo with SHAP explanations
├── config/                 # Pipeline parameters (config.yaml)
├── data/                   # YouTube URL + category lists
├── experiments/            # MLflow experiment scripts
├── ml_data/                # All data (raw → processed → quality_report)
├── notebooks/              # Exploratory analysis & model training
├── scripts/                # CLI entry points (collect, process, convert)
├── selected_outputs/       # Model evaluation charts
├── spectrograms/           # Per-class spectrogram visualizations
├── src/                    # Core library (collectors, processors, quality)
└── tests/                  # pytest unit tests
```

---

## Future Work

- **Adapt to new domains:** The pipeline is domain-agnostic — next targets include wildlife sounds, industrial machinery, and indoor acoustic scenes
- **Data augmentation:** SpecAugment, pitch shifting, and room impulse responses to increase effective dataset size
- **Real-time inference:** Export model to ONNX and wrap with a FastAPI endpoint for sub-100 ms latency
- **Cloud data versioning:** Integrate S3/GCS storage for team-scale dataset reproducibility
