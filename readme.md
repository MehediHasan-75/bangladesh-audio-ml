# Audio ML Dataset Pipeline

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> End-to-end pipeline for collecting, processing, and quality-controlling labeled audio datasets for machine learning — from raw YouTube URLs and field recordings to clean, model-ready WAV segments.

---

## What This Solves

Building audio ML datasets is painful: sources are heterogeneous, recordings vary wildly in quality, and naive pipelines produce duplicate or silent segments that silently corrupt training. This pipeline handles all of it — dual ingestion sources, automatic quality filtering, incremental processing, and experiment tracking — so data collection becomes repeatable and scalable.

**Built to demonstrate:** data engineering at the ML boundary, production-minded pipeline design, and end-to-end ownership from raw data to trained model.

---

## Engineering Highlights

- **Dual-source ingestion** — YouTube (yt-dlp with retry + dedup) and physical recordings (8 audio + 8 video formats) converge into a single standardized output format
- **Incremental by design** — video-ID deduplication prevents re-downloading; smart sequential numbering means new recordings can be added at any time without touching existing segments
- **Quality gate** — each segment is evaluated for silence (< −45 dBFS) and audio content (< 30% non-silent = rejected) before being written to disk
- **Config-driven** — all pipeline parameters live in one YAML; change segment length, sample rate, or quality thresholds once and the entire pipeline adapts
- **Experiment tracking** — MLflow logs training runs; SHAP explanations surface feature attribution in the Streamlit demo

---

## Architecture

```
┌─────────────────────────┐      ┌──────────────────────────────┐
│  data/youtube_urls.csv  │      │  ml_data/physically_collected│
│  url, category          │      │  <category>/<file>.*         │
└────────────┬────────────┘      └──────────────┬───────────────┘
             │                                  │
     YouTubeAudioCollector           PhysicalAudioProcessor
     • yt-dlp download               • FFmpeg-backed conversion
     • exponential-backoff retry     • 8 audio + 8 video formats
     • video-ID deduplication        • auto audio extraction
             │                                  │
             └─────────────┬────────────────────┘
                           │
                    AudioProcessor
                    • slice into 10-s segments
                    • resample to 48 kHz mono
                    • smart sequential numbering
                           │
                   QualityController
                   • reject silence (< −45 dBFS)
                   • reject low-content (< 30% non-silent)
                   • validate sample rate + duration
                           │
              ml_data/processed/<category>/
              <category>_0000.wav, _0001.wav …
                           │
             ┌─────────────┴──────────────┐
       CNN Classifier            Wav2Vec2 Fine-tune
       MFCC features             raw waveform embeddings
       PyTorch                   Hugging Face Transformers
             │                            │
       MLflow run                   MLflow run
             └─────────────┬──────────────┘
                    Streamlit Demo
                    live inference + SHAP explanations
```

---

## Key Design Decisions

**Why two ingestion sources?**
YouTube provides scale and variety; physical recordings provide domain precision for sounds that are rare or absent on YouTube (e.g., specific regional vehicle types). Unifying both into one output format means downstream training code never needs to know the source.

**Why incremental processing?**
Re-downloading or re-segmenting files is expensive and error-prone. Each YouTube video is tracked by ID in `download_metadata.json`; segment numbering reads the highest existing index before writing, making the pipeline safe to interrupt and resume at any point.

**Why a quality gate before disk write?**
Silent or near-silent segments waste storage and, more critically, introduce label noise if they reach training. Filtering at collection time is cheaper than cleaning a corrupted dataset later.

**Why config-driven parameters?**
Hardcoded values in processing scripts make experimentation brittle. All thresholds and format settings live in `config/config.yaml` — a single change propagates everywhere without touching source code.

---

## Example: Bangladeshi Urban Audio Dataset

To validate the pipeline, a 10-class urban soundscape dataset was collected from Dhaka, Bangladesh — an acoustic environment not represented in existing public benchmarks (UrbanSound8K, ESC-50, AudioSet subsets).

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

CNN and Wav2Vec2 classifiers trained on the Bangladeshi urban audio dataset. Experiments tracked in MLflow — run `mlflow ui` to browse all logged runs.

### Training Curve

![Validation vs Training Curve](selected_outputs/validation_vs_training_curve.png)

### Confusion Matrix — CNN

![Confusion Matrix](selected_outputs/confusion_cnn.png)

### Per-Class Accuracy — CNN

![Per-Class Accuracy](selected_outputs/cnn_per_class_accuracy.png)

---

## Tech Stack

| Layer | Tools |
|---|---|
| Audio processing | pydub, librosa, FFmpeg |
| ML / Deep Learning | PyTorch, Hugging Face Transformers |
| Experiment tracking | MLflow |
| Data pipeline | yt-dlp, pandas, PyYAML |
| Explainability | SHAP |
| Demo | Streamlit |
| Testing | pytest |

---

## Quick Start

```bash
git clone https://github.com/MehediHasan-75/bangladesh-audio-ml.git
cd bangladesh-audio-ml

# Install system dependency
brew install ffmpeg          # macOS — apt install ffmpeg on Linux

# Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r mac_requirements.txt   # Linux: requirements.txt

# Collect from YouTube (edit data/youtube_urls.csv first)
python scripts/collect_audio.py

# Process physical recordings
python scripts/process_physical.py

# Interactive demo
streamlit run app/demo.py

# Track experiments
python experiments/track_experiment.py
mlflow ui --port 5000
```

To adapt to a new domain: edit `data/youtube_urls.csv` with your own `url,category` pairs and drop recordings into `ml_data/physically_collected/<category>/`.

---

## Project Structure

```
.
├── app/                    # Streamlit demo — live inference + SHAP
├── config/
│   └── config.yaml         # All pipeline parameters
├── data/
│   └── youtube_urls.csv    # Ingestion manifest (url, category)
├── experiments/            # MLflow experiment scripts
├── ml_data/                # Runtime data directory
│   ├── raw/                # YouTube downloads
│   ├── physically_collected/   # Field recordings by category
│   ├── processed/          # Final 10-s WAV segments
│   ├── download_metadata.json  # Dedup tracking
│   └── quality_report.csv  # Per-segment QA log
├── notebooks/              # EDA and model training
├── scripts/                # CLI entry points
├── selected_outputs/       # Model evaluation charts
├── spectrograms/           # Per-class spectrogram visualizations
├── src/
│   ├── collectors/         # YouTubeAudioCollector
│   ├── processors/         # AudioProcessor, PhysicalAudioProcessor
│   ├── quality/            # QualityController
│   └── utils/
└── tests/                  # pytest unit tests
```

---

## Future Work

- **Domain expansion** — pipeline is domain-agnostic; next targets include wildlife bioacoustics, industrial machinery fault detection, and indoor acoustic scene classification
- **Data augmentation** — SpecAugment, pitch shifting, and room impulse response convolution to improve model generalization
- **Real-time inference** — ONNX export + FastAPI wrapper for sub-100 ms latency serving
- **Cloud storage** — S3/GCS backend for team-scale dataset versioning and reproducibility
