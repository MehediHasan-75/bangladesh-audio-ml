# Audio ML Pipeline — A Complete Developer Guide

This document walks you through every concept, tool, and design decision used in this project. If you're new to audio processing, machine learning pipelines, or Python data engineering, this is your starting point.

> **Who is this for?** Junior developers and newcomers to audio ML who want to understand not just *what* the code does, but *why* it was built this way — and *how* the underlying mechanisms work.

---

## Table of Contents

- [The Big Picture](#the-big-picture)
- [Configuration: One File to Rule Them All](#configuration-one-file-to-rule-them-all)
- [Audio Fundamentals You Need to Know](#audio-fundamentals-you-need-to-know)
- [Source 1: Collecting Audio from YouTube](#source-1-collecting-audio-from-youtube)
- [Source 2: Processing Physical Recordings](#source-2-processing-physical-recordings)
- [The Audio Processor: Slicing Audio into ML-Ready Segments](#the-audio-processor-slicing-audio-into-ml-ready-segments)
- [The Quality Gate: Filtering Before It Reaches Disk](#the-quality-gate-filtering-before-it-reaches-disk)
- [Quality Controller: Post-Processing Validation](#quality-controller-post-processing-validation)
- [Data Validation with Pandera: Catching Bad Data Early](#data-validation-with-pandera-catching-bad-data-early)
- [MLflow: Tracking Experiments Like a Pro](#mlflow-tracking-experiments-like-a-pro)
- [The Streamlit Demo: Interactive Inference with Explanations](#the-streamlit-demo-interactive-inference-with-explanations)
- [Logging: Knowing What Your Pipeline Is Doing](#logging-knowing-what-your-pipeline-is-doing)
- [Testing Strategy: How the Pipeline Verifies Itself](#testing-strategy-how-the-pipeline-verifies-itself)
- [CLI Scripts: How Everything Is Wired Together](#cli-scripts-how-everything-is-wired-together)
- [Library Reference](#library-reference)

---

## The Big Picture

Think of this pipeline as a factory assembly line — but for audio data:

```
Raw Sources (YouTube / Physical Files)
         |
         |  Download + Convert
         v
    Raw Audio Files  (ml_data/raw/)
         |
         |  Slice into 10-second segments
         v
    Candidate Segments
         |
         |  Quality Gate (silence? too quiet?)
         v
    Clean Segments  (ml_data/processed/<category>/)
         |
         |  Post-processing validation
         v
    quality_report.csv
         |
    ┌────┴────┐
    |         |
  CNN      Wav2Vec2
  Model    Model
    |         |
    └────┬────┘
         |
    MLflow Tracking
         |
    Streamlit Demo
    (live inference + SHAP)
```

Each stage is a separate Python module with a single, clear responsibility. This is called the **Single Responsibility Principle** — each class does one thing and does it well. It makes the code easier to test, debug, and extend.

---

## Configuration: One File to Rule Them All

### What It Is

`config/config.yaml` holds every parameter the pipeline cares about — sample rate, quality thresholds, paths, supported file formats. Nothing is hardcoded in the Python files.

```yaml
audio:
  segment_duration_ms: 10000   # 10-second segments
  sample_rate: 48000           # 48 kHz — CD-quality
  channels: 1                  # Mono (single channel)

quality:
  silence_threshold_db: -45    # Reject segments quieter than this
  min_speech_percentage: 30    # At least 30% of segment must be non-silent

paths:
  raw_dir: "ml_data/raw"
  processed_dir: "ml_data/processed"
  physically_collected_dir: "ml_data/physically_collected"
```

### Why Config-Driven Design?

Imagine you decide to experiment with 5-second segments instead of 10-second ones. If `10000` is hardcoded in 3 different Python files, you'd need to hunt down every occurrence and change each one — and probably miss one. With a central config file, you change it once and the entire pipeline adapts automatically.

This is analogous to a `.env` file in web development — one place to tune all your settings.

### How It's Loaded

```python
import yaml

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

segment_duration = config["audio"]["segment_duration_ms"]  # 10000
silence_threshold = config["quality"]["silence_threshold_db"]  # -45
```

`yaml.safe_load()` is used instead of `yaml.load()` because `safe_load` prevents execution of arbitrary Python code embedded in YAML — a security best practice.

---

## Audio Fundamentals You Need to Know

Before diving into the code, here are a few audio concepts that appear throughout the pipeline.

### Sample Rate (Hz)

Sample rate is how many times per second an audio file records the sound level. Think of it like the frame rate of a video — higher frame rates = smoother video; higher sample rates = higher-fidelity audio.

| Sample Rate | Used For |
|---|---|
| 16,000 Hz (16 kHz) | Speech recognition models (Whisper, Wav2Vec2) |
| 44,100 Hz (44.1 kHz) | CD audio |
| **48,000 Hz (48 kHz)** | **This pipeline's output — broadcast standard** |

This pipeline outputs everything at **48 kHz mono** so the output format is consistent regardless of where the audio came from.

### Channels: Mono vs Stereo

- **Stereo**: Two channels (left ear, right ear). Most music.
- **Mono**: One channel. Simpler, smaller file size, and for environmental audio classification, the stereo separation adds no useful information — a truck sounds like a truck in both channels.

The pipeline converts everything to mono (`channels: 1`).

### dBFS (Decibels relative to Full Scale)

dBFS measures loudness in audio software. The scale runs from 0 (maximum possible loudness) downward into negative numbers.

| dBFS | What it means |
|---|---|
| 0 dBFS | Maximum loudness (clipping) |
| -6 dBFS | Very loud |
| -20 dBFS | Normal speech level |
| **-45 dBFS** | **Near-silence — this pipeline's rejection threshold** |
| -60 dBFS | Almost inaudible |
| -∞ | Complete silence |

Any segment quieter than **-45 dBFS** is almost certainly background noise or a recording gap — not useful for training a classifier.

---

## Source 1: Collecting Audio from YouTube

**File**: `src/collectors/youtube_collector.py`
**Class**: `YouTubeAudioCollector`

### What It Does

Downloads audio from YouTube URLs, converts them to WAV, and tracks which videos have already been downloaded — so re-running the script never downloads the same video twice.

### How YouTube Downloading Works (yt-dlp)

yt-dlp is a command-line tool (and Python library) that can download video and audio from YouTube. You configure it with an options dictionary:

```python
ydl_opts = {
    "format": "bestaudio/best",          # Download the best audio quality available
    "postprocessors": [{
        "key": "FFmpegExtractAudio",     # Use FFmpeg to extract just the audio
        "preferredcodec": "wav",         # Convert to WAV format
        "preferredquality": "192",       # 192 kbps quality
    }],
    "outtmpl": str(output_path),         # Where to save the file
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])
```

Think of yt-dlp as the downloader and FFmpeg as the format converter — yt-dlp fetches the raw video stream, then hands it off to FFmpeg which strips out just the audio and writes a clean WAV file.

### Deduplication: Never Download Twice

The collector maintains a JSON file (`ml_data/download_metadata.json`) that records every video ID that has been downloaded:

```json
{
  "dQw4w9WgXcQ": {
    "url": "https://youtube.com/watch?v=dQw4w9WgXcQ",
    "category": "traffic_jam",
    "downloaded_at": "2025-01-15T10:30:00"
  }
}
```

Before downloading, the collector extracts the video ID from the URL:

```python
def _extract_video_id(self, url: str) -> str | None:
    # Handles: watch?v=ID, youtu.be/ID, shorts/ID, embed/ID
    patterns = [
        r"(?:v=|youtu\.be/|shorts/|embed/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None
```

If the video ID is already in the JSON, the download is skipped. This is called **idempotent processing** — running the same command twice produces the same result without doing redundant work.

### Retry Logic with Exponential Backoff

Networks fail. YouTube rate-limits. The collector handles transient failures with an exponential backoff retry loop:

```python
def _download_with_retry(self, url: str, ydl_opts: dict, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            return True  # success
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)  # jitter
                time.sleep(wait_time)
    return False
```

**Why exponential backoff?**

Attempt 1 fails → wait 1 second
Attempt 2 fails → wait 2 seconds
Attempt 3 fails → wait 4 seconds

The doubling wait time (`2 ** attempt`) prevents hammering the server when it's struggling. The random `jitter` (`+ random.uniform(0, 1)`) prevents many parallel workers from retrying at the exact same moment — a problem known as the **thundering herd**.

---

## Source 2: Processing Physical Recordings

**File**: `src/processors/physical_audio_processor.py`
**Class**: `PhysicalAudioProcessor`

**File**: `src/processors/media_handler.py`
**Class**: `MediaHandler`

### What It Does

Handles audio and video files placed in `ml_data/physically_collected/<category>/`. These are field recordings — someone physically recorded sounds with a microphone or phone.

### Supporting 16 File Formats

Physical recordings come in all kinds of formats. The processor handles all of them:

| Audio Formats | Video Formats |
|---|---|
| .wav, .mp3, .m4a, .opus | .mp4, .mov, .avi, .mkv |
| .aac, .ogg, .flac, .wma | .webm, .flv, .wmv |

**For audio files**: pydub's `AudioSegment.from_file()` auto-detects the format and loads it:

```python
audio = AudioSegment.from_file(file_path)  # Works for any supported format
```

Under the hood, pydub delegates to FFmpeg for formats it can't natively read (like .opus, .m4a). Think of pydub as a clean Python interface sitting on top of FFmpeg.

**For video files**: FFmpeg extracts only the audio track:

```python
import subprocess

def extract_audio_from_video(self, video_path: str) -> str:
    output_path = video_path.replace(".mp4", ".mp3")
    command = [
        "ffmpeg", "-i", video_path,    # Input video
        "-q:a", "0",                   # Best audio quality
        "-map", "a",                   # Extract only the audio stream
        output_path
    ]
    subprocess.run(command, check=True, timeout=300)
    return output_path
```

The `-map a` flag tells FFmpeg: "I only want the audio track from this video — discard everything else."

### Lazy Extraction: Don't Repeat Work

Before running FFmpeg, the `MediaHandler` checks if the extracted audio file already exists:

```python
if output_path.exists():
    logger.info(f"Already extracted: {output_path}")
    return str(output_path)
```

This is the same principle as the YouTube deduplication — never redo work that was already done. Especially important since video-to-audio extraction can be slow for long files.

---

## The Audio Processor: Slicing Audio into ML-Ready Segments

**File**: `src/processors/audio_processor.py`
**Class**: `AudioProcessor`

### What It Does

Takes raw audio files (which might be 2 minutes, 10 minutes, or longer) and slices them into standardized **10-second mono WAV segments** at 48 kHz. This is the core transformation step.

### Why 10-Second Segments?

Machine learning models need fixed-size inputs. A 10-second window is long enough to contain meaningful audio events (a car horn, a bus engine rev) but short enough to be processed quickly. All popular audio datasets (UrbanSound8K, ESC-50) use segments in this range.

### How Segmentation Works

```python
from pydub import AudioSegment

def segment_audio(self, audio_path: str, category: str):
    audio = AudioSegment.from_wav(audio_path)
    segment_length_ms = 10_000  # 10 seconds in milliseconds

    for i, start_ms in enumerate(range(0, len(audio), segment_length_ms)):
        segment = audio[start_ms : start_ms + segment_length_ms]

        # Skip the last segment if it's shorter than 10 seconds
        if len(segment) < segment_length_ms:
            continue

        is_valid, dbfs, speech_pct = self.is_segment_valid(segment)
        if is_valid:
            self._export_segment(segment, category)
```

Think of it like slicing a loaf of bread — you start at one end, cut 10-second slices, and stop when you don't have enough bread left for a full slice.

### pydub: The Audio Processing Library

pydub treats audio like a list of samples. You can slice it with Python's list syntax:

```python
audio = AudioSegment.from_wav("recording.wav")

first_10_seconds = audio[0:10_000]    # Milliseconds, not seconds!
next_10_seconds  = audio[10_000:20_000]
```

The slicing syntax (`audio[start:end]`) uses **milliseconds** as the unit. This is a pydub convention — keep it in mind.

To export with specific audio parameters:

```python
segment.export(
    output_path,
    format="wav",
    parameters=["-ar", "48000", "-ac", "1"]  # 48 kHz, mono
)
```

`-ar 48000` sets the sample rate; `-ac 1` sets channels to 1 (mono). These are FFmpeg flags passed through pydub.

### Smart Sequential Numbering

When the pipeline saves a segment, it names it like `bike_0042.wav`. But what if the pipeline was interrupted earlier and `bike_0000.wav` through `bike_0041.wav` already exist?

The processor scans existing files and picks up from where it left off:

```python
def initialize_category_counter(self, category: str):
    processed_dir = Path(self.processed_dir) / category
    existing_files = list(processed_dir.glob(f"{category}_*.wav"))

    if not existing_files:
        self.category_counters[category] = 0
        return

    # Extract the number from each filename (e.g., "bike_0042.wav" → 42)
    numbers = []
    for f in existing_files:
        stem = f.stem  # "bike_0042"
        num_part = stem.split("_")[-1]  # "0042"
        numbers.append(int(num_part))

    # Start from the next number after the highest existing one
    self.category_counters[category] = max(numbers) + 1
```

This means you can safely interrupt the pipeline, run it again, and new segments will be numbered `_0043.wav`, `_0044.wav` — never overwriting or duplicating existing files.

---

## The Quality Gate: Filtering Before It Reaches Disk

**Method**: `AudioProcessor.is_segment_valid()` and `PhysicalAudioProcessor.is_segment_valid()`

This is one of the most important parts of the pipeline. A quality gate runs on every segment *before* it gets saved — if a segment doesn't pass, it's discarded immediately.

### Why Filter at Write Time?

> "Silent or near-silent segments waste storage and, more critically, introduce label noise if they reach training."

Label noise means: a segment labeled "bike" that contains only silence. When the model trains on this, it learns that silence = bike — corrupting the classifier. It's much cheaper to reject bad data here than to clean a corrupted dataset later.

### Two Quality Checks

**Check 1: Overall Loudness (dBFS)**

```python
dbfs = segment.dBfs  # pydub property: average loudness of the segment
if dbfs < self.silence_threshold_db:  # threshold: -45 dBFS
    return False, dbfs, 0.0  # reject: too quiet overall
```

`segment.dBfs` returns the RMS (Root Mean Square) loudness of the entire segment. If the whole segment is below -45 dBFS, it's essentially silence.

**Check 2: Speech/Content Percentage**

```python
from pydub.silence import detect_nonsilent

nonsilent_ranges = detect_nonsilent(
    segment,
    min_silence_len=100,      # Ignore gaps shorter than 100ms
    silence_thresh=-45,       # Anything below -45 dBFS is "silence"
)

# Calculate what percentage of the segment is non-silent
total_nonsilent_ms = sum(end - start for start, end in nonsilent_ranges)
speech_percentage = (total_nonsilent_ms / len(segment)) * 100

if speech_percentage < self.min_speech_percentage:  # threshold: 30%
    return False, dbfs, speech_percentage  # reject: too much silence
```

`detect_nonsilent` scans the audio and returns a list of `(start_ms, end_ms)` tuples representing regions that are louder than the silence threshold. If less than 30% of the segment is non-silent, the segment is rejected.

**Real-world analogy**: Imagine recording a street corner for 10 seconds. If a bus passes through in seconds 3-6 and the rest is quiet, you'd still keep it (60% non-silent). But if there's a brief horn beep in 1 second and 9 seconds of silence (10% non-silent), you'd discard it — it's not a representative sample.

---

## Quality Controller: Post-Processing Validation

**File**: `src/quality/quality_controller.py`
**Class**: `QualityController`

### What It Does

After all segments are saved, the QualityController runs a final verification pass over every saved `.wav` file in `ml_data/processed/`. It's a post-processing sanity check.

### Why a Second Validation Layer?

The quality gate in `AudioProcessor` filters based on audio content (loudness, silence). The `QualityController` checks a different set of properties — technical correctness:

```python
import librosa
import numpy as np

def verify_segment(self, audio_path: str) -> dict:
    y, sr = librosa.load(audio_path, sr=None)  # Load without resampling

    checks = {
        "sample_rate_ok": sr == 48000,                         # Correct sample rate?
        "duration_ok": abs(librosa.get_duration(y=y, sr=sr) - 10.0) < 0.2,  # ~10 seconds?
        "not_silent": np.max(np.abs(y)) > 0.01,               # Any actual signal?
    }

    checks["passes_all"] = all(checks.values())
    return checks
```

The results are exported to `ml_data/quality_report.csv` — a per-segment audit log you can inspect to understand how many segments passed and which ones failed, and why.

### librosa vs pydub: When to Use Which?

| Task | Library | Why |
|---|---|---|
| Slicing, exporting, format conversion | **pydub** | Simple Python API, millisecond indexing |
| Loading for feature extraction, duration, sample rate | **librosa** | Returns numpy arrays; integrates with ML code |
| Detecting non-silent regions | **pydub** | Built-in `detect_nonsilent` function |
| Computing MFCCs, spectrograms, zero-crossing rate | **librosa** | Industry-standard feature extraction |

They're complementary, not competing. pydub is the workhorse for manipulation; librosa is the toolbox for analysis.

---

## Data Validation with Pandera: Catching Bad Data Early

**File**: `src/processors/data_validator.py`

### What It Does

Validates the structure and content of CSV files before and after the pipeline runs, ensuring data quality at schema level.

### The Problem It Solves

Imagine someone edits `data/youtube_urls.csv` and accidentally puts a Spotify URL instead of a YouTube URL. Without validation, this would silently fail somewhere deep in the download logic — possibly after hours of processing. Pandera catches it immediately.

### Defining a Schema

```python
import pandera as pa
from pandera import Column, DataFrameSchema

YOUTUBE_URL_SCHEMA = DataFrameSchema({
    "category": Column(str, pa.Check(lambda x: x.str.len() > 0, error="category cannot be empty")),
    "url": Column(str, pa.Check(
        lambda x: x.str.contains("youtube.com|youtu.be"),
        error="URL must be a YouTube URL"
    )),
    "description": Column(str, nullable=True),  # Optional column
})
```

### Running Validation

```python
def validate_youtube_csv(csv_path: str) -> bool:
    df = pd.read_csv(csv_path)
    try:
        YOUTUBE_URL_SCHEMA.validate(df)
        logger.info(f"Validation passed: {len(df)} rows, {df['category'].nunique()} categories")
        return True
    except pa.errors.SchemaError as e:
        logger.error(f"Validation failed: {e}")
        return False
```

**Why Pandera instead of manual checks?**

Manual validation is repetitive (`if "url" not in df.columns: raise ValueError(...)`). Pandera lets you describe your data contract declaratively — what each column should look like — and it handles the validation logic for you. It also produces informative error messages that tell you exactly which row and column failed.

---

## MLflow: Tracking Experiments Like a Pro

**File**: `experiments/track_experiment.py`

### What It Does

Every time you train a model, MLflow records the hyperparameters you used, the metrics you got, and the model artifact — all in a local database you can browse in a web UI.

### Why Experiment Tracking Matters

Without tracking, you might train a RandomForest with `n_estimators=100` today and `n_estimators=200` tomorrow, get different results, but have no record of what settings produced what outcome. MLflow is like a lab notebook for ML experiments.

### How It Works

```python
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("experiments/mlruns")         # Store runs locally
mlflow.set_experiment("bangladeshi-audio-classifier") # Group runs by experiment name

with mlflow.start_run():
    # Log hyperparameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)

    # Train the model
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1)
    clf.fit(X_train, y_train)

    # Log metrics
    accuracy = clf.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_weighted", f1)

    # Save the trained model
    mlflow.sklearn.log_model(clf, "model")
```

### Browsing Runs

```bash
mlflow ui --port 5000
# Open http://localhost:5000 in your browser
```

You'll see a table of every run with its parameters and metrics, making it easy to compare which hyperparameter combination performed best.

---

## The Streamlit Demo: Interactive Inference with Explanations

**File**: `app/demo.py`

### What It Does

Provides an interactive web interface where you can upload a 10-second WAV file, see its waveform and spectrogram, and get a classification prediction with a SHAP explanation of which audio features drove the decision.

### Feature Extraction: Turning Audio into Numbers

ML models can't process raw audio waveforms directly — they need numerical features. The demo extracts 17 features from each audio file:

```python
import librosa

def extract_features(audio_path: str) -> np.ndarray:
    y, sr = librosa.load(audio_path, sr=48000, duration=10.0)

    # MFCCs: 13 coefficients — capture the "shape" of the audio spectrum
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)  # Average over time → 13 numbers

    # Additional timbral features
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rms_energy = np.mean(librosa.feature.rms(y=y))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))

    return np.hstack([mfcc_mean, spectral_centroid, spectral_bandwidth, rms_energy, zero_crossing_rate])
    # Total: 13 + 1 + 1 + 1 + 1 = 17 features
```

**What are MFCCs?** Mel-Frequency Cepstral Coefficients. They represent the "color" or texture of a sound in a way that matches how human hearing works. A truck engine and a car engine have different MFCC profiles, which is what lets the model distinguish them.

**What do the other features capture?**

| Feature | What It Captures | Intuition |
|---|---|---|
| **Spectral Centroid** | "Brightness" of the sound | High-pitched sirens have a high centroid |
| **Spectral Bandwidth** | How spread out the frequencies are | Broadband traffic noise vs. narrow-band siren |
| **RMS Energy** | Loudness of the signal | A truck is louder than a bicycle |
| **Zero Crossing Rate** | How often the signal crosses zero | Higher for noisy signals, lower for tonal sounds |

### Visualization

The demo generates two visualizations to help users understand the audio:

```python
import librosa.display
import matplotlib.pyplot as plt

# Waveform: amplitude over time
fig, ax = plt.subplots()
librosa.display.waveshow(y, sr=sr, ax=ax)
st.pyplot(fig)

# Mel spectrogram: frequency content over time
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
S_db = librosa.power_to_db(S, ref=np.max)  # Convert to dB scale
fig, ax = plt.subplots()
librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
st.pyplot(fig)
```

A **mel spectrogram** is like a heat map of sound — the x-axis is time, the y-axis is frequency (on the mel scale, which matches human perception), and the color shows loudness. Patterns in this heatmap are what CNN models learn to classify.

### SHAP: Explaining the Model's Decision

SHAP (SHapley Additive exPlanations) answers the question: "Which features most influenced this specific prediction?"

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(features)

shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[predicted_class][0],
        base_values=explainer.expected_value[predicted_class],
        feature_names=feature_names
    )
)
```

The waterfall plot shows each feature's contribution as a bar — red bars pushed the prediction toward the predicted class, blue bars pushed it away. A feature with a long red bar is a strong positive indicator for that sound category.

**Real-world analogy**: You predicted rain today. SHAP would show: "dark clouds (+80%), humidity (+60%), clear sky forecast (-40%)." You can see exactly what drove the prediction.

### Caching the Model

```python
@st.cache_resource
def load_model():
    with open("models/classifier.pkl", "rb") as f:
        return pickle.load(f)
```

`@st.cache_resource` tells Streamlit: "Load this once and reuse it for every subsequent request." Without this, the model would be reloaded from disk on every user interaction — making the app feel very slow.

---

## Logging: Knowing What Your Pipeline Is Doing

**File**: `src/utils/helpers.py`

### What It Does

Every module uses the same logging setup so you always know what the pipeline is doing — both in your terminal and in a persistent log file.

### The Logger Factory

```python
import logging
from logging.handlers import RotatingFileHandler

def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Already configured — avoid duplicate handlers

    formatter = logging.Formatter(
        "%(asctime)s — %(name)s — %(levelname)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler: logs to your terminal
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File handler: logs to ml_data/pipeline.log (rotates at 10 MB)
    file_handler = RotatingFileHandler("ml_data/pipeline.log", maxBytes=10_000_000, backupCount=3)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(level)
    return logger
```

### How to Use It

Every module gets its own named logger:

```python
# In youtube_collector.py
from src.utils.helpers import get_logger
logger = get_logger(__name__)  # __name__ = "src.collectors.youtube_collector"

logger.info("Starting download: https://youtube.com/...")
logger.warning("Retry attempt 2/3 for video dQw4w9WgXcQ")
logger.error("Download failed after 3 attempts")
```

**Why named loggers?** The `__name__` convention means each module's logs are tagged with its full module path (e.g., `src.collectors.youtube_collector`). When you have a problem, you immediately know which module generated the log line.

**Why a rotating file handler?** The `RotatingFileHandler` automatically creates a new log file when the current one reaches 10 MB, keeping up to 3 old files. Without rotation, a long-running pipeline would create a single log file that grows indefinitely.

---

## Testing Strategy: How the Pipeline Verifies Itself

**Directory**: `tests/`

### Philosophy

Tests cover the core pipeline logic — the parts that, if broken, would silently corrupt data or produce bad training sets. Synthetic audio (generated in Python) is used instead of real audio files, so tests are fast and self-contained.

### Generating Synthetic Audio for Tests

```python
import numpy as np
from pydub import AudioSegment

def make_sine_segment(duration_ms: int = 10_000, freq: int = 1000) -> AudioSegment:
    """Generate a pure sine wave tone for testing."""
    sample_rate = 48000
    samples = np.sin(2 * np.pi * freq * np.linspace(0, duration_ms / 1000, int(sample_rate * duration_ms / 1000)))
    samples_int = (samples * 32767).astype(np.int16)  # Convert to 16-bit PCM
    return AudioSegment(
        samples_int.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    )
```

A 1000 Hz sine wave is loud (it won't be rejected by the quality gate) and perfectly predictable (you know exactly what the test audio sounds like). Real audio would make tests fragile and slow.

### What's Tested

| Test File | What It Tests |
|---|---|
| `test_collector.py` | URL parsing (4 YouTube URL formats), deduplication, metadata save/load |
| `test_processor.py` | Quality gate (loud vs silent), counter resumption, 30s → 3 segments |
| `test_physical_processor.py` | Multi-format loading, starting number detection |
| `test_quality.py` | Sample rate check, duration check, silence detection |

### Example Test: Quality Gate

```python
def test_valid_segment_passes(self, processor):
    loud_segment = make_sine_segment(duration_ms=10_000, freq=1000)
    is_valid, dbfs, speech_pct = processor.is_segment_valid(loud_segment)

    assert is_valid is True
    assert dbfs > -45      # Loud enough
    assert speech_pct > 30  # Enough non-silent content


def test_silent_segment_rejected(self, processor):
    silent_segment = AudioSegment.silent(duration=10_000)
    is_valid, dbfs, speech_pct = processor.is_segment_valid(silent_segment)

    assert is_valid is False
    assert dbfs < -45  # Too quiet
```

### Running the Tests

```bash
pytest tests/ -v
```

The `-v` flag (verbose) shows each test name and whether it passed or failed, rather than just a dot for each test.

---

## CLI Scripts: How Everything Is Wired Together

The `scripts/` directory contains the entry points you actually run. Each script orchestrates the pipeline modules in a specific order.

### Collecting from YouTube

```bash
python scripts/collect_audio.py
```

**What happens internally:**
1. `YouTubeAudioCollector.download_from_csv("data/youtube_urls.csv")` — downloads new videos only
2. `AudioProcessor.process_specific_files(newly_downloaded)` — segments only the new downloads
3. `QualityController.verify_all()` — validates all segments, writes `quality_report.csv`

This script is **incremental** — run it daily to keep the dataset growing without reprocessing anything.

### Processing Physical Recordings

```bash
python scripts/process_physical.py
```

**What happens internally:**
1. `MediaHandler.preprocess_videos()` — extracts audio from any video files
2. `PhysicalAudioProcessor.process_all_categories()` — segments all recordings
3. `QualityController.verify_all()`

### Processing a Single Category

```bash
python scripts/process_physical_category.py bike
```

Useful when you've added new recordings to just one category and don't want to re-process everything.

### Launching the Demo

```bash
streamlit run app/demo.py
```

Opens a local web server (usually at `http://localhost:8501`) with the interactive classification interface.

### Viewing Experiment Results

```bash
python experiments/track_experiment.py  # Train and log a new run
mlflow ui --port 5000                  # Browse all runs at http://localhost:5000
```

---

## Library Reference

A quick reference of every library in this project and why it was chosen.

| Library | Role | Why It Was Chosen |
|---|---|---|
| **yt-dlp** | YouTube audio download | Modern, maintained; handles format conversion via FFmpeg post-processors |
| **pydub** | Audio slicing, silence detection, format export | Intuitive Python API with millisecond indexing; wraps FFmpeg |
| **librosa** | Feature extraction (MFCCs, spectrograms), audio loading | De-facto standard for audio ML; returns numpy arrays |
| **numpy** | Numerical operations on audio arrays | Fast, universal; used by librosa and for signal analysis |
| **pandas** | CSV I/O for metadata tracking | Efficient tabular data; simple read/write API |
| **pandera** | Schema validation for input/output CSVs | Declarative contracts; catches bad data with informative errors |
| **PyYAML** | Loading `config.yaml` | Human-readable config format; `safe_load` prevents code injection |
| **PyTorch** | Deep learning model training | Industry standard; GPU acceleration for CNN and Wav2Vec2 |
| **transformers** | Wav2Vec2 fine-tuning | Hugging Face ecosystem; pre-trained audio embeddings |
| **scikit-learn** | RandomForest classifier, train/test split | Quick baselines; integrates with MLflow and SHAP |
| **mlflow** | Experiment tracking | Logs params, metrics, and model artifacts; web UI included |
| **streamlit** | Interactive demo UI | No backend needed; Python-only; hot-reloads on file save |
| **shap** | Feature importance explanations | Model-agnostic; TreeExplainer for RandomForest |
| **matplotlib** | Waveform and spectrogram plots in Streamlit | librosa.display integrates directly with matplotlib |
| **pathlib** | Cross-platform file path operations | Modern Python; avoids OS-specific path separator bugs |
| **subprocess** | Running FFmpeg from Python | Standard library; no extra dependencies |
| **pytest** | Unit testing | Standard Python testing; simple fixture system |
| **tqdm** | Progress bars for long-running loops | One-line wrapper around any iterator |

---

## Next Steps

Now that you understand the full pipeline, here's a suggested learning path:

1. **Run the pipeline on a small dataset** — create 2-3 categories with a few YouTube URLs and see the segments appear in `ml_data/processed/`
2. **Inspect the quality report** — open `ml_data/quality_report.csv` and look at which segments passed and failed
3. **Read the tests** — start with `tests/test_processor.py`; the test code shows exactly how each component is expected to behave
4. **Add a new category** — place recordings in `ml_data/physically_collected/your_category/` and run `process_physical_category.py your_category`
5. **Adapt to your own domain** — the pipeline is completely domain-agnostic; any sound classification task can replace the Bangladeshi urban audio example

---

*Questions, suggestions, or corrections? Open an issue or pull request.*
