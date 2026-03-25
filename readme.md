# Bangladesh Audio Data Collection & Processing Pipeline

[![CI](https://github.com/MehediHasan-75/bangladesh-audio-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/MehediHasan-75/bangladesh-audio-ml/actions/workflows/ci.yml)

A comprehensive Python-based audio collection, processing, and quality control pipeline designed for building machine learning datasets. This project handles audio from multiple sources (YouTube and physically collected) in various formats, processes them into standardized 10-second WAV segments, and performs quality validation.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage Guide](#detailed-usage-guide)
- [Configuration](#configuration)
- [File Format Reference](#file-format-reference)
- [Troubleshooting](#troubleshooting)
- [Project Architecture](#project-architecture)

---

## ✨ Features

- **YouTube Audio Collection**: Download audio from YouTube URLs with automatic duplicate detection
- **Multi-Format Support**: Process audio in `.opus`, `.m4a`, `.mp3`, `.aac`, `.wav`, `.ogg`, `.flac`, `.wma`
- **Intelligent Segmentation**: Slice audio into 10-second segments with smart numbering
- **Smart Numbering**: Automatically continue numbering from existing segments
- **Quality Control**: Validate segments for silence, speech content, and audio properties
- **Metadata Tracking**: Record processing details in CSV files
- **Incremental Processing**: Only process new files, skip already processed content
- **Robust Error Handling**: Continue processing even if individual files fail

---

## 📁 Project Structure

```
bangladesh-audio-ml/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore rules
│
├── config/
│   └── config.yaml                     # Configuration settings
│
├── data/
│   └── youtube_urls.csv                # YouTube URLs to download
│
├── ml_data/                            # Data directory (created at runtime)
│   ├── raw/                            # YouTube downloads
│   │   ├── motorcycle_engine/
│   │   ├── car_horn/
│   │   └── [other categories]/
│   ├── physically_collected/           # Manually recorded audio
│   │   ├── bike/
│   │   ├── truck/
│   │   ├── bus/
│   │   └── [other categories]/
│   ├── processed/                      # Output segments (10s WAV files)
│   │   ├── motorcycle_engine/
│   │   ├── bike/
│   │   └── [other categories]/
│   ├── download_metadata.json          # YouTube download history
│   ├── processing_metadata.csv         # Processing details
│   ├── physical_processing_metadata.csv
│   └── quality_report.csv              # Quality control results
│
├── src/
│   ├── __init__.py
│   ├── collectors/
│   │   ├── __init__.py
│   │   └── youtube_collector.py        # YouTube downloader
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── audio_processor.py          # Audio segmentation
│   │   ├── format_converter.py         # Format conversion utilities
│   │   └── physical_audio_processor.py # Processes physical collection
│   ├── quality/
│   │   ├── __init__.py
│   │   └── quality_controller.py       # QA validation
│   └── utils/
│       ├── __init__.py
│       └── helpers.py                  # Utility functions
│
├── scripts/
│   ├── collect_audio.py                # Main: Download & process YouTube
│   ├── process_raw.py                  # Batch: Process all raw YouTube files
│   ├── process_physical.py             # Main: Process all physical audio
│   ├── process_physical_category.py    # Process single category
│   └── convert_opus.py                 # Convert OPUS to WAV
│
├── notebooks/
│   └── exploratory_analysis.ipynb      # Data exploration & visualization
│
└── tests/
    ├── __init__.py
    └── test_physical_processor.py       # Unit tests
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- FFmpeg (required by pydub for audio processing)

### Step 1: Clone Repository

```bash
git clone https://github.com/MehediHasan-75/bangladesh-audio-ml.git
cd bangladesh-audio-ml
```

### Step 2: Install FFmpeg

**macOS (using Homebrew):**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg
```

**Windows (using Chocolatey):**
```bash
choco install ffmpeg
```

**Or download from:** https://ffmpeg.org/download.html

### Step 3: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Verify Installation

```bash
# Test FFmpeg
ffmpeg -version

# Test Python packages
python -c "import pydub, librosa, pandas; print('✅ All packages installed')"
```

---

## ⚡ Quick Start

### YouTube Pipeline (5 minutes)

**1. Prepare YouTube URLs:**

Edit `data/youtube_urls.csv`:
```csv
url,category
https://www.youtube.com/watch?v=dQw4w9WgXcQ,motorcycle_engine
https://www.youtube.com/watch?v=dQw4w9WgXcQ,car_horn
https://www.youtube.com/watch?v=dQw4w9WgXcQ,motorcycle_engine
```

**2. Download and process:**

```bash
python scripts/collect_audio.py
```

**Expected output:**
```
============================================================
BANGLADESH AUDIO DATA COLLECTION
============================================================

STEP 1: Downloading from YouTube...
============================================================
📥 Downloading motorcycle_engine...
  ✅ Downloaded: sample_1.wav

STEP 2: Processing ONLY newly downloaded files...
============================================================
🎵 PROCESSING 1 NEW FILES

📁 motorcycle_engine: 1 files
[1/1] sample_1.wav (120.5s)
  ✅ Created 12/12 segments (rejected 0)

STEP 3: Quality control...
============================================================

QUALITY CONTROL REPORT
============================================================
Total files: 12
Passed: 12
Failed: 0
Pass rate: 100.0%

✓ COLLECTION COMPLETE!
New segments created: 12
============================================================
```

### Physical Audio Pipeline (5 minutes)

**1. Organize your audio files:**

```
ml_data/physically_collected/
├── bike/
│   ├── recording1.opus
│   ├── recording2.m4a
│   └── recording3.mp3
├── truck/
│   ├── sound1.aac
│   └── sound2.wav
└── bus/
    ├── audio1.flac
    └── audio2.wav
```

**2. Process all categories:**

```bash
python scripts/process_physical.py
```

**3. Or process specific category:**

```bash
python scripts/process_physical_category.py bike
```

---

## 📖 Detailed Usage Guide

### Workflow 1: Download & Process YouTube Audio (Recommended for First Run)

Use when downloading new YouTube content and you want to process it immediately.

```bash
python scripts/collect_audio.py
```

**What it does:**
1. ✅ Downloads new audio files from `data/youtube_urls.csv`
2. ✅ Checks for duplicates (avoids re-downloading)
3. ✅ Processes ONLY newly downloaded files
4. ✅ Segments into 10-second WAV files
5. ✅ Numbers sequentially (respects existing numbers)
6. ✅ Runs quality control
7. ✅ Saves metadata

**Output files created:**
- `ml_data/raw/[category]/` - Raw downloaded audio
- `ml_data/processed/[category]/[category]_0000.wav` - Segmented audio
- `ml_data/download_metadata.json` - Download history
- `ml_data/processing_metadata.csv` - Processing details
- `ml_data/quality_report.csv` - QA results

### Workflow 2: Batch Process All Raw YouTube Files

Use when you want to re-process all existing YouTube downloads (e.g., after adjusting parameters).

```bash
python scripts/process_raw.py
```

**What it does:**
1. ✅ Processes ALL files in `ml_data/raw/` (not just new ones)
2. ✅ Segments and validates
3. ✅ Runs quality control

**Warning:** This will attempt to re-segment all files. Use sparingly!

### Workflow 3: Process All Physically Collected Audio

Use when you have manually recorded audio in `ml_data/physically_collected/`.

```bash
python scripts/process_physical.py
```

**What it does:**
1. ✅ Finds all audio files in `ml_data/physically_collected/[category]/`
2. ✅ Supports multiple formats (`.opus`, `.m4a`, `.mp3`, `.aac`, `.wav`, `.ogg`, `.flac`, `.wma`)
3. ✅ Segments into 10-second chunks
4. ✅ Smart numbering (continues from existing segments)
5. ✅ Validates quality
6. ✅ Saves metadata

**Example:**

If `ml_data/processed/bike/` already has `bike_0000.wav` through `bike_0056.wav`, new segments will start from `bike_0057.wav`.

```bash
python scripts/process_physical.py
```

Output:
```
============================================================
📁 Processing category: bike
============================================================
  Found 5 audio files
  📁 bike: continue from 0057 (found 56 existing)
  [1/5] recording1.opus (45.3s)
    ✅ Created 4/4 segments (rejected 0)
  [2/5] recording2.m4a (32.1s)
    ✅ Created 3/3 segments (rejected 0)
...
```

### Workflow 4: Process Single Physical Category

Use when processing new audio for a specific category.

```bash
python scripts/process_physical_category.py bike
```

Replace `bike` with your category name.

### Workflow 5: Convert OPUS Files to WAV

Use as a preprocessing step before the main pipeline.

```bash
python scripts/convert_opus.py
```

This converts all `.opus` files in a folder to `.wav` format.

---

## ⚙️ Configuration

Edit `config/config.yaml` to customize parameters:

```yaml
# Audio Processing Configuration

audio:
  segment_duration_ms: 10000           # Segment length in milliseconds
  sample_rate: 48000                   # Output sample rate (Hz)
  channels: 1                          # Mono (1) or Stereo (2)
  
quality:
  silence_threshold_db: -45            # Silence detection threshold
  min_speech_percentage: 30            # Minimum content percentage

paths:
  base_dir: "ml_data"
  raw_dir: "ml_data/raw"
  physically_collected_dir: "ml_data/physically_collected"
  processed_dir: "ml_data/processed"

download:
  format: "wav"
  quality: "best"

supported_formats:
  - .opus
  - .m4a
  - .mp3
  - .aac
  - .wav
  - .ogg
  - .flac
  - .wma
```

### Key Parameters Explained

**`segment_duration_ms: 10000`**
- Segment length in milliseconds (10000ms = 10 seconds)
- Smaller = more segments, but may lose context
- Larger = fewer segments, but may be too long

**`silence_threshold_db: -45`**
- Volume threshold for detecting silence (in decibels)
- Lower value = more tolerant of quiet audio
- Higher value = stricter silence detection

**`min_speech_percentage: 30`**
- Minimum percentage of segment that must contain speech/sound
- Segments below this threshold are rejected
- Helps filter out mostly-silent clips

---

## 📊 File Format Reference

### Input Formats

All these audio formats are automatically supported:

| Format | Extension | Notes |
|--------|-----------|-------|
| OPUS | `.opus` | Used by YouTube, Discord |
| MP4 Audio | `.m4a` | Apple, iTunes |
| MP3 | `.mp3` | Standard MP3 format |
| AAC | `.aac` | Advanced Audio Codec |
| WAV | `.wav` | Uncompressed (largest file size) |
| OGG | `.ogg` | Open-source format |
| FLAC | `.flac` | Lossless compression |
| WMA | `.wma` | Windows Media Audio |

### Output Format

All segments are saved as:
- **Format:** WAV (PCM)
- **Sample Rate:** 48 kHz
- **Channels:** Mono (1)
- **Duration:** ~10 seconds
- **Naming:** `{category}_{number:04d}.wav`

Examples:
```
bike_0000.wav
bike_0001.wav
bike_0056.wav
truck_0000.wav
truck_0001.wav
```

### Metadata CSV Format

`processing_metadata.csv`:

| Column | Description | Example |
|--------|-------------|---------|
| original_file | Source file path | `ml_data/raw/bike/video.opus` |
| original_format | Original file format | `.opus` |
| segment_file | Output segment path | `ml_data/processed/bike/bike_0000.wav` |
| category | Category name | `bike` |
| segment_number | Segment index | `0` |
| duration_s | Segment duration | `10.0` |
| dbfs | Audio volume (dB) | `-25.5` |
| speech_percentage | Content percentage | `65.3` |

---

## 🔍 Monitoring & Inspection

### Check Processing Progress

View the last processed segment number for a category:

```bash
ls -la ml_data/processed/bike/ | tail -5
```

Output:
```
-rw-r--r--  1 user  staff  960000 Nov 17 11:30 bike_0054.wav
-rw-r--r--  1 user  staff  960000 Nov 17 11:30 bike_0055.wav
-rw-r--r--  1 user  staff  960000 Nov 17 11:30 bike_0056.wav
```

### View Quality Report

```bash
cat ml_data/quality_report.csv
```

### Check Processing Statistics

```bash
# Count segments by category
for cat in ml_data/processed/*/; do
    echo "$(basename $cat): $(ls $cat/*.wav | wc -l) files"
done
```

Output:
```
bike: 57 files
truck: 43 files
bus: 28 files
```

---

## 🐛 Troubleshooting

### Error: "FFmpeg not found"

**Solution:** Install FFmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
choco install ffmpeg
```

### Error: "ModuleNotFoundError: No module named 'pydub'"

**Solution:** Reinstall dependencies

```bash
pip install -r requirements.txt
```

### Error: "youtube_urls.csv not found"

**Solution:** Create the file with correct format:

```bash
mkdir -p data
cat > data/youtube_urls.csv << 'EOF'
url,category
https://www.youtube.com/watch?v=YOUR_VIDEO_ID,motorcycle_engine
https://www.youtube.com/watch?v=ANOTHER_ID,car_horn
EOF
```

### Error: "No new files to process"

This is normal! It means:
- All URLs in CSV were already downloaded, OR
- All URLs failed to download

Check the download history:

```bash
cat ml_data/download_metadata.json
```

### Error: "Physically collected folder not found"

**Solution:** Create the folder structure:

```bash
mkdir -p ml_data/physically_collected/{bike,truck,bus}
```

Then add your audio files to the category folders.

### Error: "Failed to load audio file: [Errno 2] No such file or directory"

This usually means pydub couldn't find the audio file or it's in an unsupported format.

**Solutions:**
1. Ensure file exists and is readable: `ls -la file.opus`
2. Try converting the file first: `python scripts/convert_opus.py`
3. Check that FFmpeg is installed and working: `ffmpeg -version`

### Segments Not Being Created

**Common causes:**

1. **Too much silence:** Segments filtered out by `min_speech_percentage`
   - Lower the threshold in `config/config.yaml`
   - Or reduce `silence_threshold_db` to `-50`

2. **Audio file too short:** Needs to be longer than `segment_duration_ms`
   - Files shorter than 10s won't produce any segments

3. **Corrupted audio file:** Cannot be decoded
   - Try converting to WAV first
   - Open in Audacity to verify

**Debug:**

Add verbose logging to see which segments are rejected:

```python
# In physical_audio_processor.py, modify print statement:
if not is_valid:
    print(f"      ⚠️  Segment {i} rejected: dbfs={dbfs:.1f}, speech%={speech_pct:.1f}%")
```

### Memory Issues with Large Files

If processing very large audio files causes memory errors:

1. **Increase available memory:**
   ```bash
   ulimit -v unlimited
   ```

2. **Process files one at a time:**
   ```bash
   python scripts/process_physical_category.py bike
   ```

3. **Split large files before processing:**
   Use Audacity or FFmpeg to split into smaller chunks first.

---

## 🏗️ Project Architecture

### Data Flow Diagram

```
YouTube URLs (CSV)
       ↓
[YouTubeAudioCollector]
       ↓
Raw Audio Files (OPUS, MP3, etc.)
       ↓
[AudioProcessor] / [PhysicalAudioProcessor]
       ↓
10-second WAV Segments
       ↓
[QualityController]
       ↓
Validated Dataset (CSV metadata)
```

### Module Responsibilities

**collectors/youtube_collector.py**
- Downloads audio from YouTube URLs
- Detects and skips duplicates
- Manages download history in JSON

**processors/audio_processor.py**
- Segments audio into 10-second chunks
- Validates audio quality
- Handles YouTube-downloaded files

**processors/physical_audio_processor.py**
- Processes manually collected audio
- Supports multiple input formats
- Smart numbering from existing files

**quality/quality_controller.py**
- Validates sample rate (48kHz)
- Checks segment duration (~10s)
- Detects silent/invalid segments
- Generates quality report

---

## 📝 Common Tasks

### Add More YouTube Videos

Edit `data/youtube_urls.csv`:

```csv
url,category
https://www.youtube.com/watch?v=dQw4w9WgXcQ,motorcycle_engine
https://www.youtube.com/watch?v=NEW_VIDEO_ID,motorcycle_engine
https://www.youtube.com/watch?v=ANOTHER_ID,car_horn
```

Then run:
```bash
python scripts/collect_audio.py
```

### Add Manually Recorded Audio

1. Create folder: `ml_data/physically_collected/your_category/`
2. Place audio files in the folder
3. Run: `python scripts/process_physical.py`

### Change Segment Length

Edit `config/config.yaml`:

```yaml
audio:
  segment_duration_ms: 5000  # 5 seconds instead of 10
```

Then re-process. Note: This creates new segment numbers starting from 0.

### Change Quality Thresholds

Edit `config/config.yaml`:

```yaml
quality:
  silence_threshold_db: -50        # More tolerant of quiet audio
  min_speech_percentage: 20        # Accept more silent segments
```

### Export All Filenames

```bash
find ml_data/processed -name "*.wav" | sort > filenames.txt
cat filenames.txt
```

### Generate Statistics

```bash
python << 'EOF'
from pathlib import Path
import pandas as pd

# Count segments by category
base = Path("ml_data/processed")
stats = {}

for category_dir in base.iterdir():
    if category_dir.is_dir():
        files = list(category_dir.glob("*.wav"))
        stats[category_dir.name] = len(files)

df = pd.DataFrame(list(stats.items()), columns=['Category', 'Segments'])
df['Duration (hours)'] = df['Segments'] * 10 / 3600  # 10s per segment
print(df.to_string(index=False))
print(f"\nTotal segments: {df['Segments'].sum()}")
print(f"Total duration: {df['Duration (hours)'].sum():.2f} hours")
EOF
```

---

## 📚 Advanced Usage

### Custom Processing Parameters

Modify scripts to use different parameters:

```python
# In scripts/process_physical.py
processor = PhysicalAudioProcessor(
    base_dir="ml_data",
    segment_duration=15000,         # 15 seconds
    silence_threshold_db=-50,       # More tolerant
    min_speech_percentage=20        # Accept quieter segments
)
```

### Batch Processing Multiple Projects

Create separate data directories:

```bash
mkdir -p project1/ml_data
mkdir -p project2/ml_data

# Process project1
cd project1
python ../scripts/process_physical.py

# Process project2
cd ../project2
python ../scripts/process_physical.py
```

### Integration with ML Training

```python
import pandas as pd
from pathlib import Path

# Load metadata
metadata = pd.read_csv("ml_data/processing_metadata.csv")

# Filter only high-quality segments
quality = pd.read_csv("ml_data/quality_report.csv")
good_segments = quality[quality['passes_all'] == True]

# Load audio for training
import librosa
for seg_path in good_segments['file']:
    y, sr = librosa.load(seg_path, sr=48000)
    # Use for model training
```

---

## 📞 Support & FAQ

**Q: How long does processing take?**
A: ~100-200 MB per minute, depending on machine. A 1-hour audio file creates ~360 segments and takes 2-3 minutes.

**Q: Can I pause and resume processing?**
A: Yes! The smart numbering system remembers the last segment, so you can safely interrupt and restart.

**Q: How do I prevent duplicate segments?**
A: The quality controller filters out segments that are mostly silent. Adjust `min_speech_percentage` in config if needed.

**Q: Can I mix YouTube and physical audio?**
A: Yes! They're processed into the same `ml_data/processed/` directory.

**Q: How do I delete segments?**
A: Simply delete the `.wav` files. The numbering won't reset unless you manually renumber.

---

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

Built for efficient audio dataset creation for machine learning projects.

---

## Version History

- **v1.0** (Nov 2025): Initial release
  - YouTube collection
  - Physical audio processing
  - Multi-format support
  - Quality control pipeline
