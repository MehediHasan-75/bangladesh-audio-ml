# Physical Audio Processing: Complete Learning Guide

## Table of Contents
1. [Fundamentals](#fundamentals)
2. [Audio Concepts](#audio-concepts)
3. [Code Walkthrough](#code-walkthrough)
4. [Hands-On Implementation](#hands-on-implementation)
5. [Debugging & Optimization](#debugging--optimization)
6. [Integration with Your ML Pipeline](#integration-with-your-ml-pipeline)

---

## Fundamentals

### Why This Matters for ML

Audio datasets require preprocessing because raw audio files are:
- **Format-heterogeneous**: Users record in different codecs (MP3, M4A, WAV, etc.)
- **Duration-inconsistent**: Recordings vary in length unpredictably
- **Quality-variable**: Some segments contain silence, noise, or inadequate signal
- **Unlabeled**: Need to organize by category and maintain provenance

Your processor solves all four problems by creating a **standardized, validated, labeled dataset**.

### Core Workflow

```
Raw Audio Files (various formats)
    ↓
Load & Parse (pydub handles format detection)
    ↓
Segment (divide into fixed-duration chunks)
    ↓
Validate (check volume & speech content)
    ↓
Export (standardized WAV format)
    ↓
Track (metadata CSV for reproducibility)
    ↓
ML Training (uniform input to models)
```

---

## Audio Concepts

### Digital Audio Representation

When you load an audio file, it becomes:
- **Sample Rate (Hz)**: How many samples per second (e.g., 48kHz = 48,000 samples/sec)
- **Bit Depth**: Precision per sample (e.g., 16-bit, 24-bit)
- **Channels**: Mono (1), Stereo (2), Surround (5.1), etc.
- **Duration**: Total time in seconds

Example: A 10-second 48kHz mono audio = 480,000 individual samples

```python
from pydub import AudioSegment

audio = AudioSegment.from_file("recording.mp3")
print(f"Duration: {audio.duration_seconds} seconds")
print(f"Channels: {audio.channels}")
print(f"Sample rate: {audio.frame_rate} Hz")
print(f"Bit depth: {audio.sample_width} bytes")
```

### Decibels (dBFS - Decibels relative to Full Scale)

dBFS measures **relative loudness** on a logarithmic scale:
- **0 dBFS**: Maximum possible volume (clipping/distortion)
- **-3 dBFS**: Half the amplitude (perceptually much quieter)
- **-45 dBFS**: Very quiet (threshold in this code)
- **-∞ dBFS**: Silence

```python
segment.dBFS  # Returns current volume in dBFS
# Example: -25.5 means moderately loud but not clipping
```

**Why -45 dBFS threshold?** 
- Filters out background noise and environment sounds
- Preserves actual speech/target audio
- Adjustable based on your microphone quality

### Non-Silent Detection

The `detect_nonsilent()` function finds time ranges where audio exceeds silence threshold:

```python
nonsilent_chunks = detect_nonsilent(
    segment,
    min_silence_len=100,    # Ignore silence gaps < 100ms
    silence_thresh=-45      # Threshold for "silence"
)
# Returns: [(start_ms, end_ms), (start_ms, end_ms), ...]
# Example: [(200, 2500), (3100, 8900)]  → Total speech: 7.2 seconds
```

Speech percentage = (Total non-silent duration / Segment duration) × 100

For a 10-second segment with 3.5 seconds of speech → 35% speech content

---

## Code Walkthrough

### 1. Initialization

```python
processor = PhysicalAudioProcessor(
    base_dir="ml_data",              # Root directory
    segment_duration=10000,           # 10 seconds in milliseconds
    silence_threshold_db=-45,         # Volume threshold
    min_speech_percentage=30          # Minimum 30% content
)
```

**Directory Structure Expected:**
```
ml_data/
├── physically_collected/     # INPUT: Raw audio files
│   ├── bike/
│   │   ├── recording1.mp3
│   │   ├── recording2.wav
│   │   └── recording3.opus
│   ├── truck/
│   │   ├── sound1.m4a
│   │   └── sound2.aac
│   └── car/
│       └── noise.flac
└── processed/                # OUTPUT: Segmented audio
    ├── bike/
    │   ├── bike_0000.wav
    │   ├── bike_0001.wav
    │   └── ...
    ├── truck/
    │   ├── truck_0000.wav
    │   └── ...
    └── car/
        └── car_0000.wav
```

### 2. Loading Audio Files

```python
def _load_audio_file(self, file_path: Path) -> Optional[AudioSegment]:
    try:
        audio = AudioSegment.from_file(str(file_path))
        return audio
    except Exception as e:
        print(f"❌ Failed to load {file_path.name}: {e}")
        return None
```

**How pydub auto-detects format:**
- Reads file extension (.mp3, .wav, etc.)
- Calls appropriate FFmpeg codec
- Returns AudioSegment object (uniform interface)

**Common errors & solutions:**
- `CouldntDecodeError`: FFmpeg not installed or corrupted file
  - Solution: `brew install ffmpeg` (macOS)
- `FileNotFoundError`: Path doesn't exist
  - Solution: Check `physically_collected/` directory structure

### 3. Segmentation Logic

```python
num_segments = len(audio) // self.segment_duration
# len(audio) returns milliseconds
# Example: 45000ms audio ÷ 10000ms = 4 segments + 5000ms leftover

for i in range(num_segments):
    start_ms = i * self.segment_duration      # 0, 10000, 20000, ...
    end_ms = (i + 1) * self.segment_duration  # 10000, 20000, 30000, ...
    segment = audio[start_ms:end_ms]
    # Python slicing syntax: audio[start:end] in milliseconds
```

**Why discard partial segments?**
- Ensures all training samples have identical length (required by neural networks)
- If you need the last 5 seconds, adjust `segment_duration` or pad with silence

### 4. Validation (Core Logic)

```python
def is_segment_valid(self, segment: AudioSegment) -> Tuple[bool, float, float]:
    dbfs = segment.dBFS  # Get volume
    
    # First check: Too quiet?
    if dbfs < self.silence_threshold_db:  # -45 dBFS
        return False, dbfs, 0.0
    
    # Second check: Enough speech content?
    nonsilent_chunks = detect_nonsilent(
        segment,
        min_silence_len=100,
        silence_thresh=self.silence_threshold_db
    )
    
    nonsilent_duration = sum(end - start for start, end in nonsilent_chunks)
    speech_percentage = (nonsilent_duration / len(segment)) * 100
    
    if speech_percentage < self.min_speech_percentage:  # 30%
        return False, dbfs, speech_percentage
    
    return True, dbfs, speech_percentage
```

**Example walkthrough:**
```
Input: 10-second segment from recording
1. Check dBFS: -32 dBFS → passes (> -45)
2. Detect silence: finds 8 seconds of speech, 2 seconds of silence
3. Calculate: 8000ms / 10000ms = 80% speech → passes (> 30%)
4. Return: (True, -32.0, 80.0)  ✅ Accepted

vs.

Input: 10-second segment of mostly noise
1. Check dBFS: -50 dBFS → fails (< -45)
2. Return: (False, -50.0, 0.0)  ❌ Rejected
```

### 5. File Numbering (Continuation Logic)

```python
def _get_starting_number(self, category: str) -> int:
    category_output = self.processed_dir / category
    category_output.mkdir(parents=True, exist_ok=True)
    
    existing_files = list(category_output.glob(f"{category}_*.wav"))
    
    if not existing_files:
        self.category_counters[category] = 0
        return 0
    
    # Extract numbers: "bike_0056.wav" → 56
    numbers = []
    for file_path in existing_files:
        stem = file_path.stem  # "bike_0056" (without .wav)
        parts = stem.split('_')  # ["bike", "0056"]
        if len(parts) >= 2 and parts[-1].isdigit():
            numbers.append(int(parts[-1]))
    
    highest_num = max(numbers)
    next_num = highest_num + 1
    self.category_counters[category] = next_num
    return next_num
```

**Why this matters:**
- You can run the processor multiple times
- New data adds to existing dataset without overwrites
- Enables incremental collection (Week 1: 100 samples, Week 2: add 50 more)

### 6. Export with Standardization

```python
segment.export(
    str(output_path),
    format="wav",
    parameters=["-ar", "48000", "-ac", "1"]
)
# -ar: audio rate (48000 Hz = 48kHz)
# -ac: audio channels (1 = mono)
```

All outputs: **48kHz, mono, WAV** (FFmpeg conversion happens automatically)

### 7. Metadata Recording

```python
self.metadata.append({
    'original_file': str(audio_path),
    'original_format': audio_path.suffix,
    'segment_file': str(output_path),
    'category': category,
    'segment_number': segment_num,
    'duration_s': 10.0,
    'dbfs': dbfs,
    'speech_percentage': speech_pct,
})
```

**Saved to CSV:** Enables querying (e.g., "which segments have < 40% speech?")

---

## Hands-On Implementation

### Step 1: Setup

```bash
# Create project directory
mkdir audio_ml_project && cd audio_ml_project

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install pydub pandas

# Install FFmpeg (required backend)
brew install ffmpeg  # macOS
```

### Step 2: Create Directory Structure

```bash
mkdir -p ml_data/physically_collected/{bike,truck,car}
# Add your audio files to these folders
```

### Step 3: Basic Test Script

```python
from pydub import AudioSegment

# Test 1: Load various formats
for fmt in ['mp3', 'wav', 'aac']:
    try:
        audio = AudioSegment.from_file(f"test_file.{fmt}")
        print(f"✅ {fmt}: {audio.duration_seconds}s @ {audio.frame_rate}Hz")
    except Exception as e:
        print(f"❌ {fmt}: {e}")

# Test 2: Check volume
audio = AudioSegment.from_file("recording.wav")
print(f"Volume: {audio.dBFS} dBFS")

# Test 3: Detect silence
from pydub.silence import detect_nonsilent
chunks = detect_nonsilent(audio, silence_thresh=-45)
print(f"Non-silent regions: {chunks}")
```

### Step 4: Run Full Pipeline

```python
from src.processors.physical_audio_processor import PhysicalAudioProcessor

processor = PhysicalAudioProcessor()
total = processor.process_all_categories()
processor.save_metadata()
processor.print_summary()
```

### Step 5: Analyze Results

```python
import pandas as pd

df = pd.read_csv("ml_data/physical_processing_metadata.csv")

# Which categories were processed?
print(df['category'].value_counts())

# What's the average speech percentage?
print(df['speech_percentage'].mean())

# Which segments have poor quality?
poor_quality = df[df['speech_percentage'] < 40]
print(f"Low-quality segments: {len(poor_quality)}")
```

---

## Debugging & Optimization

### Common Issues

**Issue 1: "No audio files found"**
```python
# Debug: Check if directory exists
from pathlib import Path
phys_dir = Path("ml_data/physically_collected")
print(f"Directory exists: {phys_dir.exists()}")
print(f"Contents: {list(phys_dir.iterdir())}")

# Check for hidden files (starting with .)
for item in phys_dir.iterdir():
    print(f"  {item.name} (hidden: {item.name.startswith('.')})")
```

**Issue 2: "CouldntDecodeError"**
```python
# FFmpeg not installed or not in PATH
import subprocess
result = subprocess.run(['ffmpeg', '-version'], capture_output=True)
print(result.returncode)  # 0 = installed, non-zero = missing

# Solution: Install or add to PATH
```

**Issue 3: Too many segments rejected**
```python
# Adjust thresholds for your audio quality
processor = PhysicalAudioProcessor(
    silence_threshold_db=-50,      # More permissive (was -45)
    min_speech_percentage=20       # Lower threshold (was 30)
)

# Analyze rejection reasons
for segment in audio.scan_segments():
    _, dbfs, speech_pct = processor.is_segment_valid(segment)
    if dbfs < -45:
        print(f"Rejected for volume: {dbfs}")
    if speech_pct < 30:
        print(f"Rejected for speech: {speech_pct}%")
```

### Performance Optimization

**For large datasets (1000+ files):**

```python
# 1. Use multiprocessing (process multiple categories in parallel)
from multiprocessing import Pool

def process_category_wrapper(category):
    processor = PhysicalAudioProcessor()
    return processor.process_category(category)

categories = ['bike', 'truck', 'car']
with Pool(processes=4) as pool:
    results = pool.map(process_category_wrapper, categories)

# 2. Reduce quality checks for speed
processor = PhysicalAudioProcessor(
    silence_threshold_db=-50  # Skip dBFS check (only speech % check)
)

# 3. Cache existing files (already implemented via category_counters)
```

---

## Integration with Your ML Pipeline

### Using Processed Audio in Jupyter

```python
import pandas as pd
from pathlib import Path
import numpy as np
from pydub import AudioSegment
import librosa

# 1. Load metadata
df = pd.read_csv("ml_data/physical_processing_metadata.csv")
processed_dir = Path("ml_data/processed")

# 2. Create dataset
def load_segment(row):
    """Load audio and extract features"""
    audio_path = row['segment_file']
    audio, sr = librosa.load(audio_path, sr=48000, mono=True)
    
    # Extract MFCC features (13-dimensional)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    
    return {
        'category': row['category'],
        'features': mfcc.mean(axis=1),  # Average across time
        'original_file': row['original_file']
    }

# 3. Vectorize
samples = [load_segment(row) for _, row in df.iterrows()]

# 4. Create PyTorch dataset
import torch
from torch.utils.data import Dataset, DataLoader

class AudioDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.categories = list(set(s['category'] for s in samples))
        self.category_to_idx = {c: i for i, c in enumerate(self.categories)}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = torch.FloatTensor(sample['features'])
        label = self.category_to_idx[sample['category']]
        return features, label

dataset = AudioDataset(samples)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 5. Train model
for batch_features, batch_labels in loader:
    # Your training logic here
    pass
```

### Feature Extraction Pipeline

```python
# For audio classification, typical features:
# 1. MFCC (Mel-frequency cepstral coefficients) - perceptual loudness
# 2. Spectral features (centroid, rolloff, zero crossing rate)
# 3. Temporal features (RMS energy)

import librosa

audio_path = "ml_data/processed/bike/bike_0000.wav"
y, sr = librosa.load(audio_path, sr=48000)

# MFCC
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# Spectral centroid
centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

# Zero crossing rate
zcr = librosa.feature.zero_crossing_rate(y)

print(f"MFCC shape: {mfcc.shape}")  # (13, time_frames)
print(f"Centroid shape: {centroid.shape}")  # (1, time_frames)
```

### Validation & Quality Checks

```python
# Before training, verify dataset integrity
def validate_processed_dataset():
    processed_dir = Path("ml_data/processed")
    
    stats = {}
    for category_dir in processed_dir.iterdir():
        if not category_dir.is_dir():
            continue
        
        audio_files = list(category_dir.glob("*.wav"))
        stats[category_dir.name] = {
            'count': len(audio_files),
            'total_duration': 0
        }
        
        for audio_file in audio_files:
            audio = AudioSegment.from_file(str(audio_file))
            assert audio.duration_seconds == 10.0, f"Wrong duration: {audio_file}"
            assert audio.channels == 1, f"Not mono: {audio_file}"
            assert audio.frame_rate == 48000, f"Wrong sample rate: {audio_file}"
            
            stats[category_dir.name]['total_duration'] += audio.duration_seconds
    
    for category, info in stats.items():
        print(f"{category}: {info['count']} files, {info['total_duration']/60:.1f} minutes")

validate_processed_dataset()
```

---

## Key Takeaways

1. **Format Agnosticism**: The processor handles any audio format pydub supports
2. **Quality Control**: Validates volume AND speech content before including segments
3. **Reproducibility**: Metadata CSV enables dataset auditing and reconstruction
4. **Incremental Building**: Can resume from last segment number for iterative collection
5. **ML-Ready Output**: Standardized 48kHz mono WAV files with consistent 10-second duration
6. **Monitoring**: Comprehensive statistics track processing effectiveness

---

## Next Steps

1. **Collect audio**: Record 5-10 samples in different categories
2. **Test the processor**: Run on small dataset to understand behavior
3. **Tune thresholds**: Adjust silence_threshold_db and min_speech_percentage for your audio quality
4. **Validate output**: Check a few WAV files manually, review metadata CSV
5. **Extract features**: Use librosa to convert WAV → MFCC features
6. **Build classifier**: Train audio classification model using processed segments

---

## Additional Resources

- **pydub**: https://github.com/jiaaro/pydub
- **librosa**: https://librosa.org/doc/latest/ (feature extraction)
- **FFmpeg**: https://ffmpeg.org/ (audio codec backend)
- **Audio DSP Basics**: Understanding dBFS, sample rates, bit depth
- **ML Dataset Best Practices**: Class balance, validation splits, metadata tracking
