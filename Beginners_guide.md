# Beginner's Learning Guide for Bangladesh Audio Data Collection \& Processing Project

## 1. **Project Overview**

This project is designed to help you collect, process, and validate audio data for machine learning tasks—common for speech, sound event detection, and audio classification projects. You'll learn how to organize files, manage multiple formats, segment audio, and ensure data quality.

***

## 2. **Python and Tool Prerequisites**

### What you need to know:

- Basic **Python (>=3.8)**
- Command line/Terminal basics (navigating directories, running scripts)
- Using **virtual environments** (venv)
- **FFmpeg** for audio handling (automatic; you don't need to code FFmpeg directly)

***

## 3. **Installation Step-by-Step**

**(A) Clone the Repository**

```bash
git clone <your-repo-url>
cd bangladesh-audio-ml
```

**(B) Install FFmpeg**

- **macOS**: `brew install ffmpeg`
- **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
- **Windows**: `choco install ffmpeg`
- [Or download from ffmpeg.org](https://ffmpeg.org/download.html)

**(C) Create a Virtual Environment and Activate**

```bash
python3 -m venv venv
source venv/bin/activate  # (macOS/Linux)
venv\Scripts\activate     # (Windows)
```

**(D) Install Python Libraries**

```bash
pip install -r requirements.txt
```


***

## 4. **Understanding the Folder Structure**

- `ml_data/raw/` — Audio downloaded from YouTube.
- `ml_data/physically_collected/` — Manually recorded data (e.g., bike, truck folders).
- `ml_data/processed/` — All segmented and validated (.wav) audio chunks.
- `data/` — Contains your `youtube_urls.csv` for YouTube links.
- `config/` — All processing settings.

***

## 5. **Running Your First Data Pipeline**

### **A. YouTube Pipeline**

1. **Edit your YouTube URLs:**
    - Open `data/youtube_urls.csv`
    - Add rows like:
`https://www.youtube.com/watch?v=ABC123,car_horn`
2. **Run the script:**

```bash
python scripts/collect_audio.py
```

    - This will download, segment, and validate new YouTube audio automatically!

### **B. Physical Recording Pipeline**

1. **Add your recordings:**
    - Create folders for each category:

```
mkdir -p ml_data/physically_collected/bike
mkdir -p ml_data/physically_collected/truck
```

    - Copy your `.opus`, `.m4a`, `.mp3`, etc. files into each category.
2. **Process them:**

```bash
python scripts/process_physical.py
```

    - Segments are created in `ml_data/processed/[category]/`.
    - Files are named like `bike_0000.wav`, `truck_0003.wav`.

***

## 6. **Exploring Results and Quality**

- Processed clips are always **10 seconds, mono, 48kHz** for ML consistency.
- Quality control:
    - See `ml_data/quality_report.csv`.
    - Segments with too much silence, wrong duration, or low volume are rejected.
- Metadata:
    - Check how every clip was created in `ml_data/processing_metadata.csv` or `physical_processing_metadata.csv`.

***

## 7. **Common Beginner Questions**

**Q: How do I get audio labels for ML?**
A: All clip filenames clearly contain the label (category), e.g. `bike_0001.wav` is labeled as `bike`.

**Q: I have different file types (.mp3, .aac, .wav, .opus)—do I need to convert them first?**
A: No! The scripts handle all common types automatically.

**Q: Can I add/remove files after processing?**
A: Yes, the scripts smartly continue numbering and skip files already processed.

**Q: How do I check the processed files?**
A: Look in `ml_data/processed/[category]/`. You can play them using any media player or with Python:

```python
import librosa
y, sr = librosa.load('ml_data/processed/bike/bike_0000.wav', sr=None)
print(sr, len(y)/sr)
```


***

## 8. **Next Steps and Experimentation**

- **Try changing settings:** Open `config/config.yaml`, modify segment length, silence threshold, etc.
- **Try processing a single category:**

```bash
python scripts/process_physical_category.py bike
```

- **Visualize clips:** Use Jupyter Notebook to plot waveforms:

```python
import librosa.display
import matplotlib.pyplot as plt

y, sr = librosa.load('ml_data/processed/bike/bike_0001.wav')
plt.figure()
librosa.display.waveshow(y, sr=sr)
plt.show()
```


***

## 9. **What to Learn Next**

- How to use the processed dataset for **machine learning** (classification, training).
- Exploring and visualizing audio features with **Python (librosa, pandas, matplotlib)**.
- Expand to multi-label, multi-class, or other advanced ML patterns.
- Collaborate: Share folders, code, and metadata easily with other researchers!

***

## 10. **Extra Resources**

- [Python documentation](https://docs.python.org/3/)
- [FFmpeg documentation](https://ffmpeg.org/documentation.html)
- [Librosa audio analysis](https://librosa.org/doc/latest/index.html)
- [Machine Learning Beginner’s Guide](https://scikit-learn.org/stable/tutorial/index.html)

***

## ☑️ Checklist for Beginners

- [ ] Python \& FFmpeg installed
- [ ] Virtual environment created \& activated
- [ ] Requirements installed
- [ ] Data files organized in `physically_collected` or YouTube links in CSV
- [ ] Main pipeline scripts run successfully
- [ ] Processed data available for ML

This guide will take you from zero setup to a working, ML-ready audio dataset—whether you’re collecting from YouTube or physical recordings!

