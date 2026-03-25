# Model Card — Bangladeshi Audio Scene Classifier

## Model Overview

| Field | Detail |
|---|---|
| **Task** | Multi-class audio scene classification |
| **Input** | 10-second mono WAV audio at 48 kHz |
| **Output** | Predicted sound category (e.g., bus, truck, siren) |
| **Architecture** | CNN / Wav2Vec2 fine-tune (see `notebooks/main1.ipynb`) |
| **Framework** | PyTorch + Hugging Face Transformers |

## Dataset

| Field | Detail |
|---|---|
| **Source** | YouTube (via yt-dlp) + physically recorded audio |
| **Geography** | Bangladesh (Dhaka and surrounding regions) |
| **Segment length** | 10 seconds |
| **Sample rate** | 48 kHz mono |
| **Labeling** | Category-level labels assigned at collection time |

### Categories

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

### Quality Filters Applied

- Segments with dBFS < −45 dB are rejected (silence)
- Segments with < 30% non-silent content are rejected
- Final segments validated for 48 kHz sample rate and ~10 s duration

## Intended Use

- Training audio classification models for Bangladeshi urban soundscapes
- Research into acoustic environments in South Asian cities
- Benchmarking audio feature extraction methods (MFCCs, spectrograms, Wav2Vec2)

## Out-of-Scope Use

- Real-time audio surveillance or monitoring of individuals
- Use outside the acoustic domain this dataset represents (non-Bangladeshi urban environments may not generalize)
- Any application requiring personally identifiable audio information

## Limitations & Known Biases

- Data is geographically biased toward Dhaka city
- YouTube sources may introduce compression artifacts
- Class imbalance may exist across categories depending on collection volume
- Background noise overlap between categories (e.g., traffic_jam vs. bus) can reduce inter-class separability

## Evaluation

See `ml_data/quality_report.csv` for per-segment QA results.
Model performance metrics (accuracy, F1, confusion matrix) are logged in `selected_outputs/`.

## Ethical Considerations

- No personally identifiable audio (voices, conversations) is intentionally included
- All YouTube content collected within yt-dlp's terms of service
- Dataset is intended solely for research purposes
