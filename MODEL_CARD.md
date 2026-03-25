# Model Card — Audio Scene Classifier (Bangladeshi Urban Soundscape)

## Overview

This model card describes classifiers trained on the example dataset produced by the [Audio ML Dataset Collection & Processing Pipeline](readme.md). The pipeline is domain-agnostic; this particular model targets Bangladeshi urban acoustic scenes.

| Field | Detail |
|---|---|
| **Task** | Multi-class audio scene classification |
| **Input** | 10-second mono WAV at 48 kHz |
| **Output** | Predicted sound category |
| **Architectures** | CNN (MFCC features) / Wav2Vec2 fine-tune |
| **Framework** | PyTorch + Hugging Face Transformers |

---

## Dataset

| Field | Detail |
|---|---|
| **Collected with** | This pipeline (YouTube + physical recordings) |
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

### Quality Filters

- Segments with dBFS < −45 dB are rejected (silence)
- Segments with < 30% non-silent content are rejected
- Final segments validated for 48 kHz sample rate and ~10 s duration

---

## Intended Use

- Training and benchmarking audio classification models on South Asian urban soundscapes
- Demonstrating the pipeline's output quality on a real-world, underrepresented acoustic domain
- Benchmarking audio feature extraction methods (MFCCs, spectrograms, Wav2Vec2 embeddings)

## Out-of-Scope Use

- Real-time audio surveillance or monitoring of individuals
- Deployment outside the acoustic domain this dataset represents
- Any application requiring personally identifiable audio information

## Limitations & Known Biases

- Data is geographically biased toward Dhaka city
- YouTube sources may introduce compression artifacts
- Class imbalance may exist across categories depending on collection volume
- Background noise overlap between categories (e.g., `traffic_jam` vs. `bus`) can reduce inter-class separability

## Evaluation

Model performance metrics (accuracy, F1, confusion matrix) are in `selected_outputs/` and logged via MLflow.

## Ethical Considerations

- No personally identifiable audio (voices, conversations) is intentionally included
- All YouTube content collected within yt-dlp's terms of service
- Dataset is intended solely for research purposes
