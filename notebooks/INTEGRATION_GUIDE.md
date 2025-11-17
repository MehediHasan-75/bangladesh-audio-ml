# How to Use Jupyter Notebooks in Your Bangladesh Audio ML Project

Complete integration guide for using the 6 machine learning notebooks with your existing audio pipeline.

---

## 📋 Project Context

Your project already has:
- ✅ **Data Collection Pipeline**: YouTube + Physical audio collection (`src/collectors/`, `src/processors/`)
- ✅ **Quality Control**: Audio validation and filtering (`src/quality/`)
- ✅ **Processed Data**: 10-second WAV segments in `ml_data/processed/[category]/`

Now we're adding:
- 📓 **ML Training Notebooks**: 6 notebooks for model training and evaluation

---

## 🚀 Complete Workflow: From Audio Collection to Model Training

### Phase 1: Prepare Audio Data (Already Done)

```bash
# 1. Collect audio from YouTube
python scripts/collect_audio.py

# 2. Process physical audio (if you have manually recorded files)
python scripts/process_physical.py

# Result: ml_data/processed/ contains 10-second WAV files organized by category
# Example structure:
# ml_data/processed/
# ├── bike/
# │   ├── bike_0000.wav
# │   ├── bike_0001.wav
# │   └── ... (57 files total)
# ├── truck/
# │   ├── truck_0000.wav
# │   ├── truck_0001.wav
# │   └── ... (43 files total)
# └── bus/
#     ├── bus_0000.wav
#     ├── bus_0001.wav
#     └── ... (28 files total)
```

**Status**: ✅ Your pipeline produces `ml_data/processed/` with categorized audio files

---

### Phase 2: Machine Learning Pipeline (New Notebooks)

#### Step 1: Set Up Notebooks Directory

```bash
# Create notebooks directory in your project
mkdir -p notebooks/

# Place all 6 notebooks here:
notebooks/
├── 01_data_exploration.ipynb
├── 02_feature_extraction.ipynb
├── 03_cnn_model.ipynb
├── 04_rnn_lstm_model.ipynb
├── 05_transformer_model.ipynb
└── 06_classical_ml_models.ipynb
```

#### Step 2: Install ML Dependencies

Your project's `requirements.txt` needs ML packages. Add these:

```bash
# In your project root, update requirements.txt
pip install librosa numpy pandas torch torchaudio scikit-learn matplotlib seaborn jupyter
```

Or create a separate ML requirements:

```bash
cat > requirements_ml.txt << 'EOF'
# Audio & Data Processing
librosa>=0.10.0
numpy>=1.24.0
pandas>=2.0.0

# Deep Learning
torch>=2.0.0
torchaudio>=2.0.0

# Machine Learning
scikit-learn>=1.3.0
matplotlib>=3.8.0
seaborn>=0.13.0

# Jupyter
jupyter>=1.0.0
EOF

pip install -r requirements_ml.txt
```

#### Step 3: Launch Notebooks in Sequence

**Start from your project root:**

```bash
# Activate your virtual environment
source venv/bin/activate  # or: venv\Scripts\activate (Windows)

# Start Jupyter
jupyter notebook
```

Then navigate to each notebook in order:

**Notebook 1: Data Exploration** (5 minutes)
```
Click: notebooks/01_data_exploration.ipynb
↓
This loads your ml_data/processed/ directory
↓
Outputs:
  - myVoices: List of audio items
  - myVoices_df: DataFrame with paths and categories
  - balanced_myVoices_df: Balanced dataset
```

**Notebook 2: Feature Extraction** (10 minutes)
```
Click: notebooks/02_feature_extraction.ipynb
↓
Uses outputs from Notebook 1
↓
Extracts MFCC features (Mel-Frequency Cepstral Coefficients)
↓
Outputs:
  - X_train, X_val, X_test: Feature arrays
  - y_train, y_val, y_test: Labels
  - label_encoder: For predictions
```

**Notebooks 3-6: Model Training** (30-60 minutes each)

Train any/all models you want:

```
③ CNN Model
   notebooks/03_cnn_model.ipynb
   └─ AudioCNN: Convolutional Neural Network
   
④ RNN/LSTM Model
   notebooks/04_rnn_lstm_model.ipynb
   └─ AudioRNN: Recurrent Neural Network with LSTM
   
⑤ Transformer Model
   notebooks/05_transformer_model.ipynb
   └─ AudioTransformer: Attention-based model
   
⑥ Classical ML Models
   notebooks/06_classical_ml_models.ipynb
   ├─ SVM: Support Vector Machine
   ├─ Random Forest: Ensemble method
   └─ GMM: Gaussian Mixture Model
```

---

## 📊 Data Flow: How Notebooks Use Your Data

```
ml_data/processed/
├── bike/
│   ├── bike_0000.wav
│   ├── bike_0001.wav
│   └── ... (57 files)
├── truck/
│   └── ... (43 files)
└── bus/
    └── ... (28 files)
    
        ↓ (Step 1: Notebook 01)
        
Loaded Audio Dataset
├── path: "ml_data/processed/bike/bike_0000.wav"
├── filename: "bike_0000.wav"
├── waveform: numpy array (48000Hz, mono)
├── sample_rate: 48000
├── category: "bike"
└── duration: 10.0 seconds

        ↓ (Step 2: Notebook 02)
        
MFCC Features Extracted
├── features: (40, 1101) - 40 MFCC coefficients
├── category: "bike"
└── [for all 128 audio files in dataset]

        ↓ Split & Balance
        
Train/Val/Test Sets
├── X_train: (102, 40, 1101) - balanced
├── X_val: (13, 40, 1101) - original
├── X_test: (13, 40, 1101) - original
└── y_train, y_val, y_test: category labels

        ↓ (Step 3-6: Notebooks 03-06)
        
Model Training & Evaluation
├── Train model on balanced training set
├── Monitor on validation set
└── Evaluate on test set
    └─ Accuracy, Precision, Recall, F1-score
    └─ Confusion Matrix
    └─ Per-class Performance
```

---

## 💻 Execution Example: Complete Workflow

### Interactive Mode (Recommended)

```bash
# Terminal 1: Start Jupyter
cd ~/bangladesh-audio-ml  # or your project directory
jupyter notebook

# This opens: http://localhost:8888
# Navigate to notebooks/01_data_exploration.ipynb
```

**In Jupyter (Browser):**

```python
# Cell 1: Run all imports and setup
# (Just click "Run" or Shift+Enter)

# Cell 2: Load audio dataset
# Output shows:
# ✓ Audio loading complete!
# Total files loaded: 128
# Category Distribution:
#   bike:   57 files
#   truck:  43 files
#   bus:    28 files

# Cell 3: Display statistics
# Output shows dataset info, durations, sample rates

# Continue through all cells...
```

Then move to next notebook:

```python
# 02_feature_extraction.ipynb
# Cell 1: Run MFCC extraction
# Output: Extracting MFCC features...
#   Processed 100/128 files
#   ✓ Feature extraction complete!

# Continue with training/val/test split...
```

Finally, train models:

```python
# 03_cnn_model.ipynb
# Cell 1: Define CNN
# Cell 2: Prepare data
# Cell 3: Train CNN
# Output: 
# Starting CNN training...
# Epoch 5/20, Train Loss: 1.2345, Val Loss: 1.1234, Val Acc: 0.7654

# At end: Test evaluation metrics
# Accuracy:  0.8462
# Precision: 0.8567
# Recall:    0.8341
# F1-score:  0.8450
```

---

## 📁 Expected Project Structure After Notebooks

```
bangladesh-audio-ml/
│
├── scripts/                              # Your existing scripts
│   ├── collect_audio.py
│   ├── process_raw.py
│   ├── process_physical.py
│   └── ...
│
├── src/                                  # Your existing modules
│   ├── collectors/
│   ├── processors/
│   ├── quality/
│   └── utils/
│
├── ml_data/
│   ├── raw/                             # YouTube downloads
│   ├── physically_collected/            # Manual recordings
│   ├── processed/                       # ✅ 10-sec WAV files (Input to ML)
│   │   ├── bike/
│   │   ├── truck/
│   │   └── bus/
│   └── [metadata files]
│
├── notebooks/                           # ✨ NEW: ML Notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_cnn_model.ipynb
│   ├── 04_rnn_lstm_model.ipynb
│   ├── 05_transformer_model.ipynb
│   └── 06_classical_ml_models.ipynb
│
├── config/
│   └── config.yaml
│
├── requirements.txt                     # Update: Add ML packages
├── requirements_ml.txt                  # ✨ NEW: ML-specific requirements
└── README.md
```

---

## 🔄 Typical Workflow

### Day 1: Collect Data

```bash
# Existing pipeline
python scripts/collect_audio.py
python scripts/process_physical.py

# Result: ml_data/processed/ has ~128 segments
```

### Day 2: Train Models

```bash
# New ML notebooks
jupyter notebook

# Open 01_data_exploration.ipynb
# → Run all cells (5 min)

# Open 02_feature_extraction.ipynb
# → Run all cells (10 min)

# Open 03_cnn_model.ipynb
# → Run all cells (30 min, includes training)

# Review results: accuracy, confusion matrix, per-class performance

# Optionally train other models (04, 05, 06)
```

### Day 3: Iterate

```bash
# Adjust hyperparameters in notebooks
# (batch size, learning rate, epochs)

# Re-run training cells

# Compare performance across models

# Use best model for predictions
```

---

## ⚙️ Configuration: Connecting Notebooks to Your Data

### Notebook 1: Data Exploration Setup

```python
# In 01_data_exploration.ipynb, Cell 1:

from pathlib import Path
import librosa

# Points to your processed audio directory
processed_folder = Path('ml_data/processed')

# The notebook automatically finds:
# ml_data/processed/bike/
# ml_data/processed/truck/
# ml_data/processed/bus/
# ... any other categories you have
```

✅ **No changes needed!** It automatically discovers your categories.

### Notebook 2: Feature Extraction Setup

```python
# In 02_feature_extraction.ipynb, Cell 2:

# Uses data from Notebook 1
# Automatically handles train/val/test split
# Parameters you can customize:

n_mfcc = 40              # MFCC coefficients
n_fft = 400              # FFT window size
hop_length = 160         # Hop length for STFT
max_padding = 176400     # Fixed audio length

test_size = 0.2          # 20% for test
val_size = 0.5           # 10% for val (of test+val)
                         # → 80% train, 10% val, 10% test
```

### Notebooks 3-6: Model Training Setup

```python
# In each model notebook:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Automatically uses GPU if available, falls back to CPU

batch_size = 32
learning_rate = 0.001
num_epochs = 20

# All uses data from Notebook 2 (X_train, y_train, etc.)
# No additional configuration needed!
```

---

## 🎯 Quick Reference: Which Notebook to Use?

### Goal: Understand My Dataset
→ **Use Notebook 01: Data Exploration**
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### Goal: Prepare Data for ML
→ **Use Notebook 02: Feature Extraction**
```bash
jupyter notebook notebooks/02_feature_extraction.ipynb
```

### Goal: Train a CNN
→ **Use Notebook 03: CNN Model**
```bash
jupyter notebook notebooks/03_cnn_model.ipynb
```

### Goal: Compare Multiple Models
→ **Use Notebooks 03, 04, 05, 06 (all of them)**
```bash
jupyter notebook notebooks/03_cnn_model.ipynb
jupyter notebook notebooks/04_rnn_lstm_model.ipynb
jupyter notebook notebooks/05_transformer_model.ipynb
jupyter notebook notebooks/06_classical_ml_models.ipynb
# Then compare results manually
```

### Goal: Run Predictions on New Audio
→ **After training**, create a simple script:
```python
import torch
import librosa
from src.models.cnn import AudioCNN

# Load trained model
model = AudioCNN(num_classes=3)
model.load_state_dict(torch.load('cnn_model.pt'))

# Load new audio
audio, sr = librosa.load('new_audio.wav', sr=None)
# Extract MFCC (same as notebook)
# Make prediction
```

---

## 📊 Expected Results

### Dataset Overview (from Notebook 01)
```
Total files loaded: 128
Category Distribution:
  bike:   57 files (44.5%)
  truck:  43 files (33.6%)
  bus:    28 files (21.9%)

Duration Statistics:
  Total: 21.3 minutes
  Average per file: 10.0 seconds
```

### Feature Shape (from Notebook 02)
```
Extracted Features DataFrame:
  Shape: (128, 2)
  Features: (40, 1101)  ← 40 MFCC coefficients × 1101 time steps
  
Training set: 102 samples (balanced)
Validation set: 13 samples
Test set: 13 samples
```

### Model Performance (from Notebooks 03-06)

**CNN Model:**
```
Accuracy:  0.8462 (11/13 correct)
Precision: 0.8567
Recall:    0.8341
F1-score:  0.8450
```

**SVM Model:**
```
Accuracy:  0.7692
Precision: 0.7654
Recall:    0.7341
F1-score:  0.7512
```

**Random Forest:**
```
Accuracy:  0.8538
Precision: 0.8621
Recall:    0.8465
F1-score:  0.8540
```

### Confusion Matrices
Each notebook generates heatmaps showing which categories are confused with each other.

---

## 🔧 Troubleshooting Common Issues

### Issue: "ModuleNotFoundError: No module named 'librosa'"

**Solution:**
```bash
pip install librosa
# or
pip install -r requirements_ml.txt
```

### Issue: "No such file or directory: ml_data/processed"

**Solution:**
```bash
# Run your data collection pipeline first
python scripts/collect_audio.py
python scripts/process_physical.py

# This creates ml_data/processed/ with audio files
```

### Issue: Notebook runs very slowly

**Reasons:**
- First run of feature extraction takes longer
- CPU training is slow (use GPU if available)
- Large dataset

**Solutions:**
```python
# Use fewer samples for testing:
extracted_features_df = extracted_features_df.head(50)  # First 50 only

# Reduce model complexity:
num_epochs = 5  # Instead of 20

# Enable GPU:
device = torch.device("cuda")  # Will use GPU if available
```

### Issue: "Out of memory" error

**Solutions:**
```python
# Reduce batch size
batch_size = 16  # Instead of 32

# Reduce number of epochs
num_epochs = 5   # Instead of 20

# Use smaller model
# Or process fewer samples at a time
```

---

## 📈 Performance Optimization

### Speed Up Training

```python
# In any model notebook:

# Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Increase batch size
batch_size = 64  # Process more samples at once

# Use fewer epochs
num_epochs = 10  # Instead of 20

# Reduce features
n_mfcc = 20  # Instead of 40 (fewer dimensions)
```

### Improve Accuracy

```python
# Use more data
# (Add more categories to your collection pipeline)

# Increase training
num_epochs = 50  # Train longer

# Try different models
# (Each has different strengths)

# Fine-tune hyperparameters
learning_rate = 0.0005  # Slower learning
```

---

## 🚀 Advanced Usage

### Save Trained Models

```python
# In notebook, after training:
torch.save(model.state_dict(), 'cnn_model.pt')

# Later, load model:
model = AudioCNN(num_classes=3)
model.load_state_dict(torch.load('cnn_model.pt'))
model.eval()
```

### Export Results

```python
# Save confusion matrix as image
import matplotlib.pyplot as plt
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')

# Export metrics to CSV
import pandas as pd
results_df = pd.DataFrame({
    'Model': ['CNN', 'RNN', 'Transformer', 'SVM', 'RF', 'GMM'],
    'Accuracy': [0.846, 0.769, 0.815, 0.769, 0.854, 0.723],
})
results_df.to_csv('model_comparison.csv', index=False)
```

### Use Notebooks as Module

```python
# Create Python script that imports notebook functions
# (Advanced usage - see modularization guide)

from src.data.audio_loader import load_audio_dataset
from src.features.mfcc_extractor import extract_features_batch
from src.training.data_splitter import split_and_balance_data
```

---

## 📞 Next Steps

1. **Organize your data**
   ```bash
   python scripts/process_physical.py  # If you have physical audio
   # Or
   python scripts/collect_audio.py     # To download from YouTube
   ```

2. **Copy notebooks to your project**
   ```bash
   mkdir -p notebooks/
   # Copy all 6 .ipynb files to notebooks/
   ```

3. **Install ML dependencies**
   ```bash
   pip install librosa numpy pandas torch torchaudio scikit-learn matplotlib seaborn jupyter
   ```

4. **Start with Notebook 1**
   ```bash
   jupyter notebook notebooks/01_data_exploration.ipynb
   ```

5. **Follow the workflow**
   01 → 02 → 03/04/05/06 (train any models you want)

---

## 📚 Learning Path

- **Beginner**: Run notebooks 01 → 02 → 03 (CNN)
- **Intermediate**: Run all notebooks 01 → 02 → 03-06 (compare all models)
- **Advanced**: Modify notebooks, combine with src/ modules, deploy to production

---

**Ready to train your first model?** Start with `jupyter notebook notebooks/01_data_exploration.ipynb`! 🚀

