"""
Bangladeshi Audio Scene Classifier — Interactive Demo
Run with: streamlit run app/demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import shap

st.set_page_config(
    page_title="Bangladeshi Audio Classifier",
    page_icon="🎙️",
    layout="wide",
)

st.title("🎙️ Bangladeshi Audio Scene Classifier")
st.markdown(
    "Upload a 10-second audio clip and the model will predict its sound category "
    "(bus, truck, siren, etc.)."
)

CATEGORIES = [
    "bike", "bus", "car", "cng_auto", "construction_noise",
    "protest", "siren", "traffic_jam", "train", "truck",
]


@st.cache_resource
def load_model():
    """Load trained model if it exists, otherwise return None."""
    import pickle
    model_path = Path("models/classifier.pkl")
    if model_path.exists():
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None


def extract_features(audio_path: str) -> np.ndarray:
    """Extract MFCC + spectral features from a WAV file."""
    y, sr = librosa.load(audio_path, sr=48000, duration=10.0)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = mfccs.mean(axis=1)

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    rms = librosa.feature.rms(y=y).mean()
    zcr = librosa.feature.zero_crossing_rate(y).mean()

    features = np.concatenate([
        mfcc_means,
        [spectral_centroid, spectral_bandwidth, rms, zcr],
    ])
    return features, y, sr


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("About")
st.sidebar.markdown(
    "**Dataset:** Bangladeshi urban audio (YouTube + physically recorded)  \n"
    "**Segment length:** 10 seconds @ 48 kHz mono  \n"
    "**Categories:** " + ", ".join(CATEGORIES)
)
st.sidebar.header("Dataset Stats")
quality_report = Path("ml_data/quality_report.csv")
if quality_report.exists():
    import pandas as pd
    df = pd.read_csv(quality_report)
    st.sidebar.metric("Total segments", len(df))
    if "passes_all" in df.columns:
        st.sidebar.metric("Passing QC", int(df["passes_all"].sum()))
        st.sidebar.metric(
            "Pass rate", f"{df['passes_all'].mean()*100:.1f}%"
        )
else:
    st.sidebar.info("No quality report found. Run the pipeline first.")

# ── Main upload area ──────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload a WAV file (10 s, 48 kHz mono)", type=["wav"])

if uploaded:
    tmp_path = Path("app/_tmp_upload.wav")
    tmp_path.write_bytes(uploaded.read())

    st.audio(str(tmp_path))

    try:
        features, y, sr = extract_features(str(tmp_path))

        # ── Waveform + spectrogram ────────────────────────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Waveform")
            fig, ax = plt.subplots(figsize=(6, 2))
            librosa.display.waveshow(y, sr=sr, ax=ax, color="#1DB954")
            ax.set_xlabel("Time (s)")
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.subheader("Mel Spectrogram")
            fig, ax = plt.subplots(figsize=(6, 2))
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
            S_dB = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(S_dB, sr=sr, x_axis="time",
                                     y_axis="mel", ax=ax)
            st.pyplot(fig)
            plt.close(fig)

        # ── Prediction ────────────────────────────────────────────────────────
        model = load_model()
        if model is not None:
            st.subheader("Prediction")
            proba = model.predict_proba([features])[0]
            pred_idx = np.argmax(proba)
            pred_label = CATEGORIES[pred_idx] if pred_idx < len(CATEGORIES) else "unknown"

            st.metric("Predicted category", pred_label.upper(),
                      delta=f"{proba[pred_idx]*100:.1f}% confidence")

            # Probability bar chart
            import pandas as pd
            prob_df = pd.DataFrame({
                "Category": CATEGORIES[:len(proba)],
                "Probability": proba,
            }).sort_values("Probability", ascending=True)
            st.bar_chart(prob_df.set_index("Category"))

            # ── SHAP explanation ──────────────────────────────────────────────
            st.subheader("Feature Importance (SHAP)")
            FEATURE_NAMES = (
                [f"MFCC_{i}" for i in range(13)]
                + ["Spectral Centroid", "Spectral Bandwidth", "RMS Energy", "ZCR"]
            )
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(features.reshape(1, -1))
            vals = shap_vals[pred_idx][0] if isinstance(shap_vals, list) else shap_vals[0]

            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ["#E74C3C" if v > 0 else "#3498DB" for v in vals]
            ax.barh(FEATURE_NAMES, vals, color=colors)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("SHAP value (impact on model output)")
            ax.set_title(f"Why '{pred_label}'?")
            st.pyplot(fig)
            plt.close(fig)

        else:
            st.info(
                "No trained model found at `models/classifier.pkl`.  \n"
                "Run `python experiments/track_experiment.py` to train and save a model."
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
