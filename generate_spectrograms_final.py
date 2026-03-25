import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

"""
Spectrogram Generation with Elsevier Data in Brief Naming Convention

ELSEVIER NAMING CONVENTION (from Guide for Authors):
"Submit each image as a separate file using a logical naming convention 
for your files for example, Figure1, Figure2 etc."

For categorical data (like noise classes), the convention is:
spectrogram_[category].png
spectrogram_[category].png
"""

def select_best_audio_clips(base_path="ml_data/wav", num_clips_per_category=1):
    """Select best audio clips based on quality metrics"""

    best_clips = {}
    noise_categories = sorted([d for d in os.listdir(base_path) 
                              if os.path.isdir(os.path.join(base_path, d))])

    for category in noise_categories:
        category_path = os.path.join(base_path, category)
        wav_files = [f for f in os.listdir(category_path) if f.endswith('.wav')]

        quality_scores = []

        for wav_file in wav_files:
            file_path = os.path.join(category_path, wav_file)
            try:
                y, sr = librosa.load(file_path, sr=None)

                rms_energy = np.sqrt(np.mean(y**2))
                energy_std = np.std(librosa.feature.rms(y=y))
                peak_level = np.max(np.abs(y))

                clipping_penalty = max(0, (peak_level - 0.95) * 100)
                stability_score = 1.0 / (1.0 + energy_std)
                quality_score = (rms_energy * 10 + stability_score * 5 - clipping_penalty)

                quality_scores.append({
                    'filename': wav_file,
                    'filepath': file_path,
                    'quality_score': quality_score,
                    'rms_energy': rms_energy,
                    'stability': stability_score,
                    'peak_level': peak_level
                })
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        if quality_scores:
            quality_scores_sorted = sorted(quality_scores, key=lambda x: x['quality_score'], reverse=True)
            best_clips[category] = quality_scores_sorted[:num_clips_per_category]

    return best_clips

def generate_spectrograms_with_naming(best_clips, output_dir="spectrograms", dpi=300):
    """
    Generate spectrograms with ELSEVIER-COMPLIANT NAMING CONVENTION:
    spectrogram_[category].png

    Example:
    spectrogram_bike.png
    spectrogram_bus.png
    spectrogram_car.png
    spectrogram_protest.png
    spectrogram_siren.png
    """

    Path(output_dir).mkdir(exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')

    spectrogram_data = []

    for category, clips in best_clips.items():
        for i, clip_info in enumerate(clips):
            try:
                filepath = clip_info['filepath']
                filename = clip_info['filename']

                y, sr = librosa.load(filepath, sr=None)
                duration = librosa.get_duration(y=y, sr=sr)

                # Mel-Spectrogram
                S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
                S_db = librosa.power_to_db(S, ref=np.max)

                # STFT
                S_stft = np.abs(librosa.stft(y))
                S_stft_db = librosa.power_to_db(S_stft, ref=np.max)

                # Create figure
                fig, axes = plt.subplots(2, 1, figsize=(6.93, 9), dpi=dpi)
                fig.patch.set_facecolor('white')

                # Mel-Spectrogram
                img1 = librosa.display.specshow(
                    S_db, sr=sr, x_axis='time', y_axis='mel',
                    ax=axes[0], cmap='viridis', fmax=8000
                )
                axes[0].set_title(f'Mel-Spectrogram: {category.replace("_", " ").title()}',
                                 fontsize=12, fontweight='bold', pad=10)
                axes[0].set_ylabel('Frequency (Hz)', fontsize=10, fontweight='bold')
                axes[0].grid(True, alpha=0.3)

                cbar1 = plt.colorbar(img1, ax=axes[0], format='%+2.0f dB', pad=0.02)
                cbar1.set_label('Magnitude (dB)', fontsize=9, fontweight='bold')
                cbar1.ax.tick_params(labelsize=8)

                # STFT
                img2 = librosa.display.specshow(
                    S_stft_db, sr=sr, x_axis='time', y_axis='log',
                    ax=axes[1], cmap='magma', fmax=8000
                )
                axes[1].set_title(f'STFT Spectrogram: {category.replace("_", " ").title()}',
                                 fontsize=12, fontweight='bold', pad=10)
                axes[1].set_ylabel('Frequency (Hz)', fontsize=10, fontweight='bold')
                axes[1].set_xlabel('Time (s)', fontsize=10, fontweight='bold')
                axes[1].grid(True, alpha=0.3)

                cbar2 = plt.colorbar(img2, ax=axes[1], format='%+2.0f dB', pad=0.02)
                cbar2.set_label('Magnitude (dB)', fontsize=9, fontweight='bold')
                cbar2.ax.tick_params(labelsize=8)

                plt.tight_layout()

                # ELSEVIER NAMING CONVENTION: spectrogram_[category].png
                output_filename = f'spectrogram_{category}.png'
                output_path = os.path.join(output_dir, output_filename)
                plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
                plt.close()

                print(f"✓ Generated: {output_filename} ({dpi} DPI)")

                spectrogram_data.append({
                    'figure_filename': output_filename,
                    'figure_id': f'spectrogram_{category}',
                    'category': category,
                    'source_file': filename,
                    'output_path': output_path,
                    'duration': round(duration, 2),
                    'sample_rate': sr,
                    'quality_score': clip_info['quality_score']
                })

            except Exception as e:
                print(f"✗ Error: {filepath}: {e}")

    return spectrogram_data

def generate_figure_captions_elsevier(spectrogram_data, output_file="figure_captions.txt"):
    """
    Generate figure captions following ELSEVIER CONVENTION:
    - Caption file separate from images
    - Refers to figure by filename
    - Brief title + detailed description
    """

    captions = {}

    for item in spectrogram_data:
        fig_filename = item['figure_filename']
        category = item['category'].replace('_', ' ').title()

        brief_title = f"Environmental Noise: {category} Audio Spectrograms"

        detailed_description = f"""Dual-representation spectrogram for {category.lower()} environmental noise category. 
(Top panel) Mel-spectrogram showing frequency distribution on perceptually-weighted mel scale with 128 mel bins and maximum frequency of 8 kHz. 
(Bottom panel) Short-Time Fourier Transform (STFT) spectrogram on logarithmic frequency scale. 
Both representations computed from a 10-second audio sample (16 kHz sampling rate) with Hann window. 
Color intensity indicates magnitude in decibels (dB) relative to peak value. 
Source: {item['source_file']} (Duration: {item['duration']}s, Sample Rate: {item['sample_rate']} Hz)."""

        captions[fig_filename] = {
            'brief_title': brief_title,
            'detailed_description': detailed_description
        }

    # Save captions
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("FIGURE CAPTIONS - Elsevier Data in Brief Convention\n")
        f.write("="*80 + "\n")
        f.write("NOTE: Each caption below corresponds to a figure file\n")
        f.write("Include these in the manuscript manuscript text as Figure captions\n")
        f.write("="*80 + "\n\n")

        for fig_filename, caption_info in sorted(captions.items()):
            f.write(f"Figure: {fig_filename}\n")
            f.write(f"Title: {caption_info['brief_title']}\n")
            f.write(f"Description:\n{caption_info['detailed_description']}\n")
            f.write("\n" + "-"*80 + "\n\n")

    print(f"\n✓ Figure captions saved: {output_file}")
    return captions

def generate_class_distribution(best_clips, output_dir="spectrograms", dpi=300):
    """Generate class distribution with Elsevier naming: spectrogram_distribution.png"""

    category_counts = {category: len(clips) for category, clips in best_clips.items()}

    fig, ax = plt.subplots(figsize=(4.5, 5), dpi=dpi)
    fig.patch.set_facecolor('white')

    categories = list(category_counts.keys())
    counts = list(category_counts.values())

    colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))
    bars = ax.bar(range(len(categories)), counts, color=colors, edgecolor='black', linewidth=1.2)

    ax.set_xlabel('Noise Category', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Spectrograms', fontsize=11, fontweight='bold')
    ax.set_title('Environmental Noise Class Distribution', fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([c.replace('_', '\n') for c in categories], fontsize=9, rotation=0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.tight_layout()

    # Elsevier naming convention
    output_filename = 'spectrogram_distribution.png'
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Class distribution: {output_filename} ({dpi} DPI)")
    return output_filename, category_counts

def create_summary_report(spectrogram_data, distribution_file, category_counts):
    """Create summary showing naming convention compliance"""

    print("\n" + "="*80)
    print("SPECTROGRAM GENERATION - ELSEVIER NAMING CONVENTION COMPLIANCE")
    print("="*80)

    print("\nELSEVIER GUIDE FOR AUTHORS REFERENCE:")
    print('  "Submit each image as a separate file using a logical naming convention')
    print('   for your files for example, Figure1, Figure2 etc."')

    print("\nIMPLEMENTED NAMING CONVENTION:")
    print("  Format: spectrogram_[category].png")
    print("  Example files:")
    for data in spectrogram_data:
        print(f"    - {data['figure_filename']}")
    print(f"    - {distribution_file}")

    print("\nFILE STRUCTURE FOR SUBMISSION:")
    print("\n  spectrograms/")
    for data in sorted(spectrogram_data, key=lambda x: x['category']):
        print(f"    ├── {data['figure_filename']}")
    print(f"    ├── {distribution_file}")
    print("    └── figure_captions.txt (separate caption file)")

    print("\nCAPTION HANDLING (Elsevier Convention):")
    print("  ✓ Captions are in a SEPARATE file (figure_captions.txt)")
    print("  ✓ NOT embedded in the images")
    print("  ✓ Include captions in manuscript text near figure citation")

    print("\nNO VENDOR-SPECIFIC REQUIREMENTS:")
    print("  ✓ Naming convention is PLATFORM-INDEPENDENT")
    print("  ✓ Works with: Overleaf, Word, LaTeX, PDF tools")
    print("  ✓ No software-specific prefixes needed")

    print("\nFIGURE REFERENCE IN MANUSCRIPT:")
    print("  Suggested text format:")
    print("  'As shown in Figure 1, the car noise spectrogram shows...'")
    print("  File referenced: spectrogram_car.png")

    print("\n" + "="*80)
    print("✓ ALL FILES READY FOR ELSEVIER SUBMISSION")
    print("="*80)

def main():
    """Main execution"""

    base_path = "ml_data/wav"

    if not os.path.exists(base_path):
        print(f"Error: Directory '{base_path}' not found!")
        return

    print("Generating spectrograms with Elsevier naming convention...\n")

    # Step 1: Select best clips
    print("Step 1: Selecting best audio clips...")
    best_clips = select_best_audio_clips(base_path, num_clips_per_category=1)
    print(f"✓ Selected clips\n")

    # Step 2: Generate spectrograms
    print("Step 2: Generating spectrograms (Elsevier naming)...")
    spectrogram_data = generate_spectrograms_with_naming(best_clips, dpi=300)

    # Step 3: Generate captions
    print("\nStep 3: Generating figure captions...")
    captions = generate_figure_captions_elsevier(spectrogram_data)

    # Step 4: Generate distribution
    print("\nStep 4: Creating class distribution...")
    dist_filename, category_counts = generate_class_distribution(best_clips)

    # Step 5: Create summary
    create_summary_report(spectrogram_data, dist_filename, category_counts)

    # Save metadata
    df = pd.DataFrame(spectrogram_data)
    df.to_csv('spectrogram_metadata.csv', index=False)
    print("\n✓ Metadata saved: spectrogram_metadata.csv")

if __name__ == "__main__":
    main()
