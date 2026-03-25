import os
import librosa
import soundfile as sf
from pathlib import Path
import numpy as np

# Paths
processed_folder = Path('ml_data/processed')
output_folder = Path('ml_data/resampled_16k')
TARGET_SAMPLE_RATE = 16000

# Create output folder
output_folder.mkdir(parents=True, exist_ok=True)

print(f"Source folder: {processed_folder}")
print(f"Output folder: {output_folder}")
print(f"Target sample rate: {TARGET_SAMPLE_RATE} Hz\n")

# Statistics
total_files = 0
success_count = 0
failed_files = []

# Iterate through each category folder
for category_folder in sorted(processed_folder.iterdir()):
    if category_folder.is_dir():
        category = category_folder.name
        audio_files = list(category_folder.glob('*.wav'))
        
        # Create output category folder
        output_category_folder = output_folder / category
        output_category_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing {category:20s}: {len(audio_files):4d} files", end=" ... ")
        
        success = 0
        for audio_path in audio_files:
            try:
                # ✅ Use librosa (better for macOS)
                waveform, sample_rate = librosa.load(str(audio_path), sr=None)
                
                # Resample to 16 kHz
                if sample_rate != TARGET_SAMPLE_RATE:
                    waveform_resampled = librosa.resample(
                        waveform, 
                        orig_sr=sample_rate, 
                        target_sr=TARGET_SAMPLE_RATE
                    )
                else:
                    waveform_resampled = waveform
                
                # ✅ Use soundfile to save
                output_path = output_category_folder / audio_path.name
                sf.write(str(output_path), waveform_resampled, TARGET_SAMPLE_RATE)
                
                success += 1
                success_count += 1
                
            except Exception as e:
                failed_files.append((category, audio_path.name, str(e)))
            
            total_files += 1
        
        print(f"✓ {success}/{len(audio_files)}")

# Summary
print(f"\n{'='*60}")
print(f"RESAMPLING COMPLETE")
print(f"{'='*60}")
print(f"Total files processed: {total_files}")
print(f"Successfully resampled: {success_count}")
print(f"Failed: {len(failed_files)}")
print(f"Success rate: {(success_count/total_files)*100:.1f}%")
print(f"Output location: {output_folder}")
print(f"{'='*60}")

if failed_files:
    print(f"\nFailed files (first 5):")
    for cat, fname, error in failed_files[:5]:
        print(f"  - {cat}/{fname}: {error}")
else:
    print(f"\n✅ All {success_count} files successfully resampled to {TARGET_SAMPLE_RATE} Hz!")
