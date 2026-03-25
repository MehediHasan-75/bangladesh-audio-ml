import os
import pandas as pd

def create_metadata_csv(base_path="ml_data/wav", output_file="metadata.csv"):
    """Create metadata.csv from audio files in ml_data/wav directory"""

    metadata_records = []

    # Get all noise categories
    noise_categories = sorted([d for d in os.listdir(base_path) 
                              if os.path.isdir(os.path.join(base_path, d))])

    for category in noise_categories:
        category_path = os.path.join(base_path, category)
        wav_files = sorted([f for f in os.listdir(category_path) if f.endswith('.wav')])

        for wav_file in wav_files:
            metadata_records.append({
                'filename': wav_file,
                'noise_category': category,
                'duration': 10,
                'sampling_rate': 16000
            })

    # Create and save DataFrame
    df = pd.DataFrame(metadata_records)
    df = df[['filename', 'noise_category', 'duration', 'sampling_rate']]
    df.to_csv(output_file, index=False)

    print(f"✓ Created {output_file} with {len(df)} entries")
    print(df.head())

if __name__ == "__main__":
    create_metadata_csv()
