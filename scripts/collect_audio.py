#!/usr/bin/env python3
"""Main collection pipeline - download and process new files only"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.collectors.youtube_collector import YouTubeAudioCollector
from src.processors.audio_processor import AudioProcessor
from src.quality.quality_controller import QualityController


def main():
    print("="*60)
    print("BANGLADESH AUDIO DATA COLLECTION")
    print("="*60)
    
    # Step 1: Download from YouTube
    print("\nSTEP 1: Downloading from YouTube...")
    collector = YouTubeAudioCollector(base_dir="ml_data")
    collector.download_from_csv("data/youtube_urls.csv")
    collector.save_metadata()
    
    newly_downloaded = collector.get_newly_downloaded_files()
    
    if len(newly_downloaded) == 0:
        print("\n" + "="*60)
        print("ℹ️  NO NEW FILES DOWNLOADED")
        print("="*60)
        return
    
    # Step 2: Process new files
    print("\n" + "="*60)
    print(f"STEP 2: Processing {len(newly_downloaded)} new files...")
    print("="*60)
    
    processor = AudioProcessor(
        base_dir="ml_data",
        segment_duration=10000,
        silence_threshold_db=-45,
        min_speech_percentage=30
    )
    
    total_segments = processor.process_specific_files(newly_downloaded)
    processor.save_metadata()
    
    # Step 3: Quality control
    if total_segments > 0:
        print("\n" + "="*60)
        print("STEP 3: Quality control...")
        qc = QualityController(base_dir="ml_data")
        qc.verify_all()
    
    print("\n" + "="*60)
    print("✓ COLLECTION COMPLETE!")
    print(f"New segments created: {total_segments}")
    print("="*60)


if __name__ == "__main__":
    main()
