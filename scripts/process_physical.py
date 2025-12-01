#!/usr/bin/env python3
"""
Process physically collected audio files
Handles multiple formats (.opus, .m4a, .mp3, .aac, etc.)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.processors.physical_audio_processor import PhysicalAudioProcessor
from src.quality.quality_controller import QualityController


def main():
    print("="*60)
    print("🎙️  PHYSICAL AUDIO PROCESSING")
    print("="*60)
    print("Processing audio from: ml_data/physically_collected/")
    print("Output destination: ml_data/processed/")
    print("="*60)
    
    # Initialize processor
    processor = PhysicalAudioProcessor(
        base_dir="ml_data",
        segment_duration=10000,        # 10 seconds
        silence_threshold_db=-45,
        min_speech_percentage=30
    )
    
    # Process all categories
    total_segments = processor.process_all_categories()
    
    # Save metadata
    processor.save_metadata(filename="physical_processing_metadata.csv")
    
    # Print summary
    processor.print_summary()
    
    # Run quality control if segments were created
    if total_segments > 0:
        print("\n" + "="*60)
        print("🔍 QUALITY CONTROL")
        print("="*60)
        qc = QualityController(base_dir="ml_data")
        qc.verify_all()
    
    print("\n" + "="*60)
    print("✅ PROCESSING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
