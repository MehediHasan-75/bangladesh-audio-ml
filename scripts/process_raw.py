#!/usr/bin/env python3
"""Process all existing raw audio files (batch processing)"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.processors.audio_processor import AudioProcessor
from src.quality.quality_controller import QualityController


def main():
    print("="*60)
    print("PROCESSING RAW AUDIO FILES")
    print("="*60)
    
    processor = AudioProcessor(base_dir="ml_data")
    segments = processor.process_all_raw_files()
    processor.save_metadata()
    
    if segments > 0:
        qc = QualityController(base_dir="ml_data")
        qc.verify_all()
    
    print(f"\n✓ Created {segments} segments")


if __name__ == "__main__":
    main()
