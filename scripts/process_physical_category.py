#!/usr/bin/env python3
"""
Process a specific category from physically collected audio
Usage: python scripts/process_physical_category.py <category_name>
Example: python scripts/process_physical_category.py bike
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.processors.physical_audio_processor import PhysicalAudioProcessor


def main():
    if len(sys.argv) < 2:
        print("❌ Usage: python scripts/process_physical_category.py <category_name>")
        print("   Example: python scripts/process_physical_category.py bike")
        sys.exit(1)
    
    category = sys.argv[1]
    
    print("="*60)
    print(f"🎙️  PROCESSING CATEGORY: {category}")
    print("="*60)
    
    processor = PhysicalAudioProcessor(
        base_dir="ml_data",
        segment_duration=10000,
        silence_threshold_db=-45,
        min_speech_percentage=30
    )
    
    segments = processor.process_category(category)
    processor.save_metadata(filename=f"physical_{category}_metadata.csv")
    processor.print_summary()
    
    print(f"\n✅ Created {segments} segments for category '{category}'")


if __name__ == "__main__":
    main()
