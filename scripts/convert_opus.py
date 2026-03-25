#!/usr/bin/env python3
"""Convert OPUS files to WAV format"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from src.processors.format_cover import AudioFormatConverter

def main():
    converter = AudioFormatConverter()
    converter.convert_opus_to_wav(
        input_folder="ml_data/raw/motorcycle_engine",
        output_folder="ml_data/raw/motorcycle_engine"
    )
    print("\n🎉 Conversion complete!")


if __name__ == "__main__":
    main()
