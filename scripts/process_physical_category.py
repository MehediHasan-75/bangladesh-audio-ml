#!/usr/bin/env python3
"""
Process a single category from physically collected audio.
Usage: python scripts/process_physical_category.py <category_name>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.processors.physical_audio_processor import PhysicalAudioProcessor
from src.utils.helpers import get_logger

logger = get_logger(__name__)


def main():
    if len(sys.argv) < 2:
        logger.error("Usage: python scripts/process_physical_category.py <category_name>")
        sys.exit(1)

    category = sys.argv[1]
    logger.info("=== PROCESSING CATEGORY: %s ===", category)

    processor = PhysicalAudioProcessor(
        base_dir="ml_data",
        segment_duration=10000,
        silence_threshold_db=-45,
        min_speech_percentage=30,
    )

    segments = processor.process_category(category)
    processor.save_metadata(filename=f"physical_{category}_metadata.csv")
    processor.print_summary()

    logger.info("Created %d segments for category '%s'", segments, category)


if __name__ == "__main__":
    main()
