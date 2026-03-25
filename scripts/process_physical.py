#!/usr/bin/env python3
"""
Process physically collected media files (audio and video).
Handles multiple formats; automatically extracts audio from video files.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.processors.media_handler import MediaHandler
from src.processors.physical_audio_processor import PhysicalAudioProcessor
from src.quality.quality_controller import QualityController
from src.utils.helpers import get_logger

logger = get_logger(__name__)


def main():
    logger.info("=== PHYSICAL MEDIA PROCESSING ===")
    logger.info("Input:  ml_data/physically_collected/")
    logger.info("Output: ml_data/processed/")

    media_handler = MediaHandler(base_dir="ml_data")
    extracted_audio_files = media_handler.preprocess_videos()

    processor = PhysicalAudioProcessor(
        base_dir="ml_data",
        segment_duration=10000,
        silence_threshold_db=-45,
        min_speech_percentage=30,
    )

    total_segments = processor.process_all_categories()
    processor.save_metadata(filename="physical_processing_metadata.csv")
    processor.print_summary()

    if total_segments > 0:
        logger.info("Running quality control…")
        qc = QualityController(base_dir="ml_data")
        qc.verify_all()

    audio_files, video_files = media_handler.get_media_files()
    logger.info(
        "Done — audio files: %d, video files: %d, extracted: %d, segments: %d",
        len(audio_files), len(video_files), len(extracted_audio_files), total_segments,
    )


if __name__ == "__main__":
    main()
