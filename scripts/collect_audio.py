#!/usr/bin/env python3
"""Main YouTube collection pipeline — download then process only new files."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.collectors.youtube_collector import YouTubeAudioCollector
from src.processors.audio_processor import AudioProcessor
from src.quality.quality_controller import QualityController
from src.utils.helpers import get_logger

logger = get_logger(__name__)


def main():
    logger.info("=== BANGLADESH AUDIO DATA COLLECTION ===")

    # Step 1: Download from YouTube
    logger.info("Step 1: Downloading from YouTube…")
    collector = YouTubeAudioCollector(base_dir="ml_data")
    collector.download_from_csv("data/youtube_urls.csv")
    collector.save_metadata()

    newly_downloaded = collector.get_newly_downloaded_files()

    if not newly_downloaded:
        logger.info("No new files downloaded — nothing to process.")
        return

    # Step 2: Process new files
    logger.info("Step 2: Processing %d new files…", len(newly_downloaded))
    processor = AudioProcessor(
        base_dir="ml_data",
        segment_duration=10000,
        silence_threshold_db=-45,
        min_speech_percentage=30,
    )
    total_segments = processor.process_specific_files(newly_downloaded)
    processor.save_metadata()
    processor.print_stats()

    # Step 3: Quality control
    if total_segments > 0:
        logger.info("Step 3: Running quality control…")
        qc = QualityController(base_dir="ml_data")
        qc.verify_all()

    logger.info("Collection complete — %d new segments created.", total_segments)


if __name__ == "__main__":
    main()
