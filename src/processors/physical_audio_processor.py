"""
Physical Audio Processor
Process manually collected audio files from various formats.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

from src.utils.helpers import get_logger

logger = get_logger(__name__)


class PhysicalAudioProcessor:
    """Process physically collected audio files from multiple formats"""

    SUPPORTED_FORMATS = {
        '.opus', '.m4a', '.mp3', '.aac', '.wav', '.ogg', '.flac', '.wma',
        '.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv',
    }

    def __init__(
        self,
        base_dir: str = "ml_data",
        segment_duration: int = 10000,
        silence_threshold_db: int = -45,
        min_speech_percentage: int = 30,
    ):
        self.base_dir = Path(base_dir)
        self.physical_dir = self.base_dir / "physically_collected"
        self.processed_dir = self.base_dir / "processed"

        self.segment_duration = segment_duration
        self.silence_threshold_db = silence_threshold_db
        self.min_speech_percentage = min_speech_percentage

        self.metadata: List[Dict] = []
        self.category_counters: Dict[str, int] = {}
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'segments_created': 0,
            'segments_rejected': 0,
        }

    def is_segment_valid(self, segment: AudioSegment) -> Tuple[bool, float, float]:
        """
        Validate segment quality based on volume and speech content.

        Returns:
            (is_valid, dbfs, speech_percentage)
        """
        dbfs = segment.dBFS

        if dbfs < self.silence_threshold_db:
            return False, dbfs, 0.0

        try:
            nonsilent_chunks = detect_nonsilent(
                segment,
                min_silence_len=100,
                silence_thresh=self.silence_threshold_db,
            )
            nonsilent_duration = sum(end - start for start, end in nonsilent_chunks)
            speech_percentage = (
                (nonsilent_duration / len(segment)) * 100 if len(segment) > 0 else 0
            )

            if speech_percentage < self.min_speech_percentage:
                return False, dbfs, speech_percentage

            return True, dbfs, speech_percentage
        except Exception:
            return dbfs > self.silence_threshold_db, dbfs, 100.0

    def _get_starting_number(self, category: str) -> int:
        """
        Find the highest existing segment number for a category.

        Args:
            category: Category folder name (e.g., 'bike', 'truck').

        Returns:
            Next available segment number (0 if no files exist).
        """
        if category in self.category_counters:
            return self.category_counters[category]

        category_output = self.processed_dir / category
        category_output.mkdir(parents=True, exist_ok=True)

        existing_files = list(category_output.glob(f"{category}_*.wav"))

        if not existing_files:
            self.category_counters[category] = 0
            logger.info("%s: starting from 0000", category)
            return 0

        numbers = [
            int(p.stem.split('_')[-1])
            for p in existing_files
            if p.stem.split('_')[-1].isdigit()
        ]

        if numbers:
            next_num = max(numbers) + 1
            self.category_counters[category] = next_num
            logger.info("%s: continuing from %04d (%d existing)",
                        category, next_num, len(existing_files))
            return next_num

        self.category_counters[category] = 0
        logger.info("%s: starting from 0000", category)
        return 0

    def _load_audio_file(self, file_path: Path) -> Optional[AudioSegment]:
        """
        Load an audio file from any supported format.

        Args:
            file_path: Path to the audio file.

        Returns:
            AudioSegment on success, None on failure.
        """
        try:
            return AudioSegment.from_file(str(file_path))
        except Exception as e:
            logger.error("Failed to load %s: %s", file_path.name, e)
            return None

    def process_audio_file(self, audio_path: Path, category: str) -> int:
        """
        Load, segment, validate, and save a single audio file.

        Args:
            audio_path: Path to source audio file.
            category: Category name (folder name).

        Returns:
            Number of valid segments created.
        """
        audio = self._load_audio_file(audio_path)
        if audio is None:
            self.stats['failed_files'] += 1
            return 0

        logger.info("%s (%.1fs, %s)", audio_path.name,
                    audio.duration_seconds, audio_path.suffix)

        category_output = self.processed_dir / category
        category_output.mkdir(parents=True, exist_ok=True)

        num_segments = len(audio) // self.segment_duration
        segments_created = 0
        segments_rejected = 0

        for i in range(num_segments):
            segment = audio[i * self.segment_duration:(i + 1) * self.segment_duration]
            is_valid, dbfs, speech_pct = self.is_segment_valid(segment)

            if not is_valid:
                segments_rejected += 1
                self.stats['segments_rejected'] += 1
                continue

            segment_num = self.category_counters[category]
            self.category_counters[category] += 1

            output_filename = f"{category}_{segment_num:04d}.wav"
            output_path = category_output / output_filename

            if output_path.exists():
                logger.warning("%s already exists, skipping", output_filename)
                continue

            segment.export(
                str(output_path),
                format="wav",
                parameters=["-ar", "48000", "-ac", "1"],
            )

            segments_created += 1
            self.stats['segments_created'] += 1

            self.metadata.append({
                'original_file': str(audio_path),
                'original_format': audio_path.suffix,
                'segment_file': str(output_path),
                'category': category,
                'segment_number': segment_num,
                'duration_s': 10.0,
                'dbfs': dbfs,
                'speech_percentage': speech_pct,
            })

        self.stats['processed_files'] += 1
        logger.info("Created %d/%d segments (rejected %d)",
                    segments_created, num_segments, segments_rejected)
        return segments_created

    def process_category(self, category: str) -> int:
        """
        Process all audio files in a category folder.

        Args:
            category: Category folder name.

        Returns:
            Total number of segments created.
        """
        category_path = self.physical_dir / category

        if not category_path.exists() or not category_path.is_dir():
            logger.warning("Category folder not found: %s", category_path)
            return 0

        audio_files = sorted(
            f for ext in self.SUPPORTED_FORMATS
            for f in category_path.glob(f"*{ext}")
        )

        if not audio_files:
            logger.warning("No audio files found in %s", category)
            return 0

        logger.info("Processing category '%s': %d files", category, len(audio_files))
        self._get_starting_number(category)

        total_segments = 0
        for idx, audio_file in enumerate(audio_files, 1):
            logger.debug("[%d/%d] %s", idx, len(audio_files), audio_file.name)
            total_segments += self.process_audio_file(audio_file, category)

        return total_segments

    def process_all_categories(self) -> int:
        """
        Process all categories in the physically_collected folder.

        Returns:
            Total number of segments created across all categories.
        """
        if not self.physical_dir.exists():
            logger.error("Physically collected folder not found: %s", self.physical_dir)
            return 0

        categories = [
            d.name for d in sorted(self.physical_dir.iterdir())
            if d.is_dir() and not d.name.startswith('.')
        ]

        if not categories:
            logger.error("No category folders found in %s", self.physical_dir)
            return 0

        logger.info("Processing %d categories: %s", len(categories), ', '.join(categories))

        total_segments = 0
        for category in categories:
            segments = self.process_category(category)
            total_segments += segments
            self.stats['total_files'] += len(list((self.physical_dir / category).iterdir()))

        return total_segments

    def save_metadata(self, filename: str = "physical_processing_metadata.csv"):
        """
        Save processing metadata to CSV.

        Args:
            filename: Output CSV filename.
        """
        if not self.metadata:
            logger.info("No metadata to save")
            return

        df = pd.DataFrame(self.metadata)
        output_path = self.base_dir / filename
        df.to_csv(output_path, index=False)
        logger.info("Metadata saved: %s", output_path)

    def print_summary(self):
        """Log processing summary statistics"""
        s = self.stats
        success_rate = (
            (s['processed_files'] / s['total_files'] * 100)
            if s['total_files'] > 0 else 0.0
        )
        logger.info(
            "Summary — files: %d, processed: %d, failed: %d, "
            "segments created: %d, rejected: %d, success rate: %.1f%%",
            s['total_files'], s['processed_files'], s['failed_files'],
            s['segments_created'], s['segments_rejected'], success_rate,
        )
