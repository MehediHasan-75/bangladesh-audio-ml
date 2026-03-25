
#!/usr/bin/env python3
"""
Process physically collected media files (audio and video)
Handles multiple formats (.opus, .m4a, .mp3, .aac, .mp4, .mov, .mkv, .webm, .avi, etc.)

Automatically extracts audio from video files for processing.
"""

import sys
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.processors.physical_audio_processor import PhysicalAudioProcessor
from src.quality.quality_controller import QualityController


# Video format constants
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv", ".wmv", ".m4v", ".3gp"}
AUDIO_EXTENSIONS = {".opus", ".m4a", ".mp3", ".aac", ".wav", ".flac", ".ogg", ".wma"}
ALL_MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS


class MediaHandler:
    """Handle video extraction and format conversion"""

    def __init__(self, base_dir: str = "ml_data"):
        self.base_dir = Path(base_dir)
        self.physically_collected = self.base_dir / "physically_collected"
        self.extracted_audio = self.base_dir / "physically_collected" / "extracted_audio"
        self.extracted_audio.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def is_video(filepath: Path) -> bool:
        """Check if file is a video format"""
        return filepath.suffix.lower() in VIDEO_EXTENSIONS

    @staticmethod
    def is_audio(filepath: Path) -> bool:
        """Check if file is an audio format"""
        return filepath.suffix.lower() in AUDIO_EXTENSIONS

    def extract_audio_from_video(
        self, video_path: Path, output_format: str = "mp3"
    ) -> Optional[Path]:
        """
        Extract audio from video using ffmpeg
        
        Args:
            video_path: Path to video file
            output_format: Output audio format (mp3, wav, aac, m4a)
            
        Returns:
            Path to extracted audio file or None if extraction failed
        """
        output_path = self.extracted_audio / f"{video_path.stem}.{output_format}"

        # Skip if already extracted
        if output_path.exists():
            print(f"  ℹ️  Already extracted: {output_path.name}")
            return output_path

        try:
            print(f"  🎬 Extracting audio from: {video_path.name}")

            # Use ffmpeg to extract audio
            # -q:a 0 = highest quality audio
            # -map a = select audio stream
            command = [
                "ffmpeg",
                "-i",
                str(video_path),
                "-q:a",
                "0",
                "-map",
                "a",
                "-y",  # Overwrite output file
                str(output_path),
            ]

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode == 0:
                print(f"  ✅ Audio extracted: {output_path.name}")
                return output_path
            else:
                print(f"  ❌ Audio extraction failed for {video_path.name}")
                print(f"     Error: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            print(f"  ❌ Audio extraction timeout for {video_path.name}")
            return None
        except Exception as e:
            print(f"  ❌ Error extracting audio from {video_path.name}: {str(e)}")
            return None

    def get_media_files(self) -> Tuple[List[Path], List[Path]]:
        """
        Get all audio and video files from physically_collected directory
        
        Returns:
            Tuple of (audio_files, video_files)
        """
        audio_files = []
        video_files = []

        if not self.physically_collected.exists():
            print(f"⚠️  Directory not found: {self.physically_collected}")
            return audio_files, video_files

        for category_dir in self.physically_collected.iterdir():
            if not category_dir.is_dir():
                continue

            # Skip the extracted_audio subdirectory
            if category_dir.name == "extracted_audio":
                continue

            for file in category_dir.rglob("*"):
                if file.is_file():
                    if self.is_audio(file):
                        audio_files.append(file)
                    elif self.is_video(file):
                        video_files.append(file)

        return audio_files, video_files

    def preprocess_videos(self) -> List[Path]:
        """
        Extract audio from all video files
        
        Returns:
            List of extracted audio file paths
        """
        _, video_files = self.get_media_files()

        if not video_files:
            print("📹 No video files found")
            return []

        extracted_paths = []

        print(f"\n{'='*60}")
        print(f"📹 VIDEO PREPROCESSING")
        print(f"{'='*60}")
        print(f"Found {len(video_files)} video file(s)")
        print(f"{'='*60}\n")

        for video_file in video_files:
            category = video_file.parent.name
            print(f"Category: {category}")

            # Try to extract as mp3 first (most compatible)
            extracted_path = self.extract_audio_from_video(video_file, "mp3")

            if extracted_path:
                extracted_paths.append(extracted_path)

        return extracted_paths


def main():
    print("=" * 60)
    print("🎙️  PHYSICAL MEDIA PROCESSING")
    print("=" * 60)
    print("Processing audio & video from: ml_data/physically_collected/")
    print("Output destination: ml_data/processed/")
    print("=" * 60)

    # Initialize handlers
    media_handler = MediaHandler(base_dir="ml_data")

    # Preprocess videos (extract audio)
    extracted_audio_files = media_handler.preprocess_videos()

    # Initialize audio processor
    processor = PhysicalAudioProcessor(
        base_dir="ml_data",
        segment_duration=10000,  # 10 seconds
        silence_threshold_db=-45,
        min_speech_percentage=30,
    )

    # Process all categories (including extracted audio)
    total_segments = processor.process_all_categories()

    # Save metadata
    processor.save_metadata(filename="physical_processing_metadata.csv")

    # Print summary
    processor.print_summary()

    # Run quality control if segments were created
    if total_segments > 0:
        print("\n" + "=" * 60)
        print("🔍 QUALITY CONTROL")
        print("=" * 60)
        qc = QualityController(base_dir="ml_data")
        qc.verify_all()

    print("\n" + "=" * 60)
    print("✅ PROCESSING COMPLETE!")
    print("=" * 60)

    # Summary
    audio_files, video_files = media_handler.get_media_files()
    print(f"\n📊 SUMMARY:")
    print(f"  • Audio files found: {len(audio_files)}")
    print(f"  • Video files found: {len(video_files)}")
    print(f"  • Audio extracted from videos: {len(extracted_audio_files)}")
    print(f"  • Total segments processed: {total_segments}")


if __name__ == "__main__":
    main()