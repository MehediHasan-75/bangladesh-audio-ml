"""Video-to-audio extraction using FFmpeg"""
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from src.utils.helpers import get_logger

logger = get_logger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv", ".wmv", ".m4v", ".3gp"}
AUDIO_EXTENSIONS = {".opus", ".m4a", ".mp3", ".aac", ".wav", ".flac", ".ogg", ".wma"}


class MediaHandler:
    """Handle video extraction and format conversion"""

    def __init__(self, base_dir: str = "ml_data"):
        self.base_dir = Path(base_dir)
        self.physically_collected = self.base_dir / "physically_collected"
        self.extracted_audio = self.physically_collected / "extracted_audio"
        self.extracted_audio.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def is_video(filepath: Path) -> bool:
        return filepath.suffix.lower() in VIDEO_EXTENSIONS

    @staticmethod
    def is_audio(filepath: Path) -> bool:
        return filepath.suffix.lower() in AUDIO_EXTENSIONS

    def extract_audio_from_video(
        self, video_path: Path, output_format: str = "mp3"
    ) -> Optional[Path]:
        """
        Extract audio from a video file using FFmpeg.

        Args:
            video_path: Path to video file.
            output_format: Output audio format (mp3, wav, aac, m4a).

        Returns:
            Path to extracted audio file, or None on failure.
        """
        output_path = self.extracted_audio / f"{video_path.stem}.{output_format}"

        if output_path.exists():
            logger.info("Already extracted: %s", output_path.name)
            return output_path

        try:
            logger.info("Extracting audio from: %s", video_path.name)
            command = [
                "ffmpeg", "-i", str(video_path),
                "-q:a", "0", "-map", "a", "-y", str(output_path),
            ]
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0:
                logger.info("Audio extracted: %s", output_path.name)
                return output_path
            else:
                logger.error("Extraction failed for %s: %s",
                             video_path.name, result.stderr)
                return None
        except subprocess.TimeoutExpired:
            logger.error("Extraction timeout for %s", video_path.name)
            return None
        except Exception as e:
            logger.error("Error extracting audio from %s: %s", video_path.name, e)
            return None

    def get_media_files(self) -> Tuple[List[Path], List[Path]]:
        """
        Scan physically_collected for audio and video files.

        Returns:
            Tuple of (audio_files, video_files).
        """
        audio_files: List[Path] = []
        video_files: List[Path] = []

        if not self.physically_collected.exists():
            logger.warning("Directory not found: %s", self.physically_collected)
            return audio_files, video_files

        for category_dir in self.physically_collected.iterdir():
            if not category_dir.is_dir() or category_dir.name == "extracted_audio":
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
        Extract audio from all discovered video files.

        Returns:
            List of extracted audio file paths.
        """
        _, video_files = self.get_media_files()

        if not video_files:
            logger.info("No video files found")
            return []

        logger.info("Preprocessing %d video file(s)", len(video_files))
        extracted_paths = []

        for video_file in video_files:
            extracted_path = self.extract_audio_from_video(video_file, "mp3")
            if extracted_path:
                extracted_paths.append(extracted_path)

        return extracted_paths
