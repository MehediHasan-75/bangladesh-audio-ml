"""YouTube audio collection module"""
import json
import csv
import re
import time
import random
from pathlib import Path
from typing import List, Dict, Optional

import yt_dlp

from src.utils.helpers import get_logger

logger = get_logger(__name__)


class YouTubeAudioCollector:
    """Downloads and manages YouTube audio files with duplicate detection"""

    _MAX_RETRIES = 3
    _BACKOFF_BASE = 2  # seconds

    def __init__(self, base_dir: str = "ml_data"):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.metadata_file = self.base_dir / "download_metadata.json"
        self.downloaded_videos = self._load_metadata()
        self.newly_downloaded = []

        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def _load_metadata(self) -> Dict:
        """Load existing download metadata with error handling"""
        if not self.metadata_file.exists():
            return {}

        try:
            with open(self.metadata_file, 'r') as f:
                content = f.read().strip()
                if not content:
                    return {}
                return json.loads(content)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("Could not load metadata, creating new: %s", e)
            return {}

    def save_metadata(self):
        """Save download metadata to JSON"""
        try:
            self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metadata_file, 'w') as f:
                json.dump(self.downloaded_videos, f, indent=2)
        except Exception as e:
            logger.error("Failed to save metadata: %s", e)

    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL"""
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com/embed/([a-zA-Z0-9_-]{11})',
            r'youtube\.com/shorts/([a-zA-Z0-9_-]{11})',
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return None

    def _download_with_retry(self, url: str, ydl_opts: dict) -> Optional[dict]:
        """
        Attempt a yt-dlp download with exponential backoff on failure.

        Args:
            url: YouTube URL to download.
            ydl_opts: yt-dlp options dict.

        Returns:
            Video info dict on success, None on failure.
        """
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    return ydl.extract_info(url, download=True)
            except Exception as exc:
                if attempt == self._MAX_RETRIES:
                    logger.error("All %d retries exhausted for %s: %s",
                                 self._MAX_RETRIES, url, exc)
                    return None
                delay = self._BACKOFF_BASE ** attempt + random.uniform(0, 1)
                logger.warning("Attempt %d/%d failed (%s). Retrying in %.1fs…",
                               attempt, self._MAX_RETRIES, exc, delay)
                time.sleep(delay)
        return None

    def download_audio(self, url: str, category: str, description: str = "") -> bool:
        """
        Download audio from YouTube URL.

        Args:
            url: YouTube URL.
            category: Category name (e.g., 'bus', 'train').
            description: Optional description.

        Returns:
            True if downloaded, False if skipped or failed.
        """
        video_id = self._extract_video_id(url)

        if not video_id:
            logger.error("Invalid URL: %s", url)
            return False

        if video_id in self.downloaded_videos:
            logger.info("Already downloaded: %s", video_id)
            return False

        category_dir = self.raw_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        output_template = str(category_dir / f"{category}_{video_id}.%(ext)s")

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_template,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
            'extract_audio': True,
        }

        logger.info("Downloading: %s (%s)", video_id, description)
        info = self._download_with_retry(url, ydl_opts)

        if info is None:
            return False

        output_file = category_dir / f"{category}_{video_id}.wav"

        if output_file.exists():
            title = info.get('title', 'Unknown')
            duration = info.get('duration', 0)

            self.downloaded_videos[video_id] = {
                'url': url,
                'category': category,
                'title': title,
                'duration': duration,
                'description': description,
                'filename': str(output_file),
            }
            self.newly_downloaded.append({
                'category': category,
                'filepath': str(output_file),
                'filename': output_file.name,
                'video_id': video_id,
            })

            logger.info("Saved: %s (%ss)", output_file.name, duration)
            return True
        else:
            logger.error("File not found after download: %s", output_file)
            return False

    def download_from_csv(self, csv_path: str):
        """
        Download all videos listed in a CSV file.

        CSV format: category,url,description

        Args:
            csv_path: Path to the CSV file.
        """
        csv_file = Path(csv_path)

        if not csv_file.exists():
            logger.error("CSV file not found: %s", csv_path)
            return

        logger.info("Reading CSV: %s", csv_path)

        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                videos = list(reader)

            if not videos:
                logger.warning("No videos found in CSV")
                return

            logger.info("Found %d videos to process", len(videos))

            success = skipped = failed = 0

            for idx, row in enumerate(videos, 1):
                category = row.get('category', '').strip()
                url = row.get('url', '').strip()
                description = row.get('description', '').strip()

                if not category or not url:
                    logger.warning("[%d/%d] Missing category or URL, skipping",
                                   idx, len(videos))
                    skipped += 1
                    continue

                logger.info("[%d/%d] %s: %s", idx, len(videos), category,
                            description[:50])

                result = self.download_audio(url, category, description)

                if result:
                    success += 1
                else:
                    video_id = self._extract_video_id(url)
                    if video_id and video_id in self.downloaded_videos:
                        skipped += 1
                    else:
                        failed += 1

            logger.info(
                "Download summary — total: %d, downloaded: %d, skipped: %d, failed: %d",
                len(videos), success, skipped, failed,
            )

        except Exception as e:
            logger.error("Error reading CSV: %s", e)

    def get_newly_downloaded_files(self) -> List[Dict]:
        """Return list of files downloaded in the current session"""
        return self.newly_downloaded
