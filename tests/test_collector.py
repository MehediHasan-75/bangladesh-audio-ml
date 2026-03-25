"""Tests for YouTubeAudioCollector"""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.collectors.youtube_collector import YouTubeAudioCollector


@pytest.fixture
def collector(tmp_path):
    return YouTubeAudioCollector(base_dir=str(tmp_path))


class TestExtractVideoId:
    def test_standard_watch_url(self, collector):
        assert collector._extract_video_id(
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        ) == "dQw4w9WgXcQ"

    def test_short_url(self, collector):
        assert collector._extract_video_id(
            "https://youtu.be/dQw4w9WgXcQ"
        ) == "dQw4w9WgXcQ"

    def test_shorts_url(self, collector):
        assert collector._extract_video_id(
            "https://www.youtube.com/shorts/dQw4w9WgXcQ"
        ) == "dQw4w9WgXcQ"

    def test_embed_url(self, collector):
        assert collector._extract_video_id(
            "https://www.youtube.com/embed/dQw4w9WgXcQ"
        ) == "dQw4w9WgXcQ"

    def test_invalid_url_returns_none(self, collector):
        assert collector._extract_video_id("https://example.com") is None


class TestMetadataPersistence:
    def test_save_and_reload(self, tmp_path):
        c = YouTubeAudioCollector(base_dir=str(tmp_path))
        c.downloaded_videos["abc123"] = {"category": "bus", "url": "https://..."}
        c.save_metadata()

        c2 = YouTubeAudioCollector(base_dir=str(tmp_path))
        assert "abc123" in c2.downloaded_videos

    def test_empty_metadata_file(self, tmp_path):
        meta = tmp_path / "ml_data" / "download_metadata.json"
        meta.parent.mkdir(parents=True)
        meta.write_text("")
        c = YouTubeAudioCollector(base_dir=str(tmp_path / "ml_data"))
        assert c.downloaded_videos == {}

    def test_corrupted_metadata_file(self, tmp_path):
        meta = tmp_path / "ml_data" / "download_metadata.json"
        meta.parent.mkdir(parents=True)
        meta.write_text("{not valid json")
        c = YouTubeAudioCollector(base_dir=str(tmp_path / "ml_data"))
        assert c.downloaded_videos == {}


class TestDownloadAudio:
    def test_skips_already_downloaded(self, collector):
        video_id = "dQw4w9WgXcQ"
        collector.downloaded_videos[video_id] = {}
        result = collector.download_audio(
            "https://youtu.be/dQw4w9WgXcQ", "bus"
        )
        assert result is False

    def test_invalid_url_returns_false(self, collector):
        result = collector.download_audio("https://example.com", "bus")
        assert result is False
