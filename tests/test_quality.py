"""Tests for QualityController"""
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.quality.quality_controller import QualityController


@pytest.fixture
def qc(tmp_path):
    return QualityController(base_dir=str(tmp_path))


class TestVerifySegment:
    def test_valid_segment(self, qc, tmp_path):
        audio_path = tmp_path / "dummy.wav"
        audio_path.touch()

        mock_y = np.ones(480000, dtype=np.float32)  # amplitude > 0.01
        with patch("librosa.load", return_value=(mock_y, 48000)), \
             patch("librosa.get_duration", return_value=10.0):
            result = qc.verify_segment(audio_path)

        assert result["valid_sr"] is True
        assert result["valid_duration"] is True
        assert result["not_silent"] == True
        assert result["passes_all"] == True

    def test_wrong_sample_rate(self, qc, tmp_path):
        audio_path = tmp_path / "dummy.wav"
        audio_path.touch()

        mock_y = np.ones(441000, dtype=np.float32)
        with patch("librosa.load", return_value=(mock_y, 44100)), \
             patch("librosa.get_duration", return_value=10.0):
            result = qc.verify_segment(audio_path)

        assert result["valid_sr"] is False
        assert result["passes_all"] is False

    def test_silent_segment(self, qc, tmp_path):
        audio_path = tmp_path / "dummy.wav"
        audio_path.touch()

        mock_y = np.zeros(480000, dtype=np.float32)
        with patch("librosa.load", return_value=(mock_y, 48000)), \
             patch("librosa.get_duration", return_value=10.0):
            result = qc.verify_segment(audio_path)

        assert result["not_silent"] == False
        assert result["passes_all"] == False

    def test_wrong_duration(self, qc, tmp_path):
        audio_path = tmp_path / "dummy.wav"
        audio_path.touch()

        mock_y = np.ones(480000, dtype=np.float32)
        with patch("librosa.load", return_value=(mock_y, 48000)), \
             patch("librosa.get_duration", return_value=8.0):  # too short
            result = qc.verify_segment(audio_path)

        assert result["valid_duration"] is False
        assert result["passes_all"] is False

    def test_load_error_returns_error_key(self, qc, tmp_path):
        audio_path = tmp_path / "missing.wav"
        with patch("librosa.load", side_effect=Exception("file not found")):
            result = qc.verify_segment(audio_path)

        assert "error" in result
