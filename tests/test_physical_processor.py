"""Tests for PhysicalAudioProcessor"""
import pytest
from pathlib import Path
from pydub import AudioSegment
from pydub.generators import Sine

from src.processors.physical_audio_processor import PhysicalAudioProcessor


def make_sine_wav(path: Path, duration_ms: int = 30000, freq: int = 440):
    """Write a sine-wave WAV to *path*."""
    audio = Sine(freq).to_audio_segment(duration=duration_ms).set_frame_rate(48000)
    audio.export(str(path), format="wav")


@pytest.fixture
def processor(tmp_path):
    return PhysicalAudioProcessor(
        base_dir=str(tmp_path),
        segment_duration=10000,
        silence_threshold_db=-45,
        min_speech_percentage=30,
    )


class TestIsSegmentValid:
    def test_audible_segment_is_valid(self, processor):
        seg = Sine(440).to_audio_segment(duration=10000).set_frame_rate(48000)
        is_valid, dbfs, pct = processor.is_segment_valid(seg)
        assert is_valid

    def test_silent_segment_is_invalid(self, processor):
        seg = AudioSegment.silent(duration=10000)
        is_valid, _, _ = processor.is_segment_valid(seg)
        assert not is_valid


class TestGetStartingNumber:
    def test_empty_category_starts_at_zero(self, processor, tmp_path):
        (tmp_path / "processed" / "bike").mkdir(parents=True)
        n = processor._get_starting_number("bike")
        assert n == 0

    def test_continues_after_highest_existing(self, processor, tmp_path):
        cat = tmp_path / "processed" / "bike"
        cat.mkdir(parents=True)
        (cat / "bike_0007.wav").touch()
        (cat / "bike_0002.wav").touch()

        n = processor._get_starting_number("bike")
        assert n == 8


class TestProcessCategory:
    def test_processes_wav_files(self, processor, tmp_path):
        cat_input = tmp_path / "physically_collected" / "bus"
        cat_input.mkdir(parents=True)
        (tmp_path / "processed" / "bus").mkdir(parents=True)

        make_sine_wav(cat_input / "audio1.wav", duration_ms=30000)

        count = processor.process_category("bus")
        assert count >= 1

    def test_missing_category_returns_zero(self, processor):
        count = processor.process_category("nonexistent_category")
        assert count == 0
