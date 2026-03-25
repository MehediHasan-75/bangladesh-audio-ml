"""Tests for AudioProcessor"""
import pytest
from pathlib import Path
from pydub import AudioSegment
from pydub.generators import Sine

from src.processors.audio_processor import AudioProcessor


def make_sine_segment(duration_ms: int = 10000, freq: int = 440) -> AudioSegment:
    """Generate a sine-wave AudioSegment for testing."""
    return Sine(freq).to_audio_segment(duration=duration_ms).set_frame_rate(48000)


@pytest.fixture
def processor(tmp_path):
    return AudioProcessor(
        base_dir=str(tmp_path),
        segment_duration=10000,
        silence_threshold_db=-45,
        min_speech_percentage=30,
    )


class TestIsSegmentValid:
    def test_loud_segment_is_valid(self, processor):
        segment = make_sine_segment(10000)
        is_valid, dbfs, pct = processor.is_segment_valid(segment)
        assert is_valid
        assert dbfs > -45
        assert pct >= 30

    def test_silent_segment_is_invalid(self, processor):
        segment = AudioSegment.silent(duration=10000)
        is_valid, dbfs, pct = processor.is_segment_valid(segment)
        assert not is_valid


class TestCategoryCounter:
    def test_starts_at_zero_for_new_category(self, processor, tmp_path):
        (tmp_path / "processed" / "bus").mkdir(parents=True)
        processor.initialize_category_counter("bus")
        assert processor.category_counters["bus"] == 0

    def test_continues_from_existing_files(self, processor, tmp_path):
        cat_dir = tmp_path / "processed" / "bus"
        cat_dir.mkdir(parents=True)
        (cat_dir / "bus_0004.wav").touch()
        (cat_dir / "bus_0009.wav").touch()

        processor.initialize_category_counter("bus")
        assert processor.category_counters["bus"] == 10

    def test_get_next_increments(self, processor, tmp_path):
        (tmp_path / "processed" / "train").mkdir(parents=True)
        n1 = processor.get_next_segment_number("train")
        n2 = processor.get_next_segment_number("train")
        assert n2 == n1 + 1


class TestSegmentAudio:
    def test_creates_wav_file(self, processor, tmp_path):
        # Create a 30-second raw WAV file
        raw_dir = tmp_path / "raw" / "bus"
        raw_dir.mkdir(parents=True)
        processed_dir = tmp_path / "processed" / "bus"
        processed_dir.mkdir(parents=True)

        audio = make_sine_segment(30000)
        src = raw_dir / "test.wav"
        audio.export(str(src), format="wav")

        count = processor.segment_audio(str(src), "bus")
        assert count >= 1
        assert any(processed_dir.glob("bus_*.wav"))
