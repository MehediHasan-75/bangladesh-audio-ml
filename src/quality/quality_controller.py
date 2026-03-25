"""Quality assurance for processed audio segments"""
from pathlib import Path
from typing import Dict

import librosa
import numpy as np
import pandas as pd

from src.utils.helpers import get_logger

logger = get_logger(__name__)


class QualityController:
    """Verify audio segments meet ML training standards"""

    def __init__(self, base_dir: str = "ml_data"):
        self.base_dir = Path(base_dir)
        self.processed_dir = self.base_dir / "processed"

    def verify_segment(self, audio_path: Path) -> Dict:
        """
        Run quality checks on a single audio segment.

        Args:
            audio_path: Path to the WAV file.

        Returns:
            Dict with check results; includes 'error' key on failure.
        """
        try:
            y, sr = librosa.load(str(audio_path), sr=None)
            duration = librosa.get_duration(y=y, sr=sr)

            checks = {
                'file': str(audio_path),
                'sample_rate': sr,
                'duration': duration,
                'valid_sr': sr == 48000,
                'valid_duration': abs(duration - 10.0) < 0.2,
                'not_silent': np.abs(y).max() > 0.01,
                'passes_all': False,
            }
            checks['passes_all'] = all([
                checks['valid_sr'],
                checks['valid_duration'],
                checks['not_silent'],
            ])

            return checks
        except Exception as e:
            logger.error("Error verifying %s: %s", audio_path.name, e)
            return {'file': str(audio_path), 'error': str(e)}

    def verify_all(self) -> pd.DataFrame:
        """
        Verify all processed audio files and generate a quality report.

        Returns:
            DataFrame with one row per segment.
        """
        audio_files = sorted(self.processed_dir.glob("**/*.wav"))
        logger.info("Verifying %d segments…", len(audio_files))

        results = [self.verify_segment(f) for f in audio_files]

        df = pd.DataFrame(results)
        report_path = self.base_dir / "quality_report.csv"
        df.to_csv(report_path, index=False)

        self._log_summary(df, report_path)
        return df

    def _log_summary(self, df: pd.DataFrame, report_path: Path):
        """Log quality control summary"""
        total = len(df)
        passed = int(df['passes_all'].sum()) if 'passes_all' in df else 0

        pass_rate = (passed / total * 100) if total > 0 else 0.0
        logger.info(
            "QC report — total: %d, passed: %d, failed: %d, pass rate: %.1f%% — saved: %s",
            total, passed, total - passed, pass_rate, report_path,
        )
