"""Quality assurance for processed audio segments"""
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict


class QualityController:
    """Verify audio segments meet ML training standards"""
    
    def __init__(self, base_dir: str = "ml_data"):
        self.base_dir = Path(base_dir)
        self.processed_dir = self.base_dir / "processed"
    
    def verify_segment(self, audio_path: Path) -> Dict:
        """Run quality checks on single audio segment"""
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
                'passes_all': False
            }
            
            checks['passes_all'] = all([
                checks['valid_sr'],
                checks['valid_duration'],
                checks['not_silent']
            ])
            
            return checks
        except Exception as e:
            return {'file': str(audio_path), 'error': str(e)}
    
    def verify_all(self) -> pd.DataFrame:
        """Verify all processed audio files and generate report"""
        results = []
        for audio_file in sorted(self.processed_dir.glob("**/*.wav")):
            result = self.verify_segment(audio_file)
            results.append(result)
        
        df = pd.DataFrame(results)
        report_path = self.base_dir / "quality_report.csv"
        df.to_csv(report_path, index=False)
        
        self._print_summary(df, report_path)
        return df
    
    def _print_summary(self, df: pd.DataFrame, report_path: Path):
        """Print quality control summary"""
        total = len(df)
        passed = df['passes_all'].sum() if 'passes_all' in df else 0
        
        print(f"\n{'='*60}")
        print(f"QUALITY CONTROL REPORT")
        print(f"{'='*60}")
        print(f"Total files: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Pass rate: {(passed/total*100):.1f}%")
        print(f"\nReport saved: {report_path}")
