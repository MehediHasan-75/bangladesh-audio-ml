"""Audio format conversion utilities"""
from pydub import AudioSegment
from pathlib import Path
from typing import List


class AudioFormatConverter:
    """Convert between audio formats (OPUS, MP3, etc. to WAV)"""
    
    @staticmethod
    def convert_opus_to_wav(input_folder: str, output_folder: str = None):
        """Convert all OPUS files in folder to WAV"""
        input_path = Path(input_folder)
        output_path = Path(output_folder) if output_folder else input_path
        output_path.mkdir(parents=True, exist_ok=True)
        
        for file_path in input_path.glob("*.opus"):
            try:
                audio = AudioSegment.from_file(str(file_path))
                output_file = output_path / f"{file_path.stem}.wav"
                audio.export(str(output_file), format="wav")
                print(f"✅ Converted: {file_path.name} → {output_file.name}")
            except Exception as e:
                print(f"❌ Failed to convert {file_path.name}: {e}")
    
    @staticmethod
    def batch_convert(files: List[str], target_format: str = "wav"):
        """Convert multiple files to target format"""
        pass
