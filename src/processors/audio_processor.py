"""Audio segmentation and processing module"""
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from pathlib import Path
import pandas as pd
from typing import List, Dict, Tuple


class AudioProcessor:
    """Segments raw audio into ML-ready chunks"""
    
    def __init__(
        self, 
        base_dir: str = "ml_data",
        segment_duration: int = 10000,
        silence_threshold_db: int = -45, 
        min_speech_percentage: int = 30 
    ):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.segment_duration = segment_duration
        self.silence_threshold_db = silence_threshold_db
        self.min_speech_percentage = min_speech_percentage
        
        self.metadata = []
        self.category_counters = {}
        self.existing_segment_counts = {}
        self.stats = {'total_processed': 0, 'kept': 0, 'rejected': 0}
    
    def is_segment_valid(self, segment: AudioSegment) -> Tuple[bool, float, float]:
        """
        Validate segment quality based on volume and speech content
        
        Args:
            segment: AudioSegment to validate
        
        Returns:
            Tuple of (is_valid, dbfs, speech_percentage)
        """
        dbfs = segment.dBFS
        
        # Check if segment is too quiet
        if dbfs < self.silence_threshold_db:
            return False, dbfs, 0.0
        
        try:
            # Detect non-silent portions
            nonsilent_chunks = detect_nonsilent(
                segment, 
                min_silence_len=100, 
                silence_thresh=self.silence_threshold_db
            )
            
            nonsilent_duration = sum(end - start for start, end in nonsilent_chunks)
            speech_percentage = (nonsilent_duration / len(segment)) * 100 if len(segment) > 0 else 0
            
            # Check if segment has enough speech/sound content
            if speech_percentage < self.min_speech_percentage:
                return False, dbfs, speech_percentage
            
            return True, dbfs, speech_percentage
            
        except Exception:
            # Fallback: just check volume
            return dbfs > self.silence_threshold_db, dbfs, 100.0
    
    def initialize_category_counter(self, category: str):
        """
        Initialize counter for a category by checking existing files
        
        Args:
            category: Category name (e.g., 'bus', 'train')
        """
        if category in self.category_counters:
            return
        
        category_output = self.processed_dir / category
        category_output.mkdir(parents=True, exist_ok=True)
        
        existing_files = list(category_output.glob(f"{category}_*.wav"))
        
        if existing_files:
            numbers = [
                int(f.stem.split('_')[-1]) 
                for f in existing_files 
                if f.stem.split('_')[-1].isdigit()
            ]
            
            if numbers:
                highest_num = max(numbers)
                self.category_counters[category] = highest_num + 1
                self.existing_segment_counts[category] = len(existing_files)
                print(f"  ✅ {category}: continue from {highest_num+1:04d}")
            else:
                self.category_counters[category] = 0
                self.existing_segment_counts[category] = 0
        else:
            self.category_counters[category] = 0
            self.existing_segment_counts[category] = 0
    
    def get_next_segment_number(self, category: str) -> int:
        """
        Get next available segment number for category
        
        Args:
            category: Category name
        
        Returns:
            Next segment number
        """
        self.initialize_category_counter(category)
        current_num = self.category_counters[category]
        self.category_counters[category] += 1
        return current_num
    
    def segment_audio(self, audio_path: str, category: str) -> int:
        """
        Split single audio file into 10s segments
        
        Args:
            audio_path: Path to audio file
            category: Category name
        
        Returns:
            Number of valid segments created (0 on error)
        """
        try:
            audio = AudioSegment.from_wav(audio_path)
            print(f"{Path(audio_path).name} ({audio.duration_seconds:.1f}s)")
            
            category_output = self.processed_dir / category
            num_segments = len(audio) // self.segment_duration
            segments_created = 0
            segments_rejected = 0
            
            for i in range(num_segments):
                segment = audio[i * self.segment_duration:(i + 1) * self.segment_duration]
                is_valid, dbfs, speech_pct = self.is_segment_valid(segment)
                self.stats['total_processed'] += 1
                
                if not is_valid:
                    segments_rejected += 1
                    self.stats['rejected'] += 1
                    continue
                
                segment_num = self.get_next_segment_number(category)
                output_path = category_output / f"{category}_{segment_num:04d}.wav"
                
                if output_path.exists():
                    continue
                
                # Export as 48kHz mono WAV
                segment.export(
                    output_path, 
                    format="wav", 
                    parameters=["-ar", "48000", "-ac", "1"]
                )
                segments_created += 1
                self.stats['kept'] += 1
                
                # Record metadata
                self.metadata.append({
                    'original_file': str(audio_path),
                    'segment_file': str(output_path),
                    'category': category,
                    'segment_number': segment_num,
                    'duration_s': 10.0,
                    'dbfs': dbfs,
                    'speech_percentage': speech_pct,
                })
            
            print(f"  ✓ {segments_created}/{num_segments} (rejected {segments_rejected})")
            return segments_created
            
        except Exception as e:
            print(f"  ✗ {str(e)}")
            return 0  # Return 0 instead of None on error
    
    def process_all_raw_files(self) -> int:
        """
        Process all files in raw directory
        
        Returns:
            Total number of segments created
        """
        print("\n" + "="*60)
        print("🔍 DETECTING EXISTING")
        print("="*60)
        
        # Check if raw directory exists
        if not self.raw_dir.exists():
            print(f"⚠️  Raw directory not found: {self.raw_dir}")
            return 0
        
        categories = [
            d.name for d in sorted(self.raw_dir.iterdir()) 
            if d.is_dir() and list(d.glob("*.wav"))
        ]
        
        if not categories:
            print("⚠️  No categories with WAV files found")
            return 0
        
        for category in categories:
            self.initialize_category_counter(category)
        
        print("\n🎵 PROCESSING ALL RAW FILES\n" + "="*60)
        total_segments = 0
        total_files = 0
        
        for category_dir in sorted(self.raw_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            
            wav_files = sorted(list(category_dir.glob("*.wav")))
            if not wav_files:
                continue
            
            total_files += len(wav_files)
            print(f"\n📁 {category_dir.name}: {len(wav_files)} files")
            
            for idx, wav_file in enumerate(wav_files, 1):
                print(f"[{idx}/{len(wav_files)}] ", end='')
                segments = self.segment_audio(str(wav_file), category_dir.name)
                total_segments += segments
        
        print(f"\n{'='*60}\n✅ {total_files} files → {total_segments} segments\n{'='*60}")
        return total_segments
    
    def process_specific_files(self, files_to_process: List[Dict]) -> int:
        """
        Process only specified files (for incremental processing)
        
        Args:
            files_to_process: List of dicts with keys 'category', 'filepath', 'filename'
        
        Returns:
            Total number of segments created
        """
        if not files_to_process:
            return 0
        
        print(f"\n🎵 PROCESSING {len(files_to_process)} NEW FILES\n" + "="*60)
        total_segments = 0
        
        # Group files by category
        files_by_category = {}
        for file_info in files_to_process:
            category = file_info['category']
            if category not in files_by_category:
                files_by_category[category] = []
            files_by_category[category].append(file_info)
        
        # Process each category
        for category in sorted(files_by_category.keys()):
            self.initialize_category_counter(category)
            files = files_by_category[category]
            print(f"\n📁 {category}: {len(files)} files")
            
            for idx, file_info in enumerate(files, 1):
                print(f"[{idx}/{len(files)}] ", end='')
                segments = self.segment_audio(file_info['filepath'], category)
                
                # Handle None return (shouldn't happen, but safety check)
                if segments is None:
                    segments = 0
                
                total_segments += segments
        
        print(f"\n{'='*60}\n✅ {total_segments} new segments\n{'='*60}")
        return total_segments
    
    def save_metadata(self, filename: str = "processing_metadata.csv"):
        """
        Export processing metadata to CSV
        
        Args:
            filename: Output CSV filename
        """
        if self.metadata:
            df = pd.DataFrame(self.metadata)
            output_path = self.base_dir / filename
            df.to_csv(output_path, index=False)
            print(f"\n✅ Metadata: {output_path}")
        else:
            print(f"\nℹ️  No metadata to save")
    
    def print_stats(self):
        """Print processing statistics"""
        print("\n" + "="*60)
        print("📊 PROCESSING STATISTICS")
        print("="*60)
        print(f"Total segments processed: {self.stats['total_processed']}")
        print(f"Kept (valid):            {self.stats['kept']}")
        print(f"Rejected (invalid):      {self.stats['rejected']}")
        
        if self.stats['total_processed'] > 0:
            keep_rate = (self.stats['kept'] / self.stats['total_processed']) * 100
            print(f"Keep rate:               {keep_rate:.1f}%")
        
        print("="*60)
