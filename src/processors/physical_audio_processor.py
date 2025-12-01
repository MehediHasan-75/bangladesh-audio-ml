"""
Physical Audio Processor
Process manually collected audio files from various formats
"""
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from pathlib import Path
import pandas as pd
from typing import List, Dict, Tuple, Optional


class PhysicalAudioProcessor:
    """Process physically collected audio files from multiple formats"""
    
    SUPPORTED_FORMATS = {'.opus', '.m4a', '.mp3', '.aac', '.wav', '.ogg', '.flac', '.wma',
                     '.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
    
    def __init__(
        self,
        base_dir: str = "ml_data",
        segment_duration: int = 10000,
        silence_threshold_db: int = -45,
        min_speech_percentage: int = 30
    ):
        self.base_dir = Path(base_dir)
        self.physical_dir = self.base_dir / "physically_collected"
        self.processed_dir = self.base_dir / "processed"
        
        self.segment_duration = segment_duration
        self.silence_threshold_db = silence_threshold_db
        self.min_speech_percentage = min_speech_percentage
        
        self.metadata = []
        self.category_counters = {}
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'segments_created': 0,
            'segments_rejected': 0
        }
    
    def is_segment_valid(self, segment: AudioSegment) -> Tuple[bool, float, float]:
        """
        Validate segment quality based on volume and speech content
        
        Returns:
            (is_valid, dbfs, speech_percentage)
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
    
    def _get_starting_number(self, category: str) -> int:
        """
        Find the highest existing segment number for a category
        
        Args:
            category: Category folder name (e.g., 'bike', 'truck')
        
        Returns:
            Next available number (0 if no files exist)
        """
        if category in self.category_counters:
            return self.category_counters[category]
        
        category_output = self.processed_dir / category
        category_output.mkdir(parents=True, exist_ok=True)
        
        # Find all existing segment files
        existing_files = list(category_output.glob(f"{category}_*.wav"))
        
        if not existing_files:
            self.category_counters[category] = 0
            print(f"  📁 {category}: starting from 0000")
            return 0
        
        # Extract numbers from filenames
        numbers = []
        for file_path in existing_files:
            stem = file_path.stem  # e.g., "bike_0056"
            parts = stem.split('_')
            if len(parts) >= 2 and parts[-1].isdigit():
                numbers.append(int(parts[-1]))
        
        if numbers:
            highest_num = max(numbers)
            next_num = highest_num + 1
            self.category_counters[category] = next_num
            print(f"  📁 {category}: continue from {next_num:04d} (found {len(existing_files)} existing)")
            return next_num
        else:
            self.category_counters[category] = 0
            print(f"  📁 {category}: starting from 0000")
            return 0
    
    def _load_audio_file(self, file_path: Path) -> Optional[AudioSegment]:
        """
        Load audio file from any supported format
        
        Args:
            file_path: Path to audio file
        
        Returns:
            AudioSegment or None if loading fails
        """
        try:
            # pydub can automatically detect format from file extension
            audio = AudioSegment.from_file(str(file_path))
            return audio
        except Exception as e:
            print(f"    ❌ Failed to load {file_path.name}: {e}")
            return None
    
    def process_audio_file(self, audio_path: Path, category: str) -> int:
        """
        Process single audio file: load, segment, validate, and save
        
        Args:
            audio_path: Path to source audio file
            category: Category name (folder name)
        
        Returns:
            Number of valid segments created
        """
        # Load audio
        audio = self._load_audio_file(audio_path)
        if audio is None:
            self.stats['failed_files'] += 1
            return 0
        
        duration_sec = audio.duration_seconds
        print(f"    📄 {audio_path.name} ({duration_sec:.1f}s, {audio_path.suffix})")
        
        # Prepare output directory
        category_output = self.processed_dir / category
        category_output.mkdir(parents=True, exist_ok=True)
        
        # Calculate number of segments
        num_segments = len(audio) // self.segment_duration
        segments_created = 0
        segments_rejected = 0
        
        # Process each segment
        for i in range(num_segments):
            start_ms = i * self.segment_duration
            end_ms = (i + 1) * self.segment_duration
            segment = audio[start_ms:end_ms]
            
            # Validate segment
            is_valid, dbfs, speech_pct = self.is_segment_valid(segment)
            
            if not is_valid:
                segments_rejected += 1
                self.stats['segments_rejected'] += 1
                continue
            
            # Get next segment number
            segment_num = self.category_counters[category]
            self.category_counters[category] += 1
            
            # Create output filename: category_0000.wav
            output_filename = f"{category}_{segment_num:04d}.wav"
            output_path = category_output / output_filename
            
            # Skip if file already exists
            if output_path.exists():
                print(f"      ⚠️  {output_filename} already exists, skipping")
                continue
            
            # Export as 48kHz mono WAV
            segment.export(
                str(output_path),
                format="wav",
                parameters=["-ar", "48000", "-ac", "1"]
            )
            
            segments_created += 1
            self.stats['segments_created'] += 1
            
            # Store metadata
            self.metadata.append({
                'original_file': str(audio_path),
                'original_format': audio_path.suffix,
                'segment_file': str(output_path),
                'category': category,
                'segment_number': segment_num,
                'duration_s': 10.0,
                'dbfs': dbfs,
                'speech_percentage': speech_pct,
            })
        
        self.stats['processed_files'] += 1
        print(f"      ✅ Created {segments_created}/{num_segments} segments (rejected {segments_rejected})")
        
        return segments_created
    
    def process_category(self, category: str) -> int:
        """
        Process all audio files in a category folder
        
        Args:
            category: Category folder name
        
        Returns:
            Total number of segments created
        """
        category_path = self.physical_dir / category
        
        if not category_path.exists() or not category_path.is_dir():
            print(f"  ⚠️  Category folder not found: {category_path}")
            return 0
        
        # Find all supported audio files
        audio_files = []
        for ext in self.SUPPORTED_FORMATS:
            audio_files.extend(category_path.glob(f"*{ext}"))
        
        audio_files = sorted(audio_files)
        
        if not audio_files:
            print(f"  ⚠️  No audio files found in {category}")
            return 0
        
        print(f"\n{'='*60}")
        print(f"📁 Processing category: {category}")
        print(f"{'='*60}")
        print(f"  Found {len(audio_files)} audio files")
        
        # Initialize counter for this category
        self._get_starting_number(category)
        
        total_segments = 0
        
        # Process each file
        for idx, audio_file in enumerate(audio_files, 1):
            print(f"  [{idx}/{len(audio_files)}]")
            segments = self.process_audio_file(audio_file, category)
            total_segments += segments
        
        return total_segments
    
    def process_all_categories(self) -> int:
        """
        Process all categories in physically_collected folder
        
        Returns:
            Total number of segments created across all categories
        """
        if not self.physical_dir.exists():
            print(f"❌ Physically collected folder not found: {self.physical_dir}")
            print(f"   Please create it and add audio files organized by category")
            return 0
        
        # Find all category folders
        categories = [
            d.name for d in sorted(self.physical_dir.iterdir())
            if d.is_dir() and not d.name.startswith('.')
        ]
        
        if not categories:
            print(f"❌ No category folders found in {self.physical_dir}")
            return 0
        
        print("\n" + "="*60)
        print("🎵 PROCESSING PHYSICALLY COLLECTED AUDIO")
        print("="*60)
        print(f"Found {len(categories)} categories: {', '.join(categories)}")
        
        total_segments = 0
        
        for category in categories:
            segments = self.process_category(category)
            total_segments += segments
            self.stats['total_files'] += len(list((self.physical_dir / category).iterdir()))
        
        return total_segments
    
    def save_metadata(self, filename: str = "physical_processing_metadata.csv"):
        """
        Save processing metadata to CSV
        
        Args:
            filename: Output CSV filename
        """
        if not self.metadata:
            print("  ℹ️  No metadata to save")
            return
        
        df = pd.DataFrame(self.metadata)
        output_path = self.base_dir / filename
        df.to_csv(output_path, index=False)
        print(f"\n✅ Metadata saved: {output_path}")
    
    def print_summary(self):
        """Print processing summary statistics"""
        print("\n" + "="*60)
        print("📊 PROCESSING SUMMARY")
        print("="*60)
        print(f"Total audio files found:    {self.stats['total_files']}")
        print(f"Successfully processed:     {self.stats['processed_files']}")
        print(f"Failed to process:          {self.stats['failed_files']}")
        print(f"Segments created:           {self.stats['segments_created']}")
        print(f"Segments rejected:          {self.stats['segments_rejected']}")
        
        if self.stats['total_files'] > 0:
            success_rate = (self.stats['processed_files'] / self.stats['total_files']) * 100
            print(f"Success rate:               {success_rate:.1f}%")
        
        print("="*60)
