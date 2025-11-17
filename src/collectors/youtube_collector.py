"""YouTube audio collection module"""
import yt_dlp
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional


class YouTubeAudioCollector:
    """Downloads and manages YouTube audio files with duplicate detection"""
    
    def __init__(self, base_dir: str = "ml_data"):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.metadata_file = self.base_dir / "download_metadata.json"
        self.downloaded_videos = self._load_metadata()
        self.newly_downloaded = []
        
        # Create directories
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
            print(f"⚠️  Warning: Could not load metadata, creating new: {e}")
            return {}
    
    def save_metadata(self):
        """Save download metadata to JSON"""
        try:
            self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metadata_file, 'w') as f:
                json.dump(self.downloaded_videos, f, indent=2)
        except Exception as e:
            print(f"❌ Failed to save metadata: {e}")
    
    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL"""
        import re
        
        # Handle different URL formats
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com/embed/([a-zA-Z0-9_-]{11})',
            r'youtube\.com/shorts/([a-zA-Z0-9_-]{11})',  # ← ADD THIS LINE
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None

    
    def download_audio(self, url: str, category: str, description: str = "") -> bool:
        """
        Download audio from YouTube URL
        
        Args:
            url: YouTube URL
            category: Category name (e.g., 'bus', 'train')
            description: Optional description
        
        Returns:
            True if downloaded, False if skipped or failed
        """
        video_id = self._extract_video_id(url)
        
        if not video_id:
            print(f"  ❌ Invalid URL: {url}")
            return False
        
        # Check if already downloaded
        if video_id in self.downloaded_videos:
            print(f"  ⏭️  Already downloaded: {video_id}")
            return False
        
        # Create category directory
        category_dir = self.raw_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        # Output filename
        output_template = str(category_dir / f"{category}_{video_id}.%(ext)s")
        
        # yt-dlp options
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
        
        try:
            print(f"  📥 Downloading: {video_id} ({description})")
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                title = info.get('title', 'Unknown')
                duration = info.get('duration', 0)
            
            # Find the downloaded file
            output_file = category_dir / f"{category}_{video_id}.wav"
            
            if output_file.exists():
                # Record metadata
                self.downloaded_videos[video_id] = {
                    'url': url,
                    'category': category,
                    'title': title,
                    'duration': duration,
                    'description': description,
                    'filename': str(output_file)
                }
                
                # Track newly downloaded
                self.newly_downloaded.append({
                    'category': category,
                    'filepath': str(output_file),
                    'filename': output_file.name,
                    'video_id': video_id
                })
                
                print(f"    ✅ Saved: {output_file.name} ({duration}s)")
                return True
            else:
                print(f"    ❌ File not found after download")
                return False
                
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            return False
    
    def download_from_csv(self, csv_path: str):
        """
        Download all videos from CSV file
        
        CSV format:
        category,url,description
        bus,https://youtu.be/abc123,Bus engine sound
        
        Args:
            csv_path: Path to CSV file
        """
        csv_file = Path(csv_path)
        
        if not csv_file.exists():
            print(f"❌ CSV file not found: {csv_path}")
            return
        
        print(f"📄 Reading CSV: {csv_path}")
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                videos = list(reader)
            
            if not videos:
                print("⚠️  No videos found in CSV")
                return
            
            print(f"Found {len(videos)} videos to process\n")
            
            success = 0
            skipped = 0
            failed = 0
            
            for idx, row in enumerate(videos, 1):
                category = row.get('category', '').strip()
                url = row.get('url', '').strip()
                description = row.get('description', '').strip()
                
                if not category or not url:
                    print(f"  [{idx}/{len(videos)}] ⚠️  Missing category or URL, skipping")
                    skipped += 1
                    continue
                
                print(f"[{idx}/{len(videos)}] {category}: {description[:50]}")
                
                result = self.download_audio(url, category, description)
                
                if result:
                    success += 1
                else:
                    # Check if it was skipped or failed
                    video_id = self._extract_video_id(url)
                    if video_id and video_id in self.downloaded_videos:
                        skipped += 1
                    else:
                        failed += 1
            
            print(f"\n{'='*60}")
            print(f"📊 DOWNLOAD SUMMARY")
            print(f"{'='*60}")
            print(f"Total: {len(videos)}")
            print(f"✅ Downloaded: {success}")
            print(f"⏭️  Skipped (already downloaded): {skipped}")
            print(f"❌ Failed: {failed}")
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
    
    def get_newly_downloaded_files(self) -> List[Dict]:
        """Return list of files downloaded in current session"""
        return self.newly_downloaded
