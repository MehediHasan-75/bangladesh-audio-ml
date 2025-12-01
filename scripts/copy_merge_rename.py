#!/usr/bin/env python3
"""
Copy and rename files from one directory to another
NEW files first, then existing
Example: Copy from processed1/cng to processed/cng_auto
"""

from pathlib import Path
import shutil


def copy_and_rename_category_new_first(
    source_category: str,
    target_category: str,
    source_base_dir: str = "ml_data/processed1",
    target_base_dir: str = "ml_data/processed"
):
    """
    Copy files from source directory to target directory with NEW files placed FIRST
    
    Strategy:
    1. Find all existing files in target
    2. Shift existing files to higher numbers
    3. Place new copied files at the beginning (starting from 0)
    
    Args:
        source_category: Source category name (e.g., 'cng')
        target_category: Target category name (e.g., 'cng_auto')
        source_base_dir: Source base directory (default: ml_data/processed1)
        target_base_dir: Target base directory (default: ml_data/processed)
    """
    source_path = Path(source_base_dir) / source_category
    target_path = Path(target_base_dir) / target_category
    
    # Check if source exists
    if not source_path.exists():
        print(f"❌ Source category not found: {source_path}")
        return
    
    # Create target directory
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Get all source files to copy
    source_files = sorted(source_path.glob(f"{source_category}_*.wav"))
    
    if not source_files:
        print(f"⚠️  No files found in {source_path}")
        return
    
    # Get existing files in target
    existing_files = sorted(target_path.glob(f"{target_category}_*.wav"))
    
    print(f"\n📁 Copy from: {source_path}")
    print(f"📁 Copy to:   {target_path}")
    print("="*60)
    print(f"New files to copy:        {len(source_files)}")
    print(f"Existing files in target: {len(existing_files)}")
    
    # Step 1: Shift existing files to higher numbers
    if existing_files:
        print(f"\n🔄 STEP 1: Shifting existing files to higher numbers...")
        
        # Get the numbers of existing files
        existing_numbers = [
            int(f.stem.split('_')[-1]) 
            for f in existing_files 
            if f.stem.split('_')[-1].isdigit()
        ]
        existing_numbers.sort()
        
        # Rename existing files to make room (start from len(source_files))
        new_start = len(source_files)
        
        for old_idx, old_file in enumerate(reversed(existing_files)):
            # Process in reverse to avoid naming conflicts
            old_num = int(old_file.stem.split('_')[-1])
            new_num = new_start + old_idx
            
            old_name = old_file.name
            new_name = f"{target_category}_{new_num:04d}.wav"
            new_path = target_path / new_name
            
            # Rename the file
            old_file.rename(new_path)
            
            if old_idx % 10 == 0 or old_idx == len(existing_files) - 1:
                print(f"  ✅ {old_name} → {new_name}")
        
        print(f"  ✓ Shifted {len(existing_files)} existing files")
    
    # Step 2: Copy new files starting from 0
    print(f"\n📥 STEP 2: Copying new files starting from 0000...")
    
    copied = 0
    for idx, source_file in enumerate(source_files):
        new_num = idx
        new_name = f"{target_category}_{new_num:04d}.wav"
        dest = target_path / new_name
        
        try:
            # Copy file
            shutil.copy2(source_file, dest)
            copied += 1
            
            # Print progress every 10 files or first/last
            if idx == 0 or idx == len(source_files) - 1 or (idx + 1) % 10 == 0:
                print(f"  ✅ [{idx+1}/{len(source_files)}] {source_file.name} → {new_name}")
                
        except Exception as e:
            print(f"  ❌ [{idx+1}/{len(source_files)}] Error copying {source_file.name}: {e}")
    
    print(f"\n✓ Done!")
    print(f"  Copied files:   {target_category}_0000.wav to {target_category}_{len(source_files)-1:04d}.wav")
    if existing_files:
        print(f"  Shifted files:  {target_category}_{len(source_files):04d}.wav to {target_category}_{len(source_files)+len(existing_files)-1:04d}.wav")
    print(f"  Total files:    {len(source_files) + len(existing_files)}")


def main():
    print("="*60)
    print("COPY AND RENAME - NEW FILES FIRST")
    print("="*60)
    
    # Copy cng from processed1 to cng_auto in processed (new files first)
    copy_and_rename_category_new_first(
        source_category="construction",
        target_category="construction_noise",
        source_base_dir="ml_data/processed1",  # ✅ Fixed
        target_base_dir="ml_data/processed"    # ✅ Fixed
    )
    
    print("\n✅ All done!")


if __name__ == "__main__":
    main()
