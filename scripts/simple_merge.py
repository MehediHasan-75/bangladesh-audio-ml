#!/usr/bin/env python3
"""Simple script to merge processed1 into processed"""

from pathlib import Path
import shutil

def merge_category(category_name, main_dir="ml_data/processed", source_dir="ml_data/processed1"):
    """Merge one category"""
    main_path = Path(main_dir) / category_name
    source_path = Path(source_dir) / category_name
    
    if not source_path.exists():
        print(f"❌ {source_path} not found")
        return
    
    # Get highest number in main
    main_path.mkdir(parents=True, exist_ok=True)
    existing = list(main_path.glob(f"{category_name}_*.wav"))
    
    if existing:
        numbers = [int(f.stem.split('_')[-1]) for f in existing if f.stem.split('_')[-1].isdigit()]
        next_num = max(numbers) + 1 if numbers else 0
    else:
        next_num = 0
    
    print(f"\n📁 {category_name}: starting from {next_num:04d}")
    
    # Copy files
    source_files = sorted(source_path.glob(f"{category_name}_*.wav"))
    
    for idx, source_file in enumerate(source_files):
        new_num = next_num + idx
        new_name = f"{category_name}_{new_num:04d}.wav"
        dest = main_path / new_name
        
        shutil.copy2(source_file, dest)
        
        if idx % 10 == 0 or idx == len(source_files) - 1:
            print(f"  ✅ [{idx+1}/{len(source_files)}] {new_name}")
    
    print(f"  ✓ Done: {len(source_files)} files")

# Main execution
if __name__ == "__main__":
    print("="*60)
    print("MERGING PROCESSED AUDIO")
    print("="*60)
    
    categories = ["train", "bike", "truck", "bus"]  # Add your categories
    
    for cat in categories:
        merge_category(cat)
    
    print("\n✅ All done!")
