import os
import re

def renumber_wavs(directory, prefix):
    """
    Renames files with pattern <prefix>_*.wav in given directory
    to a compact sequence starting from <prefix>_0000.wav.

    Example:
      (before) bus_0002.wav, bus_0010.wav
      (after)  bus_0000.wav, bus_0001.wav

    Args:
        directory (str): Path to the target directory.
        prefix (str): Filename prefix (e.g., 'bus')
    """
    # Pattern to match: <prefix>_NNNN.wav
    pattern = rf'^{re.escape(prefix)}_(\d+)\.wav$'

    files = []
    for fname in os.listdir(directory):
        m = re.match(pattern, fname)
        if m:
            files.append((int(m.group(1)), fname))

    # Sort by original integer number
    files.sort()
    # Prepare new names (avoid collision by doing two-step renaming)
    temp_names = []
    for i, (old_num, old_name) in enumerate(files):
        temp_name = f"{prefix}_tmp_{i:04d}.wav"
        os.rename(os.path.join(directory, old_name), os.path.join(directory, temp_name))
        temp_names.append(temp_name)

    # Final pass: rename temp -> serial order
    for i, temp_name in enumerate(temp_names):
        new_name = f"{prefix}_{i:04d}.wav"
        os.rename(os.path.join(directory, temp_name), os.path.join(directory, new_name))

    print(f"Renamed {len(files)} files to serial order.")

# Usage example:
folder_path = "/Users/mehedihasan/Projects/bangladeshi-audio-ml/ml_data/processed/truck"
prefix = "truck"
renumber_wavs(folder_path, prefix)
