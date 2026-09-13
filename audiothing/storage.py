"""
Audio file storage and naming management.
"""

import re
from pathlib import Path


def get_next_audio_filename(output_dir: Path, base_name: str = "audio", ext: str = ".wav") -> Path:
    """
    Finds the next incremental filename in output_dir (e.g. audio_1.wav, audio_2.wav, ...).
    Always performs a +1 so existing audio files are never overwritten.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(base_name)}_?(\d+){re.escape(ext)}$")
    existing_nums = []

    for item in output_dir.iterdir():
        if item.is_file():
            match = pattern.match(item.name)
            if match:
                existing_nums.append(int(match.group(1)))
            elif item.name == f"{base_name}{ext}":
                existing_nums.append(0)

    next_num = max(existing_nums, default=0) + 1
    return output_dir / f"{base_name}_{next_num}{ext}"
