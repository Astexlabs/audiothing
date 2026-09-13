"""Audio file storage and naming utilities."""
import re
from pathlib import Path

def get_next_audio_filename(output_dir: Path, base_name: str = "audio", ext: str = ".wav") -> Path:
    """Return next available audio filename in directory."""
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


def get_next_comparison_filenames(
    output_dir: Path,
    base_name: str = "audio",
    ext: str = ".wav",
) -> tuple[Path, Path, Path]:
    """Return non-overwriting paths for the three-way pronunciation test."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(
        rf"^{re.escape(base_name)}_?(\d+)(?:_[^.]+)?{re.escape(ext)}$"
    )
    existing_nums = []

    for item in output_dir.iterdir():
        if item.is_file():
            match = pattern.match(item.name)
            if match:
                existing_nums.append(int(match.group(1)))
            elif item.name == f"{base_name}{ext}":
                existing_nums.append(0)

    next_num = max(existing_nums, default=0) + 1
    return (
        output_dir / f"{base_name}_{next_num}_without_classifier{ext}",
        output_dir / f"{base_name}_{next_num}_with_classifier{ext}",
        output_dir / f"{base_name}_{next_num}_with_phonetic_signs{ext}",
    )
