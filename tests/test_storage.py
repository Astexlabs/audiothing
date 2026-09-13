import tempfile
import unittest
from pathlib import Path

from audiothing import get_next_comparison_filenames


class TestComparisonFilenames(unittest.TestCase):
    def test_uses_shared_number_and_skips_existing_audio_variants(self):
        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp)
            first = get_next_comparison_filenames(output_dir)
            self.assertEqual(
                [path.name for path in first],
                [
                    "audio_1_without_classifier.wav",
                    "audio_1_with_classifier.wav",
                    "audio_1_with_phonetic_signs.wav",
                ],
            )

            (output_dir / "audio_1_without_classifier.wav").touch()
            second = get_next_comparison_filenames(output_dir)
            self.assertEqual(
                [path.name for path in second],
                [
                    "audio_2_without_classifier.wav",
                    "audio_2_with_classifier.wav",
                    "audio_2_with_phonetic_signs.wav",
                ],
            )


if __name__ == "__main__":
    unittest.main()
