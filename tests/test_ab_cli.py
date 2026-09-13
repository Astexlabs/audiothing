import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import main


class TestAbCli(unittest.TestCase):
    def test_ab_synthesizes_baseline_and_classified_text(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path = root / "input.txt"
            input_path.write_text("AI uses API.", encoding="utf-8")
            args = SimpleNamespace(
                api_key="test-key",
                input=input_path,
                output_dir=root / "output",
                no_sanitize=False,
                pronounce_initialisms=False,
                phonetic_initialisms=False,
                a_b=True,
                server="test-server",
                function_id="test-function",
                voice="test-voice",
                language_code="en-US",
                sample_rate=22050,
                sentence_pause=0.25,
                paragraph_pause=0.5,
                max_chars=220,
                request_delay=0,
                text_only=False,
            )

            with patch.object(main, "parse_args", return_value=args), patch.object(
                main, "synthesize_speech"
            ) as synthesize:
                main.main()

            self.assertEqual(synthesize.call_count, 3)
            calls = synthesize.call_args_list
            self.assertEqual(
                calls[0].kwargs["output_path"].name,
                "audio_1_without_classifier.wav",
            )
            self.assertEqual(
                calls[1].kwargs["output_path"].name,
                "audio_1_with_classifier.wav",
            )
            self.assertEqual(
                calls[2].kwargs["output_path"].name,
                "audio_1_with_phonetic_signs.wav",
            )
            self.assertEqual(calls[0].kwargs["speech_units"][0][0], "AI uses API.")
            self.assertEqual(
                calls[1].kwargs["speech_units"][0][0], "ay eye uses ay pee eye."
            )
            self.assertEqual(
                calls[2].kwargs["speech_units"][0][0], "eɪ aɪ uses eɪ piː aɪ."
            )


if __name__ == "__main__":
    unittest.main()
