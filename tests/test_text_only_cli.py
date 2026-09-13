import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import main


class TestTextOnlyCli(unittest.TestCase):
    def test_text_only_prints_classified_text_without_api_or_audio(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path = root / "input.txt"
            output_dir = root / "output"
            input_path.write_text("AI uses API.", encoding="utf-8")
            args = SimpleNamespace(
                api_key="",
                input=input_path,
                output_dir=output_dir,
                no_sanitize=False,
                pronounce_initialisms=True,
                phonetic_initialisms=False,
                a_b=False,
                text_only=True,
                server="test-server",
                function_id="test-function",
                voice="test-voice",
                language_code="en-US",
                sample_rate=22050,
                sentence_pause=0.25,
                paragraph_pause=0.5,
                max_chars=220,
                request_delay=0,
            )

            stdout = io.StringIO()
            with patch.object(main, "parse_args", return_value=args), patch.object(
                main, "synthesize_speech"
            ) as synthesize, contextlib.redirect_stdout(stdout):
                main.main()

            self.assertEqual(stdout.getvalue(), "ay eye uses ay pee eye.\n")
            synthesize.assert_not_called()
            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
