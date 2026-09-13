import unittest

from audiothing import pronounce_initialisms, pronounce_initialisms_phonetic


class TestPronounceInitialisms(unittest.TestCase):
    def test_expands_two_and_three_letter_groups(self):
        self.assertEqual(
            pronounce_initialisms("AI and API improve UX."),
            "ay eye and ay pee eye improve you ex.",
        )

    def test_expands_dotted_initialisms(self):
        self.assertEqual(
            pronounce_initialisms("A.I. and U.S.A. technology"),
            "ay eye and you ess ay technology",
        )

    def test_expands_with_phonetic_signs(self):
        self.assertEqual(
            pronounce_initialisms_phonetic("AI, API, and U.S.A."),
            "eɪ aɪ, eɪ piː aɪ, and juː ɛs eɪ",
        )

    def test_preserves_common_uppercase_words_and_single_letters(self):
        self.assertEqual(
            pronounce_initialisms("A I THE US OK"),
            "A I THE US OK",
        )

    def test_known_technical_initialism_overrides_ambiguous_word(self):
        self.assertEqual(
            pronounce_initialisms("The IT team uses AI."),
            "The eye tee team uses ay eye.",
        )

    def test_preserves_mixed_case_and_longer_groups(self):
        self.assertEqual(
            pronounce_initialisms("iPhone AIModel NASA"),
            "iPhone AIModel NASA",
        )

    def test_does_not_expand_possessive_or_longer_word_prefixes(self):
        self.assertEqual(
            pronounce_initialisms("AI's APIs"),
            "AI's APIs",
        )


if __name__ == "__main__":
    unittest.main()
