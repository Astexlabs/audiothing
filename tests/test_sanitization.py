import unittest
from audiothing import sanitize_text


class TestSanitizeText(unittest.TestCase):
    def test_double_quotes_removed(self):
        self.assertEqual(sanitize_text('"Hello world"'), "Hello world")
        self.assertEqual(sanitize_text("“Curly double quotes”"), "Curly double quotes")
        self.assertEqual(sanitize_text('The "rogue" model'), "The rogue model")

    def test_backticks_removed(self):
        self.assertEqual(sanitize_text("`code_sample`"), "code sample")
        self.assertEqual(
            sanitize_text("Use `python main.py` to run"), "Use python main.py to run"
        )

    def test_single_quotes_and_contractions(self):
        # Contractions and possessives preserved
        self.assertEqual(
            sanitize_text("don't it's we'll let's O'Connor"),
            "don't it's we'll let's O'Connor",
        )
        self.assertEqual(
            sanitize_text("don’t it’s we’ll let’s"), "don't it's we'll let's"
        )
        # Quotation single quotes removed
        self.assertEqual(sanitize_text("'quoted text'"), "quoted text")
        self.assertEqual(sanitize_text("‘single curly quotes’"), "single curly quotes")

    def test_hyphens_dashes_and_ranges(self):
        # Number ranges turned into "to"
        self.assertEqual(sanitize_text("350-450K"), "350 to 450K")
        self.assertEqual(sanitize_text("10 - 20 TBs"), "10 to 20 TBs")
        # Em-dashes and en-dashes turned into pause commas
        self.assertEqual(sanitize_text("start — middle — end"), "start, middle, end")
        self.assertEqual(sanitize_text("start -- middle -- end"), "start, middle, end")
        self.assertEqual(sanitize_text("start - middle - end"), "start, middle, end")
        # Hyphenated words separated cleanly
        self.assertEqual(
            sanitize_text("non-trivial and anti-virus"), "non trivial and anti virus"
        )
        self.assertEqual(sanitize_text("rack-scale server"), "rack scale server")

    def test_underscores_removed(self):
        self.assertEqual(sanitize_text("snake_case_variable"), "snake case variable")
        self.assertEqual(sanitize_text("__dunder__"), "dunder")
        self.assertEqual(
            sanitize_text("word_with_multiple___underscores"),
            "word with multiple underscores",
        )

    def test_markdown_and_formatting(self):
        self.assertEqual(sanitize_text("# Heading\nBody text"), "Heading\nBody text")
        self.assertEqual(
            sanitize_text("This *may* be **bold** and ***italic***"),
            "This may be bold and italic",
        )
        self.assertEqual(
            sanitize_text("1) First item\n2) Second item"),
            "1. First item\n2. Second item",
        )
        self.assertEqual(
            sanitize_text("- Bullet one\n- Bullet two"), "Bullet one\nBullet two"
        )
        self.assertEqual(
            sanitize_text("hardware/software/capabilities"),
            "hardware, software, capabilities",
        )
        self.assertEqual(
            sanitize_text("model (replaced by rogue model) done."),
            "model, replaced by rogue model, done.",
        )

    def test_symbols_expanded(self):
        self.assertEqual(sanitize_text("Tom & Jerry"), "Tom and Jerry")
        self.assertEqual(sanitize_text("100%"), "100 percent")
        self.assertEqual(sanitize_text("A + B = C"), "A plus B equals C")
        self.assertEqual(sanitize_text("~50 units"), "about 50 units")


if __name__ == "__main__":
    unittest.main()
