# -*- coding: utf-8 -*-
"""Regression tests for emoji-to-text conversion in polarity_scores().

Run with: python3 -m unittest test_vaderSentiment -v
"""

import unittest

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class TestEmojiTextConversion(unittest.TestCase):
    """An emoji must not fuse with an adjacent word (issue #163)."""

    @classmethod
    def setUpClass(cls):
        cls.analyzer = SentimentIntensityAnalyzer()

    def test_leading_emoji_does_not_fuse_with_following_word(self):
        # https://github.com/cjhutto/vaderSentiment/issues/163
        # the word after a leading emoji used to be lost from the score:
        # "😍great" scored below plain "great" because "great" was not scored
        baseline = self.analyzer.polarity_scores("great")["compound"]
        spaced = self.analyzer.polarity_scores("😍 great")["compound"]
        leading = self.analyzer.polarity_scores("😍great")["compound"]
        self.assertGreater(leading, baseline)
        self.assertAlmostEqual(leading, spaced, places=4)

    def test_leading_emoji_in_sentence_does_not_fuse(self):
        baseline = self.analyzer.polarity_scores("The great show")["compound"]
        leading = self.analyzer.polarity_scores("The 😍great show")["compound"]
        self.assertGreater(leading, baseline)

    def test_trailing_emoji_still_scores(self):
        # the reverse direction, fixed by PR #57, must keep working
        spaced = self.analyzer.polarity_scores("😍 great")["compound"]
        trailing = self.analyzer.polarity_scores("great😍")["compound"]
        self.assertAlmostEqual(trailing, spaced, places=4)

    def test_leading_emoji_with_negative_word_does_not_fuse(self):
        # a fused "noseterrible" token is not in the lexicon, so the fused
        # form used to score ABOVE (less negative than) the baseline
        baseline = self.analyzer.polarity_scores("The terrible show")["compound"]
        leading = self.analyzer.polarity_scores("The 😤terrible show")["compound"]
        self.assertLessEqual(leading, baseline)


if __name__ == "__main__":
    unittest.main()
