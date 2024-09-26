import unittest
from src.experiment import Experiment


class TestUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.experiment = Experiment()

    def test_convert_datetime(self):
        test_data = {
            0.0: "neutral",
            .92348: "positive",
            -.92348:  "negative",
            0.043: "neutral",
            -0.043: "neutral",
            .051: "positive",
            -.051: "negative"
        }

        for input, expected_data in test_data.items():
            actual_output = self.experiment._categorize_sentiment_range(input)
            self.assertEqual(actual_output, expected_data)
