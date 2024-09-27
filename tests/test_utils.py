import unittest
from src.utils import Utils


class TestUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.utils = Utils()

    def test_convert_datetime(self):
        test_data = {
            1727104000: ('2024-09-23 11:06:40', '2024-09-23', 0),
            1727130000: ('2024-09-23 18:20:00', '2024-09-24', 1),  # after 4pm, move to next day
            1727490000:  ('2024-09-27 22:20:00', '2024-09-30', 3),  # Friday after 4pm move to Monday
            1727500000: ('2024-09-28 01:06:40', '2024-09-30', 2),  # Saturday to Monday
            1727590000: ('2024-09-29 02:06:40', '2024-09-30', 1),  # Sunday move to Monday
        }

        for input, expected_data in test_data.items():
            actual_output = self.utils.convert_datetime(input)
            self.assertEqual(actual_output, expected_data)
