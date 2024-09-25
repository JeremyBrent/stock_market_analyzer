import datetime
from src.consts import DATE_FORMAT


class Utils:

    @staticmethod
    def convert_datetime(timestamp: int) -> str:
        """
        Given an int timestamp, convert to
        :param timestamp:
        :return:
        """
        # Convert to datetime object
        dt_object = datetime.datetime.fromtimestamp(timestamp)
        # Convert to string
        datetime_str = dt_object.strftime(DATE_FORMAT)

        return datetime_str
