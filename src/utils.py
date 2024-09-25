import datetime as dt
from src.consts import DATE_FORMAT


class Utils:

    @staticmethod
    def convert_datetime(timestamp: int) -> tuple[str, str]:
        """
        This method will extract a human-readable date and determine for which day this news
        might impact stock prices. So for example, if the news comes from after 4pm on Tuesday,
        we might say that it will have an impact on trading Wednesday.

        :param timestamp:
        :return:
        """
        # Convert to datetime object
        original_date = dt.datetime.fromtimestamp(timestamp)
        dt_object = original_date

        # Define 4 PM as the time to compare against
        four_pm = original_date.replace(hour=16, minute=0, second=0, microsecond=0)

        # If the input time is after 4 PM, move to the next day
        if original_date > four_pm:
            dt_object = original_date + dt.timedelta(days=1)

            # Reset time, we are throwing time out
            dt_object = dt_object.replace(hour=0, minute=0, second=0, microsecond=0)

        # Ensure the date is a weekday (Monday to Friday)
        if dt_object.weekday() == 5:  # Saturday
            dt_object += dt.timedelta(days=2)  # Move to Monday
        elif dt_object.weekday() == 6:  # Sunday
            dt_object += dt.timedelta(days=1)  # Move to Monday
        elif dt_object.weekday() == 4 and dt_object > four_pm:  # Friday after 4 PM
            dt_object += dt.timedelta(days=3)  # Move to Monday

        datetime_str = dt_object.strftime(DATE_FORMAT).split(' ')[0]
        original_date_str = original_date.strftime(DATE_FORMAT)

        return original_date_str, datetime_str
