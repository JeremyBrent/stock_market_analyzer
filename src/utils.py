import datetime as dt
from src.consts import DATE_FORMAT
from zoneinfo import ZoneInfo
from typing import Union
import torch


class Utils:

    @staticmethod
    def convert_datetime(timestamp: Union[int, str]) -> tuple[str, str, int]:
        """
        This method will extract a human-readable date and determine for which day this news
        might impact stock prices. So for example, if the news comes from after 4pm on Tuesday,
        we could say that it will have an impact on trading Wednesday. This hypothesis needs to
        be more thoroughly tested

        :param timestamp: this could be either a str or int
        :return: tuple(original date string, effective date string, difference between the two)
        """
        if isinstance(timestamp, int):
            # Ensure datetime is in NY Timezone bc that's were the NYSE is located
            original_date = dt.datetime.fromtimestamp(timestamp, tz=ZoneInfo("America/New_York"))
        elif isinstance(timestamp, str):
            # Ensure datetime is in NY Timezone bc that's were the NYSE is located
            original_date = dt.datetime.fromisoformat(timestamp).astimezone(ZoneInfo('America/New_York'))
        else:
            raise Exception(f"What kind of timestamp are you inputtin, "
                            f"this is the value: '{timestamp}', "
                            f"and this is the type: '{type(timestamp)}'")

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

        # Get difference between original date and effective date
        date_diff = abs((original_date - dt_object).days)

        datetime_str = dt_object.strftime(DATE_FORMAT).split(' ')[0]
        original_date_str = original_date.strftime(DATE_FORMAT)

        return original_date_str, datetime_str, date_diff

    @staticmethod
    def get_device() -> torch.device:
        """
        Get the Torch device

        :return: torch.device
        """
        # If on an Apple Silicon Mac (M1/M2), Pytorch supports MPS Backend (Metal Performance Shaders)
        if torch.backends.mps.is_available():
            device = torch.device('mps')

        # If machine has access to trad GPU
        elif torch.cuda.is_available():
            device = torch.device('cuda')

        else:
            device = torch.device('cpu')

        return device
