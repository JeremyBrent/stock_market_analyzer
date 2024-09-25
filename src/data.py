import yfinance as yf
from src.utils import Utils


class Data:

    def __init__(self):
        self.utils = Utils()

    def get_news(self, ticker: str) -> list:
        """

        :param ticker:
        :return:
        """

        ticker = yf.Ticker(ticker)
        return [(x['title'],
                self.utils.convert_datetime(x['providerPublishTime']))
                for x in ticker.news]
