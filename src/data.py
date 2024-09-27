import pandas as pd
import yfinance as yf
from src.utils import Utils


class Data:

    def __init__(self):
        self.utils = Utils()

    def get_news(self, ticker: str) -> pd.DataFrame:
        """

        :param ticker:
        :return:
        """

        ticker = yf.Ticker(ticker)
        return pd.DataFrame([(x['title'],
                              *self.utils.convert_datetime(x['providerPublishTime']))
                             for x in ticker.news],
                            columns=['news', 'original_date', 'effective_date'])

    def get_price_history(self, ticker: str, period: str = '1mo') -> pd.DataFrame:
        ticker = yf.Ticker(ticker)
        historical_price: pd.DataFrame = ticker.history(period=period).reset_index()

        # remove timestamp
        historical_price['Date'] = (
            historical_price['Date'].astype(str).apply(lambda x: x.split(' ')[0]))

        return historical_price

    def join_price_and_news(self, price: pd.DataFrame, news: pd.DataFrame):
        # TODO: handle multiple peices of news in one day
        df = price.merge(news, left_on='Date', right_on='effective_date')
        return df

    def main(self, ticker: str):
        news_data: pd.DataFrame = self.get_news(ticker=ticker)
        price_data: pd.DataFrame = self.get_price_history(ticker=ticker)

        data = self.join_price_and_news(price=price_data, news=news_data)
        self.feature_extraction(data=data)

    def feature_extraction(self, data):
        # TODO: extract features for model
        pass

