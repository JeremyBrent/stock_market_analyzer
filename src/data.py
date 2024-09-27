import os

import pandas as pd
import yfinance as yf
from datetime import datetime as dt
from src.consts import PROJECT_ROOT_PATH
from src.utils import Utils
import requests
import pandas_ta as ta
from src.model import Model


class Data:

    def __init__(self):
        self.utils: Utils = Utils()
        self.model: Model = Model()

    def _get_news(self, ticker: str, mode: str = 'train') -> pd.DataFrame:
        """
        Get news data. If mode is train, we will create our training data. If mode is experiment,
        we will attempt to read it, otherwise, we will get news from yahoo finance.

        :param ticker: ticker to get news for
        :return: dataframe of news data
        """
        data_path = os.path.join(PROJECT_ROOT_PATH, 'data', f'{ticker}_news_training_data.pkl')

        # Create training data
        if mode == 'train':

            offset = 0
            limit = 1000
            api_token = 'demo'  # TODO: we'd need to get an api for this API
            fmt = 'json'
            file_paths = []
            data = [{}]
            # This API works with an offset, so we will extract financial news until the offset
            # returns no data or the date of the first news article is before 2020
            # (this is generally arbitrary, and we would need to
            # determine if we needed more or less data)
            while data or data[0].get('date') < '2020-01-01':

                # TODO: remove early breakage -- added for testing purposes
                if offset == 10:
                    break

                # Create a path for the offset, we will aggregate when done.
                offset_path = os.path.join(PROJECT_ROOT_PATH,
                                           'data',
                                           f'{offset}_{ticker}_news_training_data.pkl')

                if offset % 10 == 0:
                    print(f'INFO: {dt.now()}: We are on page {offset}.')

                url = (f'https://eodhd.com/api/news?s={ticker}&offset={offset}'
                       f'&limit={limit}&api_token={api_token}&fmt={fmt}')
                data = requests.get(url).json()

                # Format the data
                tmp = pd.DataFrame([(x['title'],
                                     *self.utils.convert_datetime(x['date']),
                                     *self.model.fsa_predict(x['title'])
                                     )
                                   for x in data],
                                   columns=['news',
                                            'original_date',
                                            'effective_date',
                                            'date_diff',
                                            'sentiment',
                                            'sentiment_score'])

                tmp = self._news_feature_extraction(tmp)

                # Increase the offset
                offset += 1

                # Write the data to the offset path, and add it to the total paths
                tmp.to_pickle(offset_path)
                file_paths.append(offset_path)

            # Aggregate the offset files into one large file
            df = pd.DataFrame()
            for path in file_paths:
                tmp = pd.read_pickle(path)
                df = pd.concat([df, tmp])

            # Write the large file to disk
            df.to_pickle(data_path)

            # Delete the individual files
            [os.remove(path) for path in file_paths]

            return df

        if mode == 'experiment':
            # Check to see if we have already created training data for this ticker
            if os.path.exists(data_path):
                return pd.read_pickle(data_path)
            raise Exception(f"{data_path} does not exist. You need to run 'train' mode to create "
                            f"the training data file for ticker {ticker}")

        else:
            ticker = yf.Ticker(ticker)
            news_df = pd.DataFrame([(x['title'],
                                     *self.utils.convert_datetime(x['providerPublishTime']),
                                     *self.model.fsa_predict(x['title']))
                                    for x in ticker.news],
                                   columns=['news',
                                            'original_date',
                                            'effective_date',
                                            'date_diff',
                                            'sentiment',
                                            'sentiment_score'])
            news_df = self._news_feature_extraction(news_df)
            return news_df

    @staticmethod
    def _news_feature_extraction(news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for news data

        :param news_df: dataframe of news
        :return: dataframe with features of news
        """
        # This will reduce the impact of sentiment of news that happened
        # further from the effective date
        news_df['norm_sentiment_score'] = news_df['sentiment_score'] / (news_df['date_diff'] + 1)
        return news_df

    @staticmethod
    def _price_feature_extraction(price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract various features from our price data

        :param price_df: pd.DataFrame
        :return: pd.DataFrame
        """
        # 50 and 200 day Simple Moving Average
        price_df['sma_50'] = price_df['Close'].rolling(window=50).mean()
        price_df['sma_200'] = price_df['Close'].rolling(window=200).mean()

        # Relative Strength Indicator
        price_df['rsi'] = ta.rsi(price_df['Close'], length=14)

        # Bollinger Bands, we are indexing here bc we only need bottom, middle and upper
        bollinger_bands = ta.bbands(price_df['Close'], length=20).iloc[:, :3]
        price_df = pd.concat([price_df, bollinger_bands], axis=1)

        # On Balance Volume
        price_df['obv'] = ta.obv(price_df['Close'], price_df['Volume'])

        # Golden Cross - When 50 day Moving Average crosses 200 day moving average
        price_df["gc"] = price_df['sma_50'] > price_df['sma_200']

        # Momentum
        price_df['momentum'] = ta.mom(price_df['Close'], length=10)

        return price_df

    def _get_price_history(self, ticker: str, period: str = '1y') -> pd.DataFrame:
        """
        Get price history for a ticker. It is critical that the period is no smaller than a
        year because at the end of this method we drop rows will null values because of features
        like 200 day SMA.

        :param ticker: ticker to get price history for
        :param period: valid periods can be found here:
            https://github.com/ranaroussi/yfinance/wiki/Ticker#parameters
        :return: dataframe of price data
        """
        # Get historical price data
        ticker = yf.Ticker(ticker)
        historical_price: pd.DataFrame = ticker.history(period=period).reset_index()

        # remove timestamp
        historical_price['date'] = (
            historical_price['Date'].astype(str).apply(lambda x: x.split(' ')[0]))

        historical_price['target'] = historical_price.apply(
            lambda x: 1 if x['Close'] > x['Open'] else 0, axis=1)

        historical_price = self._price_feature_extraction(historical_price)

        # Remove rows with null data, specifically something like a 200-day SMA
        # will have 200 nulls values
        historical_price.dropna(how='any', inplace=True)

        return historical_price

    @staticmethod
    def _join_price_and_news(price: pd.DataFrame, news: pd.DataFrame):
        """
        Given price and news data, merge them

        :param price: pd.DataFrame
        :param news: pd.DataFrame
        :return: pd.DataFrame
        """
        df = price.merge(news, left_on='date', right_on='effective_date')
        return df

    @staticmethod
    def _compress_news_data(news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Take all records of news and get the average norm score for a given effective date

        :param news_df: pd.DataFrame
        :return: pd.DataFrame
        """
        return news_df.groupby('effective_date')['norm_sentiment_score'].mean().reset_index()

    def main(self, ticker: str, mode: str, period: str) -> pd.DataFrame:
        """
        Get news data, price data and merge them into one dataframe

        :param ticker: ticker to query
        :param mode: mode for getting data, valid modes are ['train', 'experiment', 'infer']
        :param period: valid periods can be found here:
            https://github.com/ranaroussi/yfinance/wiki/Ticker#parameters
        :return: pd.Dataframe of news and price data
        """
        assert ticker == 'AAPL', \
            (f"Ticker needs to be AAPL, you selected {ticker}, because are using EODHD demo api "
             f"key and that will only pull Apple news.")

        assert mode in {'train', 'experiment', 'infer'}, \
            f"Mode should be in ['train', 'experiment', 'infer'] but you selected {mode}"

        assert period not in {'1d', '5d', '1mo', '3mo', '6mo'}, \
            (f"Period cannot be '1d', '5d', '1mo', '3mo', or '6mo' due to null values when "
             f"calculating metrics like 200 day simple moving average")

        # Get news data
        news_data: pd.DataFrame = self._get_news(ticker=ticker, mode=mode)
        news_data: pd.DataFrame = self._compress_news_data(news_data)

        # Get price data
        price_data: pd.DataFrame = self._get_price_history(ticker=ticker, period=period)

        # Combine news and price data
        data = self._join_price_and_news(price=price_data, news=news_data)

        return data
