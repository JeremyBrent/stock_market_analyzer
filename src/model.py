import os
import torch
import pickle
import pandas as pd
import transformers
from typing import Union
from src.utils import Utils
from src.consts import PROJECT_ROOT_PATH
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Model:
    """
    This class contains methods for both our Financial Sentiment Analysis (FSA) and our
    Price Prediction (PP) model.

    FSA is using FinBert on MPS, CUDA or CPU depending on machine.

    PP is using RandomForestClassifier
    """

    def __init__(self,
                 fsa_model_name: str = 'ProsusAI/finbert',
                 pp_model_name: str = 'RandomForestClassifier'):
        self.utils: Utils = Utils()
        self.device = self.utils.get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(fsa_model_name)
        self.fsa_model = AutoModelForSequenceClassification.from_pretrained(fsa_model_name).to(self.device)
        self.pp_model = self._pp_read_model(model_name=pp_model_name)

    def _fsa_tokenize(self, text: str) -> transformers.tokenization_utils_base.BatchEncoding:
        """
        Tokenize text

        :param text: str - text to tokenize
        :return: transformers.tokenization_utils_base.BatchEncoding - tokenized result
        """
        tokens = self.tokenizer(text,
                                padding=True,
                                truncation=True,
                                return_tensors='pt',
                                max_length=512).to(self.device)
        return tokens

    def _fsa_get_sentiment(self,
                           tokens: transformers.tokenization_utils_base.BatchEncoding
                           ) -> transformers.modeling_outputs.SequenceClassifierOutput:
        """
        Get sentiment of tokens

        :param tokens: transformers.tokenization_utils_base.BatchEncoding
        :return: transformers.modeling_outputs.SequenceClassifierOutput
        """
        output = self.fsa_model(**tokens)
        return output

    def _fsa_extract_results(self,
                             model_output: transformers.modeling_outputs.SequenceClassifierOutput
                             ) -> tuple[str, float]:
        """
        Extract results from bert model

        :param model_output: transformers.modeling_outputs.SequenceClassifierOutput
        :return: tuple (sentiment category, sentiment score)
        """
        # self.fsa_models[model]['model'].config.id2label.values() ~= ['pos', 'neg', 'neu']
        # torch.nn.functional.softmax(output.logits, dim=-1).tolist()[0] ~= [.083, .034, .883]
        results = {label: value
                   for label, value in
                   zip(self.fsa_model.config.id2label.values(),
                       torch.nn.functional.softmax(model_output.logits, dim=-1).tolist()[0])
                   }

        # Only get the sentiment with the highest score
        sentiment_category: str = max(results, key=results.get)
        sentiment_score = self._compute_score(results=results, category=sentiment_category)
        return sentiment_category, sentiment_score

    @staticmethod
    def _compute_score(results: dict[str, float], category: str) -> float:
        """
        Give the results dict and the largest category, return the final value. If the top category
        is neutral, we want to return a value between -.05 and .05. Discussed here, note that
        this is a different model, but we are exptrapolating for the time being.
        https://github.com/cjhutto/vaderSentiment?tab=readme-ov-file#about-the-scoring

        :param results: dict {'positive': .93494, 'neutral': .34592, 'negative': .349524}
        :param category: top category
        :return: sentiment score
        """
        if category == 'neutral':
            # return score between -.05 and .05 if top category is neutral.
            # TODO: is this the right approach for BERT models?
            neutral_score = (results[category] * 0.1) - 0.05
            return neutral_score + .1 if neutral_score < 0 else neutral_score

        return results[category] if category == 'positive' else results[category] * -1

    def fsa_predict(self, text: str) -> tuple[str, float]:
        """
        Predict sentiment

        :param text: text to predict on
        :return: tuple of (sentiment category, sentiment score)
        """
        tokens = self._fsa_tokenize(text)
        output = self._fsa_get_sentiment(tokens)
        results = self._fsa_extract_results(output)
        return results

    @staticmethod
    def pp_train(model,
                 features: pd.DataFrame,
                 targets: pd.Series):
        """
        Train a model

        :param model: un-trained model
        :param features: features to train on
        :param targets: targets to train on
        :return: trained model
        """
        model.fit(features, targets)
        return model

    def pp_predict(self,
                   features: pd.DataFrame,
                   model=None) -> pd.Series:
        """
        Predict price movement

        :param model: model object
        :param features: features data
        :return: predictions
        """
        assert model or self.pp_model, "We need a model!"

        model = model if model else self.pp_model

        predictions = model.predict(features)
        return predictions

    @staticmethod
    def pp_evaluate_model(y_test: pd.Series,
                          y_pred: pd.Series) -> float:
        """
        Compare y test to y pred

        :param y_test: Target of test data
        :param y_pred: Predicted targets of x_test
        :return: accuracy - float
        """
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        return accuracy

    @staticmethod
    def pp_extract_features(stock_data: pd.DataFrame,
                            features=None) -> tuple[pd.DataFrame, pd.Series]:
        """
        Extract features from data

        :param stock_data: pd.DataFrame
        :param features: optional, if None, will default to features listed below
        :return: tuple(features, targets)
        """

        if not features:
            features = ['sma_50',
                        'sma_200',
                        'rsi',
                        'obv',
                        'BBL_20_2.0',
                        'BBM_20_2.0',
                        'BBU_20_2.0',
                        'norm_sentiment_score']

        # Define X (input features) and y (target)
        x = stock_data[features]
        y = stock_data['target']
        return x, y

    @staticmethod
    def pp_save_model(model, model_name: str) -> str:
        """
        Save a model to pickle file

        :param model: Fitted model
        :param model_name: Name of model
        :return: path to saved file
        """
        model_file = os.path.join(PROJECT_ROOT_PATH, 'models', f'{model_name}.pkl')
        with open(model_file, "wb") as f:
            pickle.dump(model, f, protocol=5)

        return model_file

    @staticmethod
    def _pp_read_model(model_name: str):
        """
        Read a pickled fitted model from disk

        :param model_name: Name of model
        :return: fitted model
        """
        model_file = os.path.join(PROJECT_ROOT_PATH, 'models', f'{model_name}.pkl')
        if not os.path.exists(model_file):
            print(f"{model_file} does not exist, returning None")
            return None

        with open(model_file, "rb") as f:
            model = pickle.load(f)
        return model

    def pp_train_test_split(self,
                            stock_data: pd.DataFrame,
                            features: list = None) -> dict[str, Union[pd.DataFrame, pd.Series]]:
        """
        Given data, get the features and split into train and test data

        :param stock_data: pd.Dataframe of stock data
        :param features: features to use
        :return: dict[str, Union[pd.DataFrame, pd.Series]]
        """
        x, y = self.pp_extract_features(stock_data=stock_data, features=features)

        # Split the data into training and testing sets (80% training, 20% testing)
        x_train, x_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            test_size=0.2,
                                                            shuffle=False)

        return {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}

