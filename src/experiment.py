import os
import torch
import nltk
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.data import Data
from src.model import Model
from src.utils import Utils
from textblob import TextBlob
from multiprocessing import Pool
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from datetime import datetime as dt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.consts import CPU_COUNT, PARALLEL_CHUNK_SIZE, PROJECT_ROOT_PATH, RANDOM_STATE


class Experiment:

    def __init__(self):

        self.time = dt.now().strftime('%Y-%m-%d %H:%M:%S')
        self.model: Model = Model()
        self.data: Data = Data()
        self.utils: Utils = Utils()
        self.device: torch.device = self.utils.get_device()

        # Storing models here so we do not have to instantiate them every time we maka prediction
        self.fsa_models = {
            'finbert': {
                'method': self._fsa_bert_model,
                'tokenizer': AutoTokenizer.from_pretrained('ProsusAI/finbert'),
                'model': AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert').to(self.device)
            },
            'roberta': {
                'method': self._fsa_bert_model,
                'tokenizer': AutoTokenizer.from_pretrained(
                    'cardiffnlp/twitter-roberta-base-sentiment-latest'),
                'model': AutoModelForSequenceClassification.from_pretrained(
                    'cardiffnlp/twitter-roberta-base-sentiment-latest').to(self.device)
            },
            'fin_roberta': {
                'method': self._fsa_bert_model,
                'tokenizer': AutoTokenizer.from_pretrained(
                    'soleimanian/financial-roberta-large-sentiment'),
                'model': AutoModelForSequenceClassification.from_pretrained(
                    'soleimanian/financial-roberta-large-sentiment').to(self.device)
            },
            'textblob': {
                'method': self._fsa_text_blob
            },
            'nltk': {
                'method': self._fsa_nltk
            }
        }

        # Price Prediction Models
        self.pp_models = {
            'LogisitcRegression': {
                'model': LogisticRegression(random_state=RANDOM_STATE),
                'params': {
                    "C": np.logspace(-3, 3, 7),
                    "penalty": ["l1", "l2"]
                },
            },
            'RandomForestClassifier': {
                'model': RandomForestClassifier(random_state=RANDOM_STATE),
                'params': {
                    # TODO: add more comprehensive grid search params
                    'n_estimators': [50, 100],
                    'max_depth': [None, 10, 20],
                    'max_features': [0, 5],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'bootstrap': [True, False]
                }
            },
            'GradientBoostingClassifier': {
                'model': GradientBoostingClassifier(random_state=RANDOM_STATE),
                'params': {
                    # TODO: add more comprehensive grid search params
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'subsample': [0.8, 1.0]
                }
            },
            # TODO: these models were killing our process with Process finished with exit code
            #  139 (interrupted by signal 11: SIGSEGV). Investigate this.
            # 'LightGradientBoostingMachine': {
            #     'model': LGBMClassifier(random_state=RANDOM_STATE),
            #     'params': {
            #         # TODO: add more comprehensive grid search params
            #         'n_estimators': [50],
            #         'learning_rate': [0.01],
            #         'max_depth': [3, 5],
            #         'num_leaves': [31],
            #         'subsample': [0.8],
            #         'colsample_bytree': [0.8]
            #     }
            # },
            # 'XGBoostClassifier': {
            #     'model': XGBClassifier(random_state=RANDOM_STATE),
            #     'params': {
            #         # TODO: add more comprehensive grid search params
            #         'n_estimators': [50, 100],
            #         'learning_rate': [0.01, 0.1],
            #         'max_depth': [3, 5],
            #         'subsample': [0.8, 1.0],
            #         'colsample_bytree': [0.8, 1.0],
            #         'gamma': [0, 0.1]
            #     }
            # }
        }
        self.pp_experiment_columns = ['date', 'model', 'train_accuracy', 'test_accuracy', 'params']
        self.fsa_ground_truth_data = self._get_fsa_ground_truth_data()

        # Download vader lexicon if we do not have it
        try:
            nltk.data.find('vader_lexicon')
            print('Finding vader')
        except LookupError:
            nltk.download('vader_lexicon')
            print('Downloading vader')

    @staticmethod
    def _get_fsa_ground_truth_data() -> list[dict[str, str]]:
        """
        Read the sentiment analysis ground truth data

        :return: list of dicts
        """
        df: pd.DataFrame = pd.read_csv(os.path.join(PROJECT_ROOT_PATH,
                                                    "data",
                                                    "fsa_ground_truth.csv"))

        return [{"text": x[0], 'sentiment': x[1]} for x in df.values.tolist()]

    @staticmethod
    def _categorize_sentiment_range(value: float) -> str:
        """
        Given a sentiment float, return the sentiment.

        Threshold were determined here:
            https://github.com/cjhutto/vaderSentiment?tab=readme-ov-file#about-the-scoring

        :param value: float of sentiment
        :return: categorical sentiment
        """
        if value >= 0.05:
            return 'positive'
        elif -.05 < value < .05:
            return 'neutral'
        else:
            return 'negative'

    def _fsa_bert_model(self, text: str, model_name: str) -> str:
        """
        Run bert model. This class is generalized to work with various HuggingFace bert models

        :param text: text to get sentiment from
        :param model_name: bert model name
        :return: sentiment
        """
        tokens = self.fsa_models[model_name]['tokenizer'](text,
                                                          padding=True,
                                                          truncation=True,
                                                          return_tensors='pt',
                                                          max_length=512).to(self.device)
        output = self.fsa_models[model_name]['model'](**tokens)

        # self.fsa_models[model]['model'].config.id2label.values() ~= ['pos', 'neg', 'neu']
        # torch.nn.functional.softmax(output.logits, dim=-1).tolist()[0] ~= [.083, .034, .883]
        final_output = {label: value
                        for label, value in
                        zip(self.fsa_models[model_name]['model'].config.id2label.values(),
                            torch.nn.functional.softmax(output.logits, dim=-1).tolist()[0])
                        }

        # Only get the sentiment with the highest score
        return max(final_output, key=final_output.get)

    def _fsa_text_blob(self, text: str, model_name: str) -> str:
        """
        Run Text blob model

        :param text: text to get sentiment from
        :param model_name: Not used in this method, but needed for design pattern
        :return: sentiment
        """
        polarity: float = TextBlob(text).sentiment.polarity
        sentiment: str = self._categorize_sentiment_range(polarity)
        return sentiment

    def _fsa_nltk(self, text: str, model_name: str) -> str:
        """
        This NLTK class uses VADER (Valence Aware Dictionary and sEntiment Reasoner),
        "a lexicon and rule-based sentiment analysis tool that is specifically attuned
        to sentiments expressed in social media" (https://github.com/cjhutto/vaderSentiment)

        :param text:
        :param model_name: Not used in this method, but needed for design pattern
        :return:
        """
        result: dict[str, float] = SentimentIntensityAnalyzer().polarity_scores(text)
        sentiment: str = self._categorize_sentiment_range(result['compound'])
        return sentiment

    def _fsa_experiment_inner(self, data: dict[str, str]) -> dict[str, int]:
        """
        Inner function to parallel or sequential call

        :param data: dict, keys == 'text', 'sentiment'
        :return: accuracy dict
        """
        data_dict = collections.defaultdict(int)

        for model_name, model_info in self.fsa_models.items():
            actual = model_info['method'](text=data['text'], model_name=model_name)
            if actual == data['sentiment']:
                data_dict[model_name] += 1

        return data_dict

    def fsa_experiment(self) -> dict[str, int]:
        """
        Run sentiment analysis experiments to determine best performing model

        :return: dict where key == model and value is correct count
        """
        results_dict = collections.defaultdict(int)

        # If we only have cpu access, run experiments using parallel processing
        if self.device.type == 'cpu':
            with Pool(CPU_COUNT) as p:
                for results in tqdm(
                    p.imap(self._fsa_experiment_inner,
                           self.fsa_ground_truth_data[:100],  # TODO: remove limit
                           chunksize=PARALLEL_CHUNK_SIZE), total=len(self.fsa_ground_truth_data)
                ):
                    for model, count in results.items():
                        results_dict[model] += count

            return dict(results_dict)

        # If we are on cuda or mps
        # TODO: remove limit
        for data in tqdm(self.fsa_ground_truth_data[:100], total=len(self.fsa_ground_truth_data)):
            results = self._fsa_experiment_inner(data=data)
            for model, count in results.items():
                results_dict[model] += count

        return dict(results_dict)

    @staticmethod
    def _pp_hyperparameter_tuning(model,
                                  param_grid: dict,
                                  x_train: pd.DataFrame,
                                  y_train: pd.Series):
        """
        Hyperparameter tune a model given the parameter grid

        :param model: a 'fittable' model
        :param param_grid: dict
        :param x_train: pd.DataFrame
        :param y_train: pd.Series
        :return: a fitted model
        """
        # Using GridSearchCV for exhaustive search
        grid_search = GridSearchCV(estimator=model,
                                   param_grid=param_grid,
                                   cv=2,  # TODO: increase cross-validation count with more compute
                                   scoring='accuracy',
                                   verbose=0)

        # Fit grid search
        grid_search.fit(x_train, y_train)

        # Best hyperparameters and score
        print(f"Best Hyperparameters: {grid_search.best_params_}")
        print(f"Best Score on training data: {grid_search.best_score_:.4f}")

        return grid_search, grid_search.best_estimator_

    def pp_experiment(self, ticker: str = 'AAPL', period: str = '5y'):
        """
        Test the models in the self.pp_models and save the best model to disk

        :param ticker:
        :param period: str - valid periods can be found here:
            https://github.com/ranaroussi/yfinance/wiki/Ticker#parameters
        :return:
        """
        data = self.data.main(ticker=ticker, mode='experiment', period=period)
        train_test_data = self.model.pp_train_test_split(data)
        best_model_accuracy = 0
        experiment_data = pd.DataFrame(columns=self.pp_experiment_columns)
        for model_name, model_data in self.pp_models.items():
            print(f"Testing {model_name}")
            print("---------------------")

            # Hyper parameter tune
            grid_search_results, best_model = self._pp_hyperparameter_tuning(
                model=model_data['model'],
                param_grid=model_data['params'],
                x_train=train_test_data['x_train'],
                y_train=train_test_data['y_train'])

            # Predict and evaluate on test data
            y_pred = self.model.pp_predict(model=best_model, feature_data=train_test_data['x_test'])
            accuracy = self.model.pp_evaluate_model(y_pred=y_pred,
                                                    y_test=train_test_data['y_test'])

            # Store experiment data
            experiment_data = pd.concat([experiment_data,
                                         pd.DataFrame(data=[[self.time,
                                                            model_name,
                                                            grid_search_results.best_score_,
                                                            accuracy,
                                                            grid_search_results.best_params_]],
                                                      columns=self.pp_experiment_columns)])
            if accuracy > best_model_accuracy:
                best_model_accuracy = accuracy
                final_model = {'model': best_model, 'name': model_name}
            print(f"Accuracy on test data: {accuracy:.4f}")

        # Write experiment data
        experiment_data.to_csv(os.path.join(PROJECT_ROOT_PATH, 'experiments', 'experiments.csv'),
                               mode='a')

        # Save the best model
        print(f'{final_model["name"]} is our best performing model')
        self.model.pp_save_model(model=final_model['model'], model_name=final_model['name'])

        return final_model['model']
