import os
import torch
import nltk
import collections
import pandas as pd
from tqdm import tqdm
from textblob import TextBlob
from multiprocessing import Pool
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from src.consts import CPU_COUNT, PARALLEL_CHUNK_SIZE, PROJECT_ROOT_PATH
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Experiment:

    def __init__(self):
        # Storing models here so we do not have to instantiate them every time we maka prediction
        self.models = {
            'finbert': {
                'method': self._fsa_bert_model,
                'tokenizer': AutoTokenizer.from_pretrained('ProsusAI/finbert'),
                'model': AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
            },
            'roberta': {
                'method': self._fsa_bert_model,
                'tokenizer': AutoTokenizer.from_pretrained(
                    'cardiffnlp/twitter-roberta-base-sentiment-latest'),
                'model': AutoModelForSequenceClassification.from_pretrained(
                    'cardiffnlp/twitter-roberta-base-sentiment-latest')
            },
            'fin_roberta': {
                'method': self._fsa_bert_model,
                'tokenizer': AutoTokenizer.from_pretrained(
                    'soleimanian/financial-roberta-large-sentiment'),
                'model': AutoModelForSequenceClassification.from_pretrained(
                    'soleimanian/financial-roberta-large-sentiment')
            },
            'textblob': {
                'method': self._fsa_text_blob
            },
            'nltk': {
                'method': self._fsa_nltk
            }
        }

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

        :return:
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

        :param value:
        :return:
        """
        if value >= 0.05:
            return 'positive'
        elif -.05 < value < .05:
            return 'neutral'
        else:
            return 'negative'

    def _fsa_bert_model(self, text: str, model_name: str) -> str:
        """

        :param text:
        :param model:
        :return:
        """
        tokens = self.models[model_name]['tokenizer'](text,
                                                      padding=True,
                                                      truncation=True,
                                                      return_tensors='pt',
                                                      max_length=512)
        output = self.models[model_name]['model'](**tokens)

        # self.models[model]['model'].config.id2label.values() ~= ['pos', 'neg', 'neu']
        # torch.nn.functional.softmax(output.logits, dim=-1).tolist()[0] ~= [.083, .034, .883]
        final_output = {label: value
                        for label, value in
                        zip(self.models[model_name]['model'].config.id2label.values(),
                            torch.nn.functional.softmax(output.logits, dim=-1).tolist()[0])
                        }

        # Only get the sentiment with the highest score
        return max(final_output, key=final_output.get)

    def _fsa_text_blob(self, text: str, model: str) -> str:
        """

        :param text:
        :param model: Not used in this method, but needed for design pattern
        :return:
        """
        polarity: float = TextBlob(text).sentiment.polarity
        sentiment: str = self._categorize_sentiment_range(polarity)
        return sentiment

    def _fsa_nltk(self, text: str, model: str) -> str:
        """
        This NLTK class uses VADER (Valence Aware Dictionary and sEntiment Reasoner),
        "a lexicon and rule-based sentiment analysis tool that is specifically attuned
        to sentiments expressed in social media" (https://github.com/cjhutto/vaderSentiment)

        :param text:
        :param model: Not used in this method, but needed for design pattern
        :return:
        """
        result: dict[str, float] = SentimentIntensityAnalyzer().polarity_scores(text)
        sentiment: str = self._categorize_sentiment_range(result['compound'])
        return sentiment

    def _fsa_experiment_inner(self, data: dict[str, str]) -> dict:
        """

        :param data:
        :return:
        """
        data_dict = collections.defaultdict(int)

        for model_name, model_info in self.models.items():
            actual = model_info['method'](text=data['text'], model=model_name)
            if actual == data['sentiment']:
                data_dict[model_name] += 1

        return data_dict

    def fsa_experiment(self) -> dict[str, int]:
        """

        :return:
        """
        result_dict = collections.defaultdict(int)

        with Pool(CPU_COUNT) as p:
            for data in tqdm(
                p.imap(self._fsa_experiment_inner,
                       self.fsa_ground_truth_data[:100],  # TODO: remove limit
                       chunksize=PARALLEL_CHUNK_SIZE), total=len(self.fsa_ground_truth_data)
            ):
                for model, count in data.items():
                    result_dict[model] += count

        return dict(result_dict)
