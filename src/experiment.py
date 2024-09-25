import torch
import nltk
import collections
import pandas as pd
from tqdm import tqdm
from textblob import TextBlob
nltk.download('vader_lexicon')
from multiprocessing import Pool
from src.consts import CPU_COUNT, PARALLEL_CHUNK_SIZE
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Experiment:

    def __init__(self):

        # Storing models here so we do not have to instantiate them every time we maka prediction
        self.models = {
            'finbert': {
                'tokenizer': AutoTokenizer.from_pretrained('ProsusAI/finbert'),
                'model': AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
            },
            'textblob': ...,
            'nltk': SentimentIntensityAnalyzer(),
            'roberta': {
                'tokenizer': AutoTokenizer.from_pretrained(
                    'cardiffnlp/twitter-roberta-base-sentiment-latest'),
                'model': AutoModelForSequenceClassification.from_pretrained(
                    'cardiffnlp/twitter-roberta-base-sentiment-latest')
            },
            'fin_roberta': {
                'tokenizer': AutoTokenizer.from_pretrained(
                    'soleimanian/financial-roberta-large-sentiment'),
                'model': AutoModelForSequenceClassification.from_pretrained(
                    'soleimanian/financial-roberta-large-sentiment')
            }
        }

        # The first index in the below list needs to be a method with 2 input parameters,
        # text, model
        self.fsa_models_to_test = \
            [('finbert', self._fsa_bert_model),
             ('fin_roberta', self._fsa_bert_model),
             ('roberta', self._fsa_bert_model),
             ('nltk', self._fsa_nltk)]

        self.fsa_ground_truth_data = self._get_fsa_ground_truth_data()

    @staticmethod
    def _get_fsa_ground_truth_data() -> list[dict[str, str]]:
        """

        :return:
        """
        df: pd.DataFrame = pd.read_csv('./data/fsa_ground_truth.csv')
        return [{"text": x[0], 'sentiment': x[1]} for x in df.values.tolist()]

    def _fsa_bert_model(self, text: str, model: str) -> str:
        """

        :param text:
        :param model:
        :return:
        """
        tokens = self.models[model]['tokenizer'](text,
                                                 padding=True,
                                                 truncation=True,
                                                 return_tensors='pt')
        output = self.models[model]['model'](**tokens)

        # self.models[model]['model'].config.id2label.values() ~= ['pos', 'neg', 'neu']
        # torch.nn.functional.softmax(output.logits, dim=-1).tolist()[0] ~= [.083, .034, .883]
        final_output = {label: value
                        for label, value in
                        zip(self.models[model]['model'].config.id2label.values(),
                            torch.nn.functional.softmax(output.logits, dim=-1).tolist()[0])
                        }
        print(final_output)
        # Only get the sentiment with the highest score
        return max(final_output, key=final_output.get)

    def _fsa_text_blob(self, text: str):
        """

        :param text:
        :return:
        """
        sentiment = TextBlob(text).sentiment.polarity
        return sentiment

    def _fsa_nltk(self, text: str, model: str):
        """
        This NLTK class uses VADER (Valence Aware Dictionary and sEntiment Reasoner),
        "a lexicon and rule-based sentiment analysis tool that is specifically attuned
        to sentiments expressed in social media" (https://github.com/cjhutto/vaderSentiment)

        :param text:
        :param model:
        :return:
        """
        results = self.models[model].polarity_scores(text)

        # Threshold were determined here:
        # https://github.com/cjhutto/vaderSentiment?tab=readme-ov-file#about-the-scoring
        if results['compound'] >= 0.05:
            return 'positive'
        elif -.05 < results['compound'] < .05:
            return 'neutral'
        else:
            return 'negative'

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

    def _fsa_experiment_inner(self, data: dict[str, str]) -> dict:
        """

        :param data:
        :return:
        """
        data_dict = collections.defaultdict(int)

        for model, predict in self.fsa_models_to_test:

            actual = predict(data['text'], model)
            if actual == data['sentiment']:
                data_dict[model] += 1

        return data_dict

