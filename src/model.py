import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Model:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
        self.model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')

    def _tokenize(self, text: str) -> transformers.tokenization_utils_base.BatchEncoding:
        """
        Tokenize text

        :param text: str - text to tokenize
        :return: transformers.tokenization_utils_base.BatchEncoding - tokenized result
        """
        tokens = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        return tokens

    def _predict(self,
                 tokens: transformers.tokenization_utils_base.BatchEncoding
                 ) -> transformers.modeling_outputs.SequenceClassifierOutput:
        """

        :param tokens:
        :return:
        """
        output = self.model(**tokens)
        return output

    def _extract_results(self,
                         model_output: transformers.modeling_outputs.SequenceClassifierOutput
                         ) -> dict[str, float]:
        """

        :param model_output:
        :return:
        """
        results = {label: value
                   for label, value in
                   zip(self.model.config.id2label.values(),
                       torch.nn.functional.softmax(model_output.logits, dim=-1).tolist()[0])
                   }

        return results

    def predict(self, text):
        """

        :param text:
        :return:
        """
        tokens = self._tokenize(text)
        output = self._predict(tokens)
        results = self._extract_results(output)
        return results



