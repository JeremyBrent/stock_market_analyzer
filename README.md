# stock_market_analyzer


# Install
```commandline
git clone https://github.com/JeremyBrent/stock_market_analyzer.git
cd stock_market_analyzer
make install
```

# FSA

## Data
Ground truth financial sentiment analysis data can be found at 
./data/data.csv comes from https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis

### Notes
Data can be extracted from Kaggle programmatically, but it requires an api key that I didn't want 
a future user to need to obtain in order to run this code base

### Future Directions
Ground truth data should be augmented with datasets found 
[here](https://dl.acm.org/doi/10.1145/3649451#sec-4-2)

It is also critical that we further analyze the ground truth data to assert that it is accurate.

To construct a more robust system, it's critical that we move away for csv files in Github
to a database. I contemplated implementing a local postgres DB to store the ground truth data,
but determined that that would be out of scope of this project.

## Model
### Description
Models tested were derived from [this literature review](https://dl.acm.org/doi/10.1145/3649451). 
For example, FinBert was directly mentioned [here](https://dl.acm.org/doi/10.1145/3649451#sec-4-4-5)
and VADER was discussed [here](https://dl.acm.org/doi/10.1145/3649451#sec-4-4-4). Finbert and Roberta 
were two of the [top performing models](https://dl.acm.org/doi/10.1145/3649451#tab3) discussed in 
this literature review.

### Run
Running experiments for choosing the most accurate FSA model on the ground truth data, defined 
[here](#data), can be triggered using the following: 
```python
from src.experiment import Experiment

experimenter = Experiment()
experimenter.fsa_experiment()
```
This will run the models defined in `experimenter.fsa_models_to_test`. 

### Future Directions
More FSA models can be experimented on. To include more models in the `Experiment` class, simply 
add the model to `experimenter.fsa_models_to_test` and any new methods that are needed to run 
inference with the new model. 

These models should be replicated based on existing research found 
[here](https://dl.acm.org/doi/10.1145/3649451#sec-4-4).

We should also implement a more sophisticated metric for 
measuring the performance of the FSA models. Currently, we are only using a raw accuracy. 
