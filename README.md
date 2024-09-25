# stock_market_analyzer


## Install
```commandline
git clone https://github.com/JeremyBrent/stock_market_analyzer.git
cd stock_market_analyzer
make install
```

## Data
### FSA
Ground truth financial sentiment analysis data can be found at ./data/data.csv comes from https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis

### Notes
Data can be extracted from Kaggle programatiically, but it requires an api key that I didn't want 
a future user to need to obtain in order to run this code base

# TODO: COMBINE DATA WTIH https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news

## Model
### FSA
#### Run
Experiments for choosing the most accurate FSA model on the ground truth data, defined 
[here](#fsa), can be triggered
using the following: 
```python
from src.experiment import Experiment

experimenter = Experiment()
experimenter.fsa_experiment()
```
This will run the models defined in `experimenter.fsa_models_to_test`. 

#### Future Directions
More FSA models can be experimented on. To include more models in the `Experiment` class, simply 
add the model to `experimenter.fsa_models_to_test` and any new methods that needed to run 
inference with the new model

We could also implement a more sophisticated metric for 
measuring the performance of the FSA models. Currently, we are only using a raw accuracy. 

## Run
### Experiment
This can be used to determine which semantic analysis model performs the best


## 