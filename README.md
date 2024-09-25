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
Ground truth data can be augmented with the following data set:
https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news

It is also critical that we further analyze the ground truth data to assert that it is accurate.

## Model
### Run
Experiments for choosing the most accurate FSA model on the ground truth data, defined 
[here](#fsa), can be triggered
using the following: 
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

We should also implement a more sophisticated metric for 
measuring the performance of the FSA models. Currently, we are only using a raw accuracy. 
