# stock_market_analyzer


# Install
```commandline
git clone git@github.com:JeremyBrent/stock_market_analyzer.git
cd stock_market_analyzer
make install
source ./venv/bin/activate
```

This code has only be developed and tested on MacOS with python version 3.9.13.
If you do not have python3.9 (`make install` will break) on your machine run the following to download it via
brew (if you have any issues with brew, reference their [installation guide](https://docs.brew.sh/Installation):
```commandline
if ! command -v brew &> /dev/null
then
    echo "Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "Homebrew is already installed."
fi

brew install python@3.9
```

# The Code Base
Given the timeframe of the project, I put together a small, end-to-end project. Some of these end 
to end features include unittests, CICD with Github actions, environment creation with Make and
requirements.txt, and github branch protection rules found 
[here](https://github.com/JeremyBrent/stock_market_analyzer/settings/branch_protection_rules/54816872)
which require 1. PRs and 2. passing Github actions in order to update the main branch. Note, that 
I didn't require approvers on the branch protection rule due to the fact that there is no one else 
to review my code .... this would not be the case in a production environment and that would be 
a rule in said production environment.

With more time, I would: 
1. Add a comprehensive logging functionality, this is critical to production-worthy code
2. Expand the unittest portfolio
3. Further develop the Github actions if we were deploying this model as a service
4. Complete any todos noted throughout the codebase

# FSA

## Data
Ground truth financial sentiment analysis data can be found at 
./data/data.csv comes from https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis

### Notes
Data can be extracted from Kaggle programmatically, but it requires an api key that I didn't want 
a future user to need to obtain in order to run this code base

### Future Directions
Ground truth data should be augmented with datasets found
[here](https://dl.acm.org/doi/10.1145/3649451#sec-4-2). 
Most notably, Financial PhraseBank is one primary datasets used for financial 
sentiment analysis ([Ding et al., 2022](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10403218/#ref-15); 
[Ye, Lin & Ren, 2021](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10403218/#ref-50)), 
which was created by [Malo et al. (2014)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10403218/#ref-33). 
Financial PhraseBank contains 4,845 news sentences found on the LexisNexis database and are annotated 
by 16 people with finance backgrounds. Annotators were required to label the sentence as positive, 
negative, or neutral market sentiment 
[Malo et al. (2014)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10403218/#ref-33). 
The 4845 sentences which exist in the dataset had higher than 50% inter-annotator agreement.

To construct a more robust system, it's critical that we move away from csv files in Github
to a database. I contemplated implementing a local postgres DB to store the ground truth data,
but determined that that would be out of scope of this project.

## Model
### Description
Models tested were derived from [this literature review](https://dl.acm.org/doi/10.1145/3649451). 
For example, FinBert was directly mentioned [here](https://dl.acm.org/doi/10.1145/3649451#sec-4-4-5)
and VADER was discussed [here](https://dl.acm.org/doi/10.1145/3649451#sec-4-4-4). Finbert and Roberta 
were two of the top performing models discussed in this literature review [Du et al. (2024)](https://dl.acm.org/doi/10.1145/3649451#tab3), 
and Finbert was used as a top performer in this research [Xiao et al. (2023)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10403218/). 

### Performance
On a small test set, FinBert achieved 74% accuracy, Roberta achieved 62% accuracy, 
FinRoberta achieved 70% accuracy, TextBlob achieved 52% accuracy, and NLTK achieved 47% accuracy.

### Run
In order to run experiments to determine the most accurate FSA model on the ground truth data, defined 
[here](#data), run the following: 
```python
from src.experiment import Experiment

experimenter = Experiment()
experimenter.fsa_experiment()
```
This will run the models defined in `experimenter.models`. 

I have implemented functionality that can run FSA experiments using CUDA, 
MPS (if on Mac Silicon Chip) or parallel compute if on CPU. This allows the software decide 
the most efficient way to FSA experimentation. Run-time experimentation that I conducted gave 
general estimates that Parallel compute (on CPU) would complete in about 1 hour, and MPS device 
would complete in about 30 minutes. 

### Future Directions
More FSA models can be experimented on. To include more models in the `Experiment` class, simply 
add the model to `experimenter.models` and any new key-value pairs that are needed to run 
inference with the new model. 

Any new models should be replicated based on existing research. Some of these models can be found 
[here](https://dl.acm.org/doi/10.1145/3649451#sec-4-4).

We should also implement a more sophisticated metric for 
measuring the performance of the FSA models. Currently, we are only using a raw accuracy.

# Price Prediction Model (PP)

## Model

### Description
The Price Prediction model is trained to perform a binary classification to determine if 
price will end higher or lower for the given day. 

### Performance
Our highest performing model was a RandomForestClassifier with a test accuracy score around 72%. 
A pretty decent score consdering the scope of this project. However, this model performed 
significantly better on the test set, almost 20% better, this can be seen in `./experiments/experiments.csv`, which is suspect ... This 
will need to be investigated for data leakage, changes in data distributions between the test 
set and the train set, etc.

### Features
The current features of the model, and I talk more about feature extraction [below](#feature-extract),
include a [50 and 200 day Simple Moving Average](https://www.investopedia.com/terms/s/sma.asp),
[Relative Strength Indicator](https://www.investopedia.com/terms/r/rsi.asp), 
[On Balance Volume](https://www.investopedia.com/terms/o/onbalancevolume.asp), 
[Bottom, Middle & Lower Bollinger Bands](https://www.investopedia.com/terms/b/bollingerbands.asp) and 
a Normalized Sentiment Score of News data. 

The Norm Sentiment Score equals
`sentiment / (original date - effective date + 1)` where original date is the date at which 
the news was published and the effective date is the theoretical date when this news will next 
effect the market. Sentiment is a value between `-1.0` and `1.0` depending on the category (negative or positive) 
and is normalized to be a value between `-.05` and `.05` if the most salient cateogry is neutral. 
Calculation of the effective date can be found at `src.Utils.convert_datetime()` and extraction of 
the sentiment can be found at `src.Model._fsa_extract_results()`

### Run

#### Inference
TODO: ADD SECTION HERE

#### Experiment
To run experiments to get the best model performing Price Prediction Model, run the following code: 
```python
from src.experiment import Experiment
exp = Experiment()
exp.pp_experiment(ticker='AAPL', period='5y')
```
The code above, with perform a grid search for hyperparameter tuning over various models, get the 
best model with the best hyperparameters and save the model to disk.

### Future Directions

1. <p id="feature-extract">We will need to run more through experimentation on our features to determine if any need to
added or removed. </p>
- Some things that need to be determined are correlations between features. For 
testing numerical features, Pearson Correlation Coefficient or Spearman or Kendall Correlation 
(for Non-linear Relationships) can be used. For categorical data, a Chi Square test can be used. 
Tree based models are less sensitive to Multicollinearity compared to a logistic regression, but we 
should still have a sense of the distributions of our training data. Multicollienarity can 
result in unstable coefficients, where changes to the correlated features can have significant 
impacts on model performance, or over-fitting, where the model simply learns the same pattern from 
many of the features.
- To add more or change features in our model, you will need to update the following code:
`src.Data._price_feature_extraction()`, `src.Data._news_feature_extraction()` 
and `src.Model.pp_extract_features()`

2. We should test more models, the current solution uses no neural network architectures.
- To add more models to our experimentation class, simply add the model and respective key-value pairs
to `exp.pp_models`
 
3. Investigate significantly higher performance in RandomForest test set compared to train set 
[mentioned above](#performance)

4. We had issues running GridSearch on XGBoost and LGBoost where we getting the error: 
`Process finished with exit code 139 (interrupted by signal 11: SIGSEGV).` This would need to be more 
thoroughly debugged. These models are currently commented out in `exp.pp_models` due to this.
