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
[here](https://dl.acm.org/doi/10.1145/3649451#sec-4-2). 
Most notably, Financial PhraseBank is one primary dataset for financial area 
sentiment analysis ([Ding et al., 2022](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10403218/#ref-15); 
[Ye, Lin & Ren, 2021](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10403218/#ref-50)), 
which was created by [Malo et al. (2014)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10403218/#ref-33). 
Financial PhraseBank contains 4,845 news sentences found on the LexisNexis database and marked 
by 16 people with finance backgrounds. Annotators were required to label the sentence as positive, 
negative, or neutral market sentiment 
[Malo et al. (2014)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10403218/#ref-33). 
All 4845 sentences were kept with higher than 50% agreement

It is also critical that we further analyze the ground truth data to assert that it is accurate.

To construct a more robust system, it's critical that we move away for csv files in Github
to a database. I contemplated implementing a local postgres DB to store the ground truth data,
but determined that that would be out of scope of this project.

## Model
### Description
Models tested were derived from [this literature review](https://dl.acm.org/doi/10.1145/3649451). 
For example, FinBert was directly mentioned [here](https://dl.acm.org/doi/10.1145/3649451#sec-4-4-5)
and VADER was discussed [here](https://dl.acm.org/doi/10.1145/3649451#sec-4-4-4). Finbert and Roberta 
were two of the top performing models discussed in this literature review [Du et al. (2024)](https://dl.acm.org/doi/10.1145/3649451#tab3), 
and used as a top performer in this research [Xiao et al. (2023)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10403218/). 


### Run
Running experiments for choosing the most accurate FSA model on the ground truth data, defined 
[here](#data), can be triggered using the following: 
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

Any new models should be replicated based on existing research found 
[here](https://dl.acm.org/doi/10.1145/3649451#sec-4-4).

We should also implement a more sophisticated metric for 
measuring the performance of the FSA models. Currently, we are only using a raw accuracy. 

# The Code Base
Given the timeframe of the project, I put together a small, end-to-end project. Some of these end 
to end features include unittests, CICD with Github actions, environment creation with Make and
requirements.txt, and github branch protection rules found 
[here](https://github.com/JeremyBrent/stock_market_analyzer/settings/branch_protection_rules/54816872)
which require 1. PRs and 2. passing Github actions in order to update the main branch. Note, that 
I didn't require approvers on the branch protection rule due to the fact that there is no one else 
to review my code .... this would not be the case in a production environment and that would be 
a rule in said production environment.

With more time, some things I would build upon would be: 
1. Added a comprehensive logging functionality, this is critical to production-worthy code
2. Expanding unittest portfolio would need to build out
3. Further developing the Github actions if we were deploying this model as a service