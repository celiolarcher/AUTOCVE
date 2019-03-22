# AUTOCVE

This library is intended to be used in the search for hard voting ensembles. Based on a coevolutionary framework, it turns possible to testing multiple ensembles configurations without repetitive training and test procedure of its components.

### Currently only classification tasks are available! Regression tasks are intended to be put in work soon.

## Prerequisites

In this current version, for proper use, it is recommended to install the library in the Anaconda package, since almost all dependencies are met in a fresh install. 

This library has not yet been tested on a Windows installation, so correct functionality is not guaranteed. 

In addition the use of the py-xgboost implementation in the Anaconda Package is observed as to give a more stable execution to AUTOCVE.

## Installing

Just type the following commands:

```
git clone git@github.com:celiolarcher/AUTOCVE.git
cd AUTOCVE
pip install .
```

## Usage

```
from AUTOCVE.AUTOCVE import AUTOCVEClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

digits=load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

autocve=AUTOCVEClassifier(generations=100, grammar='grammarTPOT', n_jobs=-1)

autocve.optimize(X_train, y_train, subsample_data=1.0)

print("Best ensemble")
best_voting_ensemble=autocve.get_best_voting_ensemble()
print(best_voting_ensemble.estimators)
print("Ensemble size: "+str(len(best_voting_ensemble.estimators)))

best_voting_ensemble.fit(X_train, y_train)
print("Train Score: {:.2f}".format(best_voting_ensemble.score(X_train, y_train)))
print("Test Score: {:.2f}".format(best_voting_ensemble.score(X_test, y_test)))
```

## Procedures

| Function      | Description |
| ------------- |-------------| 
| optimize      | Optimize pipelines to the (X,y) base. X and y expect to be numeric (used pandas.get_dummies otherwise). |
| get_best_voting_ensemble     | Get the best ensemble produced in the ensemble population defined in the last generation (option recomended). |
| get_best_pipeline      | Get the pipeline with higher score in the last generation. |
| get_voting_ensemble_elite      | Get the ensemble compound by the 10% pipelines with higher score defined in the last generation. |
| get_voting_ensemble_all      | Get the ensemble compound by all the pipelines defined in the last generation. |
| get_grammar      | Get as text the grammar used in the optimization procedure. |
| get_parameters      | Get as text the parameters used in the optimization procedure. |


## Parameters

All these keyword parameters can be set in the initialization of the AUTOCVE.

| Keyword       | Description|
| ------------- |-------------| 
| random_state                  | seed used in the optimization process | 
| n_jobs                  | number of jobs scheduled in parallel in the evaluation of the a component   | 
| max_pipeline_time_secs        | maximum time allowed to a single training and test procedure of the cross-validation (None is not bounded)  |
| max_evolution_time_sec        | maximum time allowed to the whole evolutionary procedure to run  | 
| grammar  | the grammar option or path to a custom grammar used in the Context Free Genetic Program algorithm (used to specfy the algorithms) | 
| generations  | number of generations performed      | 
| population_size_components  | size of the population of components used in the ensembles | 
| mutation_rate_components  | mutation rate of the population of components | 
| crossover_rate_components  | crossover rate of the population of components | 
| population_size_ensemble  | size of the population of ensembles | 
| mutation_rate_ensemble  | mutation rate of the population of ensembles | 
| crossover_rate_ensemble  | crossover rate of the population of ensembles | 
| scoring  | score option used to evaluate the pipelines (sklearn compatible) | 
| cv_folds  | number of folds in the cross validation procedure  | 
| verbose  | verbose option | 
| elite_portion_components  | deprecated | 
| elite_portion_ensemble  | deprecated | 




## Contributions

Any suggestions are welcome to improve this work.

Despite this, as this work is part of my PhD thesis, the pull request acceptance is limited to simple fixes. 

Also, although I try to continually improve this code, I can not guaranteed an immediate fix of any requested issue.
