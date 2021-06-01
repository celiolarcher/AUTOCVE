# AUTOCVE 


This library is intended to be used in the search for hard voting ensembles. Based on a coevolutionary framework, it turns possible to testing multiple ensembles configurations without repetitive training and test procedure of its components.

The ensembles created are based on the [Voting Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html) class. In the default version, several methods implemented in the [scikit-learn](https://github.com/scikit-learn/scikit-learn) package can be used on the final ensemble as well as  the XGBClassifier of the [XGBoost](https://github.com/dmlc/xgboost) library. In addition, other [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) compatible libraries can be added in a custom grammar. 

Yet, the Auto-CVE uses the dynamic sampling holdout as an option to accelerate the evaluation of pipelines, which can make the search procedure orders of magnitude faster than using the regular cross-validation.

### Currently only classification tasks are available (although regression tasks are planned to be included as well).

#### Paper experiment scripts can be found in the [autocve_experiments](https://github.com/celiolarcher/autocve_experiments) repository.

## Prerequisites

In this current version, for proper use, it is recommended to install the library in the Anaconda package, since almost all dependencies are met in a fresh install. 

## Installing

Just type the following commands:

```
git clone git@github.com:celiolarcher/AUTOCVE.git
cd AUTOCVE
pip install -r requirements.txt
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
| optimize      | Optimize an ensemble to the (X,y) base. X and y expect to be numeric (used pandas.get_dummies otherwise). |
| get_best_voting_ensemble     | Get the best ensemble produced in the optimization procedure (recommended option). |
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
| n_jobs                  | number of jobs scheduled in parallel in the evaluation of components  | 
| max_pipeline_time_secs        | maximum time allowed to a single training and test procedure of the cross-validation (None means not time bounded)  |
| max_evolution_time_secs        | maximum time allowed to the whole evolutionary procedure to run (0 means not time bounded) | 
| grammar  | the grammar option or path to a custom grammar used in the Context Free Genetic Program algorithm (used to specfy the algorithms) | 
| generations  | number of generations performed      | 
| population_size_components  | size of the population of components used in the ensembles | 
| mutation_rate_components  | mutation rate of the population of components | 
| crossover_rate_components  | crossover rate of the population of components | 
| population_size_ensemble  | size of the population of ensembles | 
| mutation_rate_ensemble  | mutation rate of the population of ensembles | 
| crossover_rate_ensemble  | crossover rate of the population of ensembles | 
| scoring  | score option used to evaluate the pipelines (sklearn compatible) | 
| cv_evaluation_mode | True to evaluate ensembles using cross-validation and False using dynamic sampling holdout |
| cv_folds  | number of folds in the cross validation procedure  | 
| test_size | ratio of database used to evaluate models when using in dynamic sampling holdout |
| evolution_steps_after_cv | number of generations performed in ensemble population after cross-validation when using dynamic sampling holdout |
| verbose  | verbose option | 



## AutoCVE reference:

If you use AutoCVE in an academic paper, please consider to one of these papers:


Celio H. N. Larcher and Helio J. C. Barbosa. 2021. Evaluating Models with Dynamic Sampling Holdout. In Applications of Evolutionary Computation (Evostar '21).      Springer International Publishing, 729--744. DOI:https://doi.org/10.1007/978-3-030-72699-7_46


```
@inproceedings{10.1007/978-3-030-72699-7_46,
  author = {Larcher, Celio H. N. and Barbosa, Helio J. C.},
  title={Evaluating Models with Dynamic Sampling Holdout},
  year={2021},
  isbn={9783030726997},
  publisher = {Springer International Publishing},
  address={Cham},
  url = {https://doi.org/10.1007/978-3-030-72699-7_46},
  doi = {10.1007/978-3-030-72699-7_46},
  booktitle={Applications of Evolutionary Computation},
  pages={729--744},
  numpages = {16},
  keywords = {Auto-ML, Machine learning, Evolutionary algorithms},
  location = {Seville, Spain},
  series = {Evostar'21}
}
```

Celio H. N. Larcher and Helio J. C. Barbosa. 2019. Auto-CVE: a coevolutionary approach to evolve ensembles in automated machine learning. In Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '19). Association for Computing Machinery, New York, NY, USA, 392–400. DOI:https://doi.org/10.1145/3321707.3321844


```
@inproceedings{10.1145/3321707.3321844,
    author = {Larcher, Celio H. N. and Barbosa, Helio J. C.},
    title = {Auto-CVE: A Coevolutionary Approach to Evolve Ensembles in Automated Machine Learning},
    year = {2019},
    isbn = {9781450361118},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3321707.3321844},
    doi = {10.1145/3321707.3321844},
    booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference},
    pages = {392–400},
    numpages = {9},
    keywords = {ensemble methods, coevolution, supervised learning, auto-ml},
    location = {Prague, Czech Republic},
    series = {GECCO'19}
}
```



## Contributions

Any suggestions are welcome to improve this work and should be directed to Celio Larcher Junior (celiolarcher@gmail.com).

Despite this, as this work is part of my Ph.D. thesis, the pull request acceptance is limited to simple fixes. 

Also, although I try to continually improve this code, I can not guarantee an immediate fix of any requested issue.
