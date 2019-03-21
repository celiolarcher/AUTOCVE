# AUTOCVE

This library is intended to be used for allows the search for voting ensembles. Based on a coevolutionary framework, it turns possible to testing multiple ensembles configurations without repetitive training and test procedure of its components.



In this current version, for proper use, it is recommended to install the library in the Anaconda package.

In addition the use of the py-xgboost library is observed as to give a more stable execution in this library.

Example of usage:

```
from AUTOCVE.AUTOCVE import AUTOCVEClassifier
from sklearn.model_selection import cross_validate
from sklearn.datasets import load_digits

digits=load_digits()
X = digits.data
y = digits.target

autocve=AUTOCVEClassifier(generations=5, grammar='grammarTPOT')

autocve.optimize(X,y, subsample_data=0.5)

print("Best ensemble")
best_voting_ensemble=p.get_best_voting_ensemble()
print(best_voting_ensemble.estimators)
print("Ensemble size: "+str(len(best_voting_ensemble.estimators)))

cv_results=cross_validate(best_voting_ensemble, X, y)
print("Train Score: "+str(cv_results["train_score"].mean()))
print("Test Score: "+str(cv_results["test_score"].mean()))
```
