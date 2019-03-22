# AUTOCVE

This library is intended to be used in the search for hard voting ensembles. Based on a coevolutionary framework, it turns possible to testing multiple ensembles configurations without repetitive training and test procedure of its components.



In this current version, for proper use, it is recommended to install the library in the Anaconda package.

In addition the use of the py-xgboost implementation in the Anaconda Package is observed as to give a more stable execution in the AUTOCVE.

## Usage:

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
