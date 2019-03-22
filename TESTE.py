import sys
import AUTOCVE.util.evaluate as evaluate
print("REF COUNT:"+str(sys.getrefcount(evaluate.evaluate_solution)))

import pandas as pd
#import AUTOCVE.util as util
from AUTOCVE.AUTOCVE import AUTOCVEClassifier
import AUTOCVE.AUTOCVE
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


digits=load_digits()
X = digits.data
y = digits.target


from sklearn.metrics import fbeta_score, make_scorer, accuracy_score
ftwo_scorer = make_scorer(fbeta_score, beta=2, average='micro')

print(callable(ftwo_scorer))

p=AUTOCVEClassifier(generations=2, grammar='grammarTPOT', population_size_components=50, elite_portion_components=0.1, mutation_rate_components=0.9, crossover_rate_components=0.9, population_size_ensemble=50, elite_portion_ensemble=0.1, mutation_rate_ensemble=0.1, crossover_rate_ensemble=0.9, scoring='balanced_accuracy', max_pipeline_time_secs=20, n_jobs=-1, verbose=1, random_state=42, cv_folds=5)
print(type(p))
print(p.get_parameters())

p.optimize(X,y, subsample_data=0.5)

print(p.__doc__)


print("Best pipeline\n")
best_pip=p.get_best_pipeline()

if isinstance(best_pip, type(Pipeline)):
    print(best_pip.steps)
else:
    print(best_pip)
cv_results=cross_validate(best_pip, X, y)
print("Train Score: "+str(cv_results["train_score"].mean()))
print("Test Score: "+str(cv_results["test_score"].mean()))

print("Ensemble best mask \n")
best_voting_ensemble=p.get_best_voting_ensemble()
print(best_voting_ensemble.estimators)
print("Ensemble size: "+str(len(best_voting_ensemble.estimators)))

cv_results_2=cross_validate(best_voting_ensemble, X, y)
print("Train Score: "+str(cv_results_2["train_score"].mean()))
print("Test Score: "+str(cv_results_2["test_score"].mean()))


print("Ensemble elite \n")
ensemble_elite=p.get_voting_ensemble_elite()
print(ensemble_elite.estimators)

cv_results_2=cross_validate(ensemble_elite, X, y)
print("Train Score: "+str(cv_results_2["train_score"].mean()))
print("Test Score: "+str(cv_results_2["test_score"].mean()))


print("Ensemble all \n")
ensemble_all=p.get_voting_ensemble_all()
print(ensemble_all.estimators)

cv_results_2=cross_validate(ensemble_all, X, y)
print("Train Score: "+str(cv_results_2["train_score"].mean()))
print("Test Score: "+str(cv_results_2["test_score"].mean()))




#print(p.get_grammar())

print(dir(p))
print(dir(AUTOCVE.AUTOCVE))
#print(AUTOCVE.runAUTOCVE(data))

print("REF COUNT:"+str(sys.getrefcount(evaluate.evaluate_solution)))

