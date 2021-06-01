from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from joblib import Parallel, delayed
from joblib import dump, load
from .joblib_silent_timeout import ParallelSilentTimeout
from joblib.externals.loky.process_executor import TerminatedWorkerError
from functools import partial
from .make_pipeline import make_pipeline_str
from multiprocessing import TimeoutError
import numpy as np
import importlib
import warnings
import time
import tempfile
import os


y_last_test_set=None

def log_warning_output(message, category, filename, lineno, file=None, line=None):
    with open("log_warning_methods.log","a+") as file:
        file.write(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())+": "+ warnings.formatwarning(message, category, filename, lineno)+"\n")


warnings.showwarning=log_warning_output

class ScorerHandler():
    def __init__(self, y_pred, ohe_proba):
        self.y_pred=y_pred
        self.ohe_proba=ohe_proba
    def fit(self):
        pass
    def predict(self, X):
        return self.y_pred
    def predict_proba(self, X):
        return self.ohe_proba.transform(self.y_pred.reshape(-1,1)).toarray()


def evaluate_population(pipelines_population,X,y,scoring,n_jobs,timeout_pip_sec, split=None, N_SPLITS=5, verbose=1, TEST_SIZE=0.3, RANDOM_STATE=42):
    try:
        global y_last_test_set
        y_last_test_set=None

        if split is None:
            
            list_train_index=[]
            list_test_index=[]
            y_last_test_set=[]
            split=StratifiedKFold(n_splits=N_SPLITS, shuffle=False)

            for train_index, test_index in split.split(X,y):
                list_train_index.append(train_index)
                list_test_index.append(test_index)
                if len(y_last_test_set)==0:
                    y_last_test_set=y[test_index]
                else:
                    y_last_test_set=np.concatenate([y_last_test_set,y[test_index]])

            predict_length=y.shape[0]

        else:

            train_index, test_index=next(split)
            list_train_index=[train_index]
            list_test_index=[test_index]
            y_last_test_set=y[test_index]
            predict_length=len(test_index)
        
        pipelines_population=pipelines_population.split("|")


        temp_folder = tempfile.mkdtemp()
        filename_train = os.path.join(temp_folder, 'autocve_X_train.mmap')
        filename_test = os.path.join(temp_folder, 'autocve_X_test.mmap')


        metric_population=[]
        predict_population=[]

        evaluate_pipe_timeout=partial(evaluate_solution,verbose=verbose)

        for train_index, test_index in zip(list_train_index,list_test_index):

            X_train=X[train_index,:]
            X_test=X[test_index,:]
            if os.path.exists(filename_train): os.unlink(filename_train)
            if os.path.exists(filename_test): os.unlink(filename_test)
            _ = dump(X_train, filename_train)
            _ = dump(X_test, filename_test)
            X_train = load(filename_train, mmap_mode='r') 
            X_test = load(filename_test, mmap_mode='r') 



            result_pipeline=ParallelSilentTimeout(n_jobs=n_jobs, backend="loky",timeout=timeout_pip_sec)(delayed(evaluate_pipe_timeout)(pipeline_str, X_train, X_test, y[train_index], y[test_index]) for pipeline_str in pipelines_population if pipeline_str is not None) 

            metric_population_cv=[]
            predict_population_cv=[]

            next_pipe=iter(result_pipeline)

            for pipe_id, pipe_str in enumerate(pipelines_population):
                if pipe_str is None:
                    metric_population_cv.append(None)
                    predict_population_cv.append(None)
                else:
                    result_solution=next(next_pipe)
                    if isinstance(result_solution,TimeoutError):
                        if verbose>0:
                            print("Timeout reach for pipeline: "+str(pipe_str))        
                        metric_population_cv.append(None)
                        predict_population_cv.append(None)
                        pipelines_population[pipe_id]=None
                    elif isinstance(result_solution,TerminatedWorkerError):
                        if verbose>0:
                            print("Worker error for pipeline: "+str(pipe_str))        
                        metric_population_cv.append(None)
                        predict_population_cv.append(None)
                        pipelines_population[pipe_id]=None
                    elif(result_solution is None):
                        metric_population_cv.append(None)
                        predict_population_cv.append(None)
                        pipelines_population[pipe_id]=None
                    else:
                        if 'needs_proba=True' in scoring._factory_args() or "needs_threshold=True" in scoring._factory_args():
                            ohe_proba = OneHotEncoder(handle_unknown='ignore').fit(y[test_index].reshape(-1,1))
                        else:
                            ohe_proba = None

                        metric_population_cv.append([scoring(ScorerHandler(result_solution, ohe_proba=ohe_proba), None, y[test_index])])
                        predict_population_cv.append(result_solution) 
            
            del result_pipeline

            if len(metric_population)==0:
                metric_population=metric_population_cv
                predict_population=predict_population_cv
            else:
                metric_population=[None if value is None or old_list is None else old_list+value for old_list, value in zip(metric_population, metric_population_cv)]
                predict_population=[None if value is None or old_list is None else np.concatenate([old_list,value]) for old_list, value in zip(predict_population, predict_population_cv)]


        metric_population=[None if metrics is None else np.mean(metrics) for metrics in metric_population]

        try:
            os.unlink(filename_train)
            os.unlink(filename_test)
            os.rmdir(temp_folder)
        except Exception as E:
            pass

        return metric_population,predict_population, predict_length

    except (KeyboardInterrupt, SystemExit) as e:
       try:
             os.unlink(filename_train)
             os.unlink(filename_test)
             os.rmdir(temp_folder)
       except Exception as e:
             pass 

       return None, None, -1
    

def evaluate_solution(pipeline_str, X_train, X_test, y_train, y_test,verbose=1):
    pipeline=make_pipeline_str(pipeline_str,verbose)
    
    if pipeline is None:
        return None
    try:
        pipeline.fit(X_train,y_train)
    except Exception as e:
        if verbose>0:
            print("Pipeline fit error: "+str(pipeline))
            print(str(e))
        return None

    predict_data=[]
    try:
        predict_data=pipeline.predict(X_test)
    except Exception as e:
        if verbose>0:
            print("Pipeline predict error: "+str(pipeline))
            print(str(e))
        return None

    return predict_data


def evaluate_predict_vector(predict_vector,scoring):
    if 'needs_proba=True' in scoring._factory_args() or "needs_threshold=True" in scoring._factory_args():
        ohe_proba = OneHotEncoder(handle_unknown='ignore').fit(y_last_test_set.reshape(-1,1))
    else:
        ohe_proba = None

    return scoring(ScorerHandler(predict_vector, ohe_proba=ohe_proba), None, y_last_test_set)
