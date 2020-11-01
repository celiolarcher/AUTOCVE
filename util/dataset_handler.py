from joblib import dump, load
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.utils import check_X_y
import sklearn

if sklearn.__version__>='0.22':
    from sklearn.impute import SimpleImputer as Imputer
else:
    from sklearn.preprocessing import Imputer

import os
import tempfile
import pandas as pd
import numpy as np

def load_dataset(data_X, data_y, subsample_data, FIXED_SPLIT=True, TEST_SIZE=0.3, RANDOM_STATE=42, LIMIT_SPLIT=1000000):
    if isinstance(data_X,pd.DataFrame):
        data_X=pd.get_dummies(data_X)
        data_X=data_X.values

    if np.any(np.isnan(data_X)):
        imputer=Imputer(strategy="median")
        imputer.fit(data_X)
        data_X=imputer.transform(data_X)


    if isinstance(data_y,pd.Series):
        data_y=data_y.values

    if not isinstance(data_X,np.ndarray) or not isinstance(data_y,np.ndarray):
        raise "Incompatible dataset type. Must be pandas or np.ndarray."

    if subsample_data < 1.0 and FIXED_SPLIT is True:
        data_X, _, data_y, _ = train_test_split(data_X, data_y, train_size=subsample_data, random_state=RANDOM_STATE)

    data_X, data_y = check_X_y(data_X, data_y, accept_sparse=False) #change for true when the sparse grammar is ready

    temp_folder = tempfile.mkdtemp()
    filename = os.path.join(temp_folder, 'autocve_joblib.mmap')
    if os.path.exists(filename): os.unlink(filename)
    _ = dump(data_X, filename)
    data_X_memmap = load(filename, mmap_mode='r') 

    if FIXED_SPLIT is True:
        split=None
    else:
        TRAIN_SIZE=1-TEST_SIZE
        split=StratifiedShuffleSplit(train_size=(TRAIN_SIZE*subsample_data), test_size=(TEST_SIZE*subsample_data), random_state=RANDOM_STATE, n_splits=LIMIT_SPLIT).split(data_X_memmap,data_y)

    return data_X_memmap, data_y, split, filename, temp_folder, data_X.shape[1]


def unload_dataset(filename, temp_folder):
    os.unlink(filename)
    os.rmdir(temp_folder)
    
    return 1

## Return data to the last cross-validation in the appropriate size (subsampling) during Dynamic Sampling run.
def get_subsample(data_X, data_y, split, RANDOM_STATE=42):

    if split is None:
        raise "Split variable must be set."

    train_index, test_index=next(split)

    if len(train_index)+len(test_index)==len(data_y):
       return data_X, data_y
    else:
        data_X_sub, _, data_y_sub, _ = train_test_split(data_X, data_y, train_size=len(train_index)+len(test_index), random_state=RANDOM_STATE)
        return data_X_sub, data_y_sub
