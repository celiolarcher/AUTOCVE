from sklearn.externals.joblib import dump, load
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.utils import check_X_y
from sklearn.preprocessing import Imputer
import os
import tempfile
import pandas as pd
import numpy as np

def load_dataset(data_X, data_y, subsample_data, RANDOM_STATE=42):
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

    if subsample_data < 1.0:
        data_X, _, data_y, _ = train_test_split(data_X, data_y, train_size=subsample_data, random_state=RANDOM_STATE)

    data_X, data_y = check_X_y(data_X, data_y, accept_sparse=False) #change for true when the sparse grammar is ready

    temp_folder = tempfile.mkdtemp()
    filename = os.path.join(temp_folder, 'autocve_joblib.mmap')
    if os.path.exists(filename): os.unlink(filename)
    _ = dump(data_X, filename)
    data_X_memmap = load(filename, mmap_mode='r') 

    return data_X_memmap, data_y, filename, temp_folder, data_X.shape[1]


def unload_dataset(filename, temp_folder):
    os.unlink(filename)
    os.rmdir(temp_folder)
    
    return 1
