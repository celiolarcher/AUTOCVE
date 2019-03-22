#include "python_interface.h"

#define BUFFER_SIZE 10000


PythonInterface::PythonInterface(int n_jobs, PyObject* timeout_pip_sec, PyObject *scoring, int cv_folds, int verbose){
    this->n_jobs=n_jobs, this->timeout_pip_sec=timeout_pip_sec, this->cv_folds=cv_folds, this->verbose=verbose;
    this->evaluate_function_py=PythonInterface::load_python_function("evaluate","evaluate_population");
    this->evaluate_predict_vector_py=PythonInterface::load_python_function("evaluate","evaluate_predict_vector");
    this->make_pipeline_py=PythonInterface::load_python_function("make_pipeline","make_pipeline_str");
    this->make_voting_ensemble_py=PythonInterface::load_python_function("make_pipeline","make_voting_ensemble");
    this->load_dataset_py=PythonInterface::load_python_function("dataset_handler","load_dataset");
    this->unload_dataset_py=PythonInterface::load_python_function("dataset_handler","unload_dataset");
    this->load_scoring_py=PythonInterface::load_python_function("scoring_handler","load_scoring");

    if(!this->evaluate_function_py || !this->evaluate_predict_vector_py || !this->make_pipeline_py || !this->make_voting_ensemble_py || !this->load_dataset_py || !this->unload_dataset_py || !this->load_scoring_py){
        throw "Problem while load python interface";
    }

    this->scoring=PyObject_CallFunction(this->load_scoring_py,"O", scoring);
    if(!this->scoring)
        throw "Bad scoring parameter. Expected a string metric in sklearn.metric.SCORER or a scorer function.\n";

    this->data_X=NULL,this->data_y=NULL;
    this->filename_dataset=NULL, this->folder_dataset=NULL;

    if(_import_array() < 0)//Must be called just one time
        throw  "Library numpy.core.multiarray failed to import";

}


PythonInterface::~PythonInterface(){
    if(this->data_X || this->data_y) this->unload_dataset();    

    Py_XDECREF(this->evaluate_function_py);
    Py_XDECREF(this->evaluate_predict_vector_py);
    Py_XDECREF(this->make_pipeline_py);
    Py_XDECREF(this->make_voting_ensemble_py);
    Py_XDECREF(this->load_dataset_py);
    Py_XDECREF(this->unload_dataset_py);
    Py_XDECREF(this->load_scoring_py);
    Py_XDECREF(this->scoring);
}

int PythonInterface::load_dataset(PyObject *raw_data_X, PyObject *raw_data_y, double subsample_data){
    if(!raw_data_X || !raw_data_y) return NULL;    

    PyObject *return_func=PyObject_CallFunction(this->load_dataset_py, "OOd", raw_data_X, raw_data_y, subsample_data);

    if(!return_func)
        return NULL;
    
    this->data_X=PyTuple_GetItem(return_func, 0);
    this->data_y=PyTuple_GetItem(return_func, 1);
    this->filename_dataset=PyTuple_GetItem(return_func, 2);
    this->folder_dataset=PyTuple_GetItem(return_func, 3);
    this->n_feat_dataset=PyLong_AsLong(PyTuple_GetItem(return_func, 4));
    Py_INCREF(this->data_X);
    Py_INCREF(this->data_y);
    Py_INCREF(this->filename_dataset);
    Py_INCREF(this->folder_dataset);
    Py_XDECREF(return_func);

    return 1;
}


int PythonInterface::unload_dataset(){
    Py_XDECREF(this->data_X);
    Py_XDECREF(this->data_y);
    this->data_X=NULL;
    this->data_y=NULL;

    PyObject *return_func=PyObject_CallFunction(this->unload_dataset_py,"OO", this->filename_dataset, this->folder_dataset);

    Py_XDECREF(this->filename_dataset);
    Py_XDECREF(this->folder_dataset);
    if(!return_func)
        return NULL;
    else 
        Py_XDECREF(return_func);

    return 1;
}


PyObject* PythonInterface::load_python_function(const char *module_str, const char* function_str){
    char lib_imported[BUFFER_SIZE];
    strcpy(lib_imported,"AUTOCVE.util.");
    strcat(lib_imported,module_str);
    PyObject *autocve_module=PyImport_ImportModule(lib_imported);

    if(!autocve_module){
        fprintf(stderr, "Cannot import %s module\n",module_str);
        return NULL;
    }

    PyObject *global_dict=PyModule_GetDict(autocve_module);

    if(!global_dict){
        Py_XDECREF(autocve_module);    
        fprintf(stderr, "Cannot get the dictionary from %s module\n",module_str);
        return NULL;
    }
    
    PyObject *loaded_function=PyDict_GetItemString(global_dict, function_str);

    if (loaded_function && PyCallable_Check(loaded_function)){
        Py_INCREF(loaded_function);
        Py_XDECREF(autocve_module);    
        return loaded_function;
    }

    Py_XDECREF(autocve_module);    
    if (PyErr_Occurred())
        PyErr_Print();

    fprintf(stderr, "Cannot find %s function\n",function_str);

    return NULL;        
}

int PythonInterface::evaluate_pipelines_cv(char *pipeline_evaluated, PyObject **pipeline_score, PyObject **result_obj, int *predict_size){
    if(!this->data_X || !this->data_y) return NULL;    

    PyObject *return_func;

    return_func=PyObject_CallFunction(this->evaluate_function_py, "sOOOiOii", pipeline_evaluated, this->data_X, this->data_y, this->scoring, this->n_jobs, this->timeout_pip_sec, this->cv_folds, this->verbose);
    
    if(!return_func)
        return NULL; 

    (*pipeline_score)=PyTuple_GetItem(return_func, 0);
    (*result_obj)=PyTuple_GetItem(return_func, 1);
    (*predict_size)=PyLong_AsLong(PyTuple_GetItem(return_func, 2));

    Py_INCREF((*pipeline_score));
    Py_INCREF((*result_obj));

    Py_XDECREF(return_func);    
    
    return 1;
}

int PythonInterface::evaluate_predict_vector(PyObject *predict_vector, double *return_score){
    PyObject *return_func=PyObject_CallFunction(this->evaluate_predict_vector_py, "OO", predict_vector, this->scoring);

    if(!return_func)
        return NULL; 

    (*return_score)=PyFloat_AsDouble(return_func);

    Py_XDECREF(return_func);    
    
    return 1;
}

PyObject *PythonInterface::make_pipeline_from_str(char *pipeline_str){
    PyObject *return_func=PyObject_CallFunction(this->make_pipeline_py,"s",pipeline_str);

    return return_func;
}

PyObject *PythonInterface::make_voting_ensemble_from_str(char *population_str){
    PyObject *return_func=PyObject_CallFunction(this->make_voting_ensemble_py,"s",population_str);

    return return_func;
}

int PythonInterface::get_n_feat_dataset(){
    return this ->n_feat_dataset;
}
