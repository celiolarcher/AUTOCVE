#ifndef PYTHONINTERFACEH
#define PYTHONINTERFACEH

#ifndef NPY_NO_DEPRECATED_API  
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION  //Setting the minimum version for Numpy API to 1.7
#endif

#include <Python.h>
#include <numpy/arrayobject.h>

class PythonInterface{

    private: PyObject *evaluate_function_py;
    PyObject *evaluate_predict_vector_py;
    PyObject *make_pipeline_py;
    PyObject *make_voting_ensemble_py;
    PyObject *load_dataset_py;
    PyObject *unload_dataset_py;
    PyObject *get_subsample_py;
    PyObject *load_scoring_py;
    PyObject *data_X; 
    PyObject *data_y; 
    PyObject *split_dataset;
    PyObject *filename_dataset;
    PyObject *folder_dataset;
    PyObject *scoring;
    int n_feat_dataset;
    int cv_folds;
    double test_size;

    int n_jobs;
    PyObject* timeout_pip_sec;
    int verbose;

    public: PythonInterface(int n_jobs, PyObject* timeout_pip_sec, PyObject *scoring, int cv_folds, double test_size, int verbose);
    ~PythonInterface();
    int load_dataset(PyObject *data_X, PyObject *data_y, double subsample_data, int cv_evaluation);
    int unload_dataset();
    int get_n_feat_dataset();
    int evaluate_pipelines(char *pipeline_evaluated, PyObject **pipeline_score, PyObject **result_obj, int *predict_size);
    int evaluate_pipelines_cv(char *pipeline_evaluated, PyObject **pipeline_score, PyObject **result_obj, int *predict_size);
    int evaluate_predict_vector(PyObject *predict_vector, double *return_score);
    PyObject *make_pipeline_from_str(char *pipeline_str);
    PyObject *make_voting_ensemble_from_str(char *population_str);
    static PyObject* load_python_function(const char *file, const char* function);
};

#endif
