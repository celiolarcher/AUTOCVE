#ifndef AUTOCVEH
#define AUTOCVEH

#ifndef NPY_NO_DEPRECATED_API  
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION  //Setting the minimum version for Numpy API to 1.7
#endif


#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include "population.h"
#include "grammar.h"
#include "python_interface.h"

class AUTOCVEClass{
    public: Population *population;
    PopulationEnsemble *population_ensemble;
    Grammar *grammar;
    PythonInterface *python_interface;
    int seed; int n_jobs; PyObject* timeout_pip_sec; int timeout_evolution_process_sec; char *grammar_file; int generations; PyObject *scoring; int verbose; 
    int size_pop_components; double elite_portion_components; double mut_rate_components; double cross_rate_components;
    int size_pop_ensemble; double elite_portion_ensemble; double mut_rate_ensemble; double cross_rate_ensemble;  
    int cv_evaluation;
    int cv_folds;
    double test_size;
    int evolution_steps_after_cv;

    public: AUTOCVEClass(int seed, int n_jobs, PyObject* timeout_pip_sec, int timeout_evolution_process_sec, char *grammar_file, int generations, int size_pop_components, double elite_portion_components, double mut_rate_components, double cross_rate_components, int size_pop_ensemble, double elite_portion_ensemble, double mut_rate_ensemble, double cross_rate_ensemble, PyObject *scoring, int cv_evaluation, int cv_folds, double test_size, int evolution_steps_after_cv, int verbose);
    ~AUTOCVEClass();
    int run_genetic_programming(PyObject *data_X, PyObject *data_y, double subsample_data);
    PyObject *get_best_pipeline();
    PyObject *get_voting_ensemble_all();
    PyObject *get_voting_ensemble_elite();
    PyObject *get_voting_ensemble_best_mask();
    char *get_grammar_char();
    char *get_parameters_char();

    private: static char* grammar_file_handler(char *grammar_file_param);
};




#endif


