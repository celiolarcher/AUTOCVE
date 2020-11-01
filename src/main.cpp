#include <Python.h>
#include <stdio.h>
#include "AUTOCVE.h"
#include "grammar.h"
#include "python_interface.h"


typedef struct {
    PyObject_HEAD
    AUTOCVEClass *ptr_autocve;
}PyAUTOCVE;

/*Object initialization method (expect to receive the pameters of optmization process)*/
static int PyAUTOCVE_init(PyAUTOCVE *self, PyObject *args, PyObject *kwargs){
    PyObject *timeout_pip_sec=NULL; 
    PyObject *scoring=NULL;

    int seed=42;
    int timeout_evolution_process_sec=0;
    int n_jobs=1;
    int size_pop_components=50, size_pop_ensemble=50, generations=100;
    int verbose=0;
    char *grammar_file="grammarTPOT";
    double elite_portion_components=0.1, mut_rate_components=0.9, cross_rate_components=0.9;
    double elite_portion_ensemble=0.1, mut_rate_ensemble=0.1, cross_rate_ensemble=0.9;
    int cv_evaluation=0;
    int cv_folds=5;
    double test_size=0.3;
    int evolution_steps_after_cv=10;

    static char *keywords[]={"random_state","n_jobs","max_pipeline_time_secs","max_evolution_time_secs","grammar","generations","population_size_components","mutation_rate_components","crossover_rate_components","population_size_ensemble","mutation_rate_ensemble","crossover_rate_ensemble","scoring","cv_evaluation_mode","cv_folds","test_size","evolution_steps_after_cv", "verbose", NULL}; //NULL-terminated array

    if(!PyArg_ParseTupleAndKeywords(args,kwargs,"|$iiOisiiddiddOpidii",keywords, &seed, &n_jobs, &timeout_pip_sec, &timeout_evolution_process_sec, &grammar_file, &generations, &size_pop_components, &mut_rate_components, &cross_rate_components, &size_pop_ensemble, &mut_rate_ensemble, &cross_rate_ensemble, &scoring, &cv_evaluation, &cv_folds, &test_size, &evolution_steps_after_cv, &verbose)) //Function and arguments |$ before keyword args
        return NULL;

    if(timeout_pip_sec==NULL)
        timeout_pip_sec=Py_BuildValue("i", 60);
    else if(timeout_pip_sec==Py_None || (PyLong_Check(timeout_pip_sec) && PyLong_AsLong(timeout_pip_sec)>0))
        Py_XINCREF(timeout_pip_sec);
    else{
        PyErr_SetString(PyExc_TypeError, "max_pipeline_time_secs must be an integer greater than zero or None");
        return NULL;
    }

    if(scoring==NULL)
        scoring=Py_BuildValue("s","balanced_accuracy");
    else 
        Py_XINCREF(scoring);

    if(self->ptr_autocve)
        delete self->ptr_autocve;

    try{
        self->ptr_autocve=new AUTOCVEClass(seed, n_jobs, timeout_pip_sec, timeout_evolution_process_sec, grammar_file, generations, size_pop_components, elite_portion_components, mut_rate_components, cross_rate_components, size_pop_ensemble, elite_portion_ensemble, mut_rate_ensemble, cross_rate_ensemble, scoring, cv_evaluation, cv_folds, test_size, evolution_steps_after_cv, verbose);
    }catch(const char *e){
        PyErr_SetString(PyExc_Exception, e);
        return NULL;
    }


    return 1;
}

static void PyAUTOCVE_dealloc(PyAUTOCVE * self){
    delete self->ptr_autocve;
    Py_TYPE(self)->tp_free(self);
}


static PyObject *PyAUTOCVE_optimize(PyAUTOCVE *self, PyObject *args, PyObject *kwargs){
    PyObject *data_X, *data_y;
    double subsample_data=1.0;

    static char *keywords[]={"X","y","subsample_data", NULL}; //NULL-terminated array

    int return_flag;

    if(!PyArg_ParseTupleAndKeywords(args,kwargs,"OO|$d",keywords ,&data_X, &data_y, &subsample_data)) //Function and arguments |$ before keyword args
        return NULL;
    try{
        if(!(return_flag = self->ptr_autocve->run_genetic_programming(data_X,data_y,subsample_data)))
            return NULL;
    }catch(const char *e){
        PyErr_SetString(PyExc_Exception, e);
        return NULL;
    }

    if(return_flag==-1)
        return Py_BuildValue("i",0);

    return Py_BuildValue("i",1);
}

static PyObject *PyAUTOCVE_get_best(PyAUTOCVE* self, PyObject* args){
    if(!PyArg_ParseTuple(args,"")) 
        return NULL;

    PyObject *best_pip=NULL;
    try{

        if(!(best_pip=self->ptr_autocve->get_best_pipeline()))
            return NULL;
    }catch(const char *e){
        PyErr_SetString(PyExc_Exception, e);
        return NULL;
    }

    return best_pip;
}

static PyObject *PyAUTOCVE_get_voting_ensemble_all(PyAUTOCVE* self, PyObject* args){
    if(!PyArg_ParseTuple(args,"")) 
        return NULL;

    PyObject *voting_ensemble=NULL;
    try{

        if(!(voting_ensemble=self->ptr_autocve->get_voting_ensemble_all()))
            return NULL;
    }catch(const char *e){
        PyErr_SetString(PyExc_Exception, e);
        return NULL;
    }

    return voting_ensemble;
}

static PyObject *PyAUTOCVE_get_voting_ensemble_best_mask(PyAUTOCVE* self, PyObject* args){
    if(!PyArg_ParseTuple(args,"")) 
        return NULL;

    PyObject *voting_ensemble=NULL;
    try{

        if(!(voting_ensemble=self->ptr_autocve->get_voting_ensemble_best_mask()))
            return NULL;
    }catch(const char *e){
        PyErr_SetString(PyExc_Exception, e);
        return NULL;
    }

    return voting_ensemble;
}

static PyObject *PyAUTOCVE_get_voting_ensemble_elite(PyAUTOCVE* self, PyObject* args){
    if(!PyArg_ParseTuple(args,"")) 
        return NULL;

    PyObject *voting_ensemble=NULL;
    try{

        if(!(voting_ensemble=self->ptr_autocve->get_voting_ensemble_elite()))
            return NULL;
    }catch(const char *e){
        PyErr_SetString(PyExc_Exception, e);
        return NULL;
    }

    return voting_ensemble;
}


static PyObject *PyAUTOCVE_get_grammar_char(PyAUTOCVE* self, PyObject* args){
    if(!PyArg_ParseTuple(args,"")) 
        return NULL;

    char *grammar_output=NULL;
    try{

        if(!(grammar_output=self->ptr_autocve->get_grammar_char()))
            return NULL;
    }catch(const char *e){
        PyErr_SetString(PyExc_Exception, e);
        return NULL;
    }

    PyObject *grammar_py=Py_BuildValue("s",grammar_output);
    free(grammar_output);
    return grammar_py;
}

static PyObject *PyAUTOCVE_get_parameters_char(PyAUTOCVE* self, PyObject* args){
    if(!PyArg_ParseTuple(args,"")) 
        return NULL;

    char *parameters_output=NULL;
    try{

        if(!(parameters_output=self->ptr_autocve->get_parameters_char()))
            return NULL;
    }catch(const char *e){
        PyErr_SetString(PyExc_Exception, e);
        return NULL;
    }

    PyObject *parameters_py=Py_BuildValue("s",parameters_output);
    free(parameters_output);
    return parameters_py;
}


static PyMethodDef PyAUTOCVE_methods[] = {
    { "optimize", (PyCFunction)PyAUTOCVE_optimize,METH_VARARGS | METH_KEYWORDS,"Optimize an ensemble to the (X,y) base. X and y expect to be numeric (used pandas.get_dummies otherwise)." },
    { "get_best_voting_ensemble", (PyCFunction)PyAUTOCVE_get_voting_ensemble_best_mask,METH_VARARGS,"Get the best ensemble produced in the optimization procedure (recommended option)." },
    { "get_best_pipeline", (PyCFunction)PyAUTOCVE_get_best,METH_VARARGS,"Get the pipeline with higher score in the last generation." },
    { "get_voting_ensemble_elite", (PyCFunction)PyAUTOCVE_get_voting_ensemble_elite,METH_VARARGS,"Get the ensemble compound by the 10% pipelines with higher score defined in the last generation." },
    { "get_voting_ensemble_all", (PyCFunction)PyAUTOCVE_get_voting_ensemble_all,METH_VARARGS,"Get the ensemble compound by all the pipelines defined in the last generation." },
    { "get_grammar", (PyCFunction)PyAUTOCVE_get_grammar_char,METH_VARARGS,"Get as text the grammar used in the optimization procedure." },
    { "get_parameters", (PyCFunction)PyAUTOCVE_get_parameters_char,METH_VARARGS,"Get as text the parameters used in the optimization procedure." },
    {NULL}  /* Sentinel */
};

static PyTypeObject PyAUTOCVEType = {PyVarObject_HEAD_INIT(NULL, 0)
                                    "AutoCVEClassifier"   /* tp_name */
};



static struct PyMethodDef AUTOCVEMethods[] = {
    /* The cast of the function (PyCFunction) is necessary since PyCFunction values
     * only take two PyObject* parameters, and keywdarg_parrot() takes
     * three.
     */
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


static PyModuleDef AUTOCVE_module = {
    PyModuleDef_HEAD_INIT,
    "AUTOCVE",   /* name of module */
    "Module developed to generate automatic ensembles", /* module documentation, may be NULL */
    -1,//sizeof(struct module_state),       /* size of per-interpreter state of the module,or -1 if the module keeps state in global variables. */
    AUTOCVEMethods
};






PyMODINIT_FUNC PyInit_AUTOCVE(void){
    PyObject* module_py;

    PyAUTOCVEType.tp_new = PyType_GenericNew;
    PyAUTOCVEType.tp_basicsize=sizeof(PyAUTOCVE);
    PyAUTOCVEType.tp_dealloc=(destructor) PyAUTOCVE_dealloc;
    PyAUTOCVEType.tp_flags=Py_TPFLAGS_DEFAULT;
    PyAUTOCVEType.tp_doc="AUTOCVE Classifier";
    PyAUTOCVEType.tp_methods=PyAUTOCVE_methods;
    PyAUTOCVEType.tp_init=(initproc)PyAUTOCVE_init;

    if (PyType_Ready(&PyAUTOCVEType) < 0)
        return NULL;

    module_py = PyModule_Create(&AUTOCVE_module);
    if (!module_py)
        return NULL;

    Py_INCREF(&PyAUTOCVEType);
    PyModule_AddObject(module_py, "AUTOCVEClassifier", (PyObject *)&PyAUTOCVEType); 
    return module_py;
}
