#include "AUTOCVE.h"
#include "grammar.h"
#include "solution.h"
#include "python_interface.h"
#include "population.h"
#include "population_ensemble.h"
#include "utility.h"
#include <stdlib.h>  
#include <fstream>
#include <string.h>

#define BUFFER_SIZE 10000


AUTOCVEClass::AUTOCVEClass(int seed, int n_jobs, PyObject* timeout_pip_sec, int timeout_evolution_process_sec, char *grammar_file, int generations, int size_pop_components, double elite_portion_components, double mut_rate_components, double cross_rate_components, int size_pop_ensemble, double elite_portion_ensemble, double mut_rate_ensemble, double cross_rate_ensemble,  PyObject *scoring, int cv_folds, int verbose){
    this->seed=seed, this->n_jobs=n_jobs, this->timeout_pip_sec=timeout_pip_sec, this->timeout_evolution_process_sec=timeout_evolution_process_sec, this->grammar_file=AUTOCVEClass::grammar_file_handler(grammar_file), this->generations=generations, this->scoring=scoring, this->verbose=verbose;
    this->size_pop_components=size_pop_components, this->elite_portion_components=elite_portion_components, this->mut_rate_components=mut_rate_components, this->cross_rate_components=cross_rate_components;
    this->size_pop_ensemble=size_pop_ensemble, this->elite_portion_ensemble=elite_portion_ensemble, this->mut_rate_ensemble=mut_rate_ensemble, this->cross_rate_ensemble=cross_rate_ensemble;

    this->cv_folds=cv_folds;
    this->population=NULL, this->grammar=NULL, this->interface=NULL, this->population_ensemble=NULL;
    this->interface=new PythonInterface(this->n_jobs, this->timeout_pip_sec, this->scoring, this->cv_folds, this->verbose);
}

AUTOCVEClass::~AUTOCVEClass(){
    if(this->population)
        delete this->population;

    if(this->population_ensemble)
        delete this->population_ensemble;

    if(this->grammar)
        delete this->grammar;

    if(this->interface)
        delete this->interface;

    free(this->grammar_file);
   
    Py_XDECREF(this->timeout_pip_sec);
    Py_XDECREF(this->scoring);
}


int AUTOCVEClass::run_genetic_programming(PyObject *data_X, PyObject *data_y, double subsample_data){
    srand(this->seed);

    PySys_WriteStdout("LOADING DATASET\n");

    if(!this->interface->load_dataset(data_X, data_y, subsample_data))
        return NULL;

    PySys_WriteStdout("LOADED DATASET\n");

    if(this->grammar)
        delete this->grammar;

    if(this->population)
        delete this->population;

    if(this->population_ensemble)
        delete this->population_ensemble;

    this->grammar=new Grammar(this->grammar_file,this->interface);

    std::ofstream evolution_log;
    evolution_log.open("evolution.log");

    if(!evolution_log.is_open())
        throw "Cannot create evolution.log file\n";

    std::ofstream matrix_sim_log;
    matrix_sim_log.open("matrix_sim.log");

    if(!matrix_sim_log.is_open())
        throw "Cannot create matrix_sim.log file\n";

    std::ofstream evolution_ensemble_log;
    evolution_ensemble_log.open("evolution_ensemble.log");

    if(!evolution_ensemble_log.is_open())
        throw "Cannot create evolution_ensemble.log file\n";

    struct timeval start, end;
    gettimeofday(&start, NULL);

    PySys_WriteStdout("GENERATION %d\n",0);
    this->population=new Population(this->interface, this->size_pop_components, this->elite_portion_components, this->mut_rate_components, this->cross_rate_components);
    this->population_ensemble=new PopulationEnsemble(this->size_pop_ensemble,this->size_pop_components,this->elite_portion_ensemble,this->mut_rate_ensemble,this->cross_rate_ensemble);

    this->population_ensemble->init_population_random();

    int return_flag=this->population->init_population(this->grammar, this->population_ensemble);
    if(!return_flag)
        return NULL;
    if(return_flag==-1)
        throw "Population not initialized\n";

    evolution_log<<"Generation;ID_solution;Pipeline;Score;Metric\n";
    this->population->write_population(0,&evolution_log);

    matrix_sim_log<<"Generation;ID_solution;ID_solution;Similarity\n";
    this->population->write_similarity_matrix(0,&matrix_sim_log);

    evolution_ensemble_log<<"Generation;Length;Score\n";
    this->population_ensemble->write_population(0,&evolution_ensemble_log);

    gettimeofday(&end, NULL);
    int control_flag;

    double generation_time=0;
    for(int i=0;i<this->generations;i++){
        PySys_WriteStdout("GENERATION %d (%d secs)\n",i+1,(int)(end.tv_sec-start.tv_sec));

        if(!(control_flag=this->population->next_generation_selection_similarity(this->population_ensemble)))
            return NULL;

        this->population_ensemble->next_generation_similarity(this->population);

        struct timeval auxiliar_time;
        gettimeofday(&auxiliar_time, NULL);
        generation_time=auxiliar_time.tv_sec-end.tv_sec;

        gettimeofday(&end, NULL);

        if(this->timeout_evolution_process_sec && (end.tv_sec-start.tv_sec)>=this->timeout_evolution_process_sec-generation_time) control_flag=-1;

        this->population->write_population(i+1,&evolution_log);
        this->population->write_similarity_matrix(i+1,&matrix_sim_log);
        this->population_ensemble->write_population(i+1,&evolution_ensemble_log);

        if(control_flag==-1)//KeyboardException or timeout verified
            break;
    }

    PySys_WriteStdout("END PROCESS (%d secs)\n",(int)(end.tv_sec-start.tv_sec));

    evolution_log.close();
    matrix_sim_log.close();
    evolution_ensemble_log.close();

    if(!this->interface->unload_dataset())
        return NULL;

    return 1;
}

PyObject *AUTOCVEClass::get_best_pipeline(){
    if(!this->population)
        throw "Error: Need to call optimize first.";

    return this->population->get_solution_pipeline_rank_i(0);
}

PyObject *AUTOCVEClass::get_voting_ensemble_all(){
    if(!this->population)
        throw "Error: Need to call optimize first.";

    return this->population->get_population_ensemble_all();
}

PyObject *AUTOCVEClass::get_voting_ensemble_elite(){
    if(!this->population)
        throw "Error: Need to call optimize first.";

    return this->population->get_population_ensemble_elite();
}

PyObject *AUTOCVEClass::get_voting_ensemble_best_mask(){
    if(!this->population || !this->population_ensemble)
        throw "Error: Need to call optimize first.";

    return this->population->get_population_ensemble_mask_i(this->population_ensemble,0);
}


char *AUTOCVEClass::get_grammar_char(){
    if(!this->grammar)
        throw "Error: Need to call optimize first.";

    return this->grammar->print_grammar();
}

char *AUTOCVEClass::get_parameters_char(){
    char *parameters=NULL, buffer[BUFFER_SIZE];

    sprintf(buffer, "%d", this->seed);
    parameters=char_concat(char_concat(parameters, "random_state: "), buffer);

    sprintf(buffer, "%d", this->n_jobs);
    parameters=char_concat(char_concat(parameters, ", n_jobs: "), buffer);

    if(this->timeout_pip_sec==Py_None)
        sprintf(buffer, "%s", "None");
    else
        sprintf(buffer, "%ld", PyLong_AsLong(this->timeout_pip_sec));
    parameters=char_concat(char_concat(parameters, ", max_pipeline_time_secs: "),  buffer);

    sprintf(buffer, "%d", this->timeout_evolution_process_sec);
    parameters=char_concat(char_concat(parameters, ", max_evolution_time_sec: "), buffer);

    parameters=char_concat(char_concat(parameters, ", grammar: "), this->grammar_file);

    sprintf(buffer, "%d", this->generations);
    parameters=char_concat(char_concat(parameters, ", generations: "), buffer);

    sprintf(buffer, "%d", this->size_pop_components);
    parameters=char_concat(char_concat(parameters, ", population_size_components: "), buffer);

    sprintf(buffer, "%.2f", this->mut_rate_components);
    parameters=char_concat(char_concat(parameters, ", mutation_rate_components: "), buffer);

    sprintf(buffer, "%.2f", this->cross_rate_components);
    parameters=char_concat(char_concat(parameters, ", crossover_rate_components: "), buffer);

    sprintf(buffer, "%d", this->size_pop_ensemble);
    parameters=char_concat(char_concat(parameters, ", population_size_ensemble: "), buffer);

    sprintf(buffer, "%.2f", this->mut_rate_ensemble);
    parameters=char_concat(char_concat(parameters, ", mutation_rate_ensemble: "), buffer);

    sprintf(buffer, "%.2f", this->cross_rate_ensemble);
    parameters=char_concat(char_concat(parameters, ", crossover_rate_ensemble: "), buffer);

    PyObject *repr_function, *repr_return;
    if(!(repr_function=PyObject_GetAttrString(this->scoring,(char *)"__repr__")))
        return NULL;
    if(!(repr_return=PyObject_CallObject(repr_function, NULL)))
        return NULL;
    const char *repr_char;
    if(!(repr_char=PyUnicode_AsUTF8(repr_return)))
        return NULL;
    parameters=char_concat(char_concat(parameters, ", scoring: "), repr_char);

    sprintf(buffer, "%d", this->cv_folds);
    parameters=char_concat(char_concat(parameters, ", cv_folds: "), buffer);

    sprintf(buffer, "%d", this->verbose);
    parameters=char_concat(char_concat(parameters, ", verbose: "), buffer);

    return parameters;
}

char* AUTOCVEClass::grammar_file_handler(char *grammar_file_param){
    char *grammar_file_return;

    if(!strchr(grammar_file_param,'/') && !strchr(grammar_file_param,'\\')){
        PyObject *autocve_module = PyImport_ImportModule("AUTOCVE.AUTOCVE");
        PyObject *path = PyObject_GetAttrString(autocve_module, "__file__");
        const char *path_char= PyUnicode_AsUTF8(path);

        grammar_file_return=(char*)malloc(sizeof(char)*(strlen(path_char)+1));
        strcpy(grammar_file_return,path_char);
        char *last_bar=strrchr(grammar_file_return,'/');    
        *last_bar='\0';
        grammar_file_return=char_concat(grammar_file_return,"/grammar/");
        grammar_file_return=char_concat(grammar_file_return, grammar_file_param);

        Py_XDECREF(path);
        Py_XDECREF(autocve_module); 

    }else{
        grammar_file_return=(char*)malloc(sizeof(char)*(strlen(grammar_file_param)+1));
        strcpy(grammar_file_return, grammar_file_param);
    }

    return grammar_file_return;
}

