#ifndef POPULATIONH
#define POPULATIONH
#include "solution.h"
#include "grammar.h"
#include "python_interface.h"
#include "population_ensemble.h"
#include <fstream>

class Population{
    private: Solution** population;
    Solution** next_gen;
    int population_size;
    int next_gen_size;
    double cross_rate, mut_rate;
    int elite_size;
    int *population_rank;  //get the position rank of each solution in population
    double *score_population;
    double *metric_population;
    double *score_next_gen;
    double *metric_next_gen;
    double *predict_population; //in the form i*population_size+predict_map[j], i=0..predict_size, j=0..population_size
    double *predict_next_gen; //in the form i*population_size+j, i=0..predict_size, j=0..population_size
    double *similarity_matrix;
    double *similarity_matrix_next_gen;
    int predict_size;
    PythonInterface* interface_call;
    int cv_evaluation;

    public: Population(PythonInterface *interface, int size_pop, double elite_portion, double mut_rate, double cross_rate, int cv_evaluation);
    public: ~Population();
    public: int init_population(Grammar* grammar, PopulationEnsemble  *population_ensemble);
    public: int next_generation_selection_similarity(PopulationEnsemble *population_ensemble, int generation, std::ofstream *competition_log, std::ofstream *matrix_sim_next_gen_log);

    private: int evaluate_next_gen(int population_as_invalid_list);
    private: int evaluate_next_gen_cv(int population_as_buffer_option); //2 to use population as buffer of already seen solutions and 1 to use population as a list of invalid solutions
    private: int evaluate_population_cv();
    private: int evaluate_ensemble_next_gen(PopulationEnsemble *population_ensemble, int *map_next_gen_population, int changed_index);
    private: int evaluate_ensemble_population(PopulationEnsemble *population_ensemble);
    private: int update_population_score(PopulationEnsemble *population_ensemble);
    private: void breed(Solution* child1, Solution* child2);
    private: void sort_population();
    private: void update_population_with_next_gen(int population_index, int next_gen_index);
    private: double get_score_population(int i);
    private: void set_score_population(int i, double score);
    private: double get_metric_population(int i);
    private: void set_metric_population(int i, double metric);
    private: double get_score_next_gen(int i);
    private: void set_score_next_gen(int i, double score);
    private: double get_metric_next_gen(int i);
    private: void set_metric_next_gen(int i, double metric);
    private: double get_predict_population(int i, int j); //(sample_id, solution_id)
    private: void set_predict_population(int i, int j, double predict_value);   //(sample_id, solution_id)
    private: double get_predict_next_gen(int i, int j); //(sample_id, solution_id)
    private: void set_predict_next_gen(int i, int j, double predict_value);   //(sample_id, solution_id)
    private: double get_similarity_matrix(int i, int j);
    private: void set_similarity_matrix(int i, int j, double value);
    private: double get_similarity_matrix_next_gen(int i, int j);
    private: void set_similarity_matrix_next_gen(int i, int j, double value);
    private: void compute_similarity();
    private: int compute_similarity_next_gen();
    
    public: void write_population(int generation, std::ofstream *evolution_log);
    public: void write_next_gen(int generation, std::ofstream *evolution_log);
    public: void write_similarity_matrix(int generation, std::ofstream *evolution_log);
    public: void write_similarity_matrix_next_gen(int generation, std::ofstream *evolution_log);
    public: void write_competition(int generation, int index_population, int index_next_gen, std::ofstream *evolution_log);

    public: PyObject *get_solution_pipeline_rank_i(int i);
    public: PyObject *get_population_ensemble_all();
    public: PyObject *get_population_ensemble_elite();
    public: PyObject *get_population_ensemble_mask_i(PopulationEnsemble *population_ensemble, int id_ensemble);

    friend class PopulationEnsemble;
};

#endif
