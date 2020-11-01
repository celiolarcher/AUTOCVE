#ifndef POPULATIONENSEMBLEH
#define POPULATIONENSEMBLEH

#include <Python.h>
#include <stdio.h>
#include <fstream>
#include "python_interface.h"

class Population;

class PopulationEnsemble{
    private: int **population;
    int **next_gen;
    int population_size;
    int next_gen_size;
    int solution_size;
    int elite_size;
    double mut_rate;
    double cross_rate;
    double *score_population;
    double *score_next_gen;
    int *id_population;
    int *id_next_gen;
    int *length_population;
    int *length_next_gen;

    private: static unsigned int solution_id_count;

    public: PopulationEnsemble(int population_size, int solution_size, double elite_portion, double mut_rate, double cross_rate);
    public: ~PopulationEnsemble();
    public: int init_population_random();
    public: int next_generation_similarity(Population *population_components, int generation, std::ofstream *competition_log);
    private: void breed(int *child1, int *child2);
    private: int mutation_bit_by_type(int *child);
    private: int crossover_without_zero_borders(int *child1, int *child2);
    private: int *copy(int *solution);
    private: double similarity(int *individual_1, int *individual_2);
    private: void update_length_population();
    private: void update_length_next_gen();

    public: int check_valid_individual(int i);
    public: int get_element_population_i(int i, int j);
    public: double get_score_population(int i);
    public: void set_score_population(int i, double score);
    public: int get_length_population(int i);
    public: int get_new_id();
    private: double get_score_next_gen(int i);
    private: int get_length_next_gen(int i);

    private: char *vector_to_string(int *individual);
    public: void write_population(int generation, int time,  std::ofstream *evolution_log);
    private: void write_competition(int generation, int index_population, int index_next_gen, std::ofstream *competition_log);
    public: void sort_population();

    public: int evaluate_score_cv(Population *population_components);
    public: int get_population_size();
};




#endif


