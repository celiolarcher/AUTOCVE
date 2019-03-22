#ifndef POPULATIONENSEMBLEH
#define POPULATIONENSEMBLEH

#include <Python.h>
#include <stdio.h>
#include <fstream>
#include "python_interface.h"

class Population;

class PopulationEnsemble{
    public: int **population;
    int **next_gen;
    int population_size;
    int next_gen_size;
    int solution_size;
    int elite_size;
    double mut_rate;
    double cross_rate;
    double *score_population;
    double *score_next_gen;
    int *length_population;

    public: PopulationEnsemble(int population_size, int solution_size, double elite_portion, double mut_rate, double cross_rate);
    public: ~PopulationEnsemble();
    public: int init_population_random();
    public: int next_generation_similarity(Population *population_components);
    private: void breed(int *child1, int *child2);
    private: int mutation_bit_by_type(int *child);
    private: int crossover(int *child1, int *child2);
    private: int *copy(int *solution);
    private: double similarity(int *individual_1, int *individual_2);
    private: void update_length_population();
    public: void write_population(int generation, std::ofstream *evolution_log);
    private: void quick_sort_population();

    public: int get_population_size();
};




#endif


