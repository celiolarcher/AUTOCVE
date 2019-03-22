#include "population_ensemble.h"
#include "utility.h"
#include "population.h"
#include <iostream>
#define K_TOURNAMENT 2

PopulationEnsemble::PopulationEnsemble(int population_size, int solution_size, double elite_portion, double mut_rate, double cross_rate){
    this->population_size=population_size;
    this->solution_size=solution_size;
    this->elite_size=this->population_size*elite_portion;
    this->next_gen_size=0;
    this->mut_rate=mut_rate;
    this->cross_rate=cross_rate;

    this->population=(int**)malloc(sizeof(int*)*this->population_size);
    for(int i=0;i<this->population_size;i++)
            this->population[i]=(int*)malloc(sizeof(int)*this->solution_size);
    this->next_gen=NULL;

    this->score_population=(double*)malloc(sizeof(double)*this->population_size);
    this->score_next_gen=NULL;

    this->length_population=(int*)malloc(sizeof(int)*this->population_size);
}

int PopulationEnsemble::init_population_random(){ 
    for(int i=0;i<this->population_size;i++){
        for(int j=0;j<this->solution_size;j++){
            double sort=randDouble(0,1);
            if(sort<0.1)
                this->population[i][j]=1;
            else
                this->population[i][j]=0;
        }
    }

    this->update_length_population();
    for(int i=0; i<this->population_size; i++) this->score_population[i]=0;

    return 1;
}


PopulationEnsemble::~PopulationEnsemble(){
    for(int i=0;i<this->population_size;i++)
        free(this->population[i]);
    
    free(this->population);
    free(this->score_population);
    free(this->length_population);    

    if(this->next_gen)
        free(this->next_gen);
    if(this->score_next_gen)
        free(this->score_next_gen);
}

int PopulationEnsemble::next_generation_similarity(Population *population_components){
    if(this->next_gen_size!=this->population_size){
        if(this->next_gen) free(this->next_gen);
        if(this->score_next_gen) free(this->score_next_gen);
        this->next_gen_size=this->population_size;
        this->next_gen=(int**)malloc(sizeof(int*)*this->next_gen_size);
        this->score_next_gen=(double*)malloc(sizeof(double)*this->next_gen_size);
        for(int i=0;i<this->next_gen_size;i++)
            this->next_gen[i]=NULL;
    }

    this->quick_sort_population();
    int choice_mask[this->population_size];
    for(int i=0;i<this->population_size;i++)choice_mask[i]=0;

    for(int i=0;i<this->next_gen_size;i+=2){
        int parent1, parent2;

        parent1=randInt(0,this->population_size-1-i);
        int sort_position=0;
        for(int count_position=0;count_position<=parent1;sort_position++)
            if(!choice_mask[sort_position])count_position++;
        parent1=sort_position-1;
        choice_mask[parent1]=1;

        if(i<this->next_gen_size-1){
           parent2=randInt(0,this->population_size-1-i-1);                
           sort_position=0;
           for(int count_position=0;count_position<=parent2;sort_position++)
               if(!choice_mask[sort_position])count_position++;
           parent2=sort_position-1;
           choice_mask[parent2]=1;
        }else //If size of population is odd, a solution of population is sorted to complete the pairing procedure
           parent2=randInt(0,this->population_size-1);                

        int *child1, *child2;

        child1=this->copy(this->population[parent1]);
        child2=this->copy(this->population[parent2]);

        this->breed(child1,child2);

        this->next_gen[i]=child1;

        if(i<this->next_gen_size-1)        
            this->next_gen[i+1]=child2;
        else
            delete child2;
    }

    std::swap(this->population,this->next_gen);
    std::swap(this->score_population,this->score_next_gen);
    std::swap(this->population_size,this->next_gen_size);

    population_components->evaluate_ensemble_population(this);

    std::swap(this->population,this->next_gen);
    std::swap(this->score_population,this->score_next_gen);
    std::swap(this->population_size,this->next_gen_size);

    for(int i=0;i<this->population_size;i++){
        int max_sim_index=-1;
        double max_similarity=0;
        for(int j=0;j<this->next_gen_size;j++){
            if(!this->next_gen[j])continue;
            double similarity=this->similarity(this->population[i],this->next_gen[j]);

            if(max_sim_index==-1 || similarity>max_similarity){
                max_similarity=similarity;
                max_sim_index=j;
            }
        }

        int next_length_solution=0;
        for(int k=0;k<this->solution_size;k++){
            if(this->next_gen[max_sim_index][k])next_length_solution++;
        }

        if(this->score_population[i]<this->score_next_gen[max_sim_index] || (this->score_population[i]==this->score_next_gen[max_sim_index] && next_length_solution<this->length_population[i])){
            delete this->population[i];
            this->population[i]=NULL;
            std::swap(population[i],next_gen[max_sim_index]);
            std::swap(score_population[i],score_next_gen[max_sim_index]);
        }else{
            delete this->next_gen[max_sim_index];
            this->next_gen[max_sim_index]=NULL;
        }
    }

    for(int i=0;i<this->next_gen_size;i++)
        if(this->next_gen[i])delete this->next_gen[i];

    this->update_length_population();
    this->quick_sort_population();

    return 1;
}

void PopulationEnsemble::breed(int *child1, int *child2){
    double prob=randDouble(0,1);
    if(prob<cross_rate)
        this->crossover(child1,child2);

    prob=randDouble(0,1);
    if(prob<mut_rate){
        this->mutation_bit_by_type(child1);
        this->mutation_bit_by_type(child2);
     }
}

int PopulationEnsemble::mutation_bit_by_type(int *individual){
    int length_solution=0;
    for(int i=0;i<this->solution_size;i++){
        if(individual[i])length_solution++;
    }

    int choice_value=randInt(0,1);
    if(choice_value && length_solution>1){
        int count_position=randInt(0,length_solution-1);

        int place=0;
        for(int count_ones=0;count_ones<=count_position;place++){
            if(individual[place])count_ones++;
        }
        place--;

        individual[place]=!individual[place];
    }else{
        int count_position=randInt(0,this->solution_size-length_solution-1);

        int place=0;
        for(int count_zeros=0;count_zeros<=count_position;place++){
            if(!individual[place])count_zeros++;
        }
        place--;

        individual[place]=!individual[place];
    }

    return 1;
}

int PopulationEnsemble::crossover(int *child1, int *child2){
    int place=randInt(0,this->solution_size-1);
    for(int i=place;i<this->solution_size;i++)
        std::swap(child1[i],child2[i]);

    return 1;
}

int *PopulationEnsemble::copy(int *solution){
    int *solution_copy=(int*)malloc(sizeof(int)*this->solution_size);
    for(int i=0;i<this->solution_size;i++)
        solution_copy[i]=solution[i];

    return solution_copy;
}

double PopulationEnsemble::similarity(int *individual_1, int *individual_2){
    double similarity=0;
    for(int i=0;i<this->solution_size;i++)
        if(individual_1[i] && individual_1[i]==individual_2[i])similarity++;
    
    similarity/=this->solution_size;

    return similarity;
}

void PopulationEnsemble::update_length_population(){
    for(int i=0;i<this->population_size;i++){
        int length_solution=0;
        for(int j=0;j<this->solution_size;j++){
            if(this->population[i][j])length_solution++;
        }
        this->length_population[i]=length_solution;
    }
}

void PopulationEnsemble::write_population(int generation, std::ofstream *evolution_log){
    for(int i=0;i<this->population_size;i++)
        (*evolution_log)<<generation<<";"<<this->length_population[i]<<";"<<this->score_population[i]<<"\n";
}

int PopulationEnsemble::get_population_size(){
    return this->population_size;
}


void quicksort(int **pop, double *values, int *length,  int begin, int end);

void PopulationEnsemble::quick_sort_population(){
     quicksort(this->population,this->score_population, this->length_population,0,this->population_size);   
}

/*Order from max to min in score and min to max in length*/
void quicksort(int **pop, double *values, int *length, int begin, int end){
	int i, j;
    double pivo;
    int pivo_length;
	i = begin;
	j = end-1;
	pivo = values[(begin + end) / 2];
    pivo_length = length[(begin + end) / 2];
	while(i <= j){
		while((values[i] > pivo || (values[i] == pivo && length[i] < pivo_length)) && i < end)
			i++;
		
		while((values[j] < pivo || (values[j] == pivo && length[j] > pivo_length)) && j > begin)
			j--;
		
		if(i <= j){
            std::swap(pop[i],pop[j]);
            std::swap(values[i], values[j]);
            std::swap(length[i], length[j]);
			i++;
			j--;
		}
	}
	if(j > begin)
		quicksort(pop, values, length, begin, j+1);
	if(i < end)
		quicksort(pop, values, length, i, end);
}
