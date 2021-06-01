#include "population_ensemble.h"
#include "utility.h"
#include "population.h"
#include <iostream>
#define K_TOURNAMENT 2
#define BUFFER_SIZE 1000

unsigned int PopulationEnsemble::solution_id_count=0;

PopulationEnsemble::PopulationEnsemble(int population_size, int solution_size, double elite_portion, double mut_rate, double cross_rate){
    PopulationEnsemble::solution_id_count = 0;

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

    this->id_population=(int*)malloc(sizeof(int)*this->population_size);
    for(int i=0; i<this->population_size; i++) this->id_population[i]=PopulationEnsemble::get_new_id();

    this->id_next_gen=NULL;

    this->length_population=(int*)malloc(sizeof(int)*this->population_size);
    this->length_next_gen=NULL;
}

int PopulationEnsemble::init_population_random(){ 
    for(int i=0;i<this->population_size;i++){
        int no_1_set_flag = 1;
        for(int j=0;j<this->solution_size;j++){
            double sort=randDouble(0,1);
            if(sort<0.1){
                this->population[i][j]=1;
                no_1_set_flag = 0;
            }
            else
                this->population[i][j]=0;
        }
        if(no_1_set_flag)
            this->population[i][randInt(0,this->solution_size-1)]=1;
    }

    this->update_length_population();
    for(int i=0; i<this->population_size; i++) this->set_score_population(i, 0);

    return 1;
}


PopulationEnsemble::~PopulationEnsemble(){
    for(int i=0;i<this->population_size;i++)
        free(this->population[i]);
    
    free(this->population);
    free(this->score_population);
    free(this->id_population);
    free(this->length_population);    

    if(this->next_gen)
        free(this->next_gen);
    if(this->score_next_gen)
        free(this->score_next_gen);
    if(this->id_next_gen)
        free(this->id_next_gen);
    if(this->length_next_gen)
        free(this->length_next_gen);
}

int PopulationEnsemble::evaluate_score_cv(Population *population_components){
    int return_flag=population_components->evaluate_population_cv();
    if(!return_flag || return_flag==-1)
        return return_flag;

    population_components->evaluate_ensemble_population(this);
    this->sort_population();

    return 1;
}


int PopulationEnsemble::next_generation_similarity(Population *population_components, int generation, std::ofstream *competition_log){
    if(this->next_gen_size!=this->population_size){
        if(this->next_gen) free(this->next_gen);
        if(this->score_next_gen) free(this->score_next_gen);
        if(this->id_next_gen) free(this->id_next_gen);
        if(this->length_next_gen) free(this->length_next_gen);


        this->next_gen_size=this->population_size;
        this->next_gen=(int**)malloc(sizeof(int*)*this->next_gen_size);
        this->score_next_gen=(double*)malloc(sizeof(double)*this->next_gen_size);
        this->id_next_gen=(int*)malloc(sizeof(int)*this->next_gen_size);
        this->length_next_gen=(int*)malloc(sizeof(int)*this->next_gen_size);

        for(int i=0;i<this->next_gen_size;i++)
            this->next_gen[i]=NULL;
    }

    this->sort_population();
    int *choice_mask=(int*) malloc(sizeof(int)*this->population_size);
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
        this->id_next_gen[i]=PopulationEnsemble::get_new_id();

        if(i<this->next_gen_size -1){
            this->next_gen[i+1]=child2;
            this->id_next_gen[i+1]=PopulationEnsemble::get_new_id();
        }
        else
            delete child2;
    }

    this->update_length_next_gen();

    std::swap(this->population,this->next_gen);
    std::swap(this->score_population,this->score_next_gen);
    std::swap(this->id_population,this->id_next_gen);
    std::swap(this->length_population,this->length_next_gen);
    std::swap(this->population_size,this->next_gen_size);

    population_components->evaluate_ensemble_population(this);

    std::swap(this->population,this->next_gen);
    std::swap(this->score_population,this->score_next_gen);
    std::swap(this->id_population,this->id_next_gen);
    std::swap(this->length_population,this->length_next_gen);
    std::swap(this->population_size,this->next_gen_size);

    for(int i=0;i<this->population_size;i++) choice_mask[i]=0;

    for(int iterator=0;iterator<this->population_size;iterator++){
        int i;
        do{
            i=randInt(0,this->population_size-1);
        }while(choice_mask[i]);

        choice_mask[i]=1;

        int max_sim_index=-1;
        double max_similarity=0;
        for(int j=0;j<this->next_gen_size;j++){
            if(!this->next_gen[j]) continue;
            double similarity=this->similarity(this->population[i],this->next_gen[j]);

            if(max_sim_index==-1 || similarity>max_similarity){
                max_similarity=similarity;
                max_sim_index=j;
            }
        }

        if(max_sim_index==-1) continue;

        if(competition_log!=NULL)
            this->write_competition(generation, i, max_sim_index, competition_log);

        if(this->get_score_population(i)<this->get_score_next_gen(max_sim_index) || (this->get_score_population(i)==this->get_score_next_gen(max_sim_index) && this->get_length_next_gen(max_sim_index)>0 && this->get_length_next_gen(max_sim_index)<this->get_length_population(i))){
            delete this->population[i];
            this->population[i]=NULL;
            std::swap(this->population[i],this->next_gen[max_sim_index]);
            std::swap(this->id_population[i],this->id_next_gen[max_sim_index]);
            std::swap(this->score_population[i],this->score_next_gen[max_sim_index]);
            std::swap(this->length_population[i],this->length_next_gen[max_sim_index]);
        }else{
            delete this->next_gen[max_sim_index];
            this->next_gen[max_sim_index]=NULL;
        }
    }

    for(int i=0;i<this->next_gen_size;i++)
        if(this->next_gen[i])delete this->next_gen[i];

    this->sort_population();

    population_components->update_population_score(this);
    free(choice_mask);

    return 1;
}


void PopulationEnsemble::breed(int *child1, int *child2){
    double prob=randDouble(0,1);
    
    int *parent_1_copy = PopulationEnsemble::copy(child1);
    int *parent_2_copy = PopulationEnsemble::copy(child2);
    
    if(PopulationEnsemble::similarity(child1, child2) >= 0.99){
        this->mutation_bit_by_type(child1);
        this->mutation_bit_by_type(child2);
    }

    if(prob<cross_rate)
        this->crossover_without_zero_borders(child1,child2);

    prob=randDouble(0,1);
    if(prob<mut_rate){
        this->mutation_bit_by_type(child1);
        this->mutation_bit_by_type(child2);
    }
      
    if(PopulationEnsemble::similarity(child1, parent_1_copy) >= 0.99 || PopulationEnsemble::similarity(child1, parent_2_copy) >= 0.99)
        this->mutation_bit_by_type(child1);
        
    if(PopulationEnsemble::similarity(child2, parent_1_copy) >= 0.99 || PopulationEnsemble::similarity(child2, parent_2_copy) >= 0.99)
        this->mutation_bit_by_type(child2); 

    delete parent_1_copy;
    delete parent_2_copy;

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

int PopulationEnsemble::crossover_without_zero_borders(int *child1, int *child2){
    int begin_position;
    int end_position;

    for(int i=0; i<this->solution_size;i++){
        if(child1[i] || child2[i]){
            begin_position = i;
            break;
        }
    }

    for(int i=this->solution_size-1; i>=0; i--){
        if(child1[i] || child2[i]){
            end_position = i;
            break;
        }
    }    
       
    if(begin_position >= end_position)
        return 0;

    int place=randInt(begin_position+1,end_position);
    for(int i=place;i<end_position+1;i++)
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
    
    int length_ind = 0, length_ind_1 = 0, length_ind_2 = 0;
    for(int i=0;i<this->solution_size;i++){
        if(individual_1[i]) length_ind_1++;
        if(individual_2[i]) length_ind_2++;
    }

    if(length_ind_1 == 0 && length_ind_2 == 0)
        return 1.0;

    if(length_ind_1 > length_ind_2)
        similarity/=length_ind_1;
    else
        similarity/=length_ind_2;

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

void PopulationEnsemble::update_length_next_gen(){
    for(int i=0;i<this->next_gen_size;i++){
        int length_solution=0;
        if(this->next_gen[i]==NULL) continue;

        for(int j=0;j<this->solution_size;j++){
            if(this->next_gen[i][j])length_solution++;
        }
        this->length_next_gen[i]=length_solution;
    }
}

char *PopulationEnsemble::vector_to_string(int *individual){
    char buffer[BUFFER_SIZE];
    char *vector_string=NULL;

    for(int j=0; j<this->solution_size;j++){
        if(individual[j]){                
            if(vector_string)
                vector_string = char_concat(vector_string, ",");
            
            sprintf(buffer, "%d", j);
            vector_string=char_concat(vector_string, buffer);
        }
    }
    if(vector_string==NULL) 
        vector_string=char_concat(vector_string, "");

    return vector_string;
}

void PopulationEnsemble::write_population(int generation, int time, std::ofstream *evolution_log){
    for(int i=0;i<this->population_size;i++){
        char *vector_string = this->vector_to_string(this->population[i]);

        (*evolution_log)<<generation<<";"<<time<<";"<<this->id_population[i]<<";"<<this->length_population[i]<<";"<<this->score_population[i]<<";"<<vector_string<<"\n";

        free(vector_string);
    }
}

void PopulationEnsemble::write_competition(int generation, int index_population, int index_next_gen, std::ofstream *competition_log){
    char *vector_string_population = this->vector_to_string(this->population[index_population]);
    char *vector_string_next_gen = this->vector_to_string(this->next_gen[index_next_gen]);

    (*competition_log)<<generation<<";"<<this->id_population[index_population]<<";"<<this->length_population[index_population]<<";"<<this->score_population[index_population]<<";"<<vector_string_population<<";";
    (*competition_log)<<this->id_next_gen[index_next_gen]<<";"<<this->length_next_gen[index_next_gen]<<";"<<this->score_next_gen[index_next_gen]<<";"<<vector_string_next_gen<<"\n";

    free(vector_string_population);
    free(vector_string_next_gen);
}


int PopulationEnsemble::get_population_size(){
    return this->population_size;
}

int PopulationEnsemble::check_valid_individual(int i){
    if(i<0 || i>this->population_size)
        throw "Invalid population ensemble index";

    if(this->population[i])
        return 1;

    return 0;
}

int PopulationEnsemble::get_element_population_i(int i, int j){
    if(i<0 || i>this->population_size)
        throw "Invalid population ensemble index";

    if(j<0 || j>this->solution_size)
        throw "Invalid solution index in population";

    return this->population[i][j];
}

double PopulationEnsemble::get_score_population(int i){
    if(i<0 || i>this->population_size)
        throw "Invalid population ensemble index";

    return this->score_population[i];
}

void PopulationEnsemble::set_score_population(int i, double score){
    if(i<0 || i>this->population_size)
        throw "Invalid population ensemble index";

    this->score_population[i] = score;
}


int PopulationEnsemble::get_length_population(int i){
    if(i<0 || i>this->population_size)
        throw "Invalid population ensemble index";

    return this->length_population[i];
}

int PopulationEnsemble::get_new_id(){
    return PopulationEnsemble::solution_id_count++;
}

double PopulationEnsemble::get_score_next_gen(int i){
    if(i<0 || i>this->next_gen_size)
        throw "Invalid next_gen ensemble index";

    return this->score_next_gen[i];
}

int PopulationEnsemble::get_length_next_gen(int i){
    if(i<0 || i>this->next_gen_size)
        throw "Invalid next_gen ensemble index";

    return this->length_next_gen[i];
}


void insertionsort(int **pop, int *id, double *values, int *length, int size);

void PopulationEnsemble::sort_population(){
     insertionsort(this->population, this->id_population, this->score_population, this->length_population, this->population_size); 
}

/*Order from max to min in score and min to max in length*/
void insertionsort(int **pop, int *id, double *values, int *length, int size){
    int i = 1;

    while(i < size){
        int j = i;

        while(j > 0 && (values[j-1] < values[j] || (values[j-1] == values[j] && length[j-1] > length[j]))){
            std::swap(pop[j-1],pop[j]);
            std::swap(values[j-1], values[j]);
            std::swap(length[j-1], length[j]);
            std::swap(id[j-1], id[j]);

            j--;
        }

        i++;
    }
}