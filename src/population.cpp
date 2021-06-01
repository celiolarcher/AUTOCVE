#include "population.h"
#include "utility.h"
#include <stdlib.h>
#include <math.h>

#define INVALID_INDIVIDUAL_SCORE -1e10
#define INVALID_SIMILARITY_PREDICT -1
#define MAX_SIM 1

Population::Population(PythonInterface *interface, int size_pop, double elite_portion, double mut_rate, double cross_rate, int cv_evaluation){
    Solution::reset_index();

    this->population_size=size_pop;
    this->next_gen_size=0;
    this->population=(Solution**)malloc(sizeof(Solution*)*this->population_size);
    this->next_gen=NULL;
    this->population_rank=(int*)malloc(sizeof(int)*this->population_size);
    this->score_population=(double*)malloc(sizeof(double)*this->population_size);
    this->metric_population=(double*)malloc(sizeof(double)*this->population_size);
    this->score_next_gen=NULL;
    this->metric_next_gen=NULL;
    this->predict_population=NULL;  
    this->predict_next_gen=NULL;  
    this->similarity_matrix=(double*)malloc(sizeof(double)*this->population_size*(this->population_size-1)/2);
    this->similarity_matrix_next_gen=NULL;
    this->predict_size=0;
    this->cross_rate=cross_rate, this->mut_rate=mut_rate, this->elite_size=size_pop*elite_portion;
    this->interface_call=interface;
    this->cv_evaluation=cv_evaluation;

    for(int i=0;i<this->population_size;i++)
        this->population[i]=NULL;

    if(_import_array() < 0)//Must be called just one time
        throw  "Library numpy.core.multiarray failed to import";
}

int Population::init_population(Grammar* grammar, PopulationEnsemble *population_ensemble){
    if(this->next_gen_size){  //Adjusting next_gen size by next_generation procedure
        free(this->next_gen);
        free(this->score_next_gen);
        free(this->metric_next_gen);
    }

    this->next_gen_size=this->population_size;
    this->next_gen=(Solution**)malloc(sizeof(Solution*)*this->next_gen_size);
    this->score_next_gen=(double*)malloc(sizeof(double)*this->next_gen_size);
    this->metric_next_gen=(double*)malloc(sizeof(double)*this->next_gen_size);

    for(int i=0;i<this->next_gen_size;i++){
        this->next_gen[i]=new Solution(grammar);
        this->next_gen[i]->init_solution();
    }

    int return_flag;
    if(this->cv_evaluation)
        return_flag=this->evaluate_next_gen_cv(0);
    else
        return_flag=this->evaluate_next_gen(false);

    if(!return_flag || return_flag==-1)
        return return_flag;
    
    std::swap(this->population,this->next_gen);
    std::swap(this->predict_population,this->predict_next_gen);
    std::swap(this->score_population,this->score_next_gen);
    std::swap(this->metric_population,this->metric_next_gen);
    for(int i=0;i<this->population_size;i++) this->population_rank[i]=i;

    if(!this->evaluate_ensemble_population(population_ensemble))
        return NULL;

    this->compute_similarity();
    this->sort_population();
    population_ensemble->sort_population();

    return 1;
}

Population::~Population(){
    for(int i=0;i<this->population_size;i++){
        if(this->population[i])
            delete this->population[i];
    }
    for(int i=0;i<this->next_gen_size;i++){
        if(this->next_gen[i])
            delete this->next_gen[i];
    }

    if(this->predict_population)
        free(this->predict_population);

    if(this->predict_next_gen)
        free(this->predict_next_gen);

    if(this->next_gen) 
        free(this->next_gen);

    if(this->score_next_gen) 
        free(this->score_next_gen);

    if(this->metric_next_gen) 
        free(this->metric_next_gen);

    if(this->similarity_matrix_next_gen)
        free(this->similarity_matrix_next_gen);

    free(this->population);
    free(this->population_rank);
    free(this->similarity_matrix);
    free(this->score_population);
    free(this->metric_population);
}

void Population::breed(Solution* child1, Solution* child2){
    Solution *parent_1_copy = child1->copy();
    Solution *parent_2_copy = child2->copy();
    
    if(!strcmp(child1->get_string_code(),child2->get_string_code())){
        Solution::mutation(child1);
        Solution::mutation(child2);
    }

    double prob=randDouble(0,1);

    if(prob<cross_rate)
        Solution::crossover(child1,child2);

    prob=randDouble(0,1);
    if(prob<mut_rate){
        Solution::mutation(child1);
        Solution::mutation(child2);
     }

    if(!strcmp(child1->get_string_code(), parent_1_copy->get_string_code()) || !strcmp(child1->get_string_code(), parent_2_copy->get_string_code()))
        Solution::mutation(child1);
    

    if(!strcmp(child2->get_string_code(), parent_1_copy->get_string_code()) || !strcmp(child2->get_string_code(), parent_2_copy->get_string_code()))
        Solution::mutation(child2);
    
    delete parent_1_copy;
    delete parent_2_copy;
}

int Population::evaluate_next_gen(int population_as_invalid_list){
    char *pipeline_string=NULL;
    int *map_evaluation=(int*)malloc(sizeof(int)*this->next_gen_size);

    int evaluation_count=0;

    for(int i=0;i<this->next_gen_size;i++){
        int flag_computed_already=0;

        if(this->next_gen[i]==NULL) continue;
        
        if(population_as_invalid_list){
            for(int j=0;j<this->population_size;j++){
                if(this->population[j] && this->get_score_population(j)==INVALID_INDIVIDUAL_SCORE && !strcmp(this->next_gen[i]->get_string_code(),this->population[j]->get_string_code())){        
                    map_evaluation[i]=-1;		        
                    flag_computed_already=1;
                    break;
                }
            }
        }

        if(flag_computed_already) continue;
        

        if(!pipeline_string){
            pipeline_string=char_concat(pipeline_string, this->next_gen[i]->get_string_code());
            map_evaluation[i]=evaluation_count++;
        }else{                    
            int found_pipe=-1;
            for(int j=0;j<i;j++){
                if(this->next_gen[j] && !strcmp(this->next_gen[i]->get_string_code(),this->next_gen[j]->get_string_code())){
                    found_pipe=map_evaluation[j];
                    break;
                }
            }
            if(found_pipe==-1){
                pipeline_string=char_concat(char_concat(pipeline_string,"|"),this->next_gen[i]->get_string_code());
                map_evaluation[i]=evaluation_count++;
            }else{
                map_evaluation[i]=found_pipe;
            }
        }
    }


    PyObject *pipeline_score;
    PyObject *result_obj;

    if(evaluation_count>0){
        if(!this->interface_call->evaluate_pipelines(pipeline_string, &pipeline_score, &result_obj, &(this->predict_size))){
            free(pipeline_string);
            free(map_evaluation);
            return NULL;
        }

        if(pipeline_score==Py_None){
            free(pipeline_string);
            free(map_evaluation);
            return -1; //flag for KeyboardException or SystemExit
        }
    }

    if(this->predict_next_gen)
        free(this->predict_next_gen);

    this->predict_next_gen=(double*) malloc(sizeof(double)*this->next_gen_size*this->predict_size); 

    for(int i=0;i<this->next_gen_size;i++){
        if(map_evaluation[i]<0 || this->next_gen[i]==NULL){        
            this->set_metric_next_gen(i,INVALID_INDIVIDUAL_SCORE);		
            this->set_score_next_gen(i,INVALID_INDIVIDUAL_SCORE);
        }
    }
    

    for(int i=0;i<this->next_gen_size;i++){
        if(map_evaluation[i]<0 || this->next_gen[i]==NULL) continue;

	    if(PyList_GetItem(pipeline_score,map_evaluation[i])==Py_None){
            this->set_metric_next_gen(i,INVALID_INDIVIDUAL_SCORE);		
            this->set_score_next_gen(i,INVALID_INDIVIDUAL_SCORE);		
        }else{
            this->set_metric_next_gen(i,PyFloat_AsDouble(PyList_GetItem(pipeline_score,map_evaluation[i])));
            this->set_score_next_gen(i,PyFloat_AsDouble(PyList_GetItem(pipeline_score,map_evaluation[i])));

            PyObject *predict_solution=PyList_GetItem(result_obj, map_evaluation[i]);

            for(int j=0;j<this->predict_size;j++){
                this->set_predict_next_gen(j, i, *(int*)PyArray_GETPTR1((PyArrayObject *)predict_solution,j));
            }
        
	    }
    }

    if(evaluation_count>0){
        Py_XDECREF(pipeline_score);
        Py_XDECREF(result_obj);

        free(pipeline_string);
    }

    free(map_evaluation);

    return 1;
}

int Population::evaluate_next_gen_cv(int population_as_buffer_option){
    char *pipeline_string=NULL;
    int *map_evaluation=(int*)malloc(sizeof(int)*this->next_gen_size);

    int evaluation_count=0;
    for(int i=0;i<this->next_gen_size;i++){
        int flag_computed_already=0;
        
        if(this->next_gen[i]==NULL) continue;

        if(population_as_buffer_option==2){ //Population is used as buffer to already seen solutions
            for(int j=0;j<this->population_size;j++){
                if(this->population[j] && !strcmp(this->next_gen[i]->get_string_code(),this->population[j]->get_string_code())){        
                    map_evaluation[i]=-j-1;		        
                    flag_computed_already=1;
                    break;
                }
            }
        }else if(population_as_buffer_option==1){ //Population is used as invalid list of solutions
            for(int j=0;j<this->population_size;j++){
                if(this->population[j] && this->get_score_population(j)==INVALID_INDIVIDUAL_SCORE && !strcmp(this->next_gen[i]->get_string_code(),this->population[j]->get_string_code())){        
                    map_evaluation[i]=-j-1;		        
                    flag_computed_already=1;
                    break;
                }
            }
        }

        if(flag_computed_already) continue;

        if(!pipeline_string){
            pipeline_string=char_concat(pipeline_string, this->next_gen[i]->get_string_code());
            map_evaluation[i]=evaluation_count++;
        }else{                    
            int found_pipe=-1;
            for(int j=0;j<i;j++){
                if(this->next_gen[j] && !strcmp(this->next_gen[i]->get_string_code(),this->next_gen[j]->get_string_code())){
                    found_pipe=map_evaluation[j];
                    break;
                }
            }
            if(found_pipe==-1){
                pipeline_string=char_concat(char_concat(pipeline_string,"|"),this->next_gen[i]->get_string_code());
                map_evaluation[i]=evaluation_count++;
            }else{
                map_evaluation[i]=found_pipe;
            }
        }
    }

    PyObject *pipeline_score;
    PyObject *result_obj;

    if(evaluation_count>0){
        if(!this->interface_call->evaluate_pipelines_cv(pipeline_string, &pipeline_score, &result_obj, &(this->predict_size))){
            free(pipeline_string);
            free(map_evaluation);
            return NULL;
        }

        if(pipeline_score==Py_None){
            free(pipeline_string);
            free(map_evaluation);
            return -1; //flag for KeyboardException or SystemExit
        }
    }

    if(this->predict_next_gen)
        free(this->predict_next_gen);

    this->predict_next_gen=(double*) malloc(sizeof(double)*this->next_gen_size*this->predict_size); 
      
    for(int i=0;i<this->next_gen_size;i++){
        if(this->next_gen[i]==NULL){
            this->set_metric_next_gen(i,INVALID_INDIVIDUAL_SCORE);		
            this->set_score_next_gen(i,INVALID_INDIVIDUAL_SCORE);		
        }else if(map_evaluation[i]<0){        
            int position_population_buffer=-map_evaluation[i]-1;
            this->set_metric_next_gen(i,this->get_metric_population(position_population_buffer));		
            this->set_score_next_gen(i,this->get_score_population(position_population_buffer));
            if(this->get_score_next_gen(i)==INVALID_INDIVIDUAL_SCORE) continue;
            for(int k=0;k<this->predict_size;k++){
                this->set_predict_next_gen(k, i, this->get_predict_population(k,position_population_buffer));
            }
        }
    }

    for(int i=0;i<this->next_gen_size;i++){
        if(map_evaluation[i]<0 || this->next_gen[i]==NULL) continue;
	    if(PyList_GetItem(pipeline_score,map_evaluation[i])==Py_None){
            this->set_metric_next_gen(i,INVALID_INDIVIDUAL_SCORE);		
            this->set_score_next_gen(i,INVALID_INDIVIDUAL_SCORE);			    
        }else{
            this->set_metric_next_gen(i,PyFloat_AsDouble(PyList_GetItem(pipeline_score,map_evaluation[i])));
            this->set_score_next_gen(i,PyFloat_AsDouble(PyList_GetItem(pipeline_score,map_evaluation[i])));

            PyObject *predict_solution=PyList_GetItem(result_obj, map_evaluation[i]);

            for(int j=0;j<this->predict_size;j++){
                this->set_predict_next_gen(j, i, *(int*)PyArray_GETPTR1((PyArrayObject *)predict_solution,j));
            }        
	    }
    }

    if(evaluation_count>0){
       Py_XDECREF(pipeline_score);
       Py_XDECREF(result_obj);

       free(pipeline_string);
    }

    free(map_evaluation);

    return 1;
}

int Population::evaluate_population_cv(){
    if(this->next_gen_size!=this->population_size){  //Adjusting next_gen size by next_generation procedure
        free(this->next_gen);
        free(this->score_next_gen);
        free(this->metric_next_gen);

        this->next_gen_size=this->population_size;
        this->next_gen=(Solution**)malloc(sizeof(Solution*)*this->next_gen_size);
        this->score_next_gen=(double*)malloc(sizeof(double)*this->next_gen_size);
        this->metric_next_gen=(double*)malloc(sizeof(double)*this->next_gen_size);
    }

    for(int i=0;i<this->population_size;i++)
        this->next_gen[i]=this->population[i]->copy();

    int return_flag=this->evaluate_next_gen_cv(1);
    if(!return_flag || return_flag==-1)
        return return_flag;

    std::swap(this->metric_population,this->metric_next_gen);
    std::swap(this->predict_population,this->predict_next_gen);
    std::swap(this->score_population,this->score_next_gen);

    for(int i=0;i<this->next_gen_size;i++){
        delete this->next_gen[i];
        this->next_gen[i]=NULL;
    }

    return 1;
}

int Population::evaluate_ensemble_next_gen(PopulationEnsemble *population_ensemble, int *map_next_gen_population, int changed_index){
    int flag_valid_individual=0;
    double min_predict, max_predict;

    for(int i=0;i<this->predict_size;i++){
        for(int j=0;j<this->population_size;j++){
            if(!flag_valid_individual && this->get_score_next_gen(map_next_gen_population[j])!=INVALID_INDIVIDUAL_SCORE){
                min_predict=max_predict=this->get_predict_next_gen(i,map_next_gen_population[j]);
                flag_valid_individual=1;
            }
            else if(min_predict>this->get_predict_next_gen(i,map_next_gen_population[j]) && this->get_score_next_gen(map_next_gen_population[j])!=INVALID_INDIVIDUAL_SCORE) min_predict=this->get_predict_next_gen(i,map_next_gen_population[j]);
            else if(max_predict<this->get_predict_next_gen(i,map_next_gen_population[j]) && this->get_score_next_gen(map_next_gen_population[j])!=INVALID_INDIVIDUAL_SCORE) max_predict=this->get_predict_next_gen(i,map_next_gen_population[j]);
        }
    }

    if(!flag_valid_individual){
        for(int pop=0;pop<population_ensemble->get_population_size();pop++)
            population_ensemble->set_score_population(pop, INVALID_INDIVIDUAL_SCORE);
        return 1;
    }

    double *predict_ensemble=(double*)malloc(sizeof(double)*this->predict_size);
    int *class_count=(int*)malloc(sizeof(int)*(max_predict-min_predict+1));
    int elected_class=0;
    for(int pop=0;pop<population_ensemble->get_population_size();pop++){
        if(!population_ensemble->check_valid_individual(pop) || changed_index!=-1 && !population_ensemble->get_element_population_i(pop, changed_index)) continue;
        
        if(population_ensemble->get_length_population(pop)==0){
            population_ensemble->set_score_population(pop, INVALID_INDIVIDUAL_SCORE);
            continue;
        }
        
        if(population_ensemble->get_length_population(pop)==1){
            for(int j=0;j<this->population_size;j++){
                if(population_ensemble->get_element_population_i(pop, j)){
                    population_ensemble->set_score_population(pop, this->get_metric_next_gen(map_next_gen_population[j]));
                    break;
                }
            }
            continue;
        }

        for(int i=0;i<this->predict_size;i++){
            for(int j=0;j<max_predict-min_predict+1;j++) class_count[j]=0;

            for(int j=0;j<this->population_size;j++)
                if(population_ensemble->get_element_population_i(pop, j) && this->get_score_next_gen(map_next_gen_population[j])!=INVALID_INDIVIDUAL_SCORE)
                    class_count[(int)(this->get_predict_next_gen(i,map_next_gen_population[j])-min_predict)]++;
            

            elected_class=0;
            for(int j=1;j<max_predict-min_predict+1;j++)
                if(class_count[j]>class_count[elected_class]) elected_class=j;
            
            if(class_count[elected_class]==0) break;           

            predict_ensemble[i]=elected_class+min_predict;
        }


        if(class_count[elected_class]==0){
            population_ensemble->set_score_population(pop, INVALID_INDIVIDUAL_SCORE);
            continue;
        }

        npy_intp numpy_dimension[1];    
        numpy_dimension[0]=this->predict_size;
        PyObject *predict_numpy=PyArray_SimpleNewFromData(1, numpy_dimension, NPY_DOUBLE, predict_ensemble);
        double return_score;

        if(!this->interface_call->evaluate_predict_vector(predict_numpy, &return_score)){
            Py_XDECREF(predict_numpy);
            free(class_count);
            free(predict_ensemble);
            return NULL;
        }
        Py_XDECREF(predict_numpy);

        population_ensemble->set_score_population(pop, return_score);
    }

    free(class_count);
    free(predict_ensemble);

    for(int i=0;i<this->population_size;i++){

        if(this->get_score_next_gen(map_next_gen_population[i])==INVALID_INDIVIDUAL_SCORE) continue;
        
        int count_ensemble=0;
        this->set_score_next_gen(map_next_gen_population[i],0);

        for(int j=0;j<population_ensemble->get_population_size();j++){
            if(population_ensemble->check_valid_individual(j) && population_ensemble->get_element_population_i(j,i)){
                this->set_score_next_gen(map_next_gen_population[i],this->get_score_next_gen(map_next_gen_population[i])+population_ensemble->get_score_population(j));
                count_ensemble++;
            }
        }
        if(count_ensemble)
            this->set_score_next_gen(map_next_gen_population[i],this->get_score_next_gen(map_next_gen_population[i])/count_ensemble);
        else
            this->set_score_next_gen(map_next_gen_population[i],this->get_metric_next_gen(map_next_gen_population[i]));
        
    }

    return 1;
}

int Population::evaluate_ensemble_population(PopulationEnsemble *population_ensemble){
    int flag_valid_individual=0;
    double min_predict,max_predict;

    for(int i=0;i<this->predict_size;i++){
        for(int j=0;j<this->population_size;j++){
            if(!flag_valid_individual && this->get_score_population(j)!=INVALID_INDIVIDUAL_SCORE){
                min_predict=max_predict=this->get_predict_population(i,j);
                flag_valid_individual=1;
            }
            else if(min_predict>this->get_predict_population(i,j) && this->get_score_population(j)!=INVALID_INDIVIDUAL_SCORE) min_predict=this->get_predict_population(i,j);
            else if(max_predict<this->get_predict_population(i,j) && this->get_score_population(j)!=INVALID_INDIVIDUAL_SCORE) max_predict=this->get_predict_population(i,j);
        }
    }

    if(!flag_valid_individual){
        for(int pop=0;pop<population_ensemble->get_population_size();pop++)
            population_ensemble->set_score_population(pop, INVALID_INDIVIDUAL_SCORE);
        return 1;
    }

    double *predict_ensemble=(double*)malloc(sizeof(double)*this->predict_size);
    int *class_count=(int*)malloc(sizeof(int)*(max_predict-min_predict+1));
    int elected_class=0;

    for(int pop=0;pop<population_ensemble->get_population_size();pop++){
        if(!population_ensemble->check_valid_individual(pop)) continue;

        if(population_ensemble->get_length_population(pop)==0){
            population_ensemble->set_score_population(pop, INVALID_INDIVIDUAL_SCORE);
            continue;
        }
        
        if(population_ensemble->get_length_population(pop)==1){
            for(int j=0;j<this->population_size;j++){
                if(population_ensemble->get_element_population_i(pop, j)){
                    population_ensemble->set_score_population(pop, this->get_metric_population(j));
                    break;
                }
            }
            continue;
        }


        for(int i=0;i<this->predict_size;i++){
            for(int j=0;j<max_predict-min_predict+1;j++) class_count[j]=0;

            for(int j=0;j<this->population_size;j++)
                if(population_ensemble->get_element_population_i(pop, j) && this->get_score_population(j)!=INVALID_INDIVIDUAL_SCORE)
                    class_count[(int)(this->get_predict_population(i,j)-min_predict)]++;


            elected_class=0;
            for(int j=1;j<max_predict-min_predict+1;j++)
                if(class_count[j]>class_count[elected_class]) elected_class=j;

            if(class_count[elected_class]==0) break;           

            predict_ensemble[i]=elected_class+min_predict;
        }

        if(class_count[elected_class]==0){
            population_ensemble->set_score_population(pop, INVALID_INDIVIDUAL_SCORE);
            continue;
        }

        npy_intp numpy_dimension[1];    
        numpy_dimension[0]=this->predict_size;
        PyObject *predict_numpy=PyArray_SimpleNewFromData(1, numpy_dimension, NPY_DOUBLE, predict_ensemble);
        double return_score;

        if(!this->interface_call->evaluate_predict_vector(predict_numpy, &return_score)){
            Py_XDECREF(predict_numpy);
            free(class_count);
            free(predict_ensemble);
            return NULL;
        }
        Py_XDECREF(predict_numpy);

        population_ensemble->set_score_population(pop, return_score);
    }

    free(class_count);
    free(predict_ensemble);

    this->update_population_score(population_ensemble);

    return 1;
}

int Population::update_population_score(PopulationEnsemble *population_ensemble){

    for(int i=0;i<this->population_size;i++){

        if(this->get_score_population(i)==INVALID_INDIVIDUAL_SCORE) continue;

        int count_ensemble=0;
        this->set_score_population(i,0);

        for(int j=0;j<population_ensemble->get_population_size();j++){
            if(population_ensemble->check_valid_individual(j) && population_ensemble->get_element_population_i(j, i)){
                this->set_score_population(i,this->get_score_population(i)+population_ensemble->get_score_population(j));
                count_ensemble++;
            }
        }
        if(count_ensemble)
            this->set_score_population(i,this->get_score_population(i)/count_ensemble);
        else
            this->set_score_population(i,this->get_metric_population(i));
    }


    return 1;
}

int Population::next_generation_selection_similarity(PopulationEnsemble *population_ensemble, int generation, std::ofstream *competition_log, std::ofstream *matrix_sim_next_gen_log){
    if(this->next_gen_size!=2*this->population_size){  //Adjusting next_gen size by next_generation procedure
        free(this->next_gen);
        free(this->score_next_gen);
        free(this->metric_next_gen);
        if(this->similarity_matrix_next_gen)
            free(this->similarity_matrix_next_gen);

        this->next_gen_size=2*this->population_size;
        this->next_gen=(Solution**)malloc(sizeof(Solution*)*this->next_gen_size);
        this->score_next_gen=(double*)malloc(sizeof(double)*this->next_gen_size);
        this->metric_next_gen=(double*)malloc(sizeof(double)*this->next_gen_size);
        this->similarity_matrix_next_gen=(double*)malloc(sizeof(double)*this->next_gen_size*(this->next_gen_size-1)/2);
    }

    for(int i=0;i<this->population_size;i++)
        this->next_gen[i]=this->population[i]->copy();


    int *choice_mask=(int*)malloc(sizeof(int)*this->population_size);
    for(int i=0;i<this->population_size;i++)choice_mask[i]=0;

    for(int i=this->population_size;i<this->next_gen_size; i+=2){
        int parent1, parent2;

        parent1=randInt(0,this->population_size-1-(i-this->population_size));
        int sort_position=0;
        for(int count_position=0;count_position<=parent1;sort_position++)
            if(!choice_mask[sort_position])count_position++;
        parent1=sort_position-1;
        choice_mask[parent1]=1;

        if(i<this->next_gen_size-1){
           parent2=randInt(0,this->population_size-1-(i-this->population_size)-1);                
           sort_position=0;
           for(int count_position=0;count_position<=parent2;sort_position++)
               if(!choice_mask[sort_position])count_position++;
           parent2=sort_position-1;
           choice_mask[parent2]=1;
        }else //If size of population is odd, a solution of population is sorted to complete the pairing procedure
           parent2=randInt(0,this->population_size-1);                


        Solution *child1, *child2;

        child1=this->population[parent1]->copy();
        child2=this->population[parent2]->copy();
        
        if(this->get_score_population(parent1)==INVALID_INDIVIDUAL_SCORE)        
            Solution::mutation(child1);
        if(this->get_score_population(parent2)==INVALID_INDIVIDUAL_SCORE)        
            Solution::mutation(child2);
        
        this->breed(child1,child2);

        this->next_gen[i]=child1;

        if(i<this->next_gen_size-1)        
            this->next_gen[i+1]=child2;
        else
            delete child2;
    }

    free(choice_mask);

    int return_flag;
    if(this->cv_evaluation)
        return_flag=this->evaluate_next_gen_cv(2);
    else
        return_flag=this->evaluate_next_gen(true);

    if(!return_flag || return_flag==-1) //Any other exception than KeyboardException, just propagate with return NULL
        return return_flag;

    free(this->predict_population);
    this->predict_population=(double*) malloc(sizeof(double)*this->population_size*this->predict_size); 


    if(!this->compute_similarity_next_gen())
        return NULL;

    if(matrix_sim_next_gen_log!=NULL)
        this->write_similarity_matrix_next_gen(generation, matrix_sim_next_gen_log);


    int *map_next_gen=(int*)malloc(sizeof(int)*this->population_size);
    for(int i=0;i<this->population_size;i++)map_next_gen[i]=i;

    //If not using CV mode update the score of components population (new scores in the population)
    if(!this->cv_evaluation){
        if(!this->evaluate_ensemble_next_gen(population_ensemble, map_next_gen, -1))
            return NULL;

        for(int i=0;i<this->population_size;i++){
            this->set_score_population(i, this->get_score_next_gen(i));
            this->set_metric_population(i, this->get_metric_next_gen(i));
        }
    }

    this->sort_population();        

    for(int iterator=0;iterator<this->population_size;iterator++){
        int max_sim_index=-1;

        int i;
        do{
            i=randInt(0,this->population_size-1);
        }while(!this->next_gen[i]);


        for(int j=this->population_size;j<this->next_gen_size;j++){
            if(this->next_gen[j] && this->get_similarity_matrix_next_gen(i,j)!=INVALID_SIMILARITY_PREDICT && (max_sim_index==-1 || this->get_similarity_matrix_next_gen(i,j)>this->get_similarity_matrix_next_gen(i,max_sim_index)))
                max_sim_index=j;
        }

        if(max_sim_index==-1){
            int valid_candidate_flag = 0;
            for(int j=this->population_size;j<this->next_gen_size;j++)
                if(this->next_gen[j]) valid_candidate_flag = 1;
            
            if(!valid_candidate_flag){                
                this->update_population_with_next_gen(i, i);
                continue;
            } 

            do{
                max_sim_index=randInt(this->population_size,this->next_gen_size-1);
            }while(!this->next_gen[max_sim_index]);
        }


        map_next_gen[i]=max_sim_index;
        if(!this->evaluate_ensemble_next_gen(population_ensemble,map_next_gen, i))
            return NULL;

        
        if(competition_log!=NULL)
            this->write_competition(generation, i, max_sim_index, competition_log);

        if(this->get_score_next_gen(i)<this->get_score_next_gen(max_sim_index) || (this->get_score_next_gen(i)==this->get_score_next_gen(max_sim_index) && this->get_metric_next_gen(i)<this->get_metric_next_gen(max_sim_index))){
            this->update_population_with_next_gen(i, max_sim_index);

            delete this->next_gen[i];
            this->next_gen[i]=NULL;

            map_next_gen[i]=max_sim_index;
        }else{
            this->update_population_with_next_gen(i, i);

            delete this->next_gen[max_sim_index];
            this->next_gen[max_sim_index]=NULL;
            
            map_next_gen[i]=i;

            if(!this->evaluate_ensemble_next_gen(population_ensemble, map_next_gen, i))
                return NULL;
        }
    }

    free(map_next_gen);

    for(int i=0;i<this->next_gen_size;i++){
        if(this->next_gen[i]){
            delete this->next_gen[i];
            this->next_gen[i]=NULL;
        }
    }

    this->update_population_score(population_ensemble);

    this->compute_similarity();
    this->sort_population();

    return 1;
}


PyObject *Population::get_solution_pipeline_rank_i(int i){
    if(i<0 || i>this->population_size)
        throw "Invalid solution index";

    i=this->population_rank[i];

    if(this->get_score_population(i)==INVALID_INDIVIDUAL_SCORE) return Py_None; //if fit method doesn't work

    PyObject *pipeline=NULL;
    if(!(pipeline=this->interface_call->make_pipeline_from_str(this->population[i]->get_string_code())))
        return NULL;

    return pipeline;
}


PyObject *Population::get_population_ensemble_elite(){
    char *pipeline_string=NULL;

    for(int iterator=0;iterator<this->elite_size;iterator++){
        int i=this->population_rank[iterator];
        if(this->get_score_population(i)==INVALID_INDIVIDUAL_SCORE) continue; //if fit method doesn't work
        if(!pipeline_string)
            pipeline_string=char_concat(pipeline_string, this->population[i]->get_string_code());
        else        
            pipeline_string=char_concat(char_concat(pipeline_string,"|"),this->population[i]->get_string_code());
    }

    PyObject *ensemble=NULL;
    ensemble=this->interface_call->make_voting_ensemble_from_str(pipeline_string);

    free(pipeline_string);

    if(!ensemble)
        return NULL;

    return ensemble;
}

PyObject *Population::get_population_ensemble_all(){
    char *pipeline_string=NULL;

    for(int i=0;i<this->population_size;i++){
        if(this->get_score_population(i)==INVALID_INDIVIDUAL_SCORE) continue; //if fit method doesn't work
        if(!pipeline_string)
            pipeline_string=char_concat(pipeline_string, this->population[i]->get_string_code());
        else        
            pipeline_string=char_concat(char_concat(pipeline_string,"|"),this->population[i]->get_string_code());
    }

    PyObject *ensemble=NULL;
    ensemble=this->interface_call->make_voting_ensemble_from_str(pipeline_string);

    free(pipeline_string);

    if(!ensemble)
        return NULL;

    return ensemble;
}


PyObject *Population::get_population_ensemble_mask_i(PopulationEnsemble *population_ensemble, int id_ensemble){
    char *pipeline_string=NULL;

    for(int i=0;i<this->population_size;i++){
        if(population_ensemble->get_element_population_i(id_ensemble, i)){
            if(this->get_score_population(i)==INVALID_INDIVIDUAL_SCORE) continue; //if fit method doesn't work
            if(!pipeline_string)
                pipeline_string=char_concat(pipeline_string, this->population[i]->get_string_code());
            else        
                pipeline_string=char_concat(char_concat(pipeline_string,"|"),this->population[i]->get_string_code());
        }
    }

    PyObject *ensemble=NULL;
    ensemble=this->interface_call->make_voting_ensemble_from_str(pipeline_string);

    free(pipeline_string);

    if(!ensemble)
        return NULL;

    return ensemble;
}


void Population::compute_similarity(){
    for(int i=0;i<this->population_size;i++){
        for(int j=0;j<i;j++){
            double sim=0;
            if(this->get_score_population(i)==INVALID_INDIVIDUAL_SCORE || this->get_score_population(j)==INVALID_INDIVIDUAL_SCORE){
                this->set_similarity_matrix(i,j,INVALID_SIMILARITY_PREDICT);
                continue;
            }

            for(int k=0;k<this->predict_size;k++)
                sim+=this->get_predict_population(k,i)==this->get_predict_population(k,j) ? 1 : 0; //equal comparison
       
            sim=sim/this->predict_size;
            this->set_similarity_matrix(i,j,sim);
        }
    }
}

int Population::compute_similarity_next_gen(){
    for(int i=0;i<this->next_gen_size;i++){
        for(int j=0;j<i;j++){
            double sim=0;
            if(this->get_score_next_gen(i)==INVALID_INDIVIDUAL_SCORE || this->get_score_next_gen(j)==INVALID_INDIVIDUAL_SCORE){
                this->set_similarity_matrix_next_gen(i,j,INVALID_SIMILARITY_PREDICT);
                continue;
            }

            for(int k=0;k<this->predict_size;k++)
                sim+=this->get_predict_next_gen(k,i)==this->get_predict_next_gen(k,j) ? 1 : 0; //equal comparison
       
            sim=sim/this->predict_size;
            this->set_similarity_matrix_next_gen(i,j,sim);
        }
    }

    return 1;
}

void Population::write_next_gen(int generation, std::ofstream *evolution_log){
    for(int i=0;i<this->next_gen_size;i++)
        (*evolution_log)<<generation<<";"<<this->next_gen[i]->get_id()<<";"<<this->next_gen[i]->get_string_code()<<";"<<this->get_score_next_gen(i)<<";"<<this->get_metric_next_gen(i)<<"\n";
}

void Population::write_population(int generation, std::ofstream *evolution_log){
    for(int i=0;i<this->population_size;i++)
        (*evolution_log)<<generation<<";"<<this->population[i]->get_id()<<";"<<this->population[i]->get_string_code()<<";"<<this->get_score_population(i)<<";"<<this->get_metric_population(i)<<"\n";
}


void Population::write_similarity_matrix(int generation, std::ofstream *evolution_log){
    for(int i=0;i<this->population_size;i++)
        for(int j=0;j<this->population_size;j++)
            (*evolution_log)<<generation<<";"<<this->population[i]->get_id()<<";"<<this->population[j]->get_id()<<";"<<this->get_similarity_matrix(i,j)<<"\n";
}

void Population::write_similarity_matrix_next_gen(int generation, std::ofstream *evolution_log){
    for(int i=0;i<this->population_size;i++){
        for(int j=this->population_size;j<this->next_gen_size;j++){
            if(this->next_gen[i] && this->next_gen[j])
                (*evolution_log)<<generation<<";"<<this->next_gen[i]->get_id()<<";"<<this->next_gen[j]->get_id()<<";"<<this->get_similarity_matrix_next_gen(i,j)<<"\n";
            else if(this->next_gen[i])
                (*evolution_log)<<generation<<";"<<this->next_gen[i]->get_id()<<";"<<"NULL"<<";"<<this->get_similarity_matrix_next_gen(i,j)<<"\n";
            else if(this->next_gen[j])
                (*evolution_log)<<generation<<";"<<"NULL"<<";"<<this->next_gen[j]->get_id()<<";"<<this->get_similarity_matrix_next_gen(i,j)<<"\n";
            else
                (*evolution_log)<<generation<<";"<<"NULL"<<";"<<"NULL"<<";"<<this->get_similarity_matrix_next_gen(i,j)<<"\n";
        }
    }
}

void Population::write_competition(int generation, int index_population, int index_next_gen, std::ofstream *competition_log){
    (*competition_log)<<generation<<";"<<this->next_gen[index_population]->get_id()<<";"<<this->next_gen[index_population]->get_string_code()<<";"<<this->get_score_next_gen(index_population)<<";"<<this->get_metric_next_gen(index_population)<<";";
    (*competition_log)<<this->next_gen[index_next_gen]->get_id()<<";"<<this->next_gen[index_next_gen]->get_string_code()<<";"<<this->get_score_next_gen(index_next_gen)<<";"<<this->get_metric_next_gen(index_next_gen)<<"\n";
}


void Population::update_population_with_next_gen(int population_index, int next_gen_index){
    if(!strcmp(this->population[population_index]->get_string_code(),this->next_gen[next_gen_index]->get_string_code())){
        delete this->next_gen[next_gen_index];
        this->next_gen[next_gen_index]=NULL;
    }else{
        delete this->population[population_index];
        this->population[population_index]=NULL;
        std::swap(this->population[population_index], this->next_gen[next_gen_index]);
    }

    this->set_score_population(population_index, this->get_score_next_gen(next_gen_index));
    this->set_metric_population(population_index, this->get_metric_next_gen(next_gen_index));

    for(int j=0;j<this->predict_size;j++)
        this->set_predict_population(j, population_index, this->get_predict_next_gen(j, next_gen_index));   
}

double Population::get_score_population(int i){
    if(i<0 || i>=this->population_size)
        throw "Invalid index in score_population";
    return this->score_population[i];
}


void Population::set_score_population(int i, double score){
    if(i<0 || i>=this->population_size)
        throw "Invalid index in score_population";

    this->score_population[i]=score;
}

double Population::get_metric_population(int i){
    if(i<0 || i>=this->population_size)
        throw "Invalid index in metric_population";
    return this->metric_population[i];
}


void Population::set_metric_population(int i, double metric){
    if(i<0 || i>=this->population_size)
        throw "Invalid index in metric_population";

    this->metric_population[i]=metric;
}


double Population::get_score_next_gen(int i){
    if(i<0 || i>=this->next_gen_size)
        throw "Invalid index in score_next_gen";

    return this->score_next_gen[i];
}

void Population::set_score_next_gen(int i, double score){
    if(i<0 || i>=this->next_gen_size)
        throw "Invalid index in score_next_gen";

    this->score_next_gen[i]=score;
}

double Population::get_metric_next_gen(int i){
    if(i<0 || i>=this->next_gen_size)
        throw "Invalid index in metric_next_gen";

    return this->metric_next_gen[i];
}

void Population::set_metric_next_gen(int i, double metric){
    if(i<0 || i>=this->next_gen_size)
        throw "Invalid index in metric_next_gen";

    this->metric_next_gen[i]=metric;
}

double Population::get_predict_population(int i, int j){ //(sample_id, solution_id)
    if(i<0 || i>=this->predict_size || j<0 || j>=this->population_size)
        throw "Invalid index in predict_population";
    return this->predict_population[i*this->population_size+j];
}

void Population::set_predict_population(int i, int j, double predict_value){ //(sample_id, solution_id)
    if(i<0 || i>=this->predict_size || j<0 || j>=this->population_size)
        throw "Invalid index in predict_population";

    this->predict_population[i*this->population_size+j]=predict_value;
}


double Population::get_predict_next_gen(int i, int j){ //(sample_id, solution_id)
    if(i<0 || i>=this->predict_size || j<0 || j>=this->next_gen_size)
        throw "Invalid index in next_gen";
    return this->predict_next_gen[i*this->next_gen_size+j];
}

void Population::set_predict_next_gen(int i, int j, double predict_value){ //(sample_id, solution_id)
    if(i<0 || i>=this->predict_size || j<0 || j>=this->next_gen_size)
        throw "Invalid index in next_gen";

    this->predict_next_gen[i*this->next_gen_size+j]=predict_value;
}


double Population::get_similarity_matrix(int i, int j){
    if(i<0 || i>=this->population_size || j<0 || j>=this->population_size)
        throw "Invalid index in similarity_matrix";

    if(i==j) return MAX_SIM;

    if(i<j) std::swap(i,j);

    return this->similarity_matrix[i*(i-1)/2+j];
}

void Population::set_similarity_matrix(int i, int j, double value){
    if(i<0 || i>=this->population_size || j<0 || j>=this->population_size)
        throw "Invalid index in similarity_matrix";

    if(i==j)
        throw "Invalid index in similarity_matrix";

    if(i<j) std::swap(i,j);

    this->similarity_matrix[i*(i-1)/2+j]=value;
}
 
double Population::get_similarity_matrix_next_gen(int i, int j){
    if(i<0 || i>=this->next_gen_size || j<0 || j>=this->next_gen_size)
        throw "Invalid index in similarity_matrix";

    if(i==j) return MAX_SIM;

    if(i<j) std::swap(i,j);

    return this->similarity_matrix_next_gen[i*(i-1)/2+j];
}

void Population::set_similarity_matrix_next_gen(int i, int j, double value){
    if(i<0 || i>=this->next_gen_size || j<0 || j>=this->next_gen_size)
        throw "Invalid index in similarity_matrix";

    if(i==j)
        throw "Invalid index in similarity_matrix";

    if(i<j) std::swap(i,j);

    this->similarity_matrix_next_gen[i*(i-1)/2+j]=value;
}

void insertionsort(Solution **pop, double *values, int *id_solution, int size);

void Population::sort_population(){
     insertionsort(this->population,this->score_population,this->population_rank,this->population_size);
}


void insertionsort(Solution **pop, double *values, int *id_solution, int size){
    int i = 1;

    while(i < size){
        int j = i;

        while(j > 0 && values[id_solution[j-1]] < values[id_solution[j]]){
            std::swap(id_solution[j-1], id_solution[j]);
            j--;
        }

        i++;
    }
}