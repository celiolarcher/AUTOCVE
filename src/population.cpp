#include "population.h"
#include "utility.h"
#include <stdlib.h>
#include <math.h>
#define INVALID_INDIVIDUAL_SCORE -1
#define INVALID_SIMILARITY_PREDICT -1
#define MAX_SIM 1

Population::Population(PythonInterface *interface, int size_pop, double elite_portion, double mut_rate, double cross_rate){
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

    int return_flag=this->evaluate_next_gen_cv(false);

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
    this->quick_sort_population();

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

    Solution::reset_index();

    free(this->population);
    free(this->population_rank);
    free(this->similarity_matrix);
    free(this->score_population);
    free(this->metric_population);
}

void Population::breed(Solution* child1, Solution* child2){
    double prob=randDouble(0,1);

    if(prob<cross_rate)
        Solution::crossover(child1,child2);

    prob=randDouble(0,1);
    if(prob<mut_rate){
        Solution::mutation(child1);
        Solution::mutation(child2);
     }
}


int Population::evaluate_next_gen_cv(int population_as_buffer){
    char *pipeline_string=NULL;
    int *map_evaluation=(int*)malloc(sizeof(int)*this->next_gen_size);

    int evaluation_count=0;
    for(int i=0;i<this->next_gen_size;i++){
        int flag_computed_already=0;
        
        if(population_as_buffer){
            for(int j=0;j<this->population_size;j++){
                if(this->population[j] && !strcmp(this->next_gen[i]->get_string_code(),this->population[j]->get_string_code())){        
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
                if(!strcmp(this->next_gen[i]->get_string_code(),this->next_gen[j]->get_string_code())){
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

    if(this->predict_next_gen)
        free(this->predict_next_gen);

    this->predict_next_gen=(double*) malloc(sizeof(double)*this->next_gen_size*this->predict_size); 
      
    for(int i=0;i<this->next_gen_size;i++){
        if(map_evaluation[i]<0){        
            int position_population_buffer=-map_evaluation[i]-1;
            this->set_metric_next_gen(i,this->get_metric_population(position_population_buffer));		
            this->set_score_next_gen(i,this->get_metric_population(position_population_buffer));
            if(this->get_score_next_gen(i)==INVALID_INDIVIDUAL_SCORE) continue;
            for(int k=0;k<this->predict_size;k++){
                this->set_predict_next_gen(k, i, this->get_predict_population(k,position_population_buffer));
            }
        }
    }

    for(int i=0;i<this->next_gen_size;i++){
        if(map_evaluation[i]<0) continue;
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

    Py_XDECREF(pipeline_score);
    Py_XDECREF(result_obj);

    free(pipeline_string);
    free(map_evaluation);

    return 1;
}

int Population::evaluate_ensemble_next_gen(PopulationEnsemble *population_ensemble, int *map_next_gen_population, int changed_index){
    int flag_valid_individual=0;
    double min_predict,max_predict;

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
            population_ensemble->score_population[pop]=INVALID_INDIVIDUAL_SCORE;
        return 1;
    }

    double *predict_ensemble=(double*)malloc(sizeof(double)*this->predict_size);
    int *class_count=(int*)malloc(sizeof(int)*(max_predict-min_predict+1));
    int elected_class=0;
    for(int pop=0;pop<population_ensemble->get_population_size();pop++){
        if(changed_index!=-1 && !population_ensemble->population[pop][changed_index]) continue;

        for(int i=0;i<this->predict_size;i++){
            for(int j=0;j<max_predict-min_predict+1;j++) class_count[j]=0;

            for(int j=0;j<this->population_size;j++)
                if(population_ensemble->population[pop][j] && this->get_score_next_gen(map_next_gen_population[j])!=INVALID_INDIVIDUAL_SCORE)
                    class_count[(int)(this->get_predict_next_gen(i,map_next_gen_population[j])-min_predict)]++;
            

            elected_class=0;
            for(int j=1;j<max_predict-min_predict+1;j++)
                if(class_count[j]>class_count[elected_class]) elected_class=j;
            
            if(class_count[elected_class]==0) break;           

            predict_ensemble[i]=elected_class+min_predict;
        }


        if(class_count[elected_class]==0){
            population_ensemble->score_population[pop]=INVALID_INDIVIDUAL_SCORE;
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

        population_ensemble->score_population[pop]=return_score;
    }

    free(class_count);
    free(predict_ensemble);

    for(int i=0;i<this->population_size;i++){

        if(this->get_score_next_gen(map_next_gen_population[i])==INVALID_INDIVIDUAL_SCORE) continue;

        int count_ensemble=0;
        this->set_score_next_gen(map_next_gen_population[i],0);

        for(int j=0;j<population_ensemble->get_population_size();j++){
            if(population_ensemble->population[j][i]){
                this->set_score_next_gen(map_next_gen_population[i],this->get_score_next_gen(map_next_gen_population[i])+population_ensemble->score_population[j]);
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
            population_ensemble->score_population[pop]=INVALID_INDIVIDUAL_SCORE;
        return 1;
    }

    double *predict_ensemble=(double*)malloc(sizeof(double)*this->predict_size);
    int *class_count=(int*)malloc(sizeof(int)*(max_predict-min_predict+1));
    int elected_class=0;
    for(int pop=0;pop<population_ensemble->get_population_size();pop++){
        for(int i=0;i<this->predict_size;i++){
            for(int j=0;j<max_predict-min_predict+1;j++) class_count[j]=0;

            for(int j=0;j<this->population_size;j++)
                if(population_ensemble->population[pop][j] && this->get_score_population(j)!=INVALID_INDIVIDUAL_SCORE)
                    class_count[(int)(this->get_predict_population(i,j)-min_predict)]++;


            elected_class=0;
            for(int j=1;j<max_predict-min_predict+1;j++)
                if(class_count[j]>class_count[elected_class]) elected_class=j;

            if(class_count[elected_class]==0) break;           

            predict_ensemble[i]=elected_class+min_predict;
        }

        if(class_count[elected_class]==0){
            population_ensemble->score_population[pop]=INVALID_INDIVIDUAL_SCORE;
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

        population_ensemble->score_population[pop]=return_score;
    }

    free(class_count);
    free(predict_ensemble);

    for(int i=0;i<this->population_size;i++){

        if(this->get_score_population(i)==INVALID_INDIVIDUAL_SCORE) continue;

        int count_ensemble=0;
        this->set_score_population(i,0);

        for(int j=0;j<population_ensemble->get_population_size();j++){
            if(population_ensemble->population[j][i]){
                this->set_score_population(i,this->get_score_population(i)+population_ensemble->score_population[j]);
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

int Population::next_generation_selection_similarity(PopulationEnsemble *population_ensemble){
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


    int choice_mask[this->population_size];
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

        this->breed(child1,child2);

        this->next_gen[i]=child1;

        if(i<this->next_gen_size-1)        
            this->next_gen[i+1]=child2;
        else
            delete child2;
    }

    int return_flag=this->evaluate_next_gen_cv(true);

    if(!return_flag || return_flag==-1) //Any other exception than KeyboardException, just propagate with return NULL
        return return_flag;


    free(this->predict_population);
    this->predict_population=(double*) malloc(sizeof(double)*this->population_size*this->predict_size); 

    if(!this->compute_similarity_next_gen())
        return NULL;

    int *map_next_gen=(int*)malloc(sizeof(int)*this->population_size);
    for(int i=0;i<this->population_size;i++)map_next_gen[i]=i;

    if(!this->evaluate_ensemble_next_gen(population_ensemble,map_next_gen, -1))
        return NULL;

    for(int iterator=0;iterator<this->population_size;iterator++){
        int max_sim_index=-1;
        int i=this->population_rank[iterator];
        for(int j=this->population_size;j<this->next_gen_size;j++){
            if(this->next_gen[j] && this->get_similarity_matrix_next_gen(i,j)!=INVALID_SIMILARITY_PREDICT && (max_sim_index==-1 || this->get_similarity_matrix_next_gen(i,j)>this->get_similarity_matrix_next_gen(i,max_sim_index)))

                max_sim_index=j;
        }

        if(max_sim_index==-1){
            do{
                max_sim_index=randInt(this->population_size,this->next_gen_size-1);
            }while(!this->next_gen[max_sim_index]);
        }


        map_next_gen[i]=max_sim_index;
        if(!this->evaluate_ensemble_next_gen(population_ensemble,map_next_gen, i))
            return NULL;

        if(this->get_score_next_gen(i)<this->get_score_next_gen(max_sim_index)){
            delete this->next_gen[i];
            this->next_gen[i]=NULL;
            delete this->population[i];
            this->population[i]=this->next_gen[max_sim_index];
            this->next_gen[max_sim_index]=NULL;
            this->set_score_population(i,this->get_score_next_gen(max_sim_index));
            this->set_metric_population(i,this->get_metric_next_gen(max_sim_index));

            for(int j=0;j<this->predict_size;j++)
                 this->set_predict_population(j,i,this->get_predict_next_gen(j,max_sim_index));   

            map_next_gen[i]=max_sim_index;
        }else{
            delete this->next_gen[i];
            this->next_gen[i]=NULL;
            delete this->next_gen[max_sim_index];
            this->next_gen[max_sim_index]=NULL;
            this->set_score_population(i,this->get_score_next_gen(i));
            this->set_metric_population(i,this->get_metric_next_gen(i));

            for(int j=0;j<this->predict_size;j++)
                 this->set_predict_population(j,i,this->get_predict_next_gen(j,i));   

            map_next_gen[i]=i;

            if(!this->evaluate_ensemble_next_gen(population_ensemble,map_next_gen, i))
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

    if(!this->evaluate_ensemble_population(population_ensemble))
        return NULL;

    this->compute_similarity();
    this->quick_sort_population();

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
        if(population_ensemble->population[id_ensemble][i]){
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

void Population::write_population(int generation, std::ofstream *evolution_log){
    for(int i=0;i<this->population_size;i++)
        (*evolution_log)<<generation<<";"<<this->population[i]->get_id()<<";"<<this->population[i]->get_string_code()<<";"<<this->get_score_population(i)<<";"<<this->get_metric_population(i)<<"\n";
}


void Population::write_similarity_matrix(int generation, std::ofstream *evolution_log){
    for(int i=0;i<this->population_size;i++)
        for(int j=0;j<this->population_size;j++)
            (*evolution_log)<<generation<<";"<<this->population[i]->get_id()<<";"<<this->population[j]->get_id()<<";"<<this->get_similarity_matrix(i,j)<<"\n";
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

void quicksort(Solution **pop, double *values, int *id_solution, int begin, int end);
void Population::quick_sort_population(){
     quicksort(this->population,this->score_population,this->population_rank,0,this->population_size);   
}

/*Order from max to min*/
void quicksort(Solution **pop, double *values, int *id_solution, int begin, int end){
	int i, j;
    double pivo;
	i = begin;
	j = end-1;
	pivo = values[id_solution[(begin + end) / 2]];
	while(i <= j){
		while(values[id_solution[i]] > pivo && i < end)
			i++;
		
		while(values[id_solution[j]] < pivo && j > begin)
			j--;
		
		if(i <= j){
            std::swap(id_solution[i], id_solution[j]);
			i++;
			j--;
		}
	}
	if(j > begin)
		quicksort(pop, values, id_solution, begin, j+1);
	if(i < end)
		quicksort(pop, values, id_solution, i, end);
}
