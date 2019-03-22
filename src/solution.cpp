#include "solution.h"
#include <stdlib.h>
#include <string.h>
#include <stack>
#include "utility.h"

#define MAX_DEEP 20

#include <iostream>

unsigned int Solution::solution_id_count=0;

Solution::Solution(Grammar *grammar){
    this->root=(Solution::Node*)malloc(sizeof(Solution::Node));
    this->root->nonterminal=grammar->get_start_grammar();
    this->root->children_count=0;
    this->root->children=NULL;

    this->id=Solution::solution_id_count++;

    this->root->keyword=(char*)malloc(sizeof(char)*(strlen(grammar->get_start_grammar()->get_nonterminal_keyword())+1));
    strcpy(this->root->keyword,grammar->get_start_grammar()->get_nonterminal_keyword());

    this->string_code=NULL;
}

Solution::~Solution(){
    this->root->free_subtree();
    free(this->root->keyword);
    free(this->root);
    free(this->string_code);
}

void Solution::init_solution(){
    this->root->expand_node_rand(0);
    this->interpret_derivation_tree();
}

void Solution::reset_index(){
    Solution::solution_id_count=0;
}


Solution::Node *Solution::Node::new_children(Grammar::Term *term){
    this->children_count++;

    if(!this->children)
      this->children=(Solution::Node*)malloc(sizeof(Solution::Node));
    else
      this->children=(Solution::Node*)realloc(this->children,sizeof(Solution::Node)*this->children_count);

    this->children[this->children_count-1].set_value(term);

    return &(this->children[this->children_count-1]);
}

void Solution::Node::set_value(Grammar::Term *term){
    this->nonterminal=term->get_nonterminal();
    this->children=NULL;
    this->children_count=0;
    this->keyword=(char*)malloc(sizeof(char)*(strlen(term->get_keyword())+1));
    strcpy(this->keyword,term->get_keyword());
}

int Solution::Node::expand_node_rand(int deep){
    if(!this->nonterminal) return 0;
    if(deep>MAX_DEEP){
        std::cout<<"Max deep reached.\n";
        return 0;
    }

    int expression_choice=randInt(0,this->nonterminal->get_expression_count()-1);
    Grammar::Expression* expression_aux=this->nonterminal->get_expression(expression_choice);

    for(int i=0;i<expression_aux->get_term_count();i++)
        this->new_children(expression_aux->get_term(i));

    for(int i=0;i<this->children_count;i++){
        if(this->children[i].nonterminal)
            this->children[i].expand_node_rand(deep+1);
    }

    return 1;
}

/*Mutation procedure. VERIFY if a mutation equivalent to the original subtree is valid.*/
void Solution::mutation(Solution* sol){
    double sum_wheel=sol->sum_probability(NULL);
    double choice_value=randDouble(0,sum_wheel);

    Node* choice_node=sol->pick_node_probability(NULL,choice_value);

    choice_node->free_subtree();

    choice_node->expand_node_rand(0);

    sol->interpret_derivation_tree();
}

void Solution::crossover(Solution* sol1, Solution* sol2){
    Node* choice_node_sol1=NULL;
    Node* choice_node_sol2=NULL;

    double sum_wheel;
    double choice_value;

    do{
        sum_wheel=sol1->sum_probability(NULL);
        choice_value=randDouble(0,sum_wheel);

        choice_node_sol1=sol1->pick_node_probability(NULL,choice_value);

        sum_wheel=sol2->sum_probability(choice_node_sol1->nonterminal);

        if(sum_wheel==0)
            choice_node_sol2=NULL;
        else{
            choice_value=randDouble(0,sum_wheel);
            choice_node_sol2=sol2->pick_node_probability(choice_node_sol1->nonterminal,choice_value);
        }
        
    }while(!choice_node_sol2);

    Solution::Node::swap_subtrees(choice_node_sol1, choice_node_sol2);

    sol1->interpret_derivation_tree();
    sol2->interpret_derivation_tree();
}



void Solution::interpret_derivation_tree(){
    if(this->string_code)
        free(this->string_code);
    
    this->string_code=NULL;
    std::stack <Node*> node_stack;
    Node *node_aux=this->root;
    if(!node_aux->children){
	    this->string_code=char_concat(this->string_code,node_aux->keyword);
        return; //end of process. There isn't nonterminal to solve.
    }

    do{
        for(int j=node_aux->children_count-1;j>=0;j--){
            node_stack.push(&node_aux->children[j]); 
        }

        do{
            node_aux=node_stack.top();
            if(!node_aux->nonterminal)
	            this->string_code=char_concat(this->string_code,node_stack.top()->keyword);
            node_stack.pop();

        }while(!node_aux->children && !node_stack.empty());
    }while(node_aux->children);
}

/*Sum all nonterminals in the tree who coincided with nonterminal_ref (if NULL sum among all nonterminals)*/
double Solution::sum_probability(Grammar::NonTerminal* nonterminal_ref){
    if(!this->root->nonterminal)
        throw "Start nonterminal not found.";

    return this->root->sum_probability_subtree(nonterminal_ref);
}

double Solution::Node::sum_probability_subtree(Grammar::NonTerminal* nonterminal_ref){
    double local_sum=0;

    if(!nonterminal_ref || this->nonterminal==nonterminal_ref)
        local_sum+=1;

    for(int i=0;i<this->children_count;i++)
         if(this->children[i].nonterminal)        
            local_sum+=this->children[i].sum_probability_subtree(nonterminal_ref);
    
    return local_sum;
}

/*Pick a node based on the propapility_select value and nonterminal_ref (roulette wheel method)*/
Solution::Node* Solution::pick_node_probability(Grammar::NonTerminal* nonterminal_ref, double probability_select){
    return this->root->pick_node_probability_subtree(nonterminal_ref, &probability_select);
}

Solution::Node* Solution::Node::pick_node_probability_subtree(Grammar::NonTerminal* nonterminal_ref, double *probability_select){
    if(!nonterminal_ref || this->nonterminal==nonterminal_ref)
        *probability_select-=1;

    if(*probability_select<0) return this;

    Node* node_return=NULL;

    for(int i=0;i<this->children_count;i++)
         if(this->children[i].nonterminal)       
            if((node_return=this->children[i].pick_node_probability_subtree(nonterminal_ref,probability_select)))
                return node_return;
    
    return node_return;
}

char* Solution::get_string_code(){
    return this->string_code;
}

int Solution::get_id(){
    return this->id;
}

void Solution::Node::free_subtree(){
    if(!this->children) return;

    for(int i=0;i<this->children_count;i++)
       this->children[i].free_children_i();

    free(this->children);  
    this->children_count=0;
    this->children=NULL; 
}

void Solution::Node::free_children_i(){
    for(int i=0;i<this->children_count;i++)
       this->children[i].free_children_i();

    if(this->children)
        free(this->children);
    
    free(this->keyword);
}

void Solution::Node::swap_subtrees(Solution::Node* node1, Solution::Node* node2){
    std::swap(node1->children_count,node2->children_count);
    std::swap(node1->children,node2->children);
}


Solution::Solution(){
    this->root=(Solution::Node*)malloc(sizeof(Solution::Node));
    this->id=solution_id_count++;
}

void Solution::Node::copy_subtree(Solution::Node* copy, Solution::Node* original){
    copy->nonterminal=original->nonterminal;
    copy->keyword=(char*)malloc(sizeof(char)*(strlen(original->keyword)+1));
    strcpy(copy->keyword,original->keyword);

    copy->children_count=original->children_count;
    if(copy->children_count)
        copy->children=(Solution::Node*)malloc(sizeof(Solution::Node)*copy->children_count);
    else
        copy->children=NULL;

    for(int i=0;i<copy->children_count;i++)
        Solution::Node::copy_subtree(&(copy->children[i]),&(original->children[i]));
}

Solution* Solution::copy(){
    Solution* copy=new Solution();

    Solution::Node::copy_subtree(copy->root,this->root);

    copy->string_code=(char*)malloc(sizeof(char)*(strlen(this->string_code)+1));
    strcpy(copy->string_code,this->string_code);

    return copy;
}

void Solution::print_tree(){
    this->root->print_tree(0);
}

void Solution::Node::print_tree(int deep){
    for(int i=0;i<deep;i++) std::cout<<"\t";
    std::cout<<this->keyword;

    for(int i=0;i<this->children_count;i++)
        this->children[i].print_tree(deep+1);

    if(!this->children) std::cout<<"\n";
}
