#include "grammar.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stack>
#include <string.h>
#include <math.h> 
#include "utility.h"

#define NTRIES 5 
#define BUFFERSIZE 5000
#define INTERVALCOUNT 20


char *process_token(char* buffer, int *nonterminal);

/*Load grammar. First, the procedure reads a file containing the grammar described. After that, it connects all nonterminals with the respective expression.*/
Grammar::Grammar(const char *filename, PythonInterface *interface){
    std::ifstream grammar_file;
    grammar_file.open(filename);
    if(!grammar_file.is_open())
        throw "GRAMMAR NOT FOUND";

    this->nonterminal_count=0;
    this->nonterminal_set=NULL;
    this->constint_count=0;
    this->constfloat_count=0;
    this->nfeat_const=interface->get_n_feat_dataset();

    char line[BUFFERSIZE];
    while(grammar_file.getline(line,BUFFERSIZE)){
        int nonterminal_flag;

	    char* buffer=line;

	    char *p_token;
	    if(!(p_token=process_token(buffer,&nonterminal_flag))) continue;
	
	    NonTerminal *nonterminal_aux;

	    int search_nonterminal;
	    if((search_nonterminal=find_nonterminal(p_token))==-1)
     	  nonterminal_aux=this->new_nonterminal(p_token);
	    else
      	  nonterminal_aux=&this->nonterminal_set[search_nonterminal];


	    buffer=strstr(line,"::=")+3;
	
        char *expression, *expression_end;
	    for(expression=buffer;expression;expression=expression_end){
	      
	      Grammar::Expression *expression_aux=nonterminal_aux->new_expression();
	        
            char *term;
	        expression_end=strstr(expression,"|");
	        if(expression_end)expression_end++;
	        for(term=expression;term && (!expression_end || term<expression_end);term=strstr(term," ")){
	             if(term)term++;
	             if(term[0]!='\"' && term[0]!='\'' && term[0]!='<') continue;
		         p_token=process_token(term,&nonterminal_flag);
		         expression_aux->new_term(p_token,nonterminal_flag);
	        }
	        
	    }	
    }

    for(int i=0;i<this->nonterminal_count;i++){
      for(int j=0;j<this->nonterminal_set[i].expression_count;j++){
	    Grammar::Expression* expression_aux=&this->nonterminal_set[i].expression_set[j]; //A direct reference to expression_set is needed, given the realloc command in the nonterminal_set array
        for(int k=0;k<expression_aux->term_count;k++){
            Grammar::Term* term_aux=&expression_aux->term_list[k];
		    if(strstr(term_aux->keyword,"CONSTINT")){
		      char *end_ptr=NULL;
		      int begin_interval;
              begin_interval=strtol(&term_aux->keyword[9],&end_ptr,10);
              if(term_aux->keyword+9==end_ptr){
                char error_msg[BUFFERSIZE];
                sprintf(error_msg, "Bad interval defined in %s \n", term_aux->keyword);
                throw error_msg;
              }

		      int end_interval;
              char *check_error=NULL;
              end_interval=strtol(end_ptr+1,&check_error,10);
              if(end_ptr+1==check_error){
                char error_msg[BUFFERSIZE];
                sprintf(error_msg, "Bad interval defined in %s \n", term_aux->keyword);
                throw error_msg;
              }

		      if(begin_interval>end_interval)std::swap(begin_interval,end_interval);
		        this->process_interval_Int(term_aux,begin_interval,end_interval);

		    }else if(strstr(term_aux->keyword,"CONSTFLOAT")){
		      char *end_ptr=NULL;
		      float begin_interval;
              begin_interval=strtod(&term_aux->keyword[11],&end_ptr);
              if(term_aux->keyword+11==end_ptr){
                char error_msg[BUFFERSIZE];
                sprintf(error_msg, "Bad interval defined in %s \n", term_aux->keyword);
                throw error_msg;
              }
		  
              float end_interval;
              char *check_error=NULL;
              end_interval=strtod(end_ptr+1,&check_error);
              if(end_ptr+1==check_error){
                char error_msg[BUFFERSIZE];
                sprintf(error_msg, "Bad interval defined in %s \n", term_aux->keyword);
                throw error_msg;
              }

		      if(begin_interval>end_interval)std::swap(begin_interval,end_interval);
		        this->process_interval_Float(term_aux,begin_interval,end_interval);

		    }
	    }
      }
    }

    /*Need to update the nonterminal pointers in each terminal after every nonterminal addition (because of realloc command)*/
    for(int i=0;i<this->nonterminal_count;i++){
        Grammar::NonTerminal *nonterminal_aux=&this->nonterminal_set[i];
        for(int j=0;j<nonterminal_aux->expression_count;j++){
            Grammar::Expression* expression_aux=&nonterminal_aux->expression_set[j];
            for(int k=0;k<expression_aux->term_count;k++){
                Grammar::Term* term_aux=&expression_aux->term_list[k];

                if(!term_aux->nonterminal_flag) continue;

                int index_nonterminal;
                if((index_nonterminal=this->find_nonterminal(term_aux->keyword))==-1){
                    throw "Bad grammar specification. Nonterminal rule not found.";
                }

                term_aux->nonterminal=&(this->nonterminal_set[index_nonterminal]);
            }
        }
    }
    
    this->start_grammar=&this->nonterminal_set[0];
}

Grammar::~Grammar(){
    for(int i=0;i<this->nonterminal_count;i++){
        this->nonterminal_set[i].delete_expression_set();
        free(this->nonterminal_set[i].keyword);
    }

    free(this->nonterminal_set);
}

int Grammar::process_interval_Int(Grammar::Term *term_interval, int begin_interval, int end_interval){
   int size_char=snprintf((char*)NULL,0,"%d",constint_count);
   char *p_token=(char *)malloc(sizeof(char)*(strlen(term_interval->keyword)+size_char+1+1)); //+1 for '#' character and +1 for '\0'
   strcpy(p_token,term_interval->keyword);
   strcat(p_token,"#");
   sprintf(p_token+strlen(p_token),"%d",constint_count);
   
   Grammar::NonTerminal* nonterminal_aux=this->new_nonterminal(p_token);
   p_token=(char *)malloc(sizeof(char)*(strlen(p_token)+1));
   strcpy(p_token,nonterminal_aux->keyword);
   free(term_interval->keyword);
   term_interval->set_value(p_token,true);

   int step=round(end_interval-begin_interval)/(float)INTERVALCOUNT;
   if(step<1) step=1;
   for(int i=begin_interval;i<end_interval-step*0.5;i=round(i+step)){ //avoiding that in the end of the list two elements are more close than 0.5*step given that the end of the interval is added always
    Grammar::Expression* expression_aux=nonterminal_aux->new_expression();
    size_char=snprintf((char*)NULL,0,"%d",i);
    char *buffer=(char*)malloc(sizeof(char)*(size_char+1));
    sprintf(buffer,"%d",i);
    expression_aux->new_term(buffer,false);
  }

  Grammar::Expression* expression_aux=nonterminal_aux->new_expression();
  size_char=snprintf((char*)NULL,0,"%d",end_interval);
  char *buffer=(char*)malloc(sizeof(char)*(size_char+1));
  sprintf(buffer,"%d",end_interval);
  expression_aux->new_term(buffer,false);


  term_interval->nonterminal=nonterminal_aux;
  this->constint_count++;
  
  return 1;
}

int Grammar::process_interval_Float(Grammar::Term *term_interval, float begin_interval, float end_interval){
   int size_char=snprintf((char*)NULL,0,"%d",constfloat_count);
   char *p_token=(char *)malloc(sizeof(char)*(strlen(term_interval->keyword)+size_char+1+1)); //+1 for '#' character and +1 for '\0'
   strcpy(p_token,term_interval->keyword);
   strcat(p_token,"#");
   sprintf(p_token+strlen(p_token),"%d",constfloat_count);

   Grammar::NonTerminal* nonterminal_aux=this->new_nonterminal(p_token);
   p_token=(char *)malloc(sizeof(char)*(strlen(p_token)+1));
   strcpy(p_token,nonterminal_aux->keyword);
   free(term_interval->keyword);
   term_interval->set_value(p_token,true);

   float step=(end_interval-begin_interval)/(float)INTERVALCOUNT;
   for(float i=begin_interval;i<end_interval-step*0.5;i=(i+step)){  //avoiding that in the end of the list two elements are more close than 0.5*step given that the end of the interval is added always
    Grammar::Expression* expression_aux=nonterminal_aux->new_expression();
    size_char=snprintf((char*)NULL,0,"%.3g",i);
    char *buffer=(char*)malloc(sizeof(char)*(size_char+1));
    sprintf(buffer,"%.3g",i);
    expression_aux->new_term(buffer,false); 
  }

  Grammar::Expression* expression_aux=nonterminal_aux->new_expression();
  size_char=snprintf((char*)NULL,0,"%.3g",end_interval);
  char *buffer=(char*)malloc(sizeof(char)*(size_char+1));
  sprintf(buffer,"%.3g",end_interval);
  expression_aux->new_term(buffer,false);


  term_interval->nonterminal=nonterminal_aux;
  this->constfloat_count++;
  
  return 1;
}

std::ostream& operator<<(std::ostream &out, Grammar &grammar){
    char *grammar_out=grammar.print_grammar();
    out << grammar_out;
    free(grammar_out);
    return out;
}

/*Print loaded grammar. Check how to deallocate the grammar_out vector.*/

char *Grammar::print_grammar(){
    char *grammar_out=NULL;
    for(int i=0;i<this->nonterminal_count;i++){
        Grammar::NonTerminal *nonterminal_aux=&this->nonterminal_set[i];
        grammar_out=char_concat(char_concat(char_concat(grammar_out,"<"),nonterminal_aux->keyword),"> ::= ");
        for(int j=0;j<nonterminal_aux->expression_count;j++){
            Grammar::Expression* expression_aux=&nonterminal_aux->expression_set[j];
            for(int k=0;k<expression_aux->term_count;k++){
                Grammar::Term* term_aux=&expression_aux->term_list[k];

		if(term_aux->nonterminal_flag)
		     grammar_out=char_concat(char_concat(char_concat(grammar_out,"<"),term_aux->keyword),">");
                else
		     grammar_out=char_concat(char_concat(char_concat(grammar_out,"\""),term_aux->keyword),"\"");
             }
	     grammar_out=char_concat(grammar_out,"|");
        }
        grammar_out=char_concat(grammar_out,"\n");
    }
  
    return grammar_out;
}

/*Return the next token or NULL if it isn't found */
char *process_token(char *buffer, int *nonterminal){
    if(!buffer || buffer[0]=='#') return NULL;

    char *index_start_NT, *index_start_T, *token;
    index_start_NT=strstr(buffer,"<");
    index_start_T=strstr(buffer,"\"");
    if(!index_start_T) index_start_T=strstr(buffer,"\'");
    
    if(index_start_NT && (!index_start_T || index_start_NT<index_start_T)){
        char *index_end=strstr(index_start_NT+1,">");
	    if(!index_end){
            char error_msg[BUFFERSIZE];
            sprintf(error_msg, "Cannot find the end of nonterminal in %s \n", buffer);
            throw error_msg;
	    }

	    token=(char*)malloc(sizeof(char)*(index_end-index_start_NT-2+2));//end-start-(2 '\"' character) plus '\0' plus start character
	    index_start_NT++;
	    int i=0;
	    for(i=0;index_start_NT<index_end;token[i++]=*index_start_NT++);
	    token[i]='\0';
        *nonterminal=1;
    }else if(index_start_T){
        char *index_end;
	    if(index_start_T[0]=='\"') index_end=strstr(index_start_T+1,"\"");
	    else if(index_start_T[0]=='\'') index_end=strstr(index_start_T+1,"\'");
	    if(!index_end){
            char error_msg[BUFFERSIZE];
            sprintf(error_msg, "Cannot find the end of terminal in %s \n", buffer);
            throw error_msg;
	    }
	
	    token=(char*)malloc(sizeof(char)*(index_end-index_start_T-2+2));//end-start-(2 '\"' character) plus '\0' plus start character
	    index_start_T++;
	    int i=0;
	    for(i=0;index_start_T<index_end;token[i++]=*index_start_T++);
	    token[i]='\0';
        *nonterminal=0;
    }else return NULL;

    return token;
}

int Grammar::find_nonterminal(char *token){
    for(int i=0;i<this->nonterminal_count;i++)
        if(!strcmp(this->nonterminal_set[i].keyword,token)) return i;

    return -1;    
}


Grammar::NonTerminal* Grammar::new_nonterminal(char *token){
    this->nonterminal_count++;
	if(!this->nonterminal_set)
	  this->nonterminal_set=(Grammar::NonTerminal*)malloc(sizeof(Grammar::NonTerminal));
	else
	  this->nonterminal_set=(Grammar::NonTerminal*)realloc(this->nonterminal_set,sizeof(Grammar::NonTerminal)*this->nonterminal_count);
    this->nonterminal_set[this->nonterminal_count-1].set_value(token);

    return &(this->nonterminal_set[this->nonterminal_count-1]);

}

Grammar::Expression* Grammar::NonTerminal::new_expression(){
    this->expression_count++;
    if(!this->expression_set)
      this->expression_set=(Grammar::Expression*)malloc(sizeof(Grammar::Expression));
    else
      this->expression_set=(Grammar::Expression*)realloc(this->expression_set,sizeof(Grammar::Expression)*this->expression_count);

    this->expression_set[this->expression_count-1].set_value();

    return &(this->expression_set[this->expression_count-1]);
}

Grammar::Term* Grammar::Expression::new_term(char *token, int nonterminal_flag){
    this->term_count++;
    if(!this->term_list)
      this->term_list=(Grammar::Term*)malloc(sizeof(Grammar::Term));
    else
      this->term_list=(Grammar::Term*)realloc(this->term_list,sizeof(Grammar::Term)*this->term_count);

    this->term_list[this->term_count-1].set_value(token, nonterminal_flag);

    return &(this->term_list[this->term_count-1]);
}

void Grammar::NonTerminal::set_value(char* token){
    this->keyword=token;
    this->expression_set=NULL;
    this->expression_count=0;
}

void Grammar::Expression::set_value(){
    this->term_count=0;
    this->term_list=NULL;
}

void Grammar::Term::set_value(char *token, int nonterminal_flag){
    this->keyword=token;
    this->nonterminal=NULL;
    this->nonterminal_flag=nonterminal_flag;
}


int Grammar::NonTerminal::get_expression_count(){
    return this->expression_count;
}

Grammar::Expression* Grammar::NonTerminal::get_expression(int expression){
    if(expression<0 || expression>this->expression_count)
        throw "Invalid expression choice";

    return &this->expression_set[expression];
}

int Grammar::Expression::get_term_count(){
    return this->term_count;
}

Grammar::Term* Grammar::Expression::get_term(int term){
    if(term<0 || term>this->term_count)
        throw "Invalid term choice";
    
    return &this->term_list[term];
}


Grammar::NonTerminal* Grammar::Term::get_nonterminal(){
    return this->nonterminal;
}

char* Grammar::Term::get_keyword(){
    return keyword;
}

char* Grammar::NonTerminal::get_nonterminal_keyword(){
    return this->keyword;
}

Grammar::NonTerminal* Grammar::get_start_grammar(){
    return this->start_grammar;
}

void Grammar::NonTerminal::delete_expression_set(){
    for(int i=0;i<this->expression_count;i++)
        this->expression_set[i].delete_term_list();

    free(this->expression_set);
}

void Grammar::Expression::delete_term_list(){
    for(int i=0;i<this->term_count;i++)
        free(this->term_list[i].keyword);
    
    free(this->term_list);   
}

