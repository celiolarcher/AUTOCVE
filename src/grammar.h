#ifndef GRAMMARH
#define GRAMMARH
#include <iostream>
#include "python_interface.h"

class Solution;

class Grammar{
    class NonTerminal;
    
    class Term{
        private:char *keyword;
        NonTerminal *nonterminal;  
        int nonterminal_flag;
    
        private: void set_value(char *token, int nonterminal_flag);
        public: NonTerminal* get_nonterminal();
        public: char* get_keyword();

        friend class Grammar;
    };

    class Expression{
        private:Term *term_list;
        int term_count;

        private: Term* new_term(char *token, int nonterminal_flag); //Needed function for allocate more space in the token array (Array of structure, not pointers)
        private: void set_value();
        private: void delete_term_list();
        public: int get_term_count();
        public: Term* get_term(int term);

        friend class Grammar;
    };

    class NonTerminal{
        private:char *keyword;
        Expression *expression_set;
        int expression_count;

        private: void set_value(char *token);
        private: void delete_expression_set();
        private: Expression* new_expression(); //Needed function for allocate more space in the expression array (Array of structure, not pointers)
        public: int get_expression_count();
        public: Expression* get_expression(int expression);
        public: char* get_nonterminal_keyword();

        friend class Grammar;
    };

    private:
    NonTerminal *nonterminal_set;
    NonTerminal *start_grammar;
    int nonterminal_count;
    int constint_count;
    int constfloat_count;
    int nfeat_const;

    public: Grammar(const char *filename, PythonInterface* interface);
    public: ~Grammar();
    private: NonTerminal* new_nonterminal(char *token);
    private: int find_nonterminal(char *token);
    private: int process_interval_Int(Term *term_interval, int interval_begin, int interval_end);
    private: int process_interval_Float(Term *term_interval, float interval_begin, float interval_end);
    public: NonTerminal* get_start_grammar();

    friend std::ostream& operator<<(std::ostream &out, Grammar &sol);  //Format loaded grammar
    public: char* print_grammar();
    friend class Solution;
};

#endif
