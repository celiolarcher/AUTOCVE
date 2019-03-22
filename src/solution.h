#ifndef SOLUTIONH
#define SOLUTIONH
#include "grammar.h"


class Solution{
    class Node{
        private:Grammar::NonTerminal *nonterminal;
        Node *children;
        int children_count;
        char *keyword;

        private: Node *new_children(Grammar::Term *term);
        private: void set_value(Grammar::Term *term);
        private: void free_subtree();
        private: void free_children_i();
        private: int expand_node_rand(int deep);
        private: double sum_probability_subtree(Grammar::NonTerminal* nonterminal_ref);
        private: Node* pick_node_probability_subtree(Grammar::NonTerminal* nonterminal_ref, double *probability_select);
        private: static void swap_subtrees(Node* node1, Node* node2);
        private: static void copy_subtree(Node* copy, Node* original);
        public: void print_tree(int deep);

        friend class Solution;    
    };
    
    private: Node *root;
    private: char* string_code;
    private: unsigned int id;
    public: Solution(Grammar *grammar);
    private: Solution();
    public: ~Solution();
    public: void init_solution();
    public: Solution* copy();
    

    private: double sum_probability(Grammar::NonTerminal* nonterminal_ref);
    private: Node* pick_node_probability(Grammar::NonTerminal* nonterminal_ref, double probability_select);
    public: static void mutation(Solution* sol);
    public: static void crossover(Solution* sol1, Solution* sol2);

    private: void interpret_derivation_tree();
    public: char* get_string_code();
    public: void print_tree();

    private: static unsigned int solution_id_count;
    public: static void reset_index();
    public: int get_id();
};


#endif
