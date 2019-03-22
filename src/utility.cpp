#include "utility.h"
#include <string.h>
#include <cstdlib>

char *char_concat(char *c1, const char *c2){  //concatenate two strings with dynamic allocation
  char *concatenated;
  if(!c1){
    concatenated=(char*)malloc(sizeof(char)*(strlen(c2)+1));
    strcpy(concatenated,c2);
  }
  else{
    concatenated=(char*)realloc(c1,sizeof(char)*(strlen(c1)+strlen(c2)+1));
    strcat(concatenated,c2);
  }
  return concatenated;
}


double randDouble(double min, double max){
    double f = (double)rand() / RAND_MAX;
    return min + f*(max - min);
}

int randInt(int min, int max){
    return (rand() % (max-min+1)) + min;
}
