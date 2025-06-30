/*
* This program takes in input a text and a string, being two arrays of char values, and 
* computes how many times the string appears in the text. In particular:
* - function findString receives in input the text and the string and saves in each 
*   position i of a third vector called match, 1 if an occurrence of the string has been 
*   found in the text starting the index i, 0 otherwise.
* - function countMatches receives the vector match in input and count the number of values 
*   equal to 1 (i.e., it counts the number of occurrences of the string in the text).
* - the main function receives as arguments the size of text and string and (for the sake 
*   of brevity) generates randomly the content of the two vectors, invokes the two 
*   functions above and prints the result on the screen.
*/

#include <stdio.h>
#include <stdlib.h>

#define MAXVAL 2

void printText(char *text, int num);
void findString(char* text, int textDim, char* str, int strDim, char* match);
int countMatches(char *match, int num);

//display a vector of numbers on the screen
void printText(char *text, int num) {
  int i;
  for(i=0; i<num; i++)
    printf("%c", text[i]);
  printf("\n");    
}

//kernel function 1: identify strings in the text
void findString(char* text, int textDim, char* str, int strDim, char* match) {
  int i, j, ok;
  for(i=0; i<textDim-strDim+1; i++){
    for(j=0, ok=1; j<strDim && ok; j++){
      if(text[i+j]!=str[j])
        ok=0;
    }
    match[i] = ok;
  }
}

//kernel function 2: count matches
int countMatches(char *match, int num) {
  int i, count;
  for(i=0, count=0; i<num; i++)
    count+=match[i];
  return count;    
}

int main(int argc, char **argv) {
  char *text, *str, *match;
  int count;
  int textDim, strDim;
  int i;

  //read arguments
  if(argc!=3){
    printf("Please specify sizes of the two input vectors\n");
    return 0;
  }
  textDim=atoi(argv[1]);
  strDim=atoi(argv[2]);
  
  //allocate memory for the three vectors
  text = (char*) malloc(sizeof(char) * (textDim));
  if(!text){
    printf("Error: malloc failed\n");
    return 1;
  }
  str = (char*) malloc(sizeof(char) * (strDim));
  if(!str){
    printf("Error: malloc failed\n");
    return 1;
  }
  match = (char*) malloc(sizeof(char) * (textDim-strDim+1));
  if(!match){
    printf("Error: malloc failed\n");
    return 1;
  }

  //initialize input vectors
  srand(0);
  for(i=0; i<textDim; i++)
    text[i] = rand()%MAXVAL +'a';

  for(i=0; i<strDim; i++)
    str[i] = rand()%MAXVAL +'a';
  
  //execute on CPU  
  findString(text, textDim, str, strDim, match);
  count = countMatches(match, textDim-strDim+1);
 
  //print results
  printText(text, textDim);
  printText(str, strDim);
  printf("%d\n", count);
  
  free(text);
  free(str);
  free(match);
  
  return 0;
}
