#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#define MAX_LENGTH 50000000

double get_time() // function to get the time of day in seconds
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

void sequential_histogram(char *data, unsigned int *histogram, int length) {
  for (int i = 0; i < length; i++) {
    int alphabet_position = data[i] - 'a';
    if (alphabet_position >= 0 && alphabet_position < 26) //check if we have an alphabet char
      histogram[alphabet_position / 6]++;                 //we group the letters into blocks of 6
  }
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Please provide a filename as an argument.\n");
    return 1;
  }

  const char *filename = argv[1];
  FILE *fp             = fopen(filename, "read");

  // unsigned char text[MAX_LENGTH];
  char *text = (char *) malloc(sizeof(char) * MAX_LENGTH);
  size_t len = 0;
  size_t read;
  unsigned int histogram[5] = {0};

  if (fp == NULL)
    exit(EXIT_FAILURE);

  while ((read = getline(&text, &len, fp)) != -1) { printf("Retrieved line of length %ld:\n", read); }
  fclose(fp);

  sequential_histogram(text, histogram, len);

  printf("a-f: %d, g-l: %d, m-r: %d, s-x: %d, y-z: %d\n",
         histogram[0],
         histogram[1],
         histogram[2],
         histogram[3],
         histogram[4]);

  return 1;
}