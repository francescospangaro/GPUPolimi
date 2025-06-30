#include <stdio.h>
#include <stdlib.h>

#define CHANNELS 3
#define OUT_FN_CPU "output.pgm"
#define BLURDIM 10

int save_ppm_image(const char* filename, unsigned char* image, unsigned int width, unsigned int height);
int save_pgm_image(const char* filename, unsigned char* image, unsigned int width, unsigned int height);
int load_ppm_image(const char* filename, unsigned char** image, unsigned int* width, unsigned int* height);
int load_pgm_image(const char* filename, unsigned char** image, unsigned int* width, unsigned int* height);
void rgb2gray(unsigned char* input, unsigned char* output, unsigned int width, unsigned int height);
void blur(unsigned char* input, unsigned char* output, unsigned int width, unsigned int height);


void rgb2gray(unsigned char* input, unsigned char* output, unsigned int width, unsigned int height){
  int i, j;
  unsigned char redValue, greenValue, blueValue, grayValue;
  //loop on all pixels and convert from RGB to gray scale
  for(i=0; i<height; i++){
    for(j=0; j<width; j++){
      redValue = input[(i*width + j)*3];
      greenValue = input[(i*width + j)*3+1];
      blueValue = input[(i*width + j)*3+2];
      grayValue = (unsigned char) (0.299*redValue + 0.587*greenValue + 0.114*blueValue);
      output[i*width + j] = grayValue;
    }
  }
}

void blur(unsigned char* input, unsigned char* output, unsigned int width, unsigned int height){
  int i, j, h, k, sum, count;
  //loop on all pixels and to compute the mean value of the intensity together with the 8 neighbor pixels
  for(i=0; i<height; i++){
    for(j=0; j<width; j++){
      count = 0;
      sum = 0;
      for(h=-BLURDIM; h<=BLURDIM; h++)
        for(k=-BLURDIM; k<=BLURDIM; k++) 
          if(i+h>=0 && i+h<height && j+k>=0 && j+k<width){
            count++;
            sum = sum + input[(i+h)*width + (j+k)];
          }
      output[i*width + j] = (float) sum / count;
    }
  }
}

int main(int argc, char* argv[]) {
  char* inputfile;
  unsigned int height, width;
  unsigned char *input, *gray, *output;
  int nPixels;
  int err;

  //read arguments
  if(argc!=2){
    printf("Please specify ppm input file name\n");
    return 0;
  }
  inputfile = argv[1];
  
  //load input image
  err = load_ppm_image(inputfile, &input, &width, &height);
  if(err)
    return 1;
  nPixels = width * height;

  //allocate memory for gray image
  gray = (unsigned char*) malloc(sizeof(unsigned char) * nPixels);
  if(!gray){
    printf("Error with malloc\n");
    free(input);
    return 1;
  }

  //allocate memory for output image
  output = (unsigned char*) malloc(sizeof(unsigned char) * nPixels);
  if(!output){
    printf("Error with malloc\n");
    free(gray);
    free(input);
    return 1;
  }
  
  //process image
  rgb2gray(input, gray, width, height);
  blur(gray, output, width, height);

  //save output image
  err = save_pgm_image(OUT_FN_CPU, output, width, height);
  if(err){
    free(input);
    free(gray);
    free(output);
    return 1;
  }

  //cleanup
  free(input);
  free(gray);
  free(output);

  return 0;
}

int save_ppm_image(const char* filename, unsigned char* image, unsigned int width, unsigned int height) {
  FILE *f; //output file handle

  //open the output file and write header info for PPM filetype
  f = fopen(filename, "wb");
  if (f == NULL){
    fprintf(stderr, "Error opening 'output.ppm' output file\n");
    return -1;
  }
  fprintf(f, "P6\n");
  fprintf(f, "%d %d\n%d\n", width, height, 255);
  fwrite(image, sizeof(unsigned char), height*width*CHANNELS, f);
  fclose(f);
  return 0;
}

int save_pgm_image(const char* filename, unsigned char* image, unsigned int width, unsigned int height) {
  FILE *f; //output file handle

  //open the output file and write header info for PPM filetype
  f = fopen(filename, "wb");
  if (f == NULL){
    fprintf(stderr, "Error opening 'output.ppm' output file\n");
    return -1;
  }
  fprintf(f, "P5\n");
  fprintf(f, "%d %d\n%d\n", width, height, 255);
  fwrite(image, sizeof(unsigned char), height*width, f);
  fclose(f);
  return 0;
}


int load_ppm_image(const char* filename, unsigned char** image, unsigned int* width, unsigned int* height) {
  FILE *f; //input file handle
  char temp[256];
  unsigned int s;

  //open the input file and write header info for PPM filetype
  f = fopen(filename, "rb");
  if (f == NULL){
    fprintf(stderr, "Error opening '%s' input file\n", filename);
    return -1;
  }
  fscanf(f, "%s\n", temp);
  fscanf(f, "%d %d\n", width, height);
  fscanf(f, "%d\n",&s);

  *image = (unsigned char*) malloc(sizeof(unsigned char)* (*width) * (*height) * CHANNELS);
  if(*image)
    fread(*image, sizeof(unsigned char), (*width) * (*height) * CHANNELS, f);
  else{
    printf("Error with malloc\n");
    return -1;
  }

  fclose(f);
  return 0;
}

int load_pgm_image(const char* filename, unsigned char** image, unsigned int* width, unsigned int* height) {
  FILE *f; //input file handle
  char temp[256];
  unsigned int s;

  //open the input file and write header info for PPM filetype
  f = fopen(filename, "rb");
  if (f == NULL){
    fprintf(stderr, "Error opening '%s' input file\n", filename);
    return -1;
  }
  fscanf(f, "%s\n", temp);
  fscanf(f, "%d %d\n", width, height);
  fscanf(f, "%d\n",&s);

  *image = (unsigned char*) malloc(sizeof(unsigned char)* (*width) * (*height));
  if(*image)
    fread(*image, sizeof(unsigned char), (*width) * (*height), f);
  else{
    printf("Error with malloc\n");
    return -1;
  }

  fclose(f);
  return 0;
}

