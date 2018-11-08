#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "matrixIO.h"

typedef struct {
  int *values;
  int rows;
  int columns;
} matrix_t;

void
mat_add(int *c, int *a, int *b, int m, int n)
{
  for (int i = 0;  i < m;  i++) {
    for (int j = 0;  j < n;  j++) {
      MAT_ELT(c, n, i, j) += MAT_ELT(a, n, i, j) + MAT_ELT(b, n, i, j);
    }
  }
}

int
compareMatrix(int *a, int *b, int m, int n)
{
  for (int i = 0;  i < m;  i++) {
    for (int j = 0;  j < n;  j++) {
      if (MAT_ELT(a, n, i, j) != MAT_ELT(b, n, i, j)) {
        fprintf(stderr, "Mismatch found at i: %d, j: %d... %d != %d\n", i, j, MAT_ELT(a, n, i, j), MAT_ELT(b, n, i, j));
        return 1;
      }
    }
  }
  return 0;
}

/* Print an optional message, usage information, and exit in error.
 */
void
usage(char *prog_name, char *msge)
{
    if (msge && strlen(msge)) {
        fprintf(stderr, "\n%s\n\n", msge);
    }

    fprintf(stderr, "usage: %s [flags]\n", prog_name);
    fprintf(stderr, "  -h                print help\n");
    fprintf(stderr, "  -a <input file A> input file for matrix A\n");
    fprintf(stderr, "  -b <input file B> input file for matrix B\n");
    fprintf(stderr, "  -o <output file>  set output file\n");
    fprintf(stderr, "  -c <compare file> file to compare matrix to\n");

    exit(1);
}

int
main(int argc, char **argv)
{
  char *prog_name = argv[0];

  char* input_file_A = NULL;
  char* input_file_B = NULL;
  char* output_file = NULL;
  char* compare_file = NULL;

  int character;
  while ((character = getopt(argc, argv, "ha:b:o:c:")) != -1) {
      switch (character) {
          case 'a':
              input_file_A = optarg;
              break;
          case 'b':
              input_file_B = optarg;
              break;
          case 'o':
              output_file = optarg;
              break;
          case 'c':
              compare_file = optarg;
              break;
          case 'h':
          default:
              usage(prog_name, "");
      }
  }

  if (!input_file_A) {
      usage(prog_name, "No input A file specified");
  }
  if (!input_file_B) {
      usage(prog_name, "No input B file specified");
  }
  if (!output_file) {
      usage(prog_name, "No output file specified");
  }
  if (strcmp(input_file_A, output_file) == 0 || strcmp(input_file_B, output_file) == 0) {
    usage(prog_name, "An input and an output file can't be the same");
  }
  if (compare_file && strcmp(compare_file, output_file) == 0) {
    usage(prog_name, "The compare and output file can't be the same");
  }
  
  fprintf(stderr, "Loading matrix A...");
  matrix_t A;
  A.values = read_matrix(&(A.rows), &(A.columns), input_file_A);
  fprintf(stderr, "Done\n");

  fprintf(stderr, "Loading matrix B...");
  matrix_t B;
  B.values = read_matrix(&(B.rows), &(B.columns), input_file_B);
  fprintf(stderr, "Done\n");

  fprintf(stderr, "Checking dimensions...");
  if(A.columns != B.columns) {
    fprintf(stderr, "Dimensions are not correct, A's columns, %d, should equal B's columns, %d", A.columns, B.columns);
    exit(1);
  } else if(A.rows != B.rows) {
    fprintf(stderr, "Dimensions are not correct, A's rows, %d, should equal B's rows, %d", A.rows, B.rows);
    exit(1);
  }
  fprintf(stderr, "Done\n");

  fprintf(stderr, "Adding A + B...");
  matrix_t output;
  output.rows = A.rows;
  output.columns = A.columns;
  output.values = calloc(output.rows * output.columns, sizeof(int));
  mat_add(output.values, A.values, B.values, A.rows, A.columns);
  fprintf(stderr, "Done\n");

  fprintf(stderr, "Saving output...");
  write_matrix(output.values, output.rows, output.columns, output_file);
  fprintf(stderr, "Done\n");

  if(compare_file) {
    matrix_t compare;

    fprintf(stderr, "Loading comparision matrix...");
    compare.values = read_matrix(&(compare.rows), &(compare.columns), compare_file);
    fprintf(stderr, "Done\n");

    fprintf(stderr, "Comparing...");
    if(output.rows != compare.rows || output.columns != compare.columns) {
      fprintf(stderr, "Done\n\n---Matrices do not match---\n\n");
    }
    else if (compareMatrix(compare.values, output.values, output.rows, output.columns) == 0) {
      fprintf(stderr, "Done\n\n---Matrices match---\n\n");
    }
    else {
      fprintf(stderr, "Done\n\n---Matrices do not match---\n\n");
    }
    free(compare.values);
  }

  fprintf(stderr, "Cleaning up...");
  free(output.values);
  free(A.values);
  free(B.values);
  fprintf(stderr, "Done\n");
}
