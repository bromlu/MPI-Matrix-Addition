#include <stdio.h>
#include <stdlib.h>
#include "matrixIO.h"

int * read_matrix(int *rows, int *columns, char *file_name) {
    FILE *file;

    if((file = fopen(file_name, "r")) == NULL) {
        fprintf(stderr, "Couldn't read from %s, please try another file", file_name);
        exit(1);
    }

    fscanf(file, "%d %d\n", rows, columns);
    int number_of_elements = (*rows) * (*columns);
    int *matrix = malloc(number_of_elements * sizeof(int));

    for(int i = 0; i < number_of_elements; i++) {
        fscanf(file, "%d", matrix + i);
    }

    fclose(file);

    return matrix;
}

void write_matrix(int *matrix, int rows, int columns, char *file_name) {
    FILE *file;

    if((file = fopen(file_name, "w")) == NULL) {
        fprintf(stderr, "Couldn't write to %s, please try another file", file_name);
        exit(1);
    }

    fprintf(file, "%d %d\n", rows, columns);
    for(int r = 0; r < rows; r++) {
        for(int c = 0; c < columns; c++) {
            fprintf(file, " %5d", MAT_ELT(matrix, columns, r, c));
        }
        fprintf(file, "\n");
    }

    fclose(file);
}