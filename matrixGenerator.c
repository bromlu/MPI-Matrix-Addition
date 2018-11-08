#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include "matrixIO.h"

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
    fprintf(stderr, "  -o <output file>  set output file\n");
    fprintf(stderr, "  -r <rows>         number of rows to generate, defaults to 100:\n");
    fprintf(stderr, "  -c <columns>      number of columns to generate, defaults to 100:\n");
    fprintf(stderr, "  -m <max>          max integer to generate, defaults to 999:\n");

    exit(1);
}

int * generate_matrix(int rows, int columns, int max) {

    int number_of_elements = rows * columns;
    int *matrix = malloc(number_of_elements * sizeof(int));

    srand(time(0));

    for(int i = 0; i < number_of_elements; i++) {
        matrix[i] = rand() % (max + 1);
    }

    return matrix;
}

int
main(int argc, char **argv)
{
    char *prog_name = argv[0];

    int rows = 100;
    int columns = 100;
    int max = 999;
    char* output_file = NULL;

    int character;
    while ((character = getopt(argc, argv, "ho:r:c:m:")) != -1) {
        switch (character) {
            case 'o':
                output_file = optarg;
                break;
            case 'r':
                rows = atoi(optarg);
                break;
            case 'c':
                columns = atoi(optarg);
                break;
            case 'm':
                max = atoi(optarg);
                break;
            case 'h':
            default:
                usage(prog_name, "");
        }
    }

    if (!output_file) {
        usage(prog_name, "No output file specified");
    }

    int *matrix = generate_matrix(rows, columns, max);
    write_matrix(matrix, rows, columns, output_file);
}
