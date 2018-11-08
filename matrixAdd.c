#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include "matrixIO.h"
#include "mpi.h"

#define ONE_BILLION (double)1000000000.0

typedef struct {
  int *values;
  int rows;
  int columns;
} matrix_t;

typedef struct {
  int rows;
  int columns;
} matrix_info_t;

/* Return the current time. */
double now(void)
{
  struct timespec current_time;
  clock_gettime(CLOCK_REALTIME, &current_time);
  return current_time.tv_sec + (current_time.tv_nsec / ONE_BILLION);
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
    fprintf(stderr, "  -h                     print help\n");
    fprintf(stderr, "  -a <input file A>      input file for matrix A\n");
    fprintf(stderr, "  -b <input file B>      input file for matrix B\n");
    fprintf(stderr, "  -o <output file>       set output file\n");
    fprintf(stderr, "  -n <number of threads> number of threads to use\n");

    exit(1);
}

/* gets the number of elements a given processor must add */
int
getElementsForProcessor(int processor, int num_procs, int rows, int columns) {
    int totalElements = rows * columns;
    int elementsForProcessor = totalElements / num_procs;

    if(processor == num_procs - 1) {
        elementsForProcessor = totalElements - elementsForProcessor * (num_procs - 1);
    }

    return elementsForProcessor;
}

/* Multiplies the matrix */ 
int *
add(int num_procs, int rank, int rows, int columns, int* A, int* B) {
    MPI_Status status;
    int* rtn = 0;

    int elementsForProcessor = getElementsForProcessor(rank, num_procs, rows, columns);

    // Receives parts of A and B 
    if(rank != 0) {
        A = malloc(elementsForProcessor * sizeof(int));
        MPI_Recv(A, elementsForProcessor, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        B = malloc(elementsForProcessor * sizeof(int));
        MPI_Recv(B, elementsForProcessor, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    }

    // Sets up C
    int *C = calloc(elementsForProcessor, sizeof(int));

    // Add A and B
    for(int i = 0; i < elementsForProcessor; i++) {
        C[i] = A[i] + B[i];
    } 

    if(rank != 0) {
        MPI_Send(C, elementsForProcessor, MPI_INT, 0, 0, MPI_COMM_WORLD);
    } else {
        rtn = C;
    }

    return rtn;
}

/* Splits a matrix into groups of rows and sends each group to a different processor */
int *
split_and_send(matrix_t M, int num_procs) {    
    // Split and send groups of rows in A
    int elementsPerProcessor = M.rows * M.columns / num_procs;
    int* rtn = malloc(elementsPerProcessor * sizeof(int));

    for (int p = num_procs - 1; p > -1; p--) {
        int elementsForProcessor = getElementsForProcessor(p, num_procs, M.rows, M.columns);
        int offset = p * elementsPerProcessor;

        if(p == 0) {
            memcpy(rtn, M.values, elementsForProcessor * sizeof(int));
        } else {
            MPI_Send(M.values + offset, elementsForProcessor, MPI_INT, p, 0, MPI_COMM_WORLD);
        }
    }
    return rtn;
}

/* Gathers elements of C from all other processors and then returns C */
int* 
gather_and_stitch(int num_procs, int rank, int rows, int columns, int* C) {
    MPI_Status status;

    int *Answer = malloc(rows * columns * sizeof(int));
    int offset = 0;

    for(int p = 0; p < num_procs; p++) {
        int elementsForProcessor = getElementsForProcessor(p, num_procs, rows, columns);

        if(p != 0) {
            C = malloc(elementsForProcessor * sizeof(int));
            MPI_Recv(C, elementsForProcessor, MPI_INT, p, 0, MPI_COMM_WORLD, &status);
        }

        memcpy(Answer + offset, C, elementsForProcessor * sizeof(int));
        offset += elementsForProcessor;
        free(C);
    }

    return Answer;
}

int
main(int argc, char **argv)
{
    // Gather arguments
    char *prog_name = argv[0];

    char* input_file_A = NULL;
    char* input_file_B = NULL;
    char* output_file = NULL;

    int character;
    while ((character = getopt(argc, argv, "ha:b:o:")) != -1) {
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
            case 'h':
            default:
                usage(prog_name, "");
        }
    }

    // Error checking
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

    // Initialize MPI
    int num_procs;
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // If processor 0, load, split, and send out the matrix
    if(rank == 0) {
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

        // Start the timer
        double start_time = now();

        // Send info about the size of A and B
        matrix_info_t *info = malloc(sizeof(matrix_info_t));
        info->rows = A.rows;
        info->columns = A.columns;
        for (int p = 1; p < num_procs; p++) {
            MPI_Send(info, sizeof(matrix_info_t), MPI_BYTE, p, 0, MPI_COMM_WORLD);
        }

        int* Apart = split_and_send(A, num_procs);
        int* Bpart = split_and_send(B, num_procs);

        free(A.values);
        free(B.values);

        fprintf(stderr, "Adding A + B...");
        int* Cpart = add(num_procs, rank, info->rows, info->columns, Apart, Bpart);
        fprintf(stderr, "Done\n");

        // Gather all the results
        int *answer = gather_and_stitch(num_procs, rank, info->rows, info->columns, Cpart);

        // Print time
        printf("    TOOK %5.3f seconds\n", now() - start_time);

        // Write to file
        fprintf(stderr, "Saving output...");
        write_matrix(answer, info->rows, info->columns, output_file);
        fprintf(stderr, "Done\n");

        free(Apart);
        free(Bpart);
        free(answer);
    } else {
        // Receive matrix info
        matrix_info_t *info = malloc(sizeof(matrix_info_t));
        MPI_Status status;
        MPI_Recv(info, sizeof(matrix_info_t), MPI_BYTE, 0, 0, MPI_COMM_WORLD, &status);

        // Multiply
        add(num_procs, rank, info->rows, info->columns, NULL, NULL);
        free(info);
    }

    MPI_Finalize();
}
