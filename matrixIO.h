#define MAT_ELT(mat, cols, i, j) *(mat + (i * cols) + j)

extern int * read_matrix(int *rows, int *columns, char *file_name);
extern void write_matrix(int *matrix, int rows, int columns, char *file_name);