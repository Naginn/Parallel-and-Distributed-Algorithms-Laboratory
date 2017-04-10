/* File:     mpi_floyd.c
 *
 *
 * Purpose:  Implements Floyd's algorithm in parallel for solving the all-pairs
 *        shortest path problem:  find the length of the shortest path
 *        between each pair of vertices in a directed graph.
 *
 * Input:    n, the number of vertices in digraph
 *           mat, the adjacency matrix of digraph (user prompted for text file)
 *
 * Output:   A matrix showing the costs of the shortest paths
 *
 * Compile:  mpicc -g -Wall -o mpi_floyd mpi_floyd.c
 *
 * Run:      mpiexec -n <number of processes> ./mpi_floyd
 *
 *
 * Notes:
 * 1.  The input matrix is overwritten by the matrix of lengths of shortest
 *     paths.
 * 2.  Edge lengths should be nonnegative.
 * 3.  If there is no edge between two vertices, the length is the constant
 *     INFINITY.  So input edge length should be substantially less than
 *     this constant.
 * 4.  The cost of travelling from a vertex to itself is 0.  So the adjacency
 *     matrix has zeroes on the main diagonal.
 * 5.  No error checking is done on the input.
 * 6.  The adjacency matrix is stored as a 1-dimensional array and subscripts
 *     are computed using the formula:  the entry in the ith row and jth
 *     column is mat[i*n + j]
 * 7.  The number of vertices (n) must be evenly divisible by the number of
 *    processes for this program to work correctly--i.e., n must be a multiple
 *    of p.
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

const int INFINITY = 1000000;

int getNumberVertices();
char* getFilename();
void readMatrix(char filename[], int mat[], int n);
void printMatrix(int mat[], int n);
int min(int x, int y);
void floyd(int p, int n, int local_mat[], int my_rank);


int main(int argc, char* argv[])
{
    int p;
    int my_rank;
    int n;
    int* mat;
    int* local_mat;
    int* temp_mat;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    /* Gets vertices and filename data from user */
    if (my_rank == 0)
    {
        printf("\nHow many vertices? ");
        scanf("%d", &n);

        mat = malloc(n * n * sizeof(int));

        char filename[40];
        printf("Enter the filename: ");
        scanf("%s", filename);

        readMatrix(filename, mat, n);
    }

    /* Buffer allocation for local rows */
    local_mat = malloc(n * (n/p) * sizeof(int));

    /* Buffer allocation for revised matrix */
    temp_mat = malloc(n * n * sizeof(int));

    /* Broadcasts n (number of "cities") to each processor */
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Distributes matrix among the processors */
    MPI_Scatter(mat, n * (n/p), MPI_INT,
            local_mat, n * (n/p), MPI_INT, 0, MPI_COMM_WORLD);

    /* Uses Floyd's algo to compute least cost between cities */
    floyd(p, n, local_mat, my_rank);

    /* Gathers the data */
    MPI_Gather(local_mat, n * (n/p), MPI_INT, temp_mat,
            n * (n/p), MPI_INT, 0, MPI_COMM_WORLD);

    /* Prints the matrix */
    if (my_rank == 0)
    {
        printf("The solution is:\n");
        printf("\n");
        printMatrix(temp_mat, n);
        printf("\n");
    }

    MPI_Finalize();
    return(0);
} /* main */

/* Creates adjacency matrix from file specified by user */
void readMatrix(char filename[], int mat[], int n)
{
    FILE *file;
    file = fopen(filename, "r");
    int i, j;

    for (i = 0; i < n; i++)
       for (j = 0; j < n; j++)
          fscanf(file, "%d", &mat[i * n + j]);

    fclose(file);
} /* readMatrix */

void printMatrix(int mat[], int n)
{
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            if (mat[i * n + j] == INFINITY)
                printf("i ");
            else
                printf("%d ", mat[i * n + j]);
        }
        printf("\n");
    }
} /* printMatrix */

/* Returns minimum of two ints--used in Floyd's algorithm */
int min(int x, int y)
{
    if (x < y)
        return x;
    else
        return y;
} /* min */

/* Floyd's Algorithm */
void floyd(int p, int n, int local_mat[], int my_rank)
{
    int* row_int_city;
    int local_int_city;
    int root;

    row_int_city = malloc(n * sizeof(int));

    int int_city;
    for (int_city = 0; int_city < n; int_city++) {
        root = int_city / (n / p);
        if (my_rank == root) {
            local_int_city = int_city % (n / p);
            int j;
            for (j = 0; j < n; j++)
                row_int_city[j] = local_mat[local_int_city * n + j];
        }
        MPI_Bcast(row_int_city, n, MPI_INT, root, MPI_COMM_WORLD);
        int local_city1;
        for (local_city1 = 0; local_city1 < n / p; local_city1++)
        {
            int city2;
            for (city2 = 0; city2 < n; city2++)
            {
                local_mat[local_city1 * n + city2] =
                    min(local_mat[local_city1 * n + city2],
                    local_mat[local_city1*n + int_city] + row_int_city[city2]);
            }
        }
    }
    free(row_int_city);
} /* floyd */
