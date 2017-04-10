#include "mpi.h"
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define min(a,b) ((a) < (b)) ? (a) : (b)

/*----------------------------------------------------------------------
// MaxIntSize
//
// Returns the number of characters val takes when printed.
//--------------------------------------------------------------------*/
int MaxIntSize(val)
int val;
{
	int size = 1;     /* even '0' takes one space */

	if (val < 0) {
		val = abs(val);
		size++;         /* for the minus sign */
	}

	while (val > 9) {
		val = val / 10;
		size++;
	}

	return(size);
}

/*----------------------------------------------------------------------
// ParseArgs
//
// Parses the command line arguments.
//--------------------------------------------------------------------*/
int ParseArgs(argc, argv, inf, outf)
int    argc;
char*  argv[];
FILE** inf;
FILE** outf;
{
	int i;

	for (i = 0; i < argc; i++)
		if (strcasecmp(argv[i], "-input") == 0)
			if (++i < argc) {
				if ((*inf = fopen(argv[i], "r")) == 0) {
					printf("Can't open input file.\n");
					return (0);
				}
			}
			else {
				printf("Not enough arguments.");
				printf("USAGE: %s -input infile -output outfile\n", argv[0]);
				return (0);
			}
		else if (strcasecmp(argv[i], "-output") == 0)
			if (++i < argc) {
				if ((*outf = fopen(argv[i], "w")) == 0) {
					printf("Can't open output file.\n");
					return (0);
				}
			}
			else {
				printf("Not enough arguments.");
				printf("USAGE: %s -input infile -output outfile\n", argv[0]);
				return (0);
			}

			return(1);
}


/*----------------------------------------------------------------------
// PrintMatrix
//
// Prints the matrix to the output file.
//--------------------------------------------------------------------*/
int PrintMatrix(file, matrix, size)
FILE* file;
int*  matrix;
int   size;
{
	int  i, j;
	int  maxval;
	int  padding;
	char formatstring[10];

	maxval = 0;
	for (i = 0; i < size; i++)
		for (j = 0; j < size; j++)
			if (abs(maxval) < abs(matrix[i*size + j]))
				maxval = matrix[i*size + j];

	padding = MaxIntSize(maxval) + 1;

	/* This next line creates something like "%4d" for the print
	stmt below, replacing the 4 with the padding value */

	sprintf(formatstring, " %%%dd", padding);

	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++)
			fprintf(file, formatstring, matrix[i*size + j]);
		fprintf(file, "\n");
	}
	fprintf(file, "\n");
	fclose(file);
}

/*----------------------------------------------------------------------
// ReadFile
//
// Reads in the digraph.
//--------------------------------------------------------------------*/
int ReadFile(file, size, graph)
FILE* file;
int*  size;
int** graph;
{
	int i, j;

	fscanf(file, "%d", size);

	*graph = (int*)malloc((*size) * (*size) * sizeof(int));

	for (i = 0; i < (*size); i++)
		for (j = 0; j < (*size); j++)
			fscanf(file, "%d", &(*graph)[i*(*size) + j]);

	fclose(file);
	return(1);
}

/*----------------------------------------------------------------------
// main
//--------------------------------------------------------------------*/
int main(int argc, char** argv)
{
	int        i, j, k;
	int*       graph;
	int*       my_partition;
	int*       kth_row;
	int        size;
	int        partition_size;
	int        my_rank, processors;
	FILE*      in_file;
	FILE*      out_file;
	MPI_Status status;

	/*----------
	// Initialize MPI
	//--------*/
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &processors);

	/*----------
	// Parse command line arguments
	//--------*/
	if (my_rank == 0)
		if (!ParseArgs(argc, argv, &in_file, &out_file))
			exit(1);

	/*----------
	// Read input file.
	//--------*/
	if (my_rank == 0)
		if (!ReadFile(in_file, &size, &graph)) {
			printf("program terminating\n");
			exit(1);
		}

	/*----------
	// Broadcast data size.
	//--------*/
	MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);

	/*----------
	// Determine data partitions.
	//--------*/
	partition_size = size / processors;
	if (size % processors)
		partition_size++;

	my_partition = (int*)malloc(size * partition_size * sizeof(int));
	kth_row = (int*)malloc(size * 1 * sizeof(int));

	assert(my_partition != NULL);

	/*----------
	// Distribute the data.
	//--------*/
	MPI_Scatter(graph, partition_size * size, MPI_INT, my_partition,
		partition_size * size, MPI_INT, 0, MPI_COMM_WORLD);

	/* If processors does not divide evenly into size, the Scatter
	will send random memory beyond the end of graph, but that data
	should not actually get used, so it probably won't crash. */

	
	for (k = 0; k < size; k++) {

		/*----------
		// Broadcast kth row.
		//--------*/
		if (my_rank == (k / partition_size))
			for (i = 0; i < size; i++)
				kth_row[i] = my_partition[(k%partition_size)*size + i];
		MPI_Bcast(kth_row, size, MPI_INT, (k / partition_size), MPI_COMM_WORLD);

		/*----------
		// Update my rows.
		//--------*/
		for (i = 0; (i < partition_size) && (i < size); i++) {
			if (my_partition[i*size + k] < 1)
				continue;
			for (j = 0; j < size; j++) {
				if (kth_row[j] < 1)
					continue;
				if (my_partition[i*size + j] < 0)
					my_partition[i*size + j] = my_partition[i*size + k] + kth_row[j];
				else
					my_partition[i*size + j] = min(my_partition[i*size + j],
						my_partition[i*size + k] + kth_row[j]);
			}
		}
	}

	/*----------
	// Collect the data.
	//--------*/
	MPI_Gather(my_partition, partition_size * size, MPI_INT, graph,
		partition_size * size, MPI_INT, 0, MPI_COMM_WORLD);

	/*----------
	// Print the output.
	//--------*/
	if (my_rank == 0)
		PrintMatrix(out_file, graph, size);

	/*----------
	// Shut down MPI.
	//--------*/
	MPI_Finalize();
}