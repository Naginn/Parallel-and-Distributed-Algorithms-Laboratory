
void compute_shortest_paths (int id, int p, dtype **a, int n){
	int i, j, k;
	int offset;   /* Local index of broadcast row */
	int root;     /* Process controlling row to be bcast */
	int* tmp;     /* Holds the broadcast row */
	tmp = (dtype *) malloc (n * sizeof(dtype));
	for (k = 0; k < n; k++){
		root = BLOCK_OWNER(k,p,n);
		if (root == id) {
			offset = k - BLOCK_LOW(id,p,n);
			for (j = 0; j < n; j++)
				tmp[j] = a[offset][j];
		}
		MPI_Bcast (tmp, n, MPI_TYPE, root, MPI_COMM_WORLD);
		for (i = 0; i < BLOCK_SIZE(id,p,n); i++)
			for (j = 0; j < n; j++)
				a[i][j] = MIN(a[i][j],a[i][k]+tmp[j]);
	}
	free (tmp);
}