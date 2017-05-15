__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) 
{
	// Each thread computes one element of C by accumulating results into Cvalue
	float Cvalue = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row > A.height || col > B.width) return;
	for (int e = 0; e < A.width; ++e)
		Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
	C.elements[row * C.width + col] = Cvalue;
}

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
	// Block row and column
	int blockRow = blockIdx.y, blockCol = blockIdx.x;

	// Each thread block computes one sub-matrix Csub of C
	Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

	// Each thread computes 1 element of Csub accumulating results into Cvalue
	float Cvalue = 0.0;

	// Thread row and column within Csub
	int row = threadIdx.y, col = threadIdx.x;

	// Loop over all the sub-matrices of A and B required to compute Csub
	for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) 
	{
		// Get sub-matrices Asub of A and Bsub of B
		Matrix Asub = GetSubMatrix(A, blockRow, m);
		Matrix Bsub = GetSubMatrix(B, m, blockCol);

		// Shared memory used to store Asub and Bsub respectively
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		// Load Asub and Bsub from device memory to shared memory
		// Each thread loads one element of each sub-matrix
		As[row][col] = GetElement(Asub, row, col);
		Bs[row][col] = GetElement(Bsub, row, col);
		__syncthreads();
		
		// Multiply Asub and Bsub together
		for (int e = 0; e < BLOCK_SIZE; ++e)
		Cvalue += As[row][e] * Bs[e][col];
		__syncthreads();
	}

	// Each thread writes one element of Csub to memory
	SetElement(Csub, row, col, Cvalue);
}