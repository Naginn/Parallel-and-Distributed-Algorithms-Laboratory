int N = 255; size_t size = N * sizeof(float);

float* h_tmp_sum = (float*)malloc(size);

float* d_tmp_sum;
cudaMalloc(&d_tmp_sum, size);


cudaMemcpy(d_tmp_sum, h_tmp_sum, size, cudaMemcpyHostToDevice);

__global__ void sumOfThePixels(unsigned char *source, int width, int height,unsigned char *destination)
{

int x = (blockIdx.x * blockDim.x) + threadIdx.x;

float *sum;
float *tmp_sum;

for (int i = 0; i <= width; i++)
{
    tmp_sum[x] += source[x][i] ;
}
for (int j = 0; j <= N ; j++)

    sum += tmp_sum[j];

destination[(y * width) + x] = (unsigned char)sum;
}