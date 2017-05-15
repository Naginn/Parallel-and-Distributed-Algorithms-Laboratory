// GPU computation of thresholded difference matrix
__global__ void difference_filter(int *dev_out, int *edges_1, int *edges_2, int width, int height, int threshold) 
{
	const int r = blockIdx.y * blockDim.y + threadIdx.y;
	const int c = blockIdx.x * blockDim.x + threadIdx.x;
	const int i = r * width + c;
	// Set it to 0 initially
	dev_out[i] = 0;
	int crop_size = 7;
	if (r > crop_size && c > crop_size && r < height - crop_size && c < width - crop_size && edges_1[i] != edges_2[i]) 
	{
		// Set to 255 if there is a pixel
		mismatch
		dev_out[i] = 255;
		for (int x_apron = -threshold; x_apron <= threshold; x_apron++)
		{
			for (int y_apron = -threshold; y_apron <= threshold; y_apron++)
			{
				// Ensure the requested index is within bounds of image
				if (c + x_apron > 0 && r + y_apron > 0 && c + x_apron <	width && r + y_apron < height)
				{
					// Check if there is a matching pixel in the apron, within the threshold
					if (edges_1[(r + y_apron) * width + c + x_apron] ==	edges_2[i])
					{
						// Set it back to 0 if a corresponding pixel exists within the vicinity of the match
						dev_out[i] = 0;
					}
				}
			}
		}
	}
}

// GPU computation of the difference density matrix
__global__ void spatial_difference_density_map(double *density_map, int *difference, int width, int height, int horizontal_divisions, int vertical_divisions)
{
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int i = r * width + c;
	int horizontal_block_size = width/horizontal_divisions;
	int vertical_block_size = height/vertical_divisions;
	int block_size = horizontal_block_size * vertical_block_size;
	const int scaling_factor = 1000;
	if (difference[i] != 0) 
	{
		int i = (int)(vertical_divisions * r/(double)height);
		int j = (int)(horizontal_divisions * c/(double)width);
		density_map[i * horizontal_divisions + j] += scaling_factor/(double)block_size;
	}
}

// GPU generation of the motion area estimation image
__global__ void motion_area_estimate(int *motion_area, double *density_map, int width, int height, int horizontal_divisions, int vertical_divisions, double threshold) {
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int i = r * width + c;
	int density_map_index = (int)(vertical_divisions*r/(double)height) * horizontal_divisions + (int)(horizontal_divisions*c/(double)width);
	if (density_map[density_map_index] >= threshold)
	{
		motion_area[i] = 255;
	}
	else
	{
		motion_area[i] = 0;
	}
}