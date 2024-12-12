#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define NUM_BINS 256  // Number of histogram bins (e.g., for grayscale image, 256 bins for pixel intensities)

// CUDA kernel to compute histogram in parallel
__global__ void compute_histogram(int *data, int *histogram, int data_size) {
    __shared__ int shared_histogram[NUM_BINS];

    // Initialize shared memory
    if (threadIdx.x < NUM_BINS) {
        shared_histogram[threadIdx.x] = 0;
    }
    __syncthreads();

    // Calculate the start and end indices for this thread
    int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Traverse the data in chunks and update histogram bins
    for (int i = start_idx; i < data_size; i += stride) {
        int value = data[i];
        atomicAdd(&shared_histogram[value], 1);
    }

    __syncthreads();

    // Update global histogram from shared memory
    if (threadIdx.x < NUM_BINS) {
        atomicAdd(&histogram[threadIdx.x], shared_histogram[threadIdx.x]);
    }
}

void run_histogram_kernel(int *data, int *histogram, int data_size, int num_blocks, int threads_per_block) {
    int *d_data, *d_histogram;
    cudaMalloc(&d_data, data_size * sizeof(int));
    cudaMalloc(&d_histogram, NUM_BINS * sizeof(int));

    cudaMemcpy(d_data, data, data_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_histogram, 0, NUM_BINS * sizeof(int));  // Initialize histogram to 0

    // Launch the kernel
    compute_histogram<<<num_blocks, threads_per_block>>>(d_data, d_histogram, data_size);

    // Copy result back to host
    cudaMemcpy(histogram, d_histogram, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_histogram);
}

int main() {
    // Example: generate random data
    int data_size = 100000000;  // Size of the data (e.g., pixels in an image)
    std::vector<int> data(data_size);
    for (int i = 0; i < data_size; i++) {
        data[i] = rand() % NUM_BINS;  // Random values between 0 and NUM_BINS-1
    }

    int *histogram = new int[NUM_BINS]{0};  // Histogram initialization
    int num_blocks = 128;  // Number of blocks
    int threads_per_block = 256;  // Number of threads per block

    // Run the histogram computation on the GPU
    run_histogram_kernel(data.data(), histogram, data_size, num_blocks, threads_per_block);

    // Output the histogram
    for (int i = 0; i < NUM_BINS; i++) {
        std::cout << "Bin " << i << ": " << histogram[i];
    }

    delete[] histogram;

    std::cout<<std::endl;
    return 0;
}
