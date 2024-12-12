# CUDA Histogram Computation

This project demonstrates how to compute histograms in parallel on a GPU using NVIDIA CUDA. The program calculates a histogram with 256 bins (typically for grayscale pixel intensities ranging from 0 to 255) using parallel processing for faster computation.An array of bins that counts how many times each value occurs in an input dataset


## Features

- Parallel computation of histograms using CUDA.
- Utilizes shared memory and atomic operations for efficiency.
- Scalable to different dataset sizes (1M, 2M, ..., 10M).

## Requirements

- **NVIDIA GPU** with CUDA support.
- **CUDA Toolkit** installed on your system. [Download CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).
- **C++ Compiler** (e.g., `g++` on Linux, Visual Studio on Windows).

## Installation

1. Clone the repository:

   git clone https://github.com/yourusername/cuda-histogram-computation.git

Compile the program using nvcc:

#### nvcc -o histogram_cuda histogram_cuda.cpp</br>
Run the compiled binary:

#### ./histogram_cuda </br>
The program generates a random dataset and computes the histogram, displaying the count of each bin (0 to 255).

Example Output:
plaintext
Copy code
Bin 0: 41234</br>
Bin 1: 38912</br>
...
Bin 255: 40502</br>
result may be change in your time
