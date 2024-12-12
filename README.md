# CUDA Histogram Computation

This project demonstrates how to compute histograms in parallel on a GPU using NVIDIA CUDA. The program calculates a histogram with 256 bins (typically for grayscale pixel intensities ranging from 0 to 255) using parallel processing for faster computation.

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

   ```bash
   git clone https://github.com/yourusername/cuda-histogram-computation.git
   cd cuda-histogram-computation
