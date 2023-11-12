#include "params.cuh"

__host__ void check_err(cudaError_t err, const char* step_name)
 {
    if (err != cudaSuccess) {
        fprintf(stderr, "Error during %s: %s.\n", step_name, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
