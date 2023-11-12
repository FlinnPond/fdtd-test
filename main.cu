#include <iostream>
#include "params.cuh"
#include "fdtd.cuh"

__constant__ Params pars;

__host__ void start_fdtd() {
    Params pars_h = Params();
    pars_h.init_pars();
    check_err(cudaMemcpyToSymbol(pars, &pars_h, sizeof(Params)), "copying params to device");

    int threadsPerBlock = 8;
    int threadsPerBlockY = 4;
    int blocksPerGridX = (pars_h.Nx + threadsPerBlock  - 1) / threadsPerBlock;
    int blocksPerGridY = (pars_h.Ny + threadsPerBlockY - 1) / threadsPerBlockY;
    int blocksPerGridZ = (pars_h.Nz + threadsPerBlock  - 1) / threadsPerBlock;

    dim3 blockShape(threadsPerBlock, threadsPerBlockY, threadsPerBlock);
    dim3 gridShape(blocksPerGridX, blocksPerGridY, blocksPerGridZ);

    dim3 blockShapeXY(threadsPerBlock, threadsPerBlockY);
    dim3 gridShapeXY(blocksPerGridX, blocksPerGridY);

    for (int step = 1; step <= pars_h.n_steps; ++step) {

    }
}

__host__ int main() {
    start_fdtd();
    std::cout << "Test run complete!\n";

    return 0;
}