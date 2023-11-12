#include <iostream>
#include "params.cuh"
#include "fdtd.cuh"
#include "plot.cuh"

__constant__ Params pars;

__host__ void start_fdtd() {
    Params pars_h = Params();
    pars_h.init_pars();
    pars_h.init_memory_2d();
    check_err(cudaMemcpyToSymbol(pars, &pars_h, sizeof(Params)), "copying params to device");

    int threadsPerBlock = 8;
    int threadsPerBlockY = 4;
    int blocksPerGridX = (pars_h.Nx + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGridY = (pars_h.Ny + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGridZ = (pars_h.Nz + threadsPerBlock - 1) / threadsPerBlock;

    dim3 blockShape(threadsPerBlock, threadsPerBlockY, threadsPerBlock);
    dim3 gridShape(blocksPerGridX, blocksPerGridY, blocksPerGridZ);

    dim3 blockShape2D(threadsPerBlock, threadsPerBlock);
    dim3 gridShape2D(blocksPerGridX, blocksPerGridY);

    for (int step = 1; step <= pars_h.n_steps; ++step) {
        cudaDeviceSynchronize();
        calc_fdtd_step_2d_xy<<<blockShape2D,gridShape2D>>>(
            pars_h.device.hx, 
            pars_h.device.ez, 
            pars_h.device.mu, 
            pars_h.yp
        );
        calc_fdtd_step_2d_xy<<<blockShape2D,gridShape2D>>>(
            pars_h.device.hy, 
            pars_h.device.ez, 
            pars_h.device.mu, 
            pars_h.xp
        );
        calc_fdtd_step_2d_z<<<blockShape2D,gridShape2D>>>(
            pars_h.device.hz, 
            pars_h.device.ex, 
            pars_h.device.ey, 
            pars_h.device.mu, 
            pars_h.xp, 
            pars_h.yp
        );
        cudaDeviceSynchronize();
        calc_fdtd_step_2d_xy<<<blockShape2D,gridShape2D>>>(
            pars_h.device.ex, 
            pars_h.device.hz, 
            pars_h.device.eps, 
            pars_h.ym
        );
        calc_fdtd_step_2d_xy<<<blockShape2D,gridShape2D>>>(
            pars_h.device.ey, 
            pars_h.device.hz, 
            pars_h.device.eps, 
            pars_h.xm
        );
        calc_fdtd_step_2d_z<<<blockShape2D,gridShape2D>>>(
            pars_h.device.ez, 
            pars_h.device.hy, 
            pars_h.device.hx, 
            pars_h.device.eps, 
            pars_h.xm, 
            pars_h.ym
        );

        if (step%pars_h.drop_rate == 0) {
            pars_h.extract_data_2d();
            plot(pars_h.host.ex, pars_h, "ex", step);
        }
    }
    pars_h.free_memory();
}

__host__ int main() {
    start_fdtd();
    std::cout << "Test run complete!\n";

    return 0;
}