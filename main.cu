#include <iostream>
#include <cmath>
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
    int blocksPerGridX = (pars_h.Nx + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGridY = (pars_h.Ny + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGridZ = (pars_h.Nz + threadsPerBlock - 1) / threadsPerBlock;

    dim3 blockShape(threadsPerBlock, threadsPerBlock, threadsPerBlock);
    dim3 gridShape(blocksPerGridX, blocksPerGridY, blocksPerGridZ);

    dim3 blockShape2D(threadsPerBlock, threadsPerBlock);
    dim3 gridShape2D(blocksPerGridX, blocksPerGridY);

    auto source { [](int step, Params& pars){return exp(-pow((step-pars.source_offset)*pars.dt/pars.source_width,2));} };

    plot_funtion(source, pars_h, "source");

    for (int step = 1; step <= pars_h.n_steps; ++step) {
        check_err(cudaPeekAtLastError(), "kernel");
        check_err(cudaDeviceSynchronize(), "e sync");
        inject_soft_source_2d<<<1,1>>>(pars_h.device.ez, source(step, pars_h));
        check_err(cudaPeekAtLastError(), "kernel");
        check_err(cudaDeviceSynchronize(), "src sync");
        calc_fdtd_step_2d_x<<<gridShape2D,blockShape2D>>>(
            pars_h.device.hx, 
            pars_h.device.ez, 
            pars_h.device.mu, 
            pars_h.yp
        );
        calc_fdtd_step_2d_y<<<gridShape2D,blockShape2D>>>(
            pars_h.device.hy, 
            pars_h.device.ez, 
            pars_h.device.mu, 
            pars_h.xp
        );
        calc_fdtd_step_2d_z<<<gridShape2D,blockShape2D>>>(
            pars_h.device.hz, 
            pars_h.device.ey, 
            pars_h.device.ex, 
            pars_h.device.mu, 
            pars_h.xp, 
            pars_h.yp
        );
        check_err(cudaPeekAtLastError(), "kernel");
        check_err(cudaDeviceSynchronize(), "h sync");
        calc_fdtd_step_2d_x<<<gridShape2D,blockShape2D>>>(
            pars_h.device.ex, 
            pars_h.device.hz, 
            pars_h.device.eps, 
            pars_h.ym
        );
        calc_fdtd_step_2d_y<<<gridShape2D,blockShape2D>>>(
            pars_h.device.ey, 
            pars_h.device.hz, 
            pars_h.device.eps, 
            pars_h.xm
        );
        calc_fdtd_step_2d_z<<<gridShape2D,blockShape2D>>>(
            pars_h.device.ez, 
            pars_h.device.hy, 
            pars_h.device.hx, 
            pars_h.device.eps, 
            pars_h.xm, 
            pars_h.ym
        );

        if (step%pars_h.drop_rate == 0) {
            pars_h.extract_data_2d();
            plot(pars_h.host.ez, pars_h, "ez", step);
            std::cout << "step " << step << ", dt: " << pars_h.dt << ", dr = " << pars_h.dr << "\n";
        }
    }
    pars_h.free_memory();
}

__host__ int main() {
    start_fdtd();
    std::cout << "Test run complete!\n";

    return 0;
}