// int xp = y*pars.Nx*pars.Nz + (x+1)*pars.Nz + z;
// int xm = y*pars.Nx*pars.Nz + (x-1)*pars.Nz + z;
// int zp = y*pars.Nx*pars.Nz + x*pars.Nz + z + 1;
// int zm = y*pars.Nx*pars.Nz + x*pars.Nz + z - 1;
// int yp = (y+1)*pars.Nx*pars.Nz + x*pars.Nz + z;
// int ym = (y-1)*pars.Nx*pars.Nz + x*pars.Nz + z;
#include "params.cuh"

extern __constant__ Params pars;

__global__ void calc_fdtd_step_x(
    double* e, 
    double* h1, 
    double* h2,
    double* j,
    double* ca,
    double* cb
) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;
    int c = y*pars.Nx*pars.Nz + x*pars.Nz + z;
    if (x < pars.Nx && y < pars.Ny && z < pars.Nz) {
        int zp = y*pars.Nx*pars.Nz + x*pars.Nz + z + 1;
        int yp = (y+1)*pars.Nx*pars.Nz + x*pars.Nz + z;
        e[c] = ca[c] * e[c] + cb[c] * (
            h1[yp] - h1[c] - h2[zp] + h2[c] - j[c] * pars.dr
        );
    }
}
__global__ void calc_fdtd_step_y(
    double* e, 
    double* h1, 
    double* h2,
    double* j,
    double* ca,
    double* cb
) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;
    int c = y*pars.Nx*pars.Nz + x*pars.Nz + z;
    if (x < pars.Nx && y < pars.Ny && z < pars.Nz) {
        int zp = y*pars.Nx*pars.Nz + x*pars.Nz + z + 1;
        int xp = y*pars.Nx*pars.Nz + (x+1)*pars.Nz + z;
        e[c] = ca[c] * e[c] + cb[c] * (
            h1[zp] - h1[c] - h2[xp] + h2[c] - j[c] * pars.dr
        );
    }
}
__global__ void calc_fdtd_step_z(
    double* e, 
    double* h1, 
    double* h2,
    double* j,
    double* ca,
    double* cb
) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;
    int c = y*pars.Nx*pars.Nz + x*pars.Nz + z;
    if (x < pars.Nx && y < pars.Ny && z < pars.Nz) {
        int xp = y*pars.Nx*pars.Nz + (x+1)*pars.Nz + z;
        int yp = (y+1)*pars.Nx*pars.Nz + x*pars.Nz + z;
        e[c] = ca[c] * e[c] + cb[c] * (
            h1[yp] - h1[c] - h2[xp] + h2[c] - j[c] * pars.dr
        );
    }
}

__global__ void calc_ca(
    double* ca,
    double* sigma,
    double* epsilon
) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;
    int c = y*pars.Nx*pars.Nz + x*pars.Nz + z;
    if (x < pars.Nx && y < pars.Ny && z < pars.Nz) {
        ca[c] = (1 - sigma[c] * pars.dt / (2*epsilon[c])) / 
        (1 + sigma[c] * pars.dt / (2*epsilon[c]));
    }
}

__global__ void calc_cb(
    double* cb,
    double* sigma,
    double* epsilon
) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;
    int c = y*pars.Nx*pars.Nz + x*pars.Nz + z;
    if (x < pars.Nx && y < pars.Ny && z < pars.Nz) {
        cb[c] = (pars.dt / (2*epsilon[c]*pars.dr)) / 
        (1 + sigma[c] * pars.dt / (2*epsilon[c]));
    }
}