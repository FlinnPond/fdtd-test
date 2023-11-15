// int xp = y*pars.Nx*pars.Nz + (x+1)*pars.Nz + z;
// int xm = y*pars.Nx*pars.Nz + (x-1)*pars.Nz + z;
// int zp = y*pars.Nx*pars.Nz + x*pars.Nz + z + 1;
// int zm = y*pars.Nx*pars.Nz + x*pars.Nz + z - 1;
// int yp = (y+1)*pars.Nx*pars.Nz + x*pars.Nz + z;
// int ym = (y-1)*pars.Nx*pars.Nz + x*pars.Nz + z;
#include "params.cuh"

extern __constant__ Params pars;

__global__ void calc_fdtd_step_x(
    ftype* e, 
    ftype* h1, 
    ftype* h2,
    ftype* ca,
    ftype* cb
) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;
    int c = y*pars.Nx*pars.Nz + x*pars.Nz + z;
    if (x < pars.Nx && y < pars.Ny && z < pars.Nz) {
        int zp = y*pars.Nx*pars.Nz + x*pars.Nz + z + 1;
        int yp = (y+1)*pars.Nx*pars.Nz + x*pars.Nz + z;
        e[c] = ca[c] * e[c] + cb[c] * (
            h1[yp] - h1[c] - h2[zp] + h2[c]
        );
    }
}
__global__ void calc_fdtd_step_y(
    ftype* e, 
    ftype* h1, 
    ftype* h2,
    ftype* j,
    ftype* ca,
    ftype* cb
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
    ftype* e, 
    ftype* h1, 
    ftype* h2,
    ftype* j,
    ftype* ca,
    ftype* cb
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
    ftype* ca,
    ftype* sigma,
    ftype* epsilon
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
    ftype* cb,
    ftype* sigma,
    ftype* epsilon
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

__global__ void calc_fdtd_step_2d_x(
    ftype* field1,
    ftype* field2z,
    ftype* perm,
    Offset off
) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x < pars.Nx-1 && y < pars.Ny-1 && x > 0 && y > 0) {
        int c = x*pars.Ny + y;
        int left  = (x + off.lx)*pars.Ny + y + off.ly;
        int right = (x + off.rx)*pars.Ny + y + off.ry;
        field1[c] += pars.c * pars.dt * (field2z[left] - field2z[right]) / (pars.dr * perm[c]);
    }
}

__global__ void calc_fdtd_step_2d_y(
    ftype* field1,
    ftype* field2z,
    ftype* perm,
    Offset off
) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x < pars.Nx-1 && y < pars.Ny-1 && x > 0 && y > 0) {
        int c = x*pars.Ny + y;
        int left  = (x + off.lx)*pars.Ny + y + off.ly;
        int right = (x + off.rx)*pars.Ny + y + off.ry;
        field1[c] += - pars.c * pars.dt * (field2z[left] - field2z[right]) / (pars.dr * perm[c]);
    }
}

__global__ void calc_fdtd_step_2d_z(
    ftype* field1,
    ftype* field2y,
    ftype* field2x,
    ftype* perm,
    Offset off1,
    Offset off2
) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x < pars.Nx-1 && y < pars.Ny-1 && x > 0 && y > 0) {
        int c = x*pars.Ny + y;
        int left1  = (x + off1.lx)*pars.Ny + y + off1.ly;
        int right1 = (x + off1.rx)*pars.Ny + y + off1.ry;
        int left2  = (x + off2.lx)*pars.Ny + y + off2.ly;
        int right2 = (x + off2.rx)*pars.Ny + y + off2.ry;
        field1[c] += - pars.c * pars.dt * (field2y[left1] - field2y[right1] - field2x[left2] + field2x[right2]) / (pars.dr * perm[c]);
    }
}

__global__ void inject_soft_source_2d(
    ftype* field,
    ftype value
) {
    int c = pars.source_x*pars.Ny + pars.source_y;
    field[c] += value;
}