#pragma once
#include <string>

typedef double ftype;

__host__ void check_err(cudaError_t err, const char* step_name = "?");

struct Offset {
public:
    int lx, ly, lz;
    int rx, ry, rz;
};

struct Data {
    ftype* ex;
    ftype* ey;
    ftype* ez;
    ftype* hx;
    ftype* hy;
    ftype* hz;

    ftype* mu;
    ftype* eps;
};

struct Params {
    ftype dr, dt;
    int Nx, Ny, Nz;
    ftype eps_0, mu_0, c;
    int n_steps;

    Offset xm, ym, zm, xp, yp, zp;

    Data host,device;

    Params() {}

    void init_pars() {
        dr = 1e-6;
        dt = 1e-8;
        Nx = 300;
        Ny = 300;
        Nz = 300;
        eps_0 = 8.854e-12;
        mu_0 = 1.257e-6;
        n_steps = 1000;
        c = 3e8;

        xm.lx=0; xm.ly=0; xm.lz=0; xm.rx=-1;xm.ry=0; xm.rz=0;
        ym.lx=0; ym.ly=0; ym.lz=0; ym.rx=0; ym.ry=-1;ym.rz=0;
        zm.lx=0; zm.ly=0; zm.lz=0; zm.rx=0; zm.ry=0; zm.rz=-1;
        xp.lx=1; xp.ly=0; xp.lz=0; xp.rx=0; xp.ry=0; xp.rz=0;
        yp.lx=0; yp.ly=1; yp.lz=0; yp.rx=0; yp.ry=0; yp.rz=0;
        zp.lx=0; zp.ly=0; zp.lz=1; zp.rx=0; zp.ry=0; zp.rz=0;
    }

    void init_memory_2d() {
        host.ex = (ftype*)(malloc(Nx*Ny*sizeof(ftype)));
        host.ey = (ftype*)(malloc(Nx*Ny*sizeof(ftype)));
        host.ez = (ftype*)(malloc(Nx*Ny*sizeof(ftype)));
        host.hx = (ftype*)(malloc(Nx*Ny*sizeof(ftype)));
        host.hy = (ftype*)(malloc(Nx*Ny*sizeof(ftype)));
        host.hz = (ftype*)(malloc(Nx*Ny*sizeof(ftype)));
        host.mu = (ftype*)(malloc(Nx*Ny*sizeof(ftype)));
        host.eps= (ftype*)(malloc(Nx*Ny*sizeof(ftype)));

        check_err(cudaMalloc(reinterpret_cast<void **>(&device.ex), Nx*Ny*sizeof(ftype)), "allocating");
        check_err(cudaMalloc(reinterpret_cast<void **>(&device.ey), Nx*Ny*sizeof(ftype)), "allocating");
        check_err(cudaMalloc(reinterpret_cast<void **>(&device.ez), Nx*Ny*sizeof(ftype)), "allocating");
        check_err(cudaMalloc(reinterpret_cast<void **>(&device.hx), Nx*Ny*sizeof(ftype)), "allocating");
        check_err(cudaMalloc(reinterpret_cast<void **>(&device.hy), Nx*Ny*sizeof(ftype)), "allocating");
        check_err(cudaMalloc(reinterpret_cast<void **>(&device.hz), Nx*Ny*sizeof(ftype)), "allocating");
        check_err(cudaMalloc(reinterpret_cast<void **>(&device.mu), Nx*Ny*sizeof(ftype)), "allocating");
        check_err(cudaMalloc(reinterpret_cast<void **>(&device.eps),Nx*Ny*sizeof(ftype)), "allocating");

        for (int x = 0; x < Nx; x++){
            for (int y = 0; y < Ny; y++) {
                int c = x * Ny + y;
                host.ex[c] = 0;
                host.ey[c] = 0;
                host.ez[c] = 0;
                host.hx[c] = 0;
                host.hy[c] = 0;
                host.hz[c] = 0;
                host.mu[c] = 1;
                host.eps[c]= 1;
            }
        }

        check_err(cudaMemcpy(device.ex, host.ex, Nx*Ny*sizeof(ftype), cudaMemcpyHostToDevice), "copying to device");
        check_err(cudaMemcpy(device.ey, host.ey, Nx*Ny*sizeof(ftype), cudaMemcpyHostToDevice), "copying to device");
        check_err(cudaMemcpy(device.ez, host.ez, Nx*Ny*sizeof(ftype), cudaMemcpyHostToDevice), "copying to device");
        check_err(cudaMemcpy(device.hx, host.hx, Nx*Ny*sizeof(ftype), cudaMemcpyHostToDevice), "copying to device");
        check_err(cudaMemcpy(device.hy, host.hy, Nx*Ny*sizeof(ftype), cudaMemcpyHostToDevice), "copying to device");
        check_err(cudaMemcpy(device.hz, host.hz, Nx*Ny*sizeof(ftype), cudaMemcpyHostToDevice), "copying to device");
        check_err(cudaMemcpy(device.mu, host.mu, Nx*Ny*sizeof(ftype), cudaMemcpyHostToDevice), "copying to device");
        check_err(cudaMemcpy(device.eps,host.eps,Nx*Ny*sizeof(ftype), cudaMemcpyHostToDevice), "copying to device");
    }
};

