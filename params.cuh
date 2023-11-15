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
    int drop_rate;

    int source_x, source_y, source_offset;
    ftype source_width;

    Offset xm, ym, zm, xp, yp, zp;

    Data host,device;

    Params() {}

    void init_pars();
    void init_memory_2d();
    void extract_data_2d();
    void free_memory();
};

