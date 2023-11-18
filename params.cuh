#pragma once
#include <string>

typedef double ftype;

void check_err(cudaError_t err, const char* step_name = "?");

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
    ftype eps_0, mu_0, c;
    int Nx, Ny, Nz;
    int n_steps, drop_rate;
    int dimensions;
    int xbc, ybc, zbc;

    int source_x, source_y, source_offset;
    ftype source_width;

    Offset xm, ym, zm, xp, yp, zp;

    Data host,device;

    Params() {}

    void init_pars(std::string filename = "");
    void init_memory_2d();
    void extract_data_2d();
    void free_memory();
};

