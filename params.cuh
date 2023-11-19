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
    ftype *ex, *ey, *ez, *hx, *hy, *hz;

    ftype *mu, *eps;
    ftype *pmlx, *pmly, *pmlz;
};

struct Params {
    ftype dr, dt;
    ftype eps_0, mu_0, c;
    int Nx, Ny, Nz, Np;
    int Npx, Npy, Npz;
    int n_steps, drop_rate;
    int dimensions;
    int xbc, ybc, zbc;

    int source_x, source_y, source_offset;
    ftype source_width;

    Offset xm, ym, zm, xp, yp, zp;

    Data host,device;

    Params() {}

    void calc_m(ftype* m, int c, int c_pml, int pml_size, ftype sig1, ftype sig2, ftype sig3, ftype perm0, ftype* perm);
    void init_pars(std::string filename = "");
    void init_memory_2d();
    void extract_data_2d();
    void free_memory();
};

