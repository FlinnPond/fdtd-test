#include <string>

struct Params {
    double dr, dt;
    int Nx, Ny, Nz;
    double eps_0, mu_0;
    int n_steps;

    Params() {}

    void init_pars() {
        dr = 1e-6;
        dt = 1e-8;
        Nx = 1000;
        Ny = 300;
        Nz = 300;
        eps_0 = 8.854e-12;
        mu_0 = 1.257e-6;
        n_steps = 1000;
    }

    void init_memory() {}
};