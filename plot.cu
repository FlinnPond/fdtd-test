#include <gnuplot-iostream.h>
#include <fstream>

#include "params.cuh"
#include "plot.cuh"

void plot(ftype* data, Params& pars, const char* name, int step) {
    std::ofstream outfile;
    outfile.open("data/data.txt");
    for (int x = 0; x < pars.Nx; x++){
        for (int y = 0; y < pars.Ny; y++) {
            int c = x * pars.Ny + y;
            outfile << x << " ";
            outfile << y << " ";
            outfile << data[c] << "\n";
        }
        outfile << "\n";
    }
    Gnuplot gp;
    gp << "set terminal png \n set view map \n set pm3d at b corners2color c4 \n";
    gp << "set size ratio -1\n";
    gp << "set output \"plots/" << name << std::setfill('0') << std::setw(7) << step << ".png\"\n";
    gp << "splot \"data/data.txt\" u 1:2:3 with pm3d\n";
}
