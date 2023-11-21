#include <gnuplot-iostream.h>
#include <string>
#include <fstream>

#include "params.cuh"
#include "plot.cuh"

void plot(ftype* data, Params& pars, std::string name, int step) {
    std::ofstream outfile;
    outfile.open("data/data.txt");
    for (int x = 0; x < pars.Nx; x++){
        for (int y = 0; y < pars.Ny; y++) {
            int c = x * pars.Ny + y;
            outfile << x * pars.dr << " ";
            outfile << y * pars.dr << " ";
            outfile << data[c] << "\n";
        }
        outfile << "\n";
    }
    outfile.close();
    Gnuplot gp;
    gp << "set terminal png \n set view map \n set pm3d at b corners2color c4 \n";
    gp << "set size ratio -1\n";
    gp << "set output \"" << pars.plots_path_cstr << "/" << name << std::setfill('0') << std::setw(7) << step << ".png\"\n";
    std::cout << " Saving to: \"" << pars.plots_path_cstr << "/" << name << std::setfill('0') << std::setw(7) << step << ".png\"\n";
    gp << "set title \"t = " << std::setprecision(3) << step * pars.dt << "\"\n";
    gp << "set cbrange [-0.025:0.025]\n";
    gp << "set palette model RGB \nset palette defined\n";
    gp << "splot \"data/data.txt\" u 1:2:3 with pm3d\n";
}

void plot_funtion(ftype (*func)(int, Params&), Params& pars, std::string name) {
    std::ofstream outfile;
    outfile.open("data/source_data.txt");
    for (int t = 0; t < pars.n_steps; t++) {
        outfile << t << " ";
        outfile << func(t, pars) << "\n";
    }
    outfile << "\n";
    outfile.close();
    Gnuplot gp;
    gp << "set terminal png\n";
    gp << "set output \"" << pars.plots_path_cstr << "/" << name << ".png\"\n";
    gp << "plot \"data/source_data.txt\" using 1:2 with line\n";
}