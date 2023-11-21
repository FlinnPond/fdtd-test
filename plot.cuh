#include <string>
#include "params.cuh"


void plot(ftype* data, Params& pars, std::string name, int step);
void plot_funtion(ftype (*func)(int, Params&), Params& pars, std::string name);
