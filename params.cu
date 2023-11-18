#include <fstream>
#include <json.hpp>
#include <iostream>

#include "params.cuh"
using json = nlohmann::json;

void check_err(cudaError_t err, const char* step_name)
 {
    if (err != cudaSuccess) {
        fprintf(stderr, "Error during %s: %s.\n", step_name, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
void default_if_missing(json& file, json& def) {
    for (auto& el : def.items()) {
        try {
            file.at(el.key());
        }
        catch (json::out_of_range) {
            file[el.key()] = def[el.key()];
            continue;
        }
        if (el.value().is_object()) {
            default_if_missing(file[el.key()], def[el.key()]);
        }
    }
}
void Params::init_pars(std::string filename) {
    eps_0 = 8.854e-12;
    mu_0 = 1.257e-6;
    c = 3e8;

    std::fstream f("default.json");
    json params_default = json::parse(f);
    json params_file;
    if (filename != "") {
        std::fstream p(filename);
        params_file = json::parse(p);

        default_if_missing(params_file, params_default);
    }
    else {
        params_file = params_default;
    }

    ftype max_freq = static_cast<ftype>(params_file["config"]["maximum_frequency"]);
    int n_steps_in_wave = static_cast<int>(params_file["config"]["n_steps_per_wave"]);
    bool optimize_dr = static_cast<bool>(params_file["config"]["optimize_dr"]);
    bool optimize_dt = static_cast<bool>(params_file["config"]["optimize_dt"]);
    dimensions = static_cast<int>(params_file["config"]["dimensions"]);
    drop_rate = static_cast<int>(params_file["config"]["drop_rate"]);
    n_steps = static_cast<int>(params_file["config"]["n_steps"]);

    if (optimize_dr) {
        dr = c / max_freq / n_steps_in_wave;
    }
    else {
        dr = static_cast<int>(params_file["numerical_params"]["dr"]);
    }

    if (optimize_dt) {
        dt = dr / (c*2);
    }
    else {
        dt = static_cast<int>(params_file["numerical_params"]["dt"]);
    }
    std::string src_type = static_cast<std::string>(params_file["source"]["type"]);

    if (src_type == "gauss_pulse") {
        source_x = static_cast<int>(static_cast<ftype>(params_file["source"]["gauss_pulse"]["x_pos"]) / dr);
        source_y = static_cast<int>(static_cast<ftype>(params_file["source"]["gauss_pulse"]["y_pos"]) / dr);
        source_offset = static_cast<int>(static_cast<ftype>(params_file["source"]["gauss_pulse"]["delay"]) / dt);
        if (static_cast<bool>(params_file["source"]["gauss_pulse"]["max_width"])) {
            source_width = 0.5/max_freq;
        }
        else {
            source_width = static_cast<ftype>(params_file["source"]["gauss_pulse"]["width"]);
        }
        if (optimize_dt) {
            dt = min(dt, source_width / n_steps_in_wave);
        }
    }

    Nx = static_cast<int>(static_cast<ftype>(params_file["domain"]["size_x"]) / dr);
    Ny = static_cast<int>(static_cast<ftype>(params_file["domain"]["size_y"]) / dr);
    Nz = static_cast<int>(static_cast<ftype>(params_file["domain"]["size_z"]) / dr);

    std::string xbc_str = static_cast<std::string>(params_file["boundary_conditions"]["x"]);
    std::string ybc_str = static_cast<std::string>(params_file["boundary_conditions"]["y"]);
    std::string zbc_str = static_cast<std::string>(params_file["boundary_conditions"]["z"]);

    if (xbc_str == "dirichlet") {xbc = 0;}
    else {xbc = 1;}
    if (ybc_str == "dirichlet") {ybc = 0;}
    else {ybc = 1;}
    if (zbc_str == "dirichlet") {zbc = 0;}
    else {zbc = 1;}

    
    xm.lx=0; xm.ly=0; xm.lz=0; xm.rx=-1;xm.ry=0; xm.rz=0;
    ym.lx=0; ym.ly=0; ym.lz=0; ym.rx=0; ym.ry=-1;ym.rz=0;
    zm.lx=0; zm.ly=0; zm.lz=0; zm.rx=0; zm.ry=0; zm.rz=-1;
    xp.lx=1; xp.ly=0; xp.lz=0; xp.rx=0; xp.ry=0; xp.rz=0;
    yp.lx=0; yp.ly=1; yp.lz=0; yp.rx=0; yp.ry=0; yp.rz=0;
    zp.lx=0; zp.ly=0; zp.lz=1; zp.rx=0; zp.ry=0; zp.rz=0;
}



void Params::init_memory_2d() {
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
void Params::extract_data_2d(){
    cudaMemcpy(host.ex,device.ex,Nx*Ny*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(host.ey,device.ey,Nx*Ny*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(host.ez,device.ez,Nx*Ny*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(host.hx,device.hx,Nx*Ny*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(host.hy,device.hy,Nx*Ny*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(host.hz,device.hz,Nx*Ny*sizeof(double), cudaMemcpyDeviceToHost);
}
void Params::free_memory() {
    free(host.ex);
    free(host.ey);
    free(host.ez);
    free(host.hx);
    free(host.hy);
    free(host.hz);
    free(host.mu);
    free(host.eps);
    check_err(cudaFree(device.ex), "cleaning");
    check_err(cudaFree(device.ey), "cleaning");
    check_err(cudaFree(device.ez), "cleaning");
    check_err(cudaFree(device.hx), "cleaning");
    check_err(cudaFree(device.hy), "cleaning");
    check_err(cudaFree(device.hz), "cleaning");
    check_err(cudaFree(device.mu), "cleaning");
    check_err(cudaFree(device.eps),"cleaning");
}
