#include <filesystem>
#include <fstream>
#include <json.hpp>
#include <iostream>

#include "params.cuh"
using json = nlohmann::json;
namespace fs = std::filesystem;

void check_err(cudaError_t err, const char* step_name) {
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
__host__ void try_create_dir(fs::path plots_path)
{
    auto directory_exists = fs::is_directory(plots_path);
    if (!directory_exists) {
        std::cout << "Creating directory: " << plots_path << "\n";
        auto created_new_directory = fs::create_directory(plots_path);
        if (!created_new_directory) {
            std::cout << "Cannot create directory " << plots_path << std::endl;
        }
        else {
            std::cout << "Directory " << plots_path << " successfully created!\n";
        }
    }
}
void Params::calc_m(ftype* m, int index, int index_pml, int pml_size, ftype sig1, ftype sig2, ftype sig3, ftype perm0, ftype* perm) {
    sig1 *= perm0;
    sig2 *= perm0;
    sig3 *= perm0;
    ftype m0 = 1 / dt + (sig1 + sig2) / (2 * perm0) + sig1 * sig2 * dt / (4 * perm0 * perm0);
    m[index_pml] = 1/m0 *( 1 / dt - (sig1 + sig2) / (2 * perm0) - sig1 * sig2 * dt / (4 * perm0 * perm0));
    m[index_pml + pml_size] = - 1/m0 * c / perm[index];
    m[index_pml + pml_size*2] = -1/m0 * c * dt * sig3 / (perm0 * perm[index]);
    m[index_pml + pml_size*3] = -1/m0 * dt * sig1 * sig2 * dt / (perm0 * perm0);
    m[index_pml + pml_size*4] = 0;
    m[index_pml + pml_size*5] = 0;    
}
void Params::init_pars(std::string filename) {

    std::fstream f("default.json");
    json params_default = json::parse(f);
    json params_file;

    if (filename != "") {
        if (!std::filesystem::exists(filename)) throw ("Parameters file " + filename + " not found!");
        std::fstream p(filename);
        params_file = json::parse(p);
        default_if_missing(params_file, params_default);
    }
    else {
        params_file = params_default;
    }

    std::cout << "Starting: " << static_cast<std::string>(params_file["output_directory"]) << "\n" << std::endl;
    std::string plots_path_str = "plots/" + static_cast<std::string>(params_file["output_directory"]);
    fs::path plots_path = static_cast<fs::path>(plots_path_str);
    try_create_dir("plots");
    try_create_dir("data");
    try_create_dir(plots_path);

    const std::string::size_type size = plots_path_str.size();
    plots_path_cstr = new char[size + 1];
    strcpy(plots_path_cstr, plots_path_str.c_str());

    eps_0 = 8.854e-12;
    mu_0 = 1.257e-6;
    c = 3e8;

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

    ftype Lx = static_cast<ftype>(params_file["domain"]["size_x"]);
    ftype Ly = static_cast<ftype>(params_file["domain"]["size_y"]);
    ftype Lz = static_cast<ftype>(params_file["domain"]["size_z"]);
    Nx = static_cast<int>(Lx / dr);
    Ny = static_cast<int>(Ly / dr);
    Nz = static_cast<int>(Lz / dr);

    Npx = static_cast<int>(params_file["boundary_conditions"]["pml"]["x_layers"]);
    Npy = static_cast<int>(params_file["boundary_conditions"]["pml"]["y_layers"]);
    Npz = static_cast<int>(params_file["boundary_conditions"]["pml"]["z_layers"]);

    std::string xbc_str = static_cast<std::string>(params_file["boundary_conditions"]["x"]);
    std::string ybc_str = static_cast<std::string>(params_file["boundary_conditions"]["y"]);
    std::string zbc_str = static_cast<std::string>(params_file["boundary_conditions"]["z"]);

    if      (xbc_str == "dirichlet") {xbc = 0;}
    else if (xbc_str == "periodic")  {xbc = 1;}
    if      (ybc_str == "dirichlet") {ybc = 0;}
    else if (xbc_str == "periodic")  {ybc = 1;}
    if      (zbc_str == "dirichlet") {zbc = 0;}
    else if (xbc_str == "periodic")  {zbc = 1;}
    
    xm.lx=0; xm.ly=0; xm.lz=0; xm.rx=-1;xm.ry=0; xm.rz=0;
    ym.lx=0; ym.ly=0; ym.lz=0; ym.rx=0; ym.ry=-1;ym.rz=0;
    zm.lx=0; zm.ly=0; zm.lz=0; zm.rx=0; zm.ry=0; zm.rz=-1;
    xp.lx=1; xp.ly=0; xp.lz=0; xp.rx=0; xp.ry=0; xp.rz=0;
    yp.lx=0; yp.ly=1; yp.lz=0; yp.rx=0; yp.ry=0; yp.rz=0;
    zp.lx=0; zp.ly=0; zp.lz=1; zp.rx=0; zp.ry=0; zp.rz=0;
}



void Params::init_memory_2d() {
    int domain_size = Nx*Ny*sizeof(ftype);
    int pml_size = (2*Npx*Ny+2*Npy*Nx-4*Npx*Npy)*12*sizeof(ftype);
    int pml_len = 2*Npx*Ny+2*Npy*Nx-4*Npx*Npy;
    std::cout << "domain sizes : " << Nx << " " << Ny << "\n";
    std::cout << "domain memory: " << domain_size*8 << "\n";
    std::cout << "pml    memory: " << pml_size*3 << "\n";
    host.ex = (ftype*)(malloc(domain_size));
    host.ey = (ftype*)(malloc(domain_size));
    host.ez = (ftype*)(malloc(domain_size));
    host.hx = (ftype*)(malloc(domain_size));
    host.hy = (ftype*)(malloc(domain_size));
    host.hz = (ftype*)(malloc(domain_size));
    host.mu = (ftype*)(malloc(domain_size));
    host.eps= (ftype*)(malloc(domain_size));
    host.pmlx = (ftype*)(malloc(pml_size));
    host.pmly = (ftype*)(malloc(pml_size));
    host.pmlz = (ftype*)(malloc(pml_size));

    check_err(cudaMalloc(reinterpret_cast<void **>(&device.ex), domain_size), "allocating");
    check_err(cudaMalloc(reinterpret_cast<void **>(&device.ey), domain_size), "allocating");
    check_err(cudaMalloc(reinterpret_cast<void **>(&device.ez), domain_size), "allocating");
    check_err(cudaMalloc(reinterpret_cast<void **>(&device.hx), domain_size), "allocating");
    check_err(cudaMalloc(reinterpret_cast<void **>(&device.hy), domain_size), "allocating");
    check_err(cudaMalloc(reinterpret_cast<void **>(&device.hz), domain_size), "allocating");
    check_err(cudaMalloc(reinterpret_cast<void **>(&device.mu), domain_size), "allocating");
    check_err(cudaMalloc(reinterpret_cast<void **>(&device.eps),domain_size), "allocating");
    check_err(cudaMalloc(reinterpret_cast<void **>(&device.pmlx),pml_size), "allocating");
    check_err(cudaMalloc(reinterpret_cast<void **>(&device.pmly),pml_size), "allocating");
    check_err(cudaMalloc(reinterpret_cast<void **>(&device.pmlz),pml_size), "allocating");

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
            if (x < Npx || x > Nx - Npx || y < Npy || y > Ny - Npy) {
                ftype sig_x = 0;
                ftype sig_y = 0;
                ftype sig_z = 0;
                ftype mag_sig_x = 0;
                ftype mag_sig_y = 0;
                ftype mag_sig_z = 0;

                if (x < Npx) {
                    sig_x = 1 / (2 * dt) * pow((ftype)(Npx - x) / Npx, 3);
                    mag_sig_x = 1 / (2 * dt) * pow((ftype)(Npx - x + 0.5) / Npx, 3);
                } else if (x > Nx - Npx) {
                    sig_x = 1 / (2 * dt) * pow((ftype)(x - Nx + Npx) / Npx, 3);
                    mag_sig_x = 1 / (2 * dt) * pow((ftype)(x - Nx + Npx + 0.5) / Npx, 3);
                }
                if (y < Npy) {
                    sig_y = 1 / (2 * dt) * pow((ftype)(Npy - y) / Npy, 3);
                    mag_sig_y = 1 / (2 * dt) * pow((ftype)(Npy - y + 0.5) / Npy, 3);
                } else if (y > Ny - Npy) {
                    sig_y = 1 / (2 * dt) * pow((ftype)(y - Ny + Npy) / Npy, 3);
                    mag_sig_y = 1 / (2 * dt) * pow((ftype)(y - Ny + Npy + 0.5) / Npy, 3);
                }

                int c_pml = 0;
                if (x < Npx) {
                    c_pml = x * Ny + y;
                } else if (x < Nx - Npx) {
                    c_pml = Npx * Ny + 2 * Npy * (x - Npx) + min (Npy, y) + max(0, y - Ny + Npy);
                } else {
                    c_pml = Npx * Ny + 2 * Npy * (Nx - 2 * Npx) + Ny * (x - Nx + Npx) + y;
                }
                calc_m(host.pmlx+pml_len*0, c, c_pml, pml_len, sig_y, sig_z, sig_x, eps_0, host.eps);
                calc_m(host.pmlx+pml_len*6, c, c_pml, pml_len, mag_sig_y, mag_sig_z, mag_sig_x, mu_0, host.mu);
                calc_m(host.pmly+pml_len*0, c, c_pml, pml_len, sig_x, sig_z, sig_y, eps_0, host.eps);
                calc_m(host.pmly+pml_len*6, c, c_pml, pml_len, mag_sig_x, mag_sig_z, mag_sig_y, mu_0, host.mu);
                calc_m(host.pmlz+pml_len*0, c, c_pml, pml_len, sig_x, sig_y, sig_z, eps_0, host.eps);
                calc_m(host.pmlz+pml_len*6, c, c_pml, pml_len, mag_sig_x, mag_sig_y, mag_sig_z, mu_0, host.mu);
            }
        }
    }

    check_err(cudaMemcpy(device.ex, host.ex, domain_size, cudaMemcpyHostToDevice), "copying to device");
    check_err(cudaMemcpy(device.ey, host.ey, domain_size, cudaMemcpyHostToDevice), "copying to device");
    check_err(cudaMemcpy(device.ez, host.ez, domain_size, cudaMemcpyHostToDevice), "copying to device");
    check_err(cudaMemcpy(device.hx, host.hx, domain_size, cudaMemcpyHostToDevice), "copying to device");
    check_err(cudaMemcpy(device.hy, host.hy, domain_size, cudaMemcpyHostToDevice), "copying to device");
    check_err(cudaMemcpy(device.hz, host.hz, domain_size, cudaMemcpyHostToDevice), "copying to device");
    check_err(cudaMemcpy(device.mu, host.mu, domain_size, cudaMemcpyHostToDevice), "copying to device");
    check_err(cudaMemcpy(device.eps,host.eps,domain_size, cudaMemcpyHostToDevice), "copying to device");
    check_err(cudaMemcpy(device.pmlx, host.pmlx, pml_size, cudaMemcpyHostToDevice), "copying to device");
    check_err(cudaMemcpy(device.pmly, host.pmly, pml_size, cudaMemcpyHostToDevice), "copying to device");
    check_err(cudaMemcpy(device.pmlz, host.pmlz, pml_size, cudaMemcpyHostToDevice), "copying to device");
}
void Params::extract_data_2d(){
    cudaMemcpy(host.ex, device.ex, Nx*Ny*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(host.ey, device.ey, Nx*Ny*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(host.ez, device.ez, Nx*Ny*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(host.hx, device.hx, Nx*Ny*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(host.hy, device.hy, Nx*Ny*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(host.hz, device.hz, Nx*Ny*sizeof(double), cudaMemcpyDeviceToHost);
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
    delete plots_path_cstr;
}
