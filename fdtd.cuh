__global__ void calc_fdtd_step_x(double*,double*,double*,double*,double*,double*);
__global__ void calc_fdtd_step_y(double*,double*,double*,double*,double*,double*);
__global__ void calc_fdtd_step_z(double*,double*,double*,double*,double*,double*);
__global__ void calc_ca(
    double* ca,
    double* sigma,
    double* epsilon
);
__global__ void calc_cb(
    double* ca,
    double* sigma,
    double* epsilon
);
