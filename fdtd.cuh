__global__ void calc_fdtd_step_x(ftype*,ftype*,ftype*,ftype*,ftype*,ftype*);
__global__ void calc_fdtd_step_y(ftype*,ftype*,ftype*,ftype*,ftype*,ftype*);
__global__ void calc_fdtd_step_z(ftype*,ftype*,ftype*,ftype*,ftype*,ftype*);
__global__ void calc_ca(
    ftype* ca,
    ftype* sigma,
    ftype* epsilon
);
__global__ void calc_cb(
    ftype* ca,
    ftype* sigma,
    ftype* epsilon
);
__global__ void calc_fdtd_step_2d_x(
    ftype* field1,
    ftype* field2z,
    ftype* perm,
    Offset off
);
__global__ void calc_fdtd_step_2d_y(
    ftype* field1,
    ftype* field2z,
    ftype* perm,
    Offset off
);
__global__ void calc_fdtd_step_2d_z(
    ftype* field1,
    ftype* field2y,
    ftype* field2x,
    ftype* perm,
    Offset off1,
    Offset off2
);
__global__ void inject_soft_source_2d(
    ftype* field,
    ftype value
);