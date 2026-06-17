#include <cstdint>
#include <cmath>

extern "C" {

    struct Probe_t {
        uint32_t fc, fs, c, num_elements, num_angles, time_samples;
    };

    __global__ void beamform2(
        Probe_t params,
        float* __restrict__ azimuths, float* __restrict__ x_elements, float* __restrict__ apod_weights,
        float* __restrict__ data, float* __restrict__ x_grid, float* __restrict__ z_grid, int grid_size, float* __restrict__ rf_image
    ) {

        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;

        uint32_t fs = params.fs;
        uint32_t c = params.c;
        uint32_t num_elements = params.num_elements;
        uint32_t num_angles = params.num_angles;
        uint32_t time_samples = params.time_samples;

        float fnum = 1.5f;

        if (col >= grid_size || row >= grid_size) return;

        float pixel_sum = 0;
        float x = x_grid[col];
        float z = z_grid[row];
        float z_squared = z*z;
        float c_recip = 1/(float)c;

        for(int a = 0; a < num_angles; a++) {
            float angle = azimuths[a];
            float t_tx = (z*__cosf(angle) + x*__sinf(angle)) * c_recip;

            for(int e = 0; e < num_elements; e++) {
                float sample = 0;
                float dx = x - x_elements[e];
                float t_rx = sqrtf(z_squared + dx*dx) * c_recip;
                float tau_total = t_tx + t_rx;
                float sample_idx = tau_total * fs;

                if(z/(fabsf(dx) + 1e-6f) < 2.0f*fnum) continue;

                bool valid = (sample_idx >= 0) && (sample_idx < time_samples-1);

                if(valid) {
                    int idx_floored = int(floorf(sample_idx));
                    int idx_next = idx_floored + 1;
                    float frac = sample_idx - idx_floored;
                    int stride = (a*num_elements+e)*time_samples;
                    sample = (1.0f-frac)*data[stride+idx_floored] + frac*data[stride+idx_next];
                }

                pixel_sum = fmaf(apod_weights[e], sample, pixel_sum);
            }
        }
        rf_image[row*grid_size+col] = pixel_sum;
    }

    void launch_kernel(
        Probe_t params,
        float* azimuths, float* x_elements, float* apod_weights,
        float* data, float* x_grid, float* z_grid, int grid_size, float* rf_image
    ) {
        dim3 block(16, 16);
        dim3 grid((grid_size + block.x - 1)/block.x,
                  (grid_size + block.y - 1)/block.y);
        beamform2<<<grid, block>>> (
            params, azimuths, x_elements, apod_weights,
            data, x_grid, z_grid, grid_size, rf_image
        );
        //cudaDeviceSynchronize();
    }
}