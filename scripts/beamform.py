import h5py as h5
import numpy as np

import ctypes
import cupy as cu

from Beamformer import Beamformer, Probe_t

import time

FILE1 = 'Alpinion_L3-8_CPWC_hyperechoic_scatterers.uff'
FILE2 = 'Alpinion_L3-8_CPWC_hypoechoic.uff'

drive_path = FILE2
file = h5.File(drive_path, 'r')

dset = file['channel_data']
data = dset['data'][()]
seq = dset['sequence']

# params
fc = dset['pulse']['center_frequency'][()].astype(np.uint32).item()
fs = dset['sampling_frequency'][()].astype(np.uint32).item()
c = dset['sound_speed'][()].astype(np.uint32).item()

num_elements = data.shape[1]    # 128 in this case
num_angles = data.shape[0]      # 21 in this case (21 transmit angles)
time_samples = data.shape[2]    # 4352 in this case

dt = 1/fs
t_max = time_samples * dt
z_max = (c * t_max) / 2

azimuths = [seq[k]['source']['azimuth'][()].item() for k in seq]
x_elements = dset['probe']['geometry'][0]

PIXEL_GRID = 1024

x_min = min(x_elements); x_max = max(x_elements)
z_min = 0
x_grid = np.linspace(x_min, x_max, PIXEL_GRID)
z_grid = np.linspace(z_min, 0.055, PIXEL_GRID)
xi, zi = np.meshgrid(x_grid, z_grid)

apod_weights = 0.54 - 0.46*np.cos(2*np.pi*np.arange(128)/127)

params = Probe_t(fc, fs, c, num_elements, num_angles, time_samples)

b = Beamformer(params, azimuths, x_elements, apod_weights, data, x_grid, z_grid)

get_kernel = lambda fname: f"./kernels/{fname}.so"

b.setup(get_kernel("naive"))
times_naive = b.profile(PIXEL_GRID)

b.setup(get_kernel("naive_fastmath"))
times_naive_fast = b.profile(PIXEL_GRID)

b.setup(get_kernel("beamform_fnum"))
times_restrict = b.profile(PIXEL_GRID)

print(f"NAIVE: {times_naive.mean():.3f} [ms]")
print(f"NAIVE (FAST MATH): {times_naive_fast.mean():.3f} [ms]")
print(f"FNUM_RESTRICT: {times_restrict.mean():.3f} [ms]")

b.setup(get_kernel("beamform_fnum"))
image, _ = b.run(PIXEL_GRID)
b.show_bmode(image, save_plot=True, fname="hypoechoic.png")