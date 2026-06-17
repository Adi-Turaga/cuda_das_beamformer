import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.signal import hilbert
import time

def euc_dist(x1, z1, x2, z2):
    return np.sqrt((x2 - x1)**2 + (z2 - z1)**2)

file = h5py.File(r"C:\Users\adist\DSP\Alpinion_L3-8_CPWC_hyperechoic_scatterers.uff", 'r')

dset = file['channel_data']
data = dset['data'][()]
seq = dset['sequence']

# params
fc = dset['pulse']['center_frequency'][()].item()
fs = dset['sampling_frequency'][()].item()
c = dset['sound_speed'][()].item()

num_elements = data.shape[1]    # 128
num_angles = data.shape[0]      # 21
time_samples = data.shape[2]    # 4352

dt = 1/fs
t_max = time_samples * dt
z_max = (c * t_max) / 2

azimuths = np.array([seq[k]['source']['azimuth'][()].item() for k in seq]) # Converted to numpy array
x_elements = dset['probe']['geometry'][0]

PIXEL_GRID = 1024

x_min = min(x_elements); x_max = max(x_elements)
z_min = 0
x_grid = np.linspace(x_min, x_max, PIXEL_GRID)
z_grid = np.linspace(z_min, 0.050, PIXEL_GRID)

xi, zi = np.meshgrid(x_grid, z_grid, indexing='ij')

apod_weights = 0.54 - 0.46*np.cos(2*np.pi*np.arange(128)/127)
rf_image = np.zeros((PIXEL_GRID, PIXEL_GRID))

t_rx = np.sqrt(zi[:, :, None]**2 + (xi[:, :, None] - x_elements[None, None, :])**2) / c

start = time.perf_counter()

for a_idx, a in enumerate(azimuths):
    print(f"BEAMFORMING angle {a_idx}: a")
    t_tx = (zi * np.cos(a) + xi * np.sin(a)) / c
    
    tau_total = t_tx[:, :, None] + t_rx
    sample_idx = tau_total * fs
    
    idx_floored = np.floor(sample_idx).astype(np.int32)
    idx_next = idx_floored + 1
    frac = sample_idx - idx_floored
    
    valid = (idx_floored >= 0) & (idx_next < time_samples)
    
    idx_floored_clipped = np.clip(idx_floored, 0, time_samples - 1)
    idx_next_clipped = np.clip(idx_next, 0, time_samples - 1)
    
    element_indices = np.arange(num_elements)[None, None, :]
    
    data_floor = data[a_idx, element_indices, idx_floored_clipped]
    data_next = data[a_idx, element_indices, idx_next_clipped]

    samples = (1.0 - frac) * data_floor + frac * data_next
    samples[~valid] = 0.0 
    
    pixel_sum = np.sum(samples * apod_weights, axis=2)
    
    rf_image += pixel_sum

stop = time.perf_counter()
elapsed = stop - start
print(f"Vectorized DAS execution time for {PIXEL_GRID}x{PIXEL_GRID}: {elapsed:.3f} [s]")

img_complex = hilbert(rf_image, axis=0)
envelope = np.abs(img_complex)
bmode = 20*np.log10(envelope / np.max(envelope))
bmode = np.clip(bmode, -60, 0)

plt.imshow(bmode.T, cmap='gray', aspect='auto', extent=[x_min*1e3, x_max*1e3, z_max*1e3, z_min*1e3])
plt.xlabel('X position (mm)')
plt.ylabel('Depth Z (mm)')
plt.colorbar(label='dB')
plt.show()