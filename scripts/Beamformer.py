import numpy as np
import cupy as cu
import ctypes
import shutil
import uuid

from scipy.signal import hilbert
import matplotlib.pyplot as plt

arr_ptr = lambda arr: ctypes.cast(arr.data.ptr, ctypes.POINTER(ctypes.c_float))

class Probe_t(ctypes.Structure):
    _fields_ = [
        ("fc", ctypes.c_uint32),
        ("fs", ctypes.c_uint32),
        ("c", ctypes.c_uint32),
        ("num_elements", ctypes.c_uint32),
        ("num_angles", ctypes.c_uint32),
        ("time_samples", ctypes.c_uint32)
    ]


class Beamformer():
    def __init__(self, params, azimuths, x_elements, apod_weights, data, x_grid, z_grid):
        self.params = params
        self.azimuths = cu.asarray(azimuths, dtype=cu.float32)
        self.x_elements = cu.asarray(x_elements, dtype=cu.float32)
        self.apod_weights = cu.asarray(apod_weights, dtype=cu.float32)
        self.data = cu.asarray(data, dtype=cu.float32)
        self.x_grid = cu.asarray(x_grid, dtype=cu.float32)
        self.z_grid = cu.asarray(z_grid, dtype=cu.float32)

    def setup(self, dll_path):
        dll_unique = f"./kernel_cache/kernel_{dll_path[10:-3]}_{uuid.uuid4().hex}.so"
        shutil.copy(dll_path, dll_unique)

        self.lib = ctypes.CDLL(dll_unique)
        self.lib.launch_kernel.argtypes = [
            Probe_t,                        # params struct
            ctypes.POINTER(ctypes.c_float), # azimuths
            ctypes.POINTER(ctypes.c_float), # x_elements
            ctypes.POINTER(ctypes.c_float), # apod_weights
            ctypes.POINTER(ctypes.c_float), # data
            ctypes.POINTER(ctypes.c_float), # x_grid
            ctypes.POINTER(ctypes.c_float), # z_grid
            ctypes.c_int,                   # grid_size
            ctypes.POINTER(ctypes.c_float)  # rf_image
        ]
        self.lib.launch_kernel.restype = None

    def __bmode(self, image_raw):
        img_complex = hilbert(image_raw , axis=0)
        envelope = np.abs(img_complex)
        bmode = 20*np.log10(envelope/np.max(envelope))
        bmode = np.clip(bmode, -60, 0)
        return bmode

    def show_bmode(self, image_raw, save_plot=False, fname="figure.png"):
        bmode = self.__bmode(image_raw)

        x_grid_np = cu.asnumpy(self.x_grid) * 1000   # meters -> mm
        z_grid_np = cu.asnumpy(self.z_grid) * 1000

        plt.imshow(
            bmode,
            cmap='gray',
            aspect='auto',
            extent=[x_grid_np.min(), x_grid_np.max(), z_grid_np.max(), z_grid_np.min()]
        )
        plt.xlabel("Lateral [mm]")
        plt.ylabel("Axial depth [mm]")
        plt.colorbar(label="dB")
        figname = f"./images/{fname}"
        if(save_plot): plt.savefig(figname)
        plt.show()

    def run(self, PIXEL_GRID, postprocess=False):
        self.rf_image_gpu = cu.zeros((PIXEL_GRID, PIXEL_GRID), dtype=cu.float32)

        start = cu.cuda.Event()
        stop = cu.cuda.Event()

        start.record()
        err = self.lib.launch_kernel(
            self.params,
            arr_ptr(self.azimuths), arr_ptr(self.x_elements), arr_ptr(self.apod_weights),
            arr_ptr(self.data), arr_ptr(self.x_grid), arr_ptr(self.z_grid), ctypes.c_int(PIXEL_GRID), arr_ptr(self.rf_image_gpu)
        )
        stop.record()
        stop.synchronize()

        image_raw = cu.asnumpy(self.rf_image_gpu)
        ms = cu.cuda.get_elapsed_time(start, stop)
        return image_raw, ms

    def profile(self, PIXEL_GRID, warmups=3, runs=10):
        for _ in range(warmups):
            self.run(PIXEL_GRID)

        times = []
        for _ in range(runs):
            _, ms = self.run(PIXEL_GRID)
            times.append(ms)

        times = np.array(times)
        return times