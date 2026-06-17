NVCC := nvcc
NVCCFLAGS := -shared -Xcompiler -fPIC -Wno-deprecated-gpu-targets
UFM := --use_fast_math

TARGETS := beamform_fnum.so naive.so naive_fastmath.so

all: $(TARGETS)

%.so: %.cu 
	$(NVCC) $(NVCCFLAGS) $(UFM) $< -o ./kernels/$@

naive.so: beamform_naive.cu
	$(NVCC) $(NVCCFLAGS) $< -o ./kernels/naive.so

naive_fastmath.so: beamform_naive.cu
	$(NVCC) $(NVCCFLAGS) $(UFM) $< -o ./kernels/naive_fastmath.so

.PHONY: clean all
clean:
	rm -f ./kernels/*
	rm -f ./kernel_cache/*