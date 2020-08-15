#import torch
import cupy as cp
from cupy.cuda import function
from collections import namedtuple
import numpy as np
from torch.utils.dlpack import to_dlpack

import models.pmath as pmath
import time

batch = 128
N = 128

w_size = 101

cp.random.seed(954)

x = cp.random.rand(batch, N).astype(cp.float32)
w = cp.random.rand(w_size, N).astype(cp.float32)

#x = cp.ones((batch, N), cp.float32)
#w = cp.ones((w_size, N), cp.float32)

t = float(100)

z_size = int(w_size//4) + 1

z = cp.zeros((batch,z_size), dtype=cp.float32)

kernel_ufl = cp.RawKernel(r'''
    extern "C"{
    
    #define N 128
    #define n 4

    __device__ float dot(const float* __restrict__ x,const float* __restrict__ w){

        float sum=0.0f;

        #pragma unroll N
        for (int i=0; i < N; i++){
            sum += x[i]*w[i];
        }

        return sum;

    }
    
    __global__ void z_loop(const float* __restrict__ x, const float* __restrict__ w, const double t, float *z, const int z_size, const int batch, const int w_size) {
        const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
        const int tidy = blockDim.y * blockIdx.y + threadIdx.y;


        if (tidx < z_size && tidy < batch){
            float sum=0.0f;

            #pragma unroll n
            for (int i=0; i < n; i++){
                if ((tidx*N*n + i*N) < w_size*N){
                    sum += expf(dot(&x[tidy*N],&w[tidx*N*n + i*N]) / t);
                }
            }

            __syncthreads();

            z[tidy*z_size + tidx] = sum;
        }
    }

}''', 'z_loop')

s = time.time()


kernel_ufl((int(z_size//64) + 1, batch), (64,1), (x, w, t, z, z_size, batch, w_size))
print (z)
z.sum(axis=-1)
print (time.time()-s)

z2 = cp.zeros((batch), dtype=cp.float32)

for i,x1 in enumerate(x):
    sum = 0
    for w1 in w:
        sum += cp.exp(cp.dot(x1,w1)/t)
    z2[i] = sum
print (z2 - z.sum(axis=-1))