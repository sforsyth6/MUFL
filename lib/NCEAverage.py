import torch
from torch.autograd import Function
from torch import nn
from .alias_multinomial import AliasMethod
import math

import numpy as np

import time

import cupy as cp
from torch.utils.dlpack import to_dlpack

import models.pmath as pmath

class NCEFunction(Function):
    @staticmethod
    def forward(self, x, y, memory, params, Z):
        T = params[0].item()

        batchSize = x.size(0)
        outputSize = memory.size(0)
        inputSize = memory.size(1)
        
        #select memory vectors for batch
        weight = torch.index_select(memory, 0, y)
        
        out = torch.bmm(weight.data.unsqueeze(1), x.data.unsqueeze(2)).view(batchSize).clone()

        out.div_(T).exp_()

        out.div_(Z)

        self.save_for_backward(x, memory, y, weight, out, params)
        return out

    @staticmethod
    def backward(self, gradOutput):
        x, memory, y, weight, out, params = self.saved_tensors
        T = params[0].item()
        momentum = params[1].item()

        batchSize = gradOutput.size(0)
        inputSize = memory.size(1)

        # gradients d Pm / d linear = exp(linear) / Z
        gradOutput.data.mul_(out)
        # add temperature
        gradOutput.data.div_(T)
        gradOutput = gradOutput.unsqueeze(1)
        
        # gradient of linear
        gradInput = gradOutput*weight

        # update the non-parametric data
        weight.mul_(momentum)
        weight.add_(torch.mul(x.data, 1-momentum))
        w_norm = torch.norm(weight, dim=-1, keepdim=True)
        updated_weight = weight.div(w_norm)
        memory.index_copy_(0, y, updated_weight)
        return gradInput, None, None, None, None


class NCEAverage(nn.Module):

    def __init__(self, inputSize, outputSize, T=0.1, momentum=0.5, batchSize=128):
        super(NCEAverage, self).__init__()
        self.inputSize = inputSize
        self.outputSize = outputSize

        stdv = 1. / math.sqrt(inputSize/3)
        #self.xminy = torch.zeros(batchSize,outputSize).double().cuda()

        self.register_buffer('params',torch.tensor([T,momentum]))
        self.register_buffer('memory', torch.rand(outputSize, inputSize, dtype=torch.float32).mul_(2*stdv).add_(-stdv))

    def forward(self, x, y):
        batchSize = x.size(0)
        #self.xminy *= 0.0
        #self.Z *= 0.0
        #Z_batch = self.Z[:batchSize]
        Z = self.generate_z(self.memory, x,self.params[0].item(),self.outputSize)
        out = NCEFunction.apply(x, y, self.memory, self.params, Z)
        return out

    def update_m(self, m):
        self.params[1] = m

    def generate_z(self, memory, x, T, outputSize):
        batchSize = x.size(0)
        z_size = int(outputSize//4) + 1
        z = torch.zeros((batchSize,z_size), dtype=torch.float32).cuda()

        x1 = to_dlpack(x)
        z1 = to_dlpack(z)
        memory1 = to_dlpack(memory)

        c_x = cp.fromDlpack(x1) #.astype(cp.float32)
        c_z = cp.fromDlpack(z1)
        c_mem = cp.fromDlpack(memory1) #.astype(cp.float32)
        
        kernel = self.cuda_kernel_UFL()
        kernel((int(z_size//64) + 1, (batchSize // 16) + 1), (64,16), (c_x, c_mem, T, c_z, int(z_size), int(batchSize), int(outputSize)))

        return z.sum(dim=-1) #.double()


    def cuda_kernel_UFL(self):
        #N is args.low_dim and needs to be manually changed if args.low_dim is not 128
        #n is a value I decided to use to split up the dot product, it allows for 4 calculations per thread
        
        kernel = cp.RawKernel(r'''
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
            
            __global__ void z_loop(const float* __restrict__ x, const float* __restrict__ w, const double t, float *z, 
                                    const int z_size, const int batch, const int w_size) {
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

        return kernel