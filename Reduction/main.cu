#include <iostream>
#include "util.h"
using namespace std;

__global__ void reduceNeighbored(int * in_data,int* out_data,unsigned int n){
//    printf("still running");
    unsigned int tid = threadIdx.x;
    //if(tid >= n) return;
    int* cur_data = in_data + blockIdx.x * blockDim.x;
    for(int stride = 1 ; stride < blockDim.x ; stride *= 2){
        if(tid%(2*stride) == 0){
            cur_data[tid] += cur_data[tid + stride];
        }
        __syncthreads();
    }
    if(tid == 0) out_data[blockIdx.x] = cur_data[0];
}

__global__ void reduceNeighboredLess(int * in_data , int * out_data, unsigned int n){
    unsigned int tid = threadIdx.x;
    //if(tid>=n) return;
    int* cur_data = in_data + blockIdx.x * blockDim.x;
    for(int stride = 1 ; stride < blockDim.x ; stride *= 2){
        int index = 2 * stride * tid;
        if( index < blockDim.x ){
            cur_data[index] += cur_data[index + stride];
        }
        __syncthreads();
    }
    if(tid == 0) out_data[blockIdx.x] = cur_data[0];
}

__global__ void reduceUnroll(int * in_data , int * out_data , unsigned  int n){
    unsigned int tid = threadIdx.x;
    unsigned int true_data_id = blockDim.x * blockIdx.x * 2 + threadIdx.x;
    int* cur_data = in_data + blockIdx.x * blockDim.x*2;

    if(true_data_id + blockDim.x  < n){
        in_data[true_data_id]  += in_data[true_data_id + blockDim.x];
    }
    __syncthreads();

    for(int stride = blockDim.x / 2 ; stride > 0 ; stride >>= 1){
        if(tid < stride){
            cur_data[tid] += cur_data[tid + stride];
        }
        __syncthreads();
    }
    if(tid == 0) out_data[blockIdx.x] = cur_data[0];
}

__global__ void reduceUnroll2(int * in_data , int * out_data , unsigned  int n){
    unsigned int tid = threadIdx.x;
    unsigned int true_data_id = blockDim.x * blockIdx.x * 2 + threadIdx.x;
    int* cur_data = in_data + blockIdx.x * blockDim.x*2;

    if(true_data_id + blockDim.x  < n){
        in_data[true_data_id]  += in_data[true_data_id + blockDim.x];
    }
    __syncthreads();

    for(int stride = blockDim.x / 2 ; stride > 32 ; stride >>= 1){
        if(tid < stride){
            cur_data[tid] += cur_data[tid + stride];
        }
        __syncthreads();
    }
    if(tid < 32){
        volatile int* dir_memory  = cur_data;
        dir_memory[tid] += dir_memory[tid+32];
        dir_memory[tid] += dir_memory[tid+16];
        dir_memory[tid] += dir_memory[tid +8];
        dir_memory[tid] += dir_memory[tid +4];
        dir_memory[tid] += dir_memory[tid +2];
        dir_memory[tid] += dir_memory[tid +1];
    }
    if(tid == 0) out_data[blockIdx.x] = cur_data[0];
}

__global__ void reduceUnroll3(int * in_data , int * out_data , unsigned  int n){
    unsigned int tid = threadIdx.x;
    unsigned int true_data_id = blockDim.x * blockIdx.x * 8 + threadIdx.x;
    int* cur_data = in_data + blockIdx.x * blockDim.x*8;

    if(true_data_id+7 * blockDim.x  < n){
        int a1=in_data[true_data_id];
        int a2=in_data[true_data_id+blockDim.x];
        int a3=in_data[true_data_id+2*blockDim.x];
        int a4=in_data[true_data_id+3*blockDim.x];
        int a5=in_data[true_data_id+4*blockDim.x];
        int a6=in_data[true_data_id+5*blockDim.x];
        int a7=in_data[true_data_id+6*blockDim.x];
        int a8=in_data[true_data_id+7*blockDim.x];
        in_data[true_data_id]=a1+a2+a3+a4+a5+a6+a7+a8;

    }
    __syncthreads();

    for(int stride = blockDim.x / 2 ; stride > 32 ; stride >>= 1){
        if(tid < stride){
            cur_data[tid] += cur_data[tid + stride];
        }
        __syncthreads();
    }
    if(tid < 32){
        volatile int* dir_memory  = cur_data;
        dir_memory[tid] += dir_memory[tid+32];
        dir_memory[tid] += dir_memory[tid+16];
        dir_memory[tid] += dir_memory[tid +8];
        dir_memory[tid] += dir_memory[tid +4];
        dir_memory[tid] += dir_memory[tid +2];
        dir_memory[tid] += dir_memory[tid +1];
    }
    if(tid == 0) out_data[blockIdx.x] = cur_data[0];
}

__global__ void reduceUnroll4(int * in_data , int * out_data , unsigned  int n){
    unsigned int tid = threadIdx.x;
    unsigned int true_data_id = blockDim.x * blockIdx.x * 8 + threadIdx.x;
    int* cur_data = in_data + blockIdx.x * blockDim.x*8;

    if(true_data_id+7 * blockDim.x  < n){
        int a1=in_data[true_data_id];
        int a2=in_data[true_data_id+blockDim.x];
        int a3=in_data[true_data_id+2*blockDim.x];
        int a4=in_data[true_data_id+3*blockDim.x];
        int a5=in_data[true_data_id+4*blockDim.x];
        int a6=in_data[true_data_id+5*blockDim.x];
        int a7=in_data[true_data_id+6*blockDim.x];
        int a8=in_data[true_data_id+7*blockDim.x];
        in_data[true_data_id]=a1+a2+a3+a4+a5+a6+a7+a8;

    }
    __syncthreads();

    if(blockDim.x>=1024 && tid <512)
        cur_data[tid]+=cur_data[tid+512];
    __syncthreads();
    if(blockDim.x>=512 && tid <256)
        cur_data[tid]+=cur_data[tid+256];
    __syncthreads();
    if(blockDim.x>=256 && tid <128)
        cur_data[tid]+=cur_data[tid+128];
    __syncthreads();
    if(blockDim.x>=128 && tid <64)
        cur_data[tid]+=cur_data[tid+64];
    __syncthreads();

    if(tid < 32){
        volatile int* dir_memory  = cur_data;
        dir_memory[tid] += dir_memory[tid+32];
        dir_memory[tid] += dir_memory[tid+16];
        dir_memory[tid] += dir_memory[tid +8];
        dir_memory[tid] += dir_memory[tid +4];
        dir_memory[tid] += dir_memory[tid +2];
        dir_memory[tid] += dir_memory[tid +1];
    }
    if(tid == 0) out_data[blockIdx.x] = cur_data[0];
}



int main(int argc , char** argv){
    bool bResult = false;
    int array_size = 1 << 24;
    cout<< "array_size: "<< array_size<<endl;

    int blockSize = 1024;
    dim3 block(blockSize,1);
    dim3 grid((array_size - 1)/block.x+1,1);
    cout << "grid:"<<grid.x << " block:" << block.x <<endl;

    size_t  bytes  = sizeof(int) * array_size;
    int* in_data_host = new int[array_size];
    int* o_data = new int[grid.x];
    int* tmp = new int[array_size];
    //util::init_array_int(in_data_host, array_size);
    memset(in_data_host,1,array_size);

    int cpu_sum = 0;
    for(int i = 0 ; i < array_size ; ++i) cpu_sum += in_data_host[i];
    cout<< "cpu_sum: "<<cpu_sum<<endl;

    mempcpy(tmp,in_data_host,bytes);
    double  iStart , iElaps;
    int gpu_sum = 0;

    int * in_data_dev,* out_data_dev;
    cudaMalloc(&in_data_dev,bytes);
    cudaMalloc(&out_data_dev,grid.x * sizeof(int));

    gpu_sum = 0;
    cudaMemcpy(in_data_dev,in_data_host,bytes,cudaMemcpyHostToDevice);
    reduceNeighbored<<<grid,block>>>(in_data_dev,out_data_dev,array_size);
    cudaMemcpy(o_data,out_data_dev,grid.x * sizeof(int),cudaMemcpyDeviceToHost);
    for(int i = 0 ; i < grid.x ; ++i) gpu_sum += o_data[i];
    cout<<"reduceNeighbored sum:" << gpu_sum <<endl;


    gpu_sum = 0;
    cudaMemcpy(in_data_dev,in_data_host,bytes,cudaMemcpyHostToDevice);
    reduceNeighboredLess<<<grid,block>>>(in_data_dev,out_data_dev,array_size);
    cudaMemcpy(o_data,out_data_dev,grid.x*sizeof (int),cudaMemcpyDeviceToHost);
    for(int i = 0 ; i < grid.x ; ++i) gpu_sum += o_data[i];
    cout<<"reduceNeighboredLess sum:" << gpu_sum <<endl;

    gpu_sum = 0;
    cudaMemcpy(in_data_dev,in_data_host,bytes,cudaMemcpyHostToDevice);
    reduceUnroll<<<grid,block>>>(in_data_dev,out_data_dev,array_size);
    cudaMemcpy(o_data,out_data_dev,grid.x*sizeof (int),cudaMemcpyDeviceToHost);
    for(int i = 0 ; i < grid.x ; ++i) gpu_sum += o_data[i];
    cout<<"reduceUnroll sum:" << gpu_sum <<endl;

    gpu_sum = 0;
    cudaMemcpy(in_data_dev,in_data_host,bytes,cudaMemcpyHostToDevice);
    reduceUnroll2<<<grid,block>>>(in_data_dev,out_data_dev,array_size);
    cudaMemcpy(o_data,out_data_dev,grid.x*sizeof (int),cudaMemcpyDeviceToHost);
    for(int i = 0 ; i < grid.x ; ++i) gpu_sum += o_data[i];
    cout<<"reduceUnroll2 sum:" << gpu_sum <<endl;

    gpu_sum = 0;
    cudaMemcpy(in_data_dev,in_data_host,bytes,cudaMemcpyHostToDevice);
    reduceUnroll3<<<grid,block>>>(in_data_dev,out_data_dev,array_size);
    cudaMemcpy(o_data,out_data_dev,grid.x*sizeof (int),cudaMemcpyDeviceToHost);
    for(int i = 0 ; i < grid.x ; ++i) gpu_sum += o_data[i];
    cout<<"reduceUnroll3 sum:" << gpu_sum <<endl;

    gpu_sum = 0;
    cudaMemcpy(in_data_dev,in_data_host,bytes,cudaMemcpyHostToDevice);
    reduceUnroll4<<<grid,block>>>(in_data_dev,out_data_dev,array_size);
    cudaMemcpy(o_data,out_data_dev,grid.x*sizeof (int),cudaMemcpyDeviceToHost);
    for(int i = 0 ; i < grid.x ; ++i) gpu_sum += o_data[i];
    cout<<"reduceUnroll4 sum:" << gpu_sum <<endl;

}
