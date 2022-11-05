#include <cstdio>
#include <iostream>
#include "util.h"
using namespace std;

constexpr int BDIM = 32;
constexpr int RADIUS = 4;
__constant__ int coef_dev[RADIUS+1];
__global__ void different_equation_contant_ver(int *in , int* out ){
    __shared__ int share_array[2*RADIUS+BDIM];//n+2*radius
    int idx = threadIdx.x + blockDim.x * blockIdx.x; // 1D array index;
    int cal_index = threadIdx.x + RADIUS;
    //copy elements to share_array;
    share_array[cal_index] = in[idx];
    if(threadIdx.x < RADIUS){
        share_array[cal_index - RADIUS] = in[idx-RADIUS];
        share_array[cal_index+BDIM] = in[idx+BDIM];
    }
    __syncthreads();
    int tmp = 0;

#pragma unroll
    for(int i = 1 ; i<= RADIUS ; ++i){
        tmp += coef_dev[i-1]*(share_array[cal_index+i]-share_array[cal_index-i]);
    }
    out[idx] = tmp;
    // printf("%d ",tmp);
}


__global__ void different_equation_read_baseline(int * in ,int * out ,int* coef_gmem){
    __shared__ int share_array[RADIUS*2 + BDIM];
    
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int share_array_idx = threadIdx.x + RADIUS;

    share_array[share_array_idx] = in[idx];
    if(threadIdx.x < RADIUS){
        share_array[share_array_idx-RADIUS] = in[idx - RADIUS];
        share_array[share_array_idx+BDIM] = in [idx + BDIM];
    }

    __syncthreads();

    int tmp = 0;

#pragma unroll
    for(int i = 1 ; i <= RADIUS; ++i){
        tmp += coef_gmem[i-1] * (share_array[share_array_idx+i]- share_array[share_array_idx - i]);
    }
    out[idx] = tmp;
    // printf("%d ",tmp);
    // printf("111");

}

__global__ void different_equation_read_only_ver(int * in ,int * out ,int* coef_gmem){
    //这个版本用ReadOnly Cache来实现
    __shared__ int share_array[RADIUS*2 + BDIM];
    
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int share_array_idx = threadIdx.x + RADIUS;

    share_array[share_array_idx] = in[idx];
    if(threadIdx.x < RADIUS){
        share_array[share_array_idx-RADIUS] = in[idx - RADIUS];
        share_array[share_array_idx+BDIM] = in [idx + BDIM];
    }

    __syncthreads();

    int tmp = 0;

#pragma unroll
    for(int i = 1 ; i <= RADIUS; ++i){
        tmp += __ldg(&coef_gmem[i-1]) * (share_array[share_array_idx+i]- share_array[share_array_idx - i]);
        // printf("s%d ",coef_gmem[i-1]);
    }
    out[idx] = tmp;

}

void test_constant_memory(){
    int  * array,*array_dev;
    int coef_size = RADIUS , array_size = 1 << 24;
    int coef_bytes = sizeof(int) * coef_size , array_bytes = sizeof(int) * array_size;
    auto&& [coef , coef_gmem] = util::cudaMallocHostAndDev<int>(coef_bytes);
    // coef[0] = 1 ; coef[1] = 1;coef[2] = 1;coef[3] =1;
    CHECK(cudaMallocHost(&array,array_bytes));
    for(int i = 0 ; i <array_size ; ++i ) array[i] = i;
    util::init_array_int(coef, coef_size);
    util::init_array_int(array,array_size);;
    CHECK(cudaMalloc(&array_dev,array_bytes))

    // for(int i = 0 ; i < array_size ; ++i) cout << array[i] <<" "; cout << endl;
    // for(int i = 0 ; i < coef_size ; ++i) cout << coef[i] <<" "; cout << endl;
   
    CHECK(cudaMemcpyToSymbol(coef_dev, coef, coef_bytes));
    CHECK(cudaMemcpy(coef_gmem,coef,coef_bytes,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(array_dev,array,array_bytes,cudaMemcpyHostToDevice));
    

    auto&& [ret_host,ret_dev] = util::cudaMallocHostAndDev<int>(array_bytes);
    different_equation_read_baseline<<<array_size/BDIM,BDIM>>>(array_dev,ret_dev,coef_gmem);
    cudaDeviceSynchronize();
    different_equation_contant_ver<<<array_size/BDIM,BDIM>>>(array_dev,ret_dev);
    cudaDeviceSynchronize();
    // cudaMemcpy(ret_host ,ret_dev , array_bytes,cudaMemcpyDeviceToHost);
    // for(int i = 0 ; i < array_size ; ++i) cout << ret_host[i] <<" "; cout << endl;


    // cudaMemcpy(ret_host ,ret_dev , array_bytes,cudaMemcpyDeviceToHost);
    // for(int i = 0 ; i < array_size ; ++i) cout << ret_host[i] <<" "; cout << endl;
 


    different_equation_read_only_ver<<<array_size/BDIM,BDIM>>>(array_dev,ret_dev,coef_gmem);
    cudaDeviceSynchronize();
    // cudaMemcpy(ret_host ,ret_dev , array_bytes,cudaMemcpyDeviceToHost);
    // for(int i = 0 ; i < array_size ; ++i) cout << ret_host[i] <<" "; cout << endl;


}

int main(){
    test_constant_memory();
}