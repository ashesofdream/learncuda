#include <bits/types/FILE.h>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <omp.h>
#include "util.h"
#include <atomic>
#include <vector>
using namespace std;

constexpr int N = 1 << 24;
__global__ void kernel_1(double * sum){
    for(int i = 0 ; i < N ; ++i ){
        *sum += tan(0.1) * tan(0.1);
    }
}

__global__ void kernel_2(double * sum){
    for(int i = 0 ; i < N ; ++i ){
        *sum += tan(0.1) * tan(0.1);
    }
}
__global__ void kernel_3(double * sum){
    for(int i = 0 ; i < N ; ++i ){
        *sum += tan(0.1) * tan(0.1);
    }
}
__global__ void kernel_4(double * sum){
    for(int i = 0 ; i < N ; ++i ){
        *sum += tan(0.1) * tan(0.1);
        
    }
}

void first_stream(){
    int stream_num = 3;
    dim3 block(1);
    dim3 grid(1);
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaStream_t* streams = (cudaStream_t*)malloc(sizeof(cudaStream_t)* stream_num);
    for(int i = 0 ; i < stream_num ; ++i ){
        CHECK(cudaStreamCreate(&streams[i]));
    }
    auto&& [sum_host,sum_dev]=util::cudaMallocHostAndDev<double>(sizeof(double));

    cudaEventRecord(start);
    // for(int i = 0 ; i < stream_num ; ++i){
    //     kernel_1<<<block,grid,0,streams[i]>>>(sum_dev);
    //     kernel_2<<<block,grid,0,streams[i]>>>(sum_dev);
    //     kernel_3<<<block,grid,0,streams[i]>>>(sum_dev);
    //     kernel_4<<<block,grid,0,streams[i]>>>(sum_dev);
    // }
    omp_set_num_threads(stream_num);
    #pragma omp parallel
    {
        int i = omp_get_thread_num();
        kernel_1<<<block,grid,0,streams[i]>>>(sum_dev);
        kernel_2<<<block,grid,0,streams[i]>>>(sum_dev);
        kernel_3<<<block,grid,0,streams[i]>>>(sum_dev);
        kernel_4<<<block,grid,0,streams[i]>>>(sum_dev);
    }

    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaMemcpy(sum_host, sum_dev, sizeof(double*),cudaMemcpyDeviceToHost);
    float elapse_time = 0.f;
    cudaEventElapsedTime(&elapse_time, start, stop);
    cout<<"elapse_time:"<<elapse_time<<endl;
    cout<<"sum:"<<*sum_host<<endl;
}

void test_wait_event(){
    int stream_num = 3;
    dim3 block(1);
    dim3 grid(1);
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaStream_t* streams = (cudaStream_t*)malloc(sizeof(cudaStream_t)* stream_num);
    cudaEvent_t* events = (cudaEvent_t*)malloc(sizeof(cudaEvent_t)*stream_num);
    for(int i = 0 ; i < stream_num ; ++i ){
        CHECK(cudaStreamCreate(&streams[i]));
        CHECK(cudaEventCreate(&events[i],cudaEventDisableTiming));
    }
    auto&& [sum_host,sum_dev]=util::cudaMallocHostAndDev<double>(sizeof(double));

    cudaEventRecord(start);
    omp_set_num_threads(stream_num);
    #pragma omp parallel
    {
        int i = omp_get_thread_num();
        kernel_1<<<block,grid,0,streams[i]>>>(sum_dev);
        kernel_2<<<block,grid,0,streams[i]>>>(sum_dev);
        kernel_3<<<block,grid,0,streams[i]>>>(sum_dev);
        kernel_4<<<block,grid,0,streams[i]>>>(sum_dev);

        cudaEventRecord(events[i],streams[i]);
        cudaStreamWaitEvent(streams[stream_num-1], events[i],0);
    }

    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaMemcpy(sum_host, sum_dev, sizeof(double*),cudaMemcpyDeviceToHost);
    float elapse_time = 0.f;
    cudaEventElapsedTime(&elapse_time, start, stop);
    cout<<"elapse_time:"<<elapse_time<<endl;
    cout<<"sum:"<<*sum_host<<endl;
}

template<typename T>
__device__ __inline__ T sum_grid(T my_sum){
    my_sum += __shfl_xor_sync(-1,my_sum,16);
    my_sum += __shfl_xor_sync(-1,my_sum,8);
    my_sum += __shfl_xor_sync(-1,my_sum,4);
    my_sum += __shfl_xor_sync(-1,my_sum,2);
    my_sum += __shfl_xor_sync(-1,my_sum,1);
    return my_sum;
}


const int my_warp_size = 32; 
template<typename T>
__global__ void sum_arrays(T * array , T * ret, const size_t array_size){
    //TODO: sum array
    __shared__ T warp_sum[1024/32];
    int g_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(g_idx > array_size) return;
    int warp_idx = threadIdx.x / warpSize;
    int lane_idx = threadIdx.x % warpSize;
    T my_sum = array[g_idx];
    my_sum = sum_grid<T>(my_sum);
    if(lane_idx == 0 ) warp_sum[warp_idx] = my_sum;
    __syncthreads();

    my_sum = threadIdx.x < 32 ?warp_sum[threadIdx.x]:0;
    if(threadIdx.x < my_warp_size) my_sum = sum_grid<T>(my_sum);
    if(threadIdx.x == 0) ret[blockIdx.x] = my_sum; 
}

void test_wait_overlap_and_callback(){
    float * array1,*array2,*array3;
    float* arrays[]={array1,array2,array3};
    
    float * ret1,*ret2,*ret3;
    float* rets[]={ret1,ret2,ret3};
    
    float *array1_dev,*array2_dev,*array3_dev;
    float* arrays_dev[]={array1_dev,array2_dev,array3_dev};

    float *ret1_dev,*ret2_dev,*ret3_dev;
    float* rets_dev[]={ret1_dev,ret2_dev,ret3_dev};


    dim3 grid(1);
    dim3 block(1024);

    size_t array_size = 1 << 24 ,array_bytes = array_size *sizeof(float);
    size_t ret_size = (array_size-1) / block.x+1,ret_bytes = ret_size * sizeof(float);

    //TODO:fix there
    // grid.x = ret_size;

    for(int i = 0 ; i < 3 ; ++i ) {
        cudaMallocHost(&arrays[i],array_bytes);
        cudaMalloc(&arrays_dev[i],array_bytes);
        cudaMallocHost(&rets[i],ret_bytes);
        cudaMalloc(&rets_dev[i], ret_bytes);
    }
    for(int i = 0 ; i < array_size; ++i ) arrays[0][i]=arrays[1][i]=arrays[2][i]=i;

    //prepare stream 
    int stream_num = 3;
    cudaStream_t* streams = (cudaStream_t*)malloc(sizeof(cudaStream_t)* stream_num);
    cudaEvent_t* events = (cudaEvent_t*)malloc(sizeof(cudaEvent_t)*stream_num);
    for(int i = 0 ; i < stream_num ; ++i ){
        CHECK(cudaStreamCreate(&streams[i]));
        CHECK(cudaEventCreate(&events[i],cudaEventDisableTiming));
    }


    //begin transfer and caculate
    #pragma omp parallel
    {
        int i = omp_get_thread_num();
        if(i < 3){
            cudaMemcpyAsync(arrays_dev[i], arrays[i], array_bytes, cudaMemcpyHostToDevice,streams[i]);
            sum_arrays<<<grid,block,0,streams[i]>>>(arrays_dev[i], rets_dev[i],array_size);
            cudaMemcpyAsync(rets[i], rets_dev[i], ret_bytes, cudaMemcpyDeviceToHost,streams[i]);
        }
    }


    //TODO cpu parallel and query
    vector<int> my_sums(stream_num,0);
    vector<int> wait_cnt(stream_num,0);
    #pragma omp parallel
    {
        int i = omp_get_thread_num();
        if(i < 3){
            cudaError_t stream_error=cudaStreamQuery(streams[i]);
            while (stream_error==cudaErrorNotReady) {
                ++wait_cnt[i];
                cout<<"stream["<<i<<"]"<<" wait times:"<<wait_cnt[i]<<endl;
                stream_error=cudaStreamQuery(streams[i]);
            }
            if(stream_error!=cudaSuccess){
                cout<<"error:"<<cudaGetErrorString(stream_error)<<endl;
                exit(0);
            }
            for(int j = 0 ; j < ret_size ; ++j) my_sums[i] += rets[i][j]; 
        }
        
    }

    //TODO: calculate time
    
    //TODO: print caluate time
    //TODO: print sum
    for(int i = 0 ; i < stream_num; ++i)
    cout<<"sum["<<i<<"]"<<"="<<my_sums[i]<<endl;
    float cpu_sum = 0;
    for(int i = 0; i < 1024;++i) cpu_sum += arrays[0][i];
    cout << "cpu sum:" << cpu_sum <<endl;
}


int main(){
    // first_stream();
    // test_wait_event();
    test_wait_overlap_and_callback();
    
}