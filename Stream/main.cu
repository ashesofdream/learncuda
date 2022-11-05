#include <bits/types/FILE.h>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <omp.h>
#include "util.h"
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
__global__ void sum_arrays(T * array , T* ret){
    //TODO: sum array
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

    grid.x = ret_size;

    for(int i = 0 ; i < 3 ; ++i ) {
        cudaMallocHost(&arrays[i],array_bytes);
        cudaMalloc(&arrays_dev[i],array_bytes);
        cudaMallocHost(&rets[i],ret_bytes);
        cudaMalloc(&rets_dev[i], ret_bytes);
    }
    for(int i = 0 ; i < array_size; ++i ) arrays[i][0]=arrays[i][1]=arrays[i][2]=i;
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
        cudaMemcpyAsync(arrays_dev[i], arrays[i], array_bytes, cudaMemcpyHostToDevice,streams[i]);
        sum_arrays<<<grid,block,0,streams[i]>>>(array1_dev, rets_dev[i]);
        cudaMemcpyAsync(rets[i], rets_dev[i], 1, cudaMemcpyDeviceToHost,streams[i]);
    }
    //TODO cpu parallel and query
    
    //TODO: calculate time
    

    for(int i = 0 ; i < 3 ; i++){
        cout << "array "<<i <<" ";
        for(int i = 0; i < ret_size; ++i) cout << rets[i] <<" " ;
        cout<<endl;
    }
    //TODO: print caluate time
    //TODO: print sum

}


int main(){
    first_stream();
    test_wait_event();
    
}