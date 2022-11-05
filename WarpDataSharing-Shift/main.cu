#include <cstddef>
#include <iostream>
#include "util.h"
using namespace std;


#define BDIMX 32

//this function used to change all array value  = array[2]
__global__ void shift_array(float * array){
    int idx = threadIdx.x;
    float val = array[idx];
    array[idx] = __shfl(val,2,16);
}

__global__ void shift_up(float*array,int offset){
    int  idx=  threadIdx.x;
    float val = array[idx];
    array[idx] = __shfl_up(val,offset,BDIMX);
}
__global__ void shift_down(float* array , int offset){
    int idx= threadIdx.x;
    float val = array[idx];
    array[idx] = __shfl_down(val,offset,BDIMX);
}

__global__ void butterfly_exchange(float* array){
    int idx = threadIdx.x;
    float val = array[idx];
    array[idx] = __shfl_xor(val,1,BDIMX);
}

__global__ void block_butterfly_exchange(float* array){
    int idx = threadIdx.x << 2;

    array[idx] = __shfl_xor(array[idx],1,BDIMX);
    array[idx+1] = __shfl_xor(array[idx+1],1,BDIMX);
    array[idx+2] = __shfl_xor(array[idx+2],1,BDIMX);
    array[idx+3] = __shfl_xor(array[idx+3],1,BDIMX);
}

//用shift实现归约
template<typename T>
__inline__ __device__  T shift_reduction(T mySum){
    mySum += __shfl_xor(mySum,16);
    mySum += __shfl_xor(mySum,8);
    mySum += __shfl_xor(mySum,4);
    mySum += __shfl_xor(mySum,2);
    mySum += __shfl_xor(mySum,1);
    return mySum;
}

template< typename T >
__global__ void sumArray(T* array, T* ret,size_t array_size){
    __shared__ T shared_arr[BDIMX];
    int idx = blockDim.x * blockIdx.x  + threadIdx.x;
    
    if(idx > array_size) return ;
    T my_sum = array[idx];
    

    my_sum = shift_reduction<T>(my_sum);
    // printf("%d:%f ",idx,my_sum);
    int lane_idx = threadIdx.x % warpSize;
    int warp_idx = threadIdx.x / warpSize;
    // printf("%d:%d ",idx,warp_idx);
        if(lane_idx == 0){shared_arr[warp_idx] = my_sum;}
    __syncthreads();
    my_sum = threadIdx.x < BDIMX?shared_arr[threadIdx.x]:0;
    printf("%d:%f ",threadIdx.x,my_sum);
    if(threadIdx.x==0||threadIdx.x==1) {
        //注意 如果其他线程没有执行到这句 则不能访问到其它线程的相同变量
        my_sum = shift_reduction<T>(my_sum); 
    }
    // if (warp_idx==0) my_sum = shift_reduction(my_sum);
    // printf("sumsum--%d:%f ",threadIdx.x,my_sum);
    if (threadIdx.x == 0) ret[blockIdx.x] = my_sum;
}

void test_shift(){
    size_t array_size= BDIMX*2;
    size_t array_bytes = array_size * sizeof(float);
    auto &&[array_host,array_dev] = util::cudaMallocHostAndDev<float>(array_bytes);
    auto &&[ret_host,ret_dev] = util::cudaMallocHostAndDev<float>(array_bytes);
    for(int i = 0 ; i  < array_size ; ++i ) array_host[i] = static_cast<float>(i);
    
    cudaMemcpy(array_dev, array_host, array_bytes, cudaMemcpyHostToDevice);
    // shift_array<<<1,BDIMX>>>(array_dev);
    // shift_up<<<1,BDIMX>>>(array_dev, 3);
    // shift_down<<<1,BDIMX>>> (array_dev, 3);
    // butterfly_exchange<<<1,BDIMX>>>(array_dev);
    // block_butterfly_exchange<<<1,BDIMX>>>(array_dev);
    sumArray<float><<<1,array_size>>>(array_dev,ret_dev,array_size);
    cudaMemcpy(ret_host, ret_dev, array_bytes, cudaMemcpyDeviceToHost);
    // for(int i = 0 ; i < array_size ; ++i ) cout <<array_host[i] << " ";
    // cout<<endl;
    cout << "sum:"<<ret_host[0]<<endl;
}

int main(){
    test_shift();
}