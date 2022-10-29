#include <iostream>
#include <stdio.h>
using namespace std;
__global__  void nest_helloworld(int iSize,int iDepth){
    unsigned int tid = threadIdx.x;
    printf("depth:%d blockIdx:%d threadIdx:%d \n",iDepth,blockIdx.x,tid);
    if(iSize == 1) return;
    int nthread = iSize >> 1;
    if(tid == 0 && nthread > 0){
        nest_helloworld<<<1,nthread>>>(nthread, ++iDepth);
        printf("------------> nested execution depth:%d \n",iDepth);
    }
}

int main(){
    int size = 64;
    int block_x = 1;
    dim3 block(block_x,1);
    dim3 grid((size-1)/block.x+1,1);
    nest_helloworld<<<grid,block>>>(size, 0);
    cudaGetLastError();
    cudaDeviceReset();
    return 0;
}