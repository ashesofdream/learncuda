
#include <iostream>
#include <stdlib.h>
#include "util.h"
using namespace std;

__device__ float devData;//__device__前缀等于定义了一个标识符
__global__ void printVari(){
    printf("Device: global variable is %f\n",devData);
}

void test_pinned_memory(){
    //cudaMallocHost会分配一段固定的不会被换页的内存，加速数据迁移。
    int* array_host_pinned = nullptr,*array_device = nullptr,*array_host_unpinned = nullptr;
    size_t array_size = 1 << 24;
    size_t array_bytes = array_size * sizeof(float);
    
    CHECK(cudaMallocHost(&array_host_pinned, array_bytes));//this copy will faster more
    memset(array_host_pinned, 1, array_size);
    array_host_unpinned = (int*)malloc(array_bytes);
    memset(array_host_unpinned, 1, array_size);
    CHECK(cudaMalloc(&array_device, array_bytes));

    cudaMemcpy(array_device, array_host_pinned, array_bytes, cudaMemcpyHostToDevice);
    //cudaMemcpy(array_device, array_host_unpinned, array_bytes, cudaMemcpyHostToDevice);
 
}

__global__ void print_array(int* array , int n){
    for(int i = 0 ; i < n ; ++ i ){
        printf("%x ",array[i]);
    }
    printf("\n");
}
void test_uniform_virutal_address(){
    //统一虚拟寻址UVA + 零拷贝内存 内存实际在分配主机中
    int* array_uva = NULL;
    int array_size = 10;
    int array_bytes = sizeof(int)*array_size;
    cudaHostAlloc(&array_uva,array_size,cudaHostAllocMapped);
    memset(array_uva,0xFF,array_bytes);
    for(int i = 0 ; i < array_size ; ++i) cout <<std::hex<< array_uva[i]<< " ";
    cout<<endl;
    print_array<<<1,1>>>(array_uva, array_size);
    cudaDeviceSynchronize();
}


// transform matrix

__global__ void transform_matrix_read_row(
    int* mat_a , int* mat_b , size_t row_num , size_t col_num
){
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    int row_idx = ix + iy * col_num;
    int col_idx = ix*row_num + iy;

    if(ix < col_num && iy < row_num){
        mat_b[col_idx] = mat_a[row_idx];
    }
}

__global__ void transform_matrix_read_col(
    int* mat_a , int* mat_b , size_t row_num , size_t col_num
){
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    int row_idx = iy*col_num + ix;
    int col_idx = ix*row_num + iy;
    if(ix < col_num && iy < row_num){
        mat_b[row_idx] = mat_a[col_idx];
    }
}

__global__ void transform_matrix_read_col_unrool(
    int* mat_a , int* mat_b , size_t row_num , size_t col_num
){
    int ix = threadIdx.x +(blockDim.x * blockIdx.x * 4);
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    int row_idx = iy*col_num + ix;
    int col_idx = ix*row_num + iy;
    if(ix < col_num && iy < row_num){
        mat_b[row_idx] = mat_a[col_idx];
        mat_b[row_idx + blockDim.x*1] = mat_a[col_idx + row_num*blockDim.x*1];
        mat_b[row_idx + blockDim.x*2] = mat_a[col_idx + row_num*blockDim.x*2];
        mat_b[row_idx + blockDim.x*3] = mat_a[col_idx + row_num*blockDim.x*3];
    }
}
__global__ void transform_matrix_read_col_unrool2(
    int* mat_a , int* mat_b , size_t row_num , size_t col_num
){
    int ix = threadIdx.x + blockDim.x * blockIdx.x * 2;
    int iy = threadIdx.y + blockDim.y * blockIdx.y * 2;
    int row_idx = iy*col_num + ix;
    int col_idx = ix*row_num + iy;
    if(ix < col_num && iy < row_num){
        mat_b[row_idx] = mat_a[col_idx];
        mat_b[row_idx+ blockDim.x] = mat_a[col_idx + row_num*blockDim.x];
        mat_b[row_idx+ blockDim.y*col_num] = mat_a[col_idx + blockIdx.y];
        mat_b[row_idx + blockDim.x + blockDim.y*col_num ] = mat_a[col_idx + blockIdx.y+ row_num*blockDim.x];
    }
}



__global__ void transform_matrix_Diagonal(int * MatA,int * MatB,int nx,int ny)
{
    //理论上按对角顺序访问每个block中对应的数据，避免同时访问DRAM内存中的同一分区，产生排队现象
    //实际上非常非常的慢
    int block_y = blockIdx.x;
    int block_x = (blockIdx.x + blockIdx.y)%gridDim.x;
    int ix=threadIdx.x+blockDim.x*block_x;
    int iy=threadIdx.y+blockDim.y*block_y;
    int idx_row=ix+iy*nx;
    int idx_col=ix*ny+iy;
    if (ix<nx && iy<ny)
    {
        MatB[idx_row]=MatA[idx_col];
    }
}

void transform_matrix(){
    size_t width = 128 , height = 128,
    array_size = width*height,array_bytes = array_size * sizeof(int);
    int* matrix_data = nullptr,*output_data = nullptr;
    CHECK(cudaMallocHost(&matrix_data, array_bytes));
    CHECK(cudaMallocHost(&output_data, array_bytes));
    util::init_array_int(matrix_data,array_size);

    int* matrix_data_dev = nullptr,* output_matrix_dev = nullptr;
    CHECK(cudaMalloc(&matrix_data_dev, array_bytes));
    CHECK(cudaMemcpy(matrix_data_dev, matrix_data, array_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&output_matrix_dev, array_bytes));
    
    dim3 block(32,16);
    dim3 grid((width-1)/block.x+1,(height-1)/block.y+1);
    dim3 gridUnrool4((width-1)/(block.x*4)+1,(height-1)/block.y +1);

    transform_matrix_read_col<<<grid,block>>>(matrix_data_dev, output_matrix_dev, height, width);
    cudaDeviceSynchronize();
    // CHECK(cudaMemcpy(output_data, output_matrix_dev, array_bytes, cudaMemcpyDeviceToHost));
    // util::print_matrix(output_data,width,height);
    // cout<<endl;

    transform_matrix_read_row<<<grid,block>>>(matrix_data_dev, output_matrix_dev, height, width);
    cudaDeviceSynchronize();
    // cudaMemcpy(output_data, output_matrix_dev, array_bytes, cudaMemcpyDeviceToHost);
    // util::print_matrix(output_data,width,height);
    // cout<<endl;


    transform_matrix_read_col_unrool<<<gridUnrool4,block>>>(matrix_data_dev, output_matrix_dev, height, width);
    cudaDeviceSynchronize();
    // cudaMemcpy(output_data, output_matrix_dev, array_bytes, cudaMemcpyDeviceToHost);
    // util::print_matrix(output_data,width,height);
    // cout<<endl;
    
    transform_matrix_Diagonal<<<grid,block>>>(matrix_data_dev, output_matrix_dev, height, width);
    cudaDeviceSynchronize();
    // cudaMemcpy(output_data, output_matrix_dev, array_bytes, cudaMemcpyDeviceToHost);
    // util::print_matrix(output_data,width,height);
    // cout<<endl;

    transform_matrix_read_col_unrool2<<<grid,block>>>(matrix_data_dev, output_matrix_dev, height, width);
    cudaDeviceSynchronize();
    // cudaMemcpy(output_data, output_matrix_dev, array_bytes, cudaMemcpyDeviceToHost);
    // util::print_matrix(output_data,width,height);
    // cout<<endl;
    
    
}

int main(){
    // float val = 3.14f;
    // cudaMemcpyToSymbol(devData, &val, sizeof(float));
    printVari<<<1,1>>>();
    // val = 6.28f;
    // cudaMemcpyToSymbol(devData, &val, sizeof(float));
    // printVari<<<1,1>>>();
    // cudaDeviceSynchronize();

    //test memcpy speed for pinned memory
    //test_uniform_virutal_address();
    transform_matrix();
    return EXIT_SUCCESS;
}