#include <cstdio>
#include <iostream>
#include <cstdlib>
#include "util.h"
#define BDIMX 32
#define BDIMY 32
__global__ void set_row_read_row(int * out){
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx = threadIdx.x + threadIdx.y * blockDim.x;
    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();
    out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void set_col_read_col(int* out){
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx =  threadIdx.y * blockDim.x + threadIdx.x;
    tile[threadIdx.x][threadIdx.y]  = idx;
    __syncthreads();
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void set_row_read_col_dynamic_padding(int* out){
    //通过padding 错开col_idx，减少访问同一个bank的冲突。
    extern __shared__ int tile[];
    int row_idx = threadIdx.y * (blockDim.x+1) + threadIdx.x; //padding
    int col_idx = threadIdx.x * (blockDim.x+1) + threadIdx.y;
    int g_idx = threadIdx.y * blockDim.x + threadIdx.x;
    tile[row_idx] = g_idx;
    __syncthreads();
    out[g_idx] = tile[col_idx];
}

void test_row_and_col(){
    int * array_dev ,* out_dev;
    size_t array_size = 1 << 24 , array_bytes = array_size * sizeof(int);
    cudaMalloc(&array_dev, array_bytes);
    cudaMalloc(&out_dev, array_bytes);


    set_col_read_col<<<1,dim3(32,32)>>>(out_dev);
    cudaDeviceSynchronize();
    set_row_read_row<<<1,dim3(32,32)>>>(out_dev);
    cudaDeviceSynchronize();
    set_row_read_col_dynamic_padding<<<1,dim3(32,32),32*33*sizeof(int)>>>(out_dev);
    cudaDeviceSynchronize();
}

__global__ void set_row_read_col_rectangle(int *out){
    extern __shared__ int tile2[];
    int row_idx = threadIdx.x + threadIdx.y * blockDim.x;//hangzhuxu 
    int col_idx = threadIdx.y + threadIdx.x * blockDim.y;//liezhuxu
    tile2[row_idx] = row_idx;
    
    __syncthreads();
    out[col_idx] = tile2[col_idx];
}

__global__ void set_col_read_row_rectangle(int *out){
    extern __shared__ int tile2[];
    int row_idx = threadIdx.x + threadIdx.y * blockDim.x;//hangzhuxu 
    int col_idx = threadIdx.y + threadIdx.x * blockDim.y;//liezhuxu
    tile2[col_idx] = row_idx;
    
    __syncthreads();
    out[row_idx] = tile2[row_idx];
}

__global__ void set_row_read_col_rectangle_pad(int* out){
    extern __shared__ int tile_pad[];
    int idx = threadIdx.x + threadIdx.y * (blockDim.x);
    int t_x = idx/blockDim.y;
    int t_y = idx%blockDim.y;
    int row_idx = threadIdx.x + threadIdx.y * (blockDim.x+1);
    // int col_idx = threadIdx.y + threadIdx.x * (blockDim.x+1);
    int col_idx = t_x + t_y * (blockDim.x+1);
    tile_pad[row_idx] = idx;

    __syncthreads();
    
    out[idx] = tile_pad[col_idx];
}

void test_rect(){
    const int y = 16 ,  x = 32;
    int *out_dev,*out_host;
    size_t array_size = x*y ,array_bytes = array_size*sizeof(int);
    size_t pad_row_size = (x+1)*y , pad_row_bytes = pad_row_size * sizeof(int);
    CHECK(cudaMalloc(&out_dev, array_bytes));
    CHECK(cudaMallocHost(&out_host, array_bytes));

    // set_row_read_col_rectangle<<<1,dim3(x,y),array_bytes>>>(out_dev);
    // cudaDeviceSynchronize();
    // cudaMemcpy(out_host, out_dev, array_bytes, cudaMemcpyDeviceToHost);
    // util::print_matrix(out_host, 32, 16);
    set_row_read_col_rectangle<<<1,dim3(x,y),array_bytes>>>(out_dev);
    cudaDeviceSynchronize();
    set_row_read_col_rectangle_pad<<<1,dim3(x,y),pad_row_bytes>>>(out_dev);
    cudaDeviceSynchronize();
            // cudaMemcpy(out_host, out_dev, array_bytes, cudaMemcpyDeviceToHost);
    // util::print_matrix(out_host, 32, 16);
}
__global__ void transpose_naive(int *out, int *in, const int nx, const int ny) {
 // matrix coordinate (ix,iy)
 unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
 unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;
 // transpose with boundary test
 if (ix < nx && iy < ny) {
 out[ix*ny+iy]= in[iy*nx+ix];
 }
}

__global__ void transpose_with_share(int* in ,int *out , int m , int n){
    __shared__ int trans_cache[32][32];
    int i_x = blockDim.x * blockIdx.x + threadIdx.x;
    int i_y = blockDim.y * blockIdx.y + threadIdx.y;
    if(i_x >= m || i_y >=n) return;
    int g_idx = i_y * m + i_x;

    int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    int b_row = bidx / blockDim.y;
    int b_col = bidx % blockDim.y;

    int o_x = blockIdx.y * blockDim.y + b_col;
    int o_y = blockIdx.x * blockDim.x + b_row;

    int o_idx = o_y * n + o_x;
    trans_cache[threadIdx.y][threadIdx.x] =  in[g_idx];
    __syncthreads();
    out[o_idx] = trans_cache[b_col][b_row] ;
}
__global__ void transpose_with_share_pad(int* in ,int *out , int m , int n){
    __shared__ int trans_cache[32][33];
    int i_x = blockDim.x * blockIdx.x + threadIdx.x;
    int i_y = blockDim.y * blockIdx.y + threadIdx.y;
    if(i_x >= m || i_y >=n) return;
    int g_idx = i_y * m + i_x;

    int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    int b_row = bidx / blockDim.y;
    int b_col = bidx % blockDim.y;

    int o_x = blockIdx.y * blockDim.y + b_col;
    int o_y = blockIdx.x * blockDim.x + b_row;

    int o_idx = o_y * n + o_x;
    trans_cache[threadIdx.y][threadIdx.x] =  in[g_idx];
    __syncthreads();
    out[o_idx] = trans_cache[b_col][b_row] ;
}

void test_transpose(){
    int* in_host,*out_host,*in_dev,*out_dev;
    int m = 128 , n = 128;
    dim3 block_dim(32,32);
    dim3 grid_dim(m/block_dim.x,n/block_dim.y);
    int array_size = m*n , array_bytes = array_size * sizeof(int);
    CHECK(cudaMallocHost(&in_host, array_bytes));
    CHECK(cudaMallocHost(&out_host, array_bytes));
    CHECK(cudaMalloc(&in_dev, array_bytes));
    CHECK(cudaMalloc(&out_dev, array_bytes));
    
    util::init_array_int(in_host, array_size);
    cudaMemcpy(in_dev, in_host, array_bytes, cudaMemcpyHostToDevice);
    transpose_naive<<<grid_dim,block_dim>>>(out_dev,in_dev,m,n);
    cudaDeviceSynchronize();
    transpose_with_share_pad<<<grid_dim,block_dim>>>(in_dev, out_dev, m, n);
    cudaDeviceSynchronize();
    transpose_with_share<<<grid_dim,block_dim>>>(out_dev,in_dev,m,n);
    cudaDeviceSynchronize();
    
    cudaMemcpy(out_host, out_dev, array_bytes, cudaMemcpyDeviceToHost);

    // std::cout<<"origin matirx:"<<std::endl;
    // util::print_matrix(in_host, 2, n);
    // std::cout<<"trans matrix:" <<std::endl;
    // util::print_matrix(out_host,2, n);
}

int main (){
    //test_row_and_col();
    // test_rect();
    test_transpose();
}