//
// Created by zzq on 22-10-27.
//

#ifndef LEARNCUDA_UTIL_H
#define LEARNCUDA_UTIL_H

#include <cstdio>
#include <tuple>
#include <array>
#include <cstdlib>

#define CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}
namespace util {
    void init_array_int(int *ip , int size);
    void print_matrix(int *array , int m , int n);
    template<typename T>
    std::array<T *,2> cudaMallocHostAndDev(size_t arr_bytes){
      T * host_p,* dev_p;
      cudaMallocHost(&host_p, arr_bytes);
      cudaMalloc(&dev_p, arr_bytes);
      return {host_p,dev_p};
    };
};


#endif //LEARNCUDA_UTIL_H
