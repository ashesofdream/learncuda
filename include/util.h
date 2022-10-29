//
// Created by zzq on 22-10-27.
//

#ifndef LEARNCUDA_UTIL_H
#define LEARNCUDA_UTIL_H

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
};


#endif //LEARNCUDA_UTIL_H
