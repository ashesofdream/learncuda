# WarpDataSharing
1. `__shfl(int var , int src_thread_idx , int size)`  
var是共享的变量，src_thread_idx是要共享哪个线程的变量,size是线程束的大小
2. `__shfl_up(int var, int offset,int warpsize)`  
`__shfl_down(int var, int offset,int warpsize)`  
up是获取获取当前lane_idx - offset线程的变量值  
down相反.  
3. `__shfl_xor(int var , int mask, int warpsize)`  
获取laneidx 为 lane_idx^waprsize的var变量.
4. 使用范例：数组求和 ArrayReduce