# Streams and Concurrency
## DEFINITION
1. stream封装了一串序列化（按Host指定）的可异步执行的CUDA操作
2. 利用stream可完成grid之间的并行，实现双重缓冲，管线等。
## CUDA STREAM
1. 默认不手动声明的时候默认使用NULL stream
2. 使用stream可以overlap很多操作，包括主机和GPU之间的异步计算以及数据传输，GPU的并发计算。
3. 异步传输数据必须主机pined内存（即内存不可被换页）
4. 两个检查stream当前情况的函数：
    ```
    //同步阻塞直到结束
    cudaError_t cudaStreamSynchronize(cudaStream_t stream);
    
    //返回cudaSuccess或者cudaErrorNotReady
    cudaError_t cudaStreamQuery(cudaStream_t stream);
    ```
5. 主机和GPU之间由于是使用PCIE全双工传输，因此传输数据时可同时并行主机复制到GPU，GPU复制到主机。
## Stream Scheduling
1. False Dependency 问题，即Kepler之前的架构由于其显卡在物理上上只有一条队列，而每个流都是以一个整体进入队列中，流中的每一个操作都必须按序执行（即后面的操作依赖于前面的操作完成），因此导致，后面的流也被堵塞在队列中，无法并发，只有第一个流的最后一个任务和第二个流的第一个操作可以并发（因为无依赖关系）。  
解决方法:
    >a. Kepler架构开始利用HyperQ技术，物理上创建了16个队列（之前只有一个），使多个Grid可以并行。  
    b.不使用流，依次输入原本流中的每一个操作，例如流1的操作1，流2的操作1，流1的操作2，流2的操作2，也可完成并发。
3. `cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags,
 int priority)`可设置线程的流的优先权，值越小优先权越高  
 `cudaError_t cudaDeviceGetStreamPriorityRange(int *leastPriority,
 int *greatestPriority)`返回优先权的属性  
4. 事件:
    ```
    // create two events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // record start event on the default stream
    cudaEventRecord(start);
    // execute kernel
    kernel<<<grid, block>>>(arguments);
    // record stop event on the default stream
    cudaEventRecord(stop);
    // wait until the stop event completes
    cudaEventSynchronize(stop);
    // calculate the elapsed time between two events
    float time;
    cudaEventElapsedTime(&time, start, stop);
    ...clean...
    ```
    特别的`cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0);`可以设置事件存放的流
## Stream Synchronization
1. 流可以分为异步的流(non-NULL流)和同步的流(NULL流)，异步的流又可分为Blocking Stream 和Non-blocking Stream.
2. `cudaStreamCreate`创建的流仍然默认为blkocking stream，需要通过`cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags);`设置为`cudaStreamNonBlocking`（默认为NULL）
3. 同步分为显式和隐式，显式有：
    ```
    cudaDeviceSynchronize, 
    cudaStreamSynchronize,
    cudaEventSynchronize.
    cudaStreamWaitEvent
    ```
    特别的`cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event);`将会等待一个事件触发时，才会触发。
    隐式的主要是内存操作相关API，如cudaMalloc,cudaMemcpy(相同设备上copy),memset
4. 事件的属性` cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags);`
    ```
    cudaEventDefault 线程使用cudaEventBlockingSync会旋转检查，直至事件放行
    cudaEventBlockingSync 线程使用cudaEventBlockingSync会进入阻塞态并睡眠，事件放行后唤醒线程
    cudaEventDisableTiming 禁止时间计算功能以提升性能 
    cudaEventInterprocess 表示事件将会用作进程间交互的事件
    ```
##  Concurrency and kernel excution
1. 使用OpenMP并行的创建任务加速。CMAKE添加
    ```
    FIND_PACKAGE(OpenMP REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    ```
    然后
    ```
    omp_set_num_threads(n_streams);
    #pragma omp parallel
    {
        int i = omp_get_thread_num()
        ....
    }
    ```
    与cuda类似
2. 环境变量`CUDA_DEVICE_MAX_CONNECTIONS`可设置最大的并发数（等于队列数）。当该变量大于物理的队列数时，也会造成`FALSE DEPENDECY`.
3. Default Stream (NULL STREAM) 默认流会阻塞其它所有流的运行。
4. `cudaStreamWaitEvent(等待的流s，等待的事件，标志FLAG)` 可用于阻塞s，等待事件发生，可对同一流设置多个阻塞事件。
5. Overlap 传输数据和计算的时间，分配内存时要用`cudaMallocAsync(...)`,具体实现有广度优先的深度优先的方法。如果计算的内核只有一个，那两种实际性能都没什么区别。但是如果计算的内核数大于1，且设备只有一个物理队列，则最好选择广度优先的执行方法。
6. CallBack，流的回调函数，通过`cudaStreamAddCallback(cudaStream_t stream,
 cudaStreamCallback_t callback, void *userData, unsigned int flags)` 设置，其中flags没用默认为0预留，回调函数的格式为`void CUDART_CB my_callback(cudaStream_t stream, cudaError_t status, void *data) `。回调函数有两点要注意：
    > 1.不能调用cudaapi  
    2.不能尝试同步