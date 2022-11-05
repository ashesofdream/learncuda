# ConstantMemory

1. 常量内存位于DRAM上，但是在片上都会有缓存，不超过64KB对于每一个流处理器
2. 常量内存的优化方法不同，要尽可能的访问内存的相同地址，每次访问不同的地址会导致一个线程束中的时间消耗线性放大。
3. 生命周期与应用的生命周期相同。
4. 对broadcast access pattern 优化好(broadcast access pattern,同一线程束中的线程同时引用相同的常量内存地址)

# Compare With ReadOnly Cache
1. Keplper架构的GPU主要是用显卡的纹理管线来作为全局内存的一个只读缓存，每个流处理器（SM）有48KB的缓存(常量内存64KB)，读取的数据粒度是4字节.一般情况下比离散的读L1内存会更快
2. 两种方法实现：  
    a. __ldg  

        ```
        __global__ void kernel(float* output, float* input) {
        ...
        output[idx] += __ldg(&input[idx]);
        ...
        }
        ```  
    b. const \<typename> \_\_strict__
    ```
    void kernel(float* output, const float* __restrict__ input) {
        ...
        output[idx] += input[idx];
    }

    ```
3. 相比于常量内存读的小而且要求多个线程读的同一位置才能保持高性能，只读缓存读的大而且对读的位置没那么高的要求
4. 满足broadcast access pattern的情况下，还是ConstantMemory会稍微好一点。