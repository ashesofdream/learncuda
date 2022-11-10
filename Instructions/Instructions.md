# Instructions
## Intrinsic Function
1. 固有函数，即相对于C标准库的函数（例如sqrt,exp），为GPU特有的函数，只能通过设备码访问（device code）。例如，`__dsqrt__rn`双精度的开方，`__fdividef`单精度的除法，都会有一定的加速。此外，某些标准库中的函数如三角函数实际上也已经被显卡用硬件实现了
2. `atomicAdd(int *M, int V)`原子加，`int atomicExch(int *M, int V)`原子将V写入M。显然的这些原子操作都会降低性能，没必要的时候不用。
## Single-Precision vs Double-Precision
1. 单精度和多精度的性能差距很大，复制会消耗多一倍的时间，执行也会慢很多
## Standard vs. Intrinsic Functions
1. 固有函数的速度快很多很多，但精度问题会更大，由于去特殊的加速，固有函数和cuda标准库函数以及cpu库函数计算出来的结果都是不一样的，以下是书上`__powf`的一个例子。
    ```
    Host calculated 66932852.000000
    Standard Device calculated 66932848.000000
    Intrinsic Device calculated 66932804.000000
    
    Host equals Standard? No diff=4.000000e+00
    Host equals Intrinsic? No diff=4.800000e+01
    Standard equals Intrinsic? No diff=4.400000e+01

    Mean execution time for standard function powf: 47 ms
    Mean execution time for intrinsic function __powf: 2 ms
    ```
    虽然精度问题更大了，但事实上还是一个数量级内的误差。
2. 通过编译标志代替手写固有函数如`--fmad=true`合并乘法和加法在一个周期内完成，`--use_fast_math=true`替代数学库中的函数为高效低精度的，`--pret-sqrt=true`更高精度更慢的开方，更多的参考书中P313。此外可以使用固有函数`__fmul`组织MAD(multiply and add)优化。
3. `__fmul`,`__fadd`,`__fsub`,这些函数后都会加上`_rn`最近邻取近似（默认），`_rz`向0取值，`_ru`向正无穷取值，`__d`向负无穷取值。使用时如`__fmul_rz`
## Understanding Atomic Instructions
1. 原子操作其实可以被CAS(compare-and_swap，java高并发中也很常见)操作重实现。CAS操作流程即比较目标地址上的值和预期旧值是否相等，相等则将新值赋上去，不相同则不赋值。无论是否赋值，最后都返回目标内存上的原值
2. 常用函数：`atomicAdd`,`atomicSub`,`atomicExch`,`atomicMin`,`atomicMax`,`atomicInc/Dec/And/Or/Xor/CAS`
3. 原子操作性能差的原因:  
    a. 当对共享内存或者全局内存使用原子操作，不会使用缓存，而是会立刻写回全局的。
    b. 如CAS实现加法一样，发生冲突时，会造成多次访存
    c. 在同一个warp中，访问同一个内存，极端情况下会耗费WarpSize*OperateTime的时间
4. 除了`atomicExch`,`atomicAdd`有float版本，都只有int版本，需要手动实现。具体流程，将float存储在uint格式(使用`__float2uint_rn()`)，使用int版的CAS实现需要的功能。Sample：
    ```
    __device__ float myAtomicAdd(float *address, float incr) {
    // Convert address to point to a supported type of the same size
    unsigned int *typedAddress = (unsigned int *)address;
    // Stored the expected and desired float values as an unsigned int
    float currentVal = *address;
    unsigned int expected = __float2uint_rn(currentVal);
    unsigned int desired = __float2uint_rn(currentVale + incr);
    int oldIntValue = atomicCAS(typedAddress, expected, desired);
    while (oldIntValue != expected) {
        expected = oldIntValue;
        /* 
        * Convert the value read from typedAddress to a float, increment,
        * and then convert back to an unsigned int
        */
        desired = __float2uint_rn(__uint2float_rn(oldIntValue) + incr);
        oldIntValue = atomicCAS(typedAddress, expected, desired);
        }
        return __uint2float_rn(oldIntValue);
    }
    ```
