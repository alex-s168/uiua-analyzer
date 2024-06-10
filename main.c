#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>

typedef struct {
    void*    allocationPtr;
    int64_t* alignedPtr;
    size_t   elemsOff;
    size_t   sizes;
    size_t   strides;
} I64Arr;

#define EXPAND(arr) arr.allocationPtr,arr.alignedPtr,arr.elemsOff,arr.sizes,arr.strides

extern double fn_$__arr_start_int_end__maybe_(void*,int64_t*,size_t,size_t,size_t);

extern void _$_rt_panic(uint64_t block, uint64_t inst) {
    printf("panic at %lu : %lu\n", block, inst);
    exit(1);
}

int main() {
    I64Arr arr;
    arr.sizes = 4;
    arr.strides = 1;
    arr.elemsOff = 0;
    arr.allocationPtr = aligned_alloc(64, sizeof(int64_t) * arr.sizes);
    arr.alignedPtr = arr.allocationPtr;

    arr.alignedPtr[0] = 85;
    arr.alignedPtr[1] = 105;
    arr.alignedPtr[2] = 117;
    arr.alignedPtr[3] = 97;

    double sum = fn_$__arr_start_int_end__maybe_(EXPAND(arr));

    printf("%f\n", sum);

    return 0;
}
