#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>

typedef struct {
    void *alloc;
    int64_t *aligned;
    size_t elemsOff;
    size_t sizes[2];
    size_t strides[2];
} I64A2;

extern I64A2 fn_$__arr_start_arr_start_int_end__maybe__end__maybe_vaoff(void*,int64_t*,size_t,size_t,size_t,size_t,size_t);

int main() {
    I64A2 a;
    a.alloc = calloc(3 * 3, sizeof(int64_t));
    a.aligned = a.alloc;
    a.elemsOff = 0;

    a.sizes[0] = 3;
    a.sizes[1] = 3;
    a.strides[0] = 3;
    a.strides[1] = 1;

    a.aligned[3* 0+0] = 1;
    a.aligned[3* 1+1] = 2;
    a.aligned[3* 2+2] = 3;

    I64A2 res = fn_$__arr_start_arr_start_int_end__maybe__end__maybe_vaoff(a.alloc,a.aligned,a.elemsOff,a.sizes[0],a.sizes[1],a.strides[0],a.strides[1]);
    int64_t *elems = res.aligned + res.elemsOff;
    for (size_t i = 0; i < res.sizes[0]; i ++) {
        for (size_t j = 0; j < res.sizes[1]; j ++) {
            int64_t elem = elems[i * res.strides[0] + j * res.strides[1]];
            printf("%lld ", elem);
        }
        printf("\n");
    }

    return 0;
}
