#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>

#define GenArrType(dim, of) \
    typedef struct { \
        void *alloc; \
        of *aligned; \
        size_t elemsOff; \
        size_t sizes[dim]; \
        size_t strides[dim]; \
    } Arr##dim_##of;

#define GenExtendScalar(ofTy, nameExt) \
extern Arr1_##ofTy _$_rt_extendScalar_##nameExt(Arr1_##ofTy base, size_t targetLen, ofTy fillWith) { \
    Arr1_##ofTy res; \
    res.alloc = calloc(targetLen, sizeof(ofTy)); \
    res.aligned = res.alloc; \
    res.elemsOff = 0; \
    res.sizes[0] = targetLen; \
    res.strides[0] = 1; \
    \
    size_t j = 0; \
    \
    ofTy *basePtr = base.aligned + base.elemsOff; \
    for (size_t i = 0; i < base.sizes[0]; i += base.strides[0]) { \
        res.aligned[j] = basePtr[i]; \
        j ++; \
    } \
    \
    for (; j < targetLen; j ++) { \
        res.aligned[j] = fillWith; \
    } \
    \
    return res; \
}

#define GenExtendRepeat(ofTy, nameExt) \
extern Arr1_##ofTy _$_rt_extendRepeat_##nameExt(Arr1_##ofTy base, size_t targetLen, Arr1_##ofTy fillWith) { \
    Arr1_##ofTy res; \
    res.alloc = calloc(targetLen, sizeof(ofTy)); \
    res.aligned = res.alloc; \
    res.elemsOff = 0; \
    res.sizes[0] = targetLen; \
    res.strides[0] = 1; \
    \
    size_t j = 0; \
    \
    ofTy *basePtr = base.aligned + base.elemsOff; \
    for (size_t i = 0; i < base.sizes[0]; i += base.strides[0]) { \
        res.aligned[j] = basePtr[i]; \
        j ++; \
    } \
    \
    ofTy *fillWithPtr = fillWith.aligned + fillWith.elemsOff; \
    size_t fillWithLen = fillWith.sizes[0]; \
    for (size_t x = 0; j < targetLen; j ++) { \
        res.aligned[j] = fillWithPtr[x % fillWithLen]; \
        x ++; \
    } \
    \
    return res; \
}

GenArrType(1, uint8_t)
GenExtendScalar(uint8_t, byte)
GenExtendRepeat(uint8_t, byte)

GenArrType(1, int64_t)
GenExtendScalar(int64_t, int)
GenExtendRepeat(int64_t, int)

GenArrType(1, double)
GenExtendScalar(double, float)
GenExtendRepeat(double, float)

GenArrType(1, size_t)
GenExtendScalar(size_t, size)
GenExtendRepeat(size_t, size)


GenArrType(2, int64_t)

// TODO: somehow tell mlir to use c struct arg

extern Arr2_int64_t fn_$__arr_start_arr_start_int_end__maybe__end__maybe_vaoff(Arr2_int64_t);

int main() {
    Arr2_int64_t a;
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

    Arr2_int64_t res = fn_$__arr_start_arr_start_int_end__maybe__end__maybe_vaoff(a);
    int64_t *elems = res.aligned + res.elemsOff;
    for (size_t i = 0; i < res.sizes[0]; i ++) {
        for (size_t j = 0; j < res.sizes[1]; j ++) {
            int64_t elem = elems[i * res.strides[0] + j * res.strides[1]];
            printf("%lld ", elem);
        }
        printf("\n");
    }
    free(res.alloc);

    return 0;
}
